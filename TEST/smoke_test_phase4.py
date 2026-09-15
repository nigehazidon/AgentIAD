"""Phase 4 smoke test: Normal Reference Evidence Tool on bottle + cable.

Structural checks (no model needed):

  1. the tool module carries no ground-truth dependency (static token scan);
  2. the Mode A / Mode B prompts differ **only** in the ``You are given:``
     reference-count line, and attach exactly 2 images (A) / 4 images (B);
  3. retrieval over **every** test image of both categories: Top-1 and Top-3
     both resolve, references are never duplicated, the query never appears
     among its own references, Top-1 is always a subset of Top-3, and the
     ranking is deterministic across repeated calls;
  4. the schema validator accepts well-formed replies (plain / fenced / with
     prose) and rejects malformed ones, forbidden localisation content, and
     decision keys;

Model checks (a few query images per category, both modes):

  5. every record parses (``parse_ok``) and validates against the exact schema;
  6. both modes cover the same query set and share identical decoding settings —
     only the number of references differs;
  7. the saved prompt is the expected one, and no similarity / score / label
     value ever reaches it;
  8. reference paths are traceable: on disk, inside ``train/good``, inside the
     frozen Phase-2 24-shot pool;
  9. re-running one Mode A query reproduces the identical raw output (greedy
     decoding);
 10. the explicit-references input path agrees with internal retrieval;
 11. every leakage flag on every record is the safe value;
 12. ``results/phase2/`` and ``results/phase3/`` are byte-for-byte untouched.

Usage::

    .../python smoke_test_phase4.py                 # full check
    .../python smoke_test_phase4.py --skip-model    # structural + retrieval only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "TEST") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "TEST"))

from normal_reference_tool import (  # noqa: E402
    ABNORMALITY_VALUES,
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUT_ROOT,
    DIFFERENCE_KEYS,
    GLOBAL_CONSISTENCY_VALUES,
    MODE_A_K,
    MODE_B_K,
    OUTPUT_KEYS,
    NormalReferenceEvidenceTool,
    ReferenceToolConfig,
    build_messages,
    parse_reference_output,
    prompt_text_of,
    system_prompt_for,
    write_reference_record,
)

PHASE2_ROOT = Path("/data/pfy/AgentIAD/results/phase2")
PHASE3_ROOT = Path("/data/pfy/AgentIAD/results/phase3")

#: Source patterns that would indicate the tool *reads* the annotation tree.
#: Written as path-shaped regexes so that the tool's own no-leakage flag names
#: (``"ground_truth_used_in_prompt"``) do not trip the check — only a real path
#: segment or join does.
FORBIDDEN_SOURCE_PATTERNS = (
    ("ground_truth_path", re.compile(r"ground_truth\s*[\"'/]")),
    ("gt_mask_path", re.compile(r"gt_mask\s*[\"'/]")),
    ("mask_png", re.compile(r"mask\.png")),
)

#: Resolved-path fragments that must never be touched at runtime.
FORBIDDEN_PATH_FRAGMENTS = ("ground_truth", "gt_mask", "_mask.png")

#: Every leakage flag that must hold on every record.
SAFE_LEAKAGE_FLAGS = (
    "references_all_from_train_good",
    "references_in_frozen_phase2_pool",
    "query_is_test_image",
)
UNSAFE_LEAKAGE_FLAGS = (
    "query_in_reference_pool",
    "query_among_references",
    "test_images_in_reference_pool",
    "ground_truth_used_in_retrieval",
    "ground_truth_used_in_prompt",
    "ground_truth_used_in_decision",
    "anomaly_label_used",
    "patchcore_artifact_used",
    "similarity_used_as_decision",
    "similarity_used_in_prompt",
)


def manifest(root: Path) -> dict:
    """``{relpath: [size, mtime_ns]}`` for every file under ``root``."""
    if not root.is_dir():
        return {}
    return {
        str(p.relative_to(root)): [p.stat().st_size, p.stat().st_mtime_ns]
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def query_images(data_root: Path, category: str, per_type: int) -> list[Path]:
    """Deterministically pick ``per_type`` test images from every defect folder.

    Round-robins the (sorted) test sub-folders so each defect type — and the
    ``good`` folder — is represented.
    """
    test_dir = data_root / category / "test"
    picked: list[Path] = []
    for sub in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        images = sorted(sub.glob("*.png"))
        picked.extend(images[:per_type])
    return picked


# --------------------------------------------------------------- structural
def check_static_no_gt_leakage() -> dict:
    """The tool module must contain no code path into the annotation tree."""
    src = (_REPO_ROOT / "normal_reference_tool.py").read_text()
    hits = [name for name, pattern in FORBIDDEN_SOURCE_PATTERNS if pattern.search(src)]
    return {"clean": not hits, "forbidden_patterns_found": hits}


class _StubChat:
    """A stand-in for Qwen that replays a fixed reply (no GPU needed)."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    async def ainvoke(self, messages, **kwargs):
        import types
        self.calls += 1
        assert kwargs.get("do_sample") is False, "decoding must be greedy"
        return types.SimpleNamespace(content=self.reply)


_STUB_REPLY = json.dumps({
    "global_consistency": "high",
    "meaningful_differences": [],
    "normality_evidence": ["matches the references"],
    "comparison_summary": "consistent with the normal references.",
})


def check_runtime_no_gt_read(data_root: Path, categories: list[str],
                             per_type: int) -> dict:
    """Prove at the I/O level that no annotation file is ever touched.

    Every file open / glob / rglob the tool performs while retrieving references
    and comparing a query is recorded; any path inside the annotation tree, or
    any mask file, is a leak. A stub model replays a canned reply, so this runs
    without the GPU and audits the real data path.
    """
    import builtins
    import types

    touched: list[str] = []
    real_open, real_path_open = builtins.open, Path.open
    real_glob, real_rglob = Path.glob, Path.rglob

    def _record(target) -> None:
        touched.append(str(target))

    def open_wrap(file, *a, **kw):
        _record(file)
        return real_open(file, *a, **kw)

    def path_open_wrap(self, *a, **kw):
        _record(self)
        return real_path_open(self, *a, **kw)

    def glob_wrap(self, pattern):
        _record(self)
        return real_glob(self, pattern)

    def rglob_wrap(self, pattern):
        _record(self)
        return real_rglob(self, pattern)

    builtins.open, Path.open = open_wrap, path_open_wrap
    Path.glob, Path.rglob = glob_wrap, rglob_wrap

    try:
        results: dict = {}
        for category in categories:
            stub = _StubChat(_STUB_REPLY)
            tool = NormalReferenceEvidenceTool(
                ReferenceToolConfig(
                    reference_k=MODE_B_K, data_root=str(data_root),
                    max_parse_retries=0),
                chat=stub)
            queries = query_images(data_root, category, per_type)
            records = asyncio.run(tool.compare_many(
                [{"query_image": str(q), "category": category} for q in queries]))
            results[category] = {
                "n_queries": len(queries),
                "stub_calls": stub.calls,
                "all_parse_ok": all(r["parse_ok"] for r in records),
                "leakage_flags_safe": all(
                    _leakage_safe(r["leakage"]) for r in records),
            }
    finally:
        builtins.open, Path.open = real_open, real_path_open
        Path.glob, Path.rglob = real_glob, real_rglob

    offenders = sorted({
        p for p in touched
        if any(frag in p for frag in FORBIDDEN_PATH_FRAGMENTS)
    })
    checked = sorted({
        p for p in touched
        if p.endswith(".png") or "MVTec-AD" in p
    })
    return {
        "no_annotation_file_touched": not offenders,
        "offending_paths": offenders,
        "n_paths_touched": len(touched),
        "sample_paths_touched": checked[:6],
        "per_category": results,
    }


def check_prompt_modes() -> dict:
    """The two modes must differ only in the reference-count line."""
    import normal_reference_tool as N

    a, b = system_prompt_for(MODE_A_K), system_prompt_for(MODE_B_K)
    identical_body = (
        a.replace(N._REFERENCE_LINE_BY_K[MODE_A_K], "")
        == b.replace(N._REFERENCE_LINE_BY_K[MODE_B_K], "")
    )
    return {
        "identical_outside_reference_line": identical_body,
        "modes_differ": a != b,
        "both_start_with_header": a.startswith(N._PROMPT_HEADER)
        and b.startswith(N._PROMPT_HEADER),
        "both_end_with_tail": a.endswith(N._PROMPT_TAIL) and b.endswith(N._PROMPT_TAIL),
        # The count-bearing lines must live only in _REFERENCE_LINE_BY_K, so the
        # header and tail are genuinely mode-independent.
        "header_has_no_count_line": not any(
            line in N._PROMPT_HEADER for line in N._REFERENCE_LINE_BY_K.values()),
        "tail_has_no_count_line": not any(
            line in N._PROMPT_TAIL for line in N._REFERENCE_LINE_BY_K.values()),
        "a_attaches_two_images": sum(
            1 for i in build_messages("d:q", ["d:r"], MODE_A_K)[1]["content"]
            if i.get("type") == "image_url") == 2,
        "b_attaches_four_images": sum(
            1 for i in build_messages("d:q", ["d:r1", "d:r2", "d:r3"], MODE_B_K)[1]["content"]
            if i.get("type") == "image_url") == 4,
        "reference_count_mismatch_raises": _raises(
            lambda: build_messages("d:q", ["d:r"], MODE_B_K)),
        "bad_k_raises": _raises(lambda: system_prompt_for(2)),
    }


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


def check_prompt_text_has_no_similarity(references: list[dict]) -> dict:
    """No similarity value, score or anomaly label may appear in the prompt."""
    messages = build_messages(
        "data:image/png;base64,QQ==", ["data:image/png;base64,QQ=="] * len(references),
        len(references))
    text = prompt_text_of(messages)
    leaks = []
    for ref in references:
        sim = ref["similarity"]
        if str(sim) in text or f"{float(sim):.6f}" in text or f"{float(sim):.4f}" in text:
            leaks.append(sim)
    expected = "\n".join(
        [system_prompt_for(len(references)), "Query image:"]
        + [f"Known NORMAL reference image {i} of {len(references)}:"
           for i in range(1, len(references) + 1)]
    )
    return {
        "prompt_is_exactly_prompt_plus_labels": text == expected,
        "no_similarity_value_in_prompt": not leaks,
        "leaked_values": leaks,
    }


def check_schema_validator() -> dict:
    good = json.dumps({
        "global_consistency": "high",
        "meaningful_differences": [
            {"description": "the cap sits slightly higher",
             "abnormality": "unlikely",
             "evidence": "the reference images show the same cap at varying heights"},
        ],
        "normality_evidence": ["same surface gloss as the references"],
        "comparison_summary": "the query matches the normal references.",
    })
    bbox_variant = json.dumps({
        "global_consistency": "high",
        "meaningful_differences": [
            {"description": "a mark at bbox [10, 20, 30, 40]",
             "abnormality": "likely", "evidence": "a dark line"},
        ],
        "normality_evidence": [],
        "comparison_summary": "x",
    })
    coord_variant = good.replace(
        '"the cap sits slightly higher"',
        '"a mark at coordinates 12, 40"')
    prob_variant = good.replace(
        '"the query matches the normal references."',
        '"the anomaly probability is 0.8."')
    key_variant = json.dumps({
        "global_consistency": "high",
        "meaningful_differences": [
            {"description": "d", "abnormality": "likely", "evidence": "e",
             "bbox": [1, 2, 3, 4]},
        ],
        "normality_evidence": [],
        "comparison_summary": "x",
    })
    threshold_variant = good.replace(
        '"the query matches the normal references."',
        '"below the decision threshold."')

    cases = (
        ("plain", good, True),
        ("fenced", f"```json\n{good}\n```", True),
        ("prose_wrapped", f"Sure, here it is:\n{good}\nHope that helps.", True),
        ("bad_consistency_enum", good.replace('"high"', '"very high"'), False),
        ("bad_abnormality_enum", good.replace('"unlikely"', '"maybe"'), False),
        ("missing_key", '{"global_consistency":"high"}', False),
        ("empty_summary",
         good.replace('"the query matches the normal references."', '""'), False),
        ("empty_normality_entry",
         good.replace('["same surface gloss as the references"]', '[""]'), False),
        ("bbox_in_description", bbox_variant, False),
        ("coordinates_in_description", coord_variant, False),
        ("anomaly_probability", prob_variant, False),
        ("decision_threshold", threshold_variant, False),
        ("bbox_key_in_difference", key_variant, False),
        ("not_json", "The query looks normal to me.", False),
        ("empty", "", False),
    )
    return {label: (parse_reference_output(text)[0] is not None) == want
            for label, text, want in cases}


# ---------------------------------------------------------------- retrieval
def check_retrieval(tool: NormalReferenceEvidenceTool, categories: list[str]) -> dict:
    """Retrieval invariants over every test image of both categories."""
    checks: dict = {}
    for category in categories:
        pool = tool.reference_pool(category)
        pool_set = {p.resolve() for p in pool}
        good_dir = (tool.data_root / category / "train" / "good").resolve()

        images = sorted((tool.data_root / category / "test").rglob("*.png"))
        n_ok = n_dup = n_self = n_subset = n_deterministic = n_rank = 0
        failures: list[dict] = []

        for query in images:
            refs3 = tool.retrieve(query, category, MODE_B_K)
            refs1 = tool.retrieve(query, category, MODE_A_K)
            paths3 = [r["image_path"] for r in refs3]
            paths1 = [r["image_path"] for r in refs1]

            ok = (len(paths3) == MODE_B_K and len(paths1) == MODE_A_K)
            dup = len(set(paths3)) != MODE_B_K or len(set(paths1)) != MODE_A_K
            self_hit = str(query.resolve()) in set(paths3) | set(paths1)
            subset = paths1[0] == paths3[0]
            again = [r["image_path"] for r in tool.retrieve(query, category, MODE_B_K)]
            deterministic = again == paths3
            sims = [r["similarity"] for r in refs3]
            rank = all(sims[i] >= sims[i + 1] for i in range(len(sims) - 1))

            n_ok += ok
            n_dup += dup
            n_self += self_hit
            n_subset += subset
            n_deterministic += deterministic
            n_rank += rank
            if not (ok and not dup and not self_hit and subset and deterministic and rank):
                failures.append({"query": str(query)})

        checks[f"{category}_n_test_images"] = len(images)
        checks[f"{category}_all_topk_resolve"] = n_ok == len(images)
        checks[f"{category}_no_duplicate_references"] = n_dup == 0
        checks[f"{category}_query_never_among_references"] = n_self == 0
        checks[f"{category}_top1_is_subset_of_top3"] = n_subset == len(images)
        checks[f"{category}_deterministic"] = n_deterministic == len(images)
        checks[f"{category}_sorted_by_similarity"] = n_rank == len(images)
        checks[f"{category}_pool_size"] = len(pool)
        checks[f"{category}_pool_unique"] = len(pool_set) == len(pool)
        checks[f"{category}_pool_all_in_train_good"] = all(
            p.resolve().is_relative_to(good_dir) for p in pool)
        checks[f"{category}_failures"] = failures[:5]

    return checks


# -------------------------------------------------------------------- model
def run_model_checks(args, tools: dict[int, NormalReferenceEvidenceTool],
                     queries: dict[str, list[Path]]) -> tuple[dict, dict[int, list[dict]]]:
    """Run both modes over the query set. Returns ``(checks, records)``."""
    checks: dict = {}
    records: dict[int, list[dict]] = {}

    async def run() -> dict:
        out: dict[int, list[dict]] = {}
        for k, tool in tools.items():
            requests = [{"query_image": str(q), "category": c}
                        for c, qs in queries.items() for q in qs]
            out[k] = await tool.compare_many(requests)

        # reproducibility: one Mode A query, twice, sequentially
        category, qs = next(iter(queries.items()))
        a = tools[MODE_A_K]
        first = await a.compare(str(qs[0]), category)
        second = await a.compare(str(qs[0]), category)

        # explicit-references input must agree with internal retrieval
        explicit = tool_refs = a.retrieve(str(qs[0]), category, MODE_A_K)
        by_input = await a.compare(str(qs[0]), category, references=explicit)
        return {"records": out, "first": first, "second": second,
                "by_input": by_input, "explicit": tool_refs}

    out = asyncio.run(run())
    records = out["records"]

    for k, recs in records.items():
        tag = f"mode{'A' if k == MODE_A_K else 'B'}"
        checks[f"{tag}_n_records"] = len(recs)
        checks[f"{tag}_all_parse_ok"] = all(r["parse_ok"] for r in recs)
        checks[f"{tag}_all_vlm_called_once"] = all(r["vlm_calls"] == 1 for r in recs)
        checks[f"{tag}_reference_k_matches"] = all(r["reference_k"] == k for r in recs)
        checks[f"{tag}_n_references_matches"] = all(
            len(r["references"]) == k for r in recs)
        checks[f"{tag}_schema_valid"] = all(
            _schema_valid(r["parsed_output"]) for r in recs if r["parse_ok"])
        checks[f"{tag}_system_prompt_saved"] = all(
            r["system_prompt"] == system_prompt_for(k) for r in recs)
        checks[f"{tag}_no_similarity_in_prompt"] = all(
            not any(str(ref["similarity"]) in r["system_prompt"]
                    for ref in r["references"])
            for r in recs)
        checks[f"{tag}_references_traceable"] = all(
            _references_traceable(r) for r in recs)
        checks[f"{tag}_leakage_flags_safe"] = all(
            _leakage_safe(r["leakage"]) for r in recs)
        checks[f"{tag}_failures"] = [
            {"query": Path(r["query_image"]).name, "violations": r["violations"]}
            for r in recs if not r["parse_ok"]
        ]

    keys = {k: {(r["category"], r["query_image"]) for r in recs}
            for k, recs in records.items()}
    checks["both_modes_same_query_set"] = (
        keys[MODE_A_K] == keys[MODE_B_K] and len(keys[MODE_A_K]) > 0)

    sig_a = tools[MODE_A_K].config.decoding_signature()
    sig_b = tools[MODE_B_K].config.decoding_signature()
    sig_a.pop("system_prompt"), sig_b.pop("system_prompt")
    checks["identical_decoding_settings"] = sig_a == sig_b
    checks["decoding_settings"] = sig_a

    checks["reproducible"] = (
        out["first"]["raw_model_output"] == out["second"]["raw_model_output"]
        and out["first"]["parse_ok"] == out["second"]["parse_ok"])
    checks["explicit_references_agree_with_retrieval"] = (
        [r["image_path"] for r in out["by_input"]["references"]]
        == [r["image_path"] for r in out["explicit"]])
    checks["explicit_references_same_output"] = (
        out["by_input"]["parsed_output"] == out["first"]["parsed_output"])

    checks["sample_outputs"] = [
        {
            "mode": r["mode"],
            "category": r["category"],
            "query": Path(r["query_image"]).parent.name + "/" + Path(r["query_image"]).name,
            "references": [Path(x["image_path"]).name for x in r["references"]],
            "similarities": [x["similarity"] for x in r["references"]],
            "global_consistency": (r["parsed_output"] or {}).get("global_consistency"),
            "n_differences": len((r["parsed_output"] or {}).get("meaningful_differences", [])),
            "abnormalities": [d["abnormality"] for d in
                              (r["parsed_output"] or {}).get("meaningful_differences", [])],
        }
        for r in records[MODE_B_K][:4]
    ]
    return checks, records


def _schema_valid(parsed: dict | None) -> bool:
    if not isinstance(parsed, dict) or set(parsed) != set(OUTPUT_KEYS):
        return False
    if parsed["global_consistency"] not in GLOBAL_CONSISTENCY_VALUES:
        return False
    if not isinstance(parsed["normality_evidence"], list):
        return False
    if not isinstance(parsed["comparison_summary"], str) or not parsed["comparison_summary"]:
        return False
    for diff in parsed["meaningful_differences"]:
        if set(diff) != set(DIFFERENCE_KEYS):
            return False
        if diff["abnormality"] not in ABNORMALITY_VALUES:
            return False
        if not diff["description"] or not diff["evidence"]:
            return False
    return True


def _references_traceable(record: dict) -> bool:
    """Every reference must exist on disk, inside train/good, with a real score."""
    for ref in record["references"]:
        path = Path(ref["image_path"])
        if not path.is_file():
            return False
        if "train/good" not in path.as_posix():
            return False
        if not isinstance(ref["similarity"], float):
            return False
    return True


def _leakage_safe(leakage: dict) -> bool:
    return (all(leakage.get(f) is True for f in SAFE_LEAKAGE_FLAGS)
            and all(leakage.get(f) is False for f in UNSAFE_LEAKAGE_FLAGS)
            and leakage.get("reference_source_split") == "train/good"
            and leakage.get("reference_pool_size") == 24)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 normal reference tool smoke test")
    parser.add_argument("--categories", default="bottle,cable")
    parser.add_argument("--per-type", type=int, default=1,
                        help="query images taken from each test defect folder")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-parse-retries", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*Kwargs passed to.*")

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    data_root = Path(args.data_root)

    before2, before3 = manifest(PHASE2_ROOT), manifest(PHASE3_ROOT)
    checks: dict = {}

    common = dict(
        model_path=args.model_path, data_root=args.data_root,
        max_new_tokens=args.max_new_tokens,
        max_parse_retries=args.max_parse_retries,
        concurrency=args.concurrency,
    )
    tools = {
        MODE_A_K: NormalReferenceEvidenceTool(
            ReferenceToolConfig(reference_k=MODE_A_K, **common)),
        MODE_B_K: NormalReferenceEvidenceTool(
            ReferenceToolConfig(reference_k=MODE_B_K, **common)),
    }
    # Modes must share the embedder instance so retrieval is provably identical.
    tools[MODE_B_K]._embedder = tools[MODE_A_K].embedder

    # ---------------------------------------------------------- structural
    checks["static_module_gt_clean"] = check_static_no_gt_leakage()
    checks["runtime_no_gt_read"] = check_runtime_no_gt_read(
        data_root, categories, args.per_type)
    checks["prompt_modes"] = check_prompt_modes()
    checks["schema_validator"] = check_schema_validator()

    first_query = query_images(data_root, categories[0], 1)[0]
    checks["prompt_shape_and_no_similarity"] = check_prompt_text_has_no_similarity(
        tools[MODE_B_K].retrieve(first_query, categories[0], MODE_B_K))

    # ----------------------------------------------------------- retrieval
    checks["retrieval"] = check_retrieval(tools[MODE_A_K], categories)

    # --------------------------------------------------------------- model
    queries = {c: query_images(data_root, c, args.per_type) for c in categories}
    n_queries = sum(len(q) for q in queries.values())
    print(f"[smoke] {n_queries} query images: "
          + ", ".join(f"{c}={len(q)}" for c, q in queries.items()))

    if not args.skip_model:
        checks["model"], records = run_model_checks(args, tools, queries)

        # persist what the smoke run produced, for inspection (same records —
        # the VLM is never called twice for the same query)
        written = 0
        for recs in records.values():
            for record in recs:
                stem = (f"{Path(record['query_image']).parent.name}"
                        f"_{Path(record['query_image']).stem}")
                write_reference_record(record, args.out_root, record["category"], stem)
                written += 1
        checks["records_written"] = written

    after2, after3 = manifest(PHASE2_ROOT), manifest(PHASE3_ROOT)
    for name, before, after in (("phase2", before2, after2), ("phase3", before3, after3)):
        checks[f"{name}_untouched"] = {
            "n_files": len(before),
            "unchanged": before == after,
            "added": sorted(set(after) - set(before))[:5],
            "removed": sorted(set(before) - set(after))[:5],
            "modified": sorted(k for k in set(before) & set(after) if before[k] != after[k])[:5],
        }

    print(json.dumps(checks, indent=2, default=str))

    retrieval = checks["retrieval"]
    required = [
        checks["static_module_gt_clean"]["clean"],
        checks["runtime_no_gt_read"]["no_annotation_file_touched"],
        all(v["all_parse_ok"] and v["leakage_flags_safe"]
            for v in checks["runtime_no_gt_read"]["per_category"].values()),
        all(v for v in checks["prompt_modes"].values() if isinstance(v, bool)),
        all(checks["schema_validator"].values()),
        all(v for v in checks["prompt_shape_and_no_similarity"].values()
            if isinstance(v, bool)),
        all(v for k, v in retrieval.items()
            if isinstance(v, bool) and not k.endswith("_failures")),
        checks["phase2_untouched"]["unchanged"],
        checks["phase3_untouched"]["unchanged"],
    ]
    if not args.skip_model:
        model = checks["model"]
        for tag in ("modeA", "modeB"):
            required += [
                model[f"{tag}_all_parse_ok"],
                model[f"{tag}_schema_valid"],
                model[f"{tag}_reference_k_matches"],
                model[f"{tag}_n_references_matches"],
                model[f"{tag}_system_prompt_saved"],
                model[f"{tag}_no_similarity_in_prompt"],
                model[f"{tag}_references_traceable"],
                model[f"{tag}_leakage_flags_safe"],
            ]
        required += [
            model["both_modes_same_query_set"],
            model["identical_decoding_settings"],
            model["reproducible"],
            model["explicit_references_agree_with_retrieval"],
            model["explicit_references_same_output"],
        ]

    passed = all(required)
    print("=" * 78)
    print("PHASE 4 SMOKE TEST", "PASSED" if passed else "FAILED")
    print("=" * 78)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
