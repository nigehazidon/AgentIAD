"""Phase 5 smoke test: Evidence Fusion Agent on bottle + cable.

Structural checks (no model needed):

  1. ``static_module_gt_clean`` — the module carries no code path into the
     annotation tree, written as path-shaped regexes so its own leakage-flag
     *names* (``ground_truth_used_in_prompt``) do not self-trip;
  2. ``system_prompt_verbatim`` — the literal and its sha256 match the pinned
     values, all seven ``Step N:`` headers and all five output keys survive, and
     the source never applies ``.format()`` to it (its JSON schema block makes
     ``.format()`` raise ``KeyError``);
  3. ``prompt_assembly`` — exactly two image payloads, the exact expected text,
     and no withheld value anywhere in it;
  4. ``region_phrase_bucketing`` — every bucket boundary, then every real
     rank-1 bbox: digit-free, stable, and position-suppressed when large;
  5. ``schema_validator`` — including the two false positives that motivated
     replacing Phase 4's guard patterns;
  6. ``bundle_loader`` — every (key, mode) resolves with no missing evidence,
     plus negative tests;
  7. ``runtime_no_gt_read`` — the I/O-level proof, extended with the check that
     matters for *this* module: **nothing under ``<data_root>/<cat>/train/`` is
     ever touched**, because fusion consumes Phase 4's text, never a reference
     image.

Model checks (a few images, both modes):

  8. every record parses and validates against the exact schema;
  9. exactly two images attached, one model call, empty violations;
 10. both modes cover the same key set and share identical decoding settings;
 11. every leakage flag on every record holds;
 12. prompt assembly is deterministic, and a lone single-item batch reproduces
     byte-identically;
 13. ``results/phase2``, ``results/phase3`` and ``results/phase4`` are untouched.

A ``posthoc_report`` block reports the confusion matrix against the defect
folder names. It is **not** part of ``required``: the module is provably
ground-truth-free, and this block is the harness reading labels the module never
saw. The distinction is deliberate.

Usage::

    .../python smoke_test_phase5.py --skip-model    # structural only
    .../python smoke_test_phase5.py                 # full check
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "TEST")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import evidence_fusion as E  # noqa: E402
from evidence_fusion import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL_PATH,
    DEFAULT_OBSERVATION_MODE,
    DEFAULT_PHASE2_MAP_ROOT,
    DEFAULT_PHASE3_ROOT,
    DEFAULT_PHASE4_ROOT,
    BundleLoader,
    EvidenceFusionAgent,
    FusionBundleError,
    FusionConfig,
    OUTPUT_KEYS,
    RESULT_VALUES,
    build_messages,
    parse_fusion_output,
    prompt_leak_violations,
    prompt_text_of,
    region_phrase,
)

PHASE2_ROOT = Path("/data/pfy/AgentIAD/results/phase2")
PHASE3_ROOT = Path("/data/pfy/AgentIAD/results/phase3")
PHASE4_ROOT = Path("/data/pfy/AgentIAD/results/phase4")
OUT_ROOT = Path("/data/pfy/AgentIAD/results/phase5")

#: Pinned from the user's system prompt. Any drift means the literal was edited.
EXPECTED_PROMPT_SHA256 = "96c5a2211f11509d9c85458923bc16b7228767a0e9bd8ce388424b8cdb67db99"

REFERENCE_MODES = ("top1_normal_reference", "top3_normal_references")

#: Source patterns that would indicate the module *reads* the annotation tree.
FORBIDDEN_SOURCE_PATTERNS = (
    ("ground_truth_path", re.compile(r"ground_truth\s*[\"'/]")),
    ("gt_mask_path", re.compile(r"gt_mask\s*[\"'/]")),
    ("mask_png", re.compile(r"mask\.png")),
    ("train_good_path", re.compile(r"[\"'/]train[\"']?\s*/\s*[\"']?good")),
)
FORBIDDEN_PATH_FRAGMENTS = ("ground_truth", "gt_mask", "_mask.png")

#: Every leakage flag on every record must be exactly this.
EXPECTED_LEAKAGE = {
    "retrieval_performed": False,
    "reference_pool_accessed": False,
    "reference_image_opened": False,
    "ground_truth_read": False,
    "gt_mask_read": False,
    "ground_truth_used_in_prompt": False,
    "ground_truth_used_in_decision": False,
    "anomaly_label_in_prompt": False,
    "defect_type_token_in_prompt": False,
    "query_image_path_in_prompt": False,
    "patch_filename_in_prompt": False,
    "reference_image_paths_in_prompt": False,
    "reference_similarity_in_prompt": False,
    "image_score_in_prompt": False,
    "region_score_in_prompt": False,
    "region_area_ratio_in_prompt": False,
    "bbox_in_prompt": False,
    "reference_count_in_prompt": False,
    # Strict where we author:
    "region_phrase_is_digit_free": True,
    "authored_text_is_digit_free": True,
    "no_numeric_value_in_prompt": True,
    "category_named_by_harness": False,
    # The upstream model text is measured, not constrained — it is the VLM's own
    # prose about the image, so it may name the object it sees. These two are
    # deliberately absent from EXPECTED_LEAKAGE and reported separately.
    # The honest pair: this module really does read PatchCore-derived region
    # metadata and attach a PatchCore-derived crop.
    "patchcore_derived_region_metadata_read": True,
    "patchcore_region_crop_attached": True,
    "patchcore_score_value_in_prompt": False,
    "prior_stage_outputs_treated_as_untrusted_text": True,
}


def manifest(root: Path) -> dict:
    """``{relpath: [size, mtime_ns]}`` for every file under ``root``."""
    if not root.is_dir():
        return {}
    return {
        str(p.relative_to(root)): [p.stat().st_size, p.stat().st_mtime_ns]
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def leakage_ok(leakage: dict) -> bool:
    return (all(leakage.get(k) is v for k, v in EXPECTED_LEAKAGE.items())
            and leakage.get("withheld_values_found_in_prompt") == []
            and leakage.get("upstream_evidence_instruction_like") == []
            and leakage.get("images_attached") == 2)


# --------------------------------------------------------------- structural
def check_static_gt_clean() -> dict:
    src = (_REPO_ROOT / "evidence_fusion.py").read_text()
    hits = [name for name, pattern in FORBIDDEN_SOURCE_PATTERNS if pattern.search(src)]
    return {"clean": not hits, "forbidden_patterns_found": hits}


def check_system_prompt() -> dict:
    import hashlib

    src = (_REPO_ROOT / "evidence_fusion.py").read_text()
    prompt = E.FUSION_SYSTEM_PROMPT
    return {
        "sha256_matches_pinned": hashlib.sha256(
            prompt.encode("utf-8")).hexdigest() == EXPECTED_PROMPT_SHA256,
        "seven_step_headers": len(re.findall(r"^Step \d+:", prompt, re.M)) == 7,
        "all_output_keys_present": all(f'"{k}"' in prompt for k in OUTPUT_KEYS),
        "all_result_values_present": all(v in prompt for v in RESULT_VALUES),
        "json_schema_braces_intact": prompt.count("{") == 1 and prompt.count("}") == 1,
        # `.format()` on this literal raises KeyError; keep it that way.
        "no_format_applied_in_source": not re.search(
            r"FUSION_SYSTEM_PROMPT\s*\.\s*format", src),
        "no_fstring_interpolation_of_prompt": not re.search(
            r"f[\"'].*\{FUSION_SYSTEM_PROMPT", src),
        "stored_by_plain_literal": bool(re.search(
            r"^FUSION_SYSTEM_PROMPT = \"\"\"", src, re.M)),
    }


class _StubChat:
    """A stand-in for Qwen that replays a fixed reply (no GPU needed)."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    async def ainvoke(self, messages, **kwargs):
        import types
        self.calls += 1
        assert kwargs.get("do_sample") is False, "decoding must be greedy"
        return types.SimpleNamespace(
            content=self.reply, additional_kwargs={"output_tokens": 42})


_STUB_REPLY = json.dumps({
    "result": "normal",
    "reason": "the object matches the expected appearance",
    "supporting_evidence": ["the surface is uniform"],
    "contradicting_evidence": [],
    "remaining_uncertainty": "",
})


def check_prompt_assembly(config: FusionConfig, category: str, stem: str) -> dict:
    """The assembled prompt must be exactly what we intend, and leak nothing."""
    loader = BundleLoader(config)
    bundle = loader.load(category, stem)
    agent = EvidenceFusionAgent(config, chat=_StubChat(_STUB_REPLY), loader=loader)
    messages = agent.build_prompt(bundle)

    text = prompt_text_of(messages)
    # The digit-free / path-free guarantees apply to the user turn we assemble,
    # NOT to the system prompt — that carries "Step 1:" .. "Step 7:" by design,
    # and "imaging/viewpoint variation".
    user_text = "\n".join(
        str(i.get("text", "")) for i in messages[1]["content"]
        if isinstance(i, dict) and i.get("type") == "text")
    n_images = E.count_images(messages)
    content = messages[1]["content"]

    withheld = bundle.withheld()
    leaks = prompt_leak_violations(messages, withheld)

    expected_order = [
        "text", "image_url", "text", "image_url", "text", "text", "text"]
    return {
        "two_messages": len(messages) == 2,
        "system_is_prompt": messages[0]["content"] == E.FUSION_SYSTEM_PROMPT,
        "exactly_two_images": n_images == 2,
        "content_item_order": [i.get("type") for i in content] == expected_order,
        "query_image_first": content[1]["image_url"]["url"].startswith(
            "data:image/png;base64,"),
        "crop_image_second": content[3]["image_url"]["url"].startswith(
            "data:image/png;base64,"),
        "blocks_numbered_1_to_5": all(
            f"{n}. " in user_text for n in range(1, 6)),
        "region_phrase_present": bundle.region_text(
            config.suppress_position_span) in user_text,
        "no_withheld_value_in_prompt": leaks == [],
        "withheld_values_checked": len(withheld),
        # The block numerals are structural; nothing else of ours may be numeric.
        "user_turn_digit_free_outside_block_markers": not re.search(
            r"\d", E.strip_block_markers(user_text)),
        "user_turn_has_no_path_separator": "/" not in user_text,
        "leakage_flags": agent.leakage_audit(bundle, messages),
    }


def check_no_crop_mode(category: str, stem: str) -> dict:
    """With the crop disabled the prompt carries one image and still validates."""
    config = FusionConfig(
        reference_mode=REFERENCE_MODES[0], phase2_map_root=DEFAULT_PHASE2_MAP_ROOT,
        phase3_root=DEFAULT_PHASE3_ROOT, phase4_root=DEFAULT_PHASE4_ROOT,
        data_root=DEFAULT_DATA_ROOT, attach_region_crop=False)
    loader = BundleLoader(config)
    messages = EvidenceFusionAgent(
        config, chat=_StubChat(_STUB_REPLY), loader=loader).build_prompt(
            loader.load(category, stem))
    return {"single_image_when_crop_disabled": E.count_images(messages) == 1}


def check_region_phrase() -> dict:
    """Bucket boundaries, then every real rank-1 bbox."""
    checks: dict = {}

    # Vertical thirds, straddling the cy = 1/3 and cy = 2/3 boundaries.
    checks["upper_at_cy_0p32"] = "upper" in region_phrase(
        [0.40, 0.24, 0.50, 0.40], 0.02)
    checks["middle_at_cy_0p34"] = "central" in region_phrase(
        [0.40, 0.30, 0.50, 0.38], 0.02)
    checks["lower_at_cy_0p68"] = "lower-central" in region_phrase(
        [0.40, 0.60, 0.50, 0.76], 0.02)
    # Horizontal thirds, straddling cx = 1/3 and cx = 2/3.
    checks["left_at_cx_0p32"] = "left" in region_phrase(
        [0.24, 0.40, 0.40, 0.50], 0.02)
    checks["right_at_cx_0p68"] = "right" in region_phrase(
        [0.60, 0.40, 0.76, 0.50], 0.02)
    # extent thresholds
    checks["small_at_0p09"] = "small" in region_phrase([0.0, 0.0, 0.3, 0.3], 0.09)
    checks["moderate_at_0p10"] = "moderate" in region_phrase([0.0, 0.0, 0.3, 0.4], 0.10)
    checks["moderate_at_0p29"] = "moderate" in region_phrase([0.0, 0.0, 0.6, 0.5], 0.29)
    checks["large_at_0p30"] = "large" in region_phrase([0.0, 0.0, 0.6, 0.5], 0.30)
    # span suppression boundary: 0.49 shown, 0.50 suppressed
    shown = region_phrase([0.0, 0.0, 0.49, 0.2], 0.05)
    hidden = region_phrase([0.0, 0.0, 0.50, 0.2], 0.05)
    checks["span_0p49_position_shown"] = "part of the image" in shown
    checks["span_0p50_position_suppressed"] = "not informative" in hidden
    # aspect boundaries
    checks["horizontally_extended_at_1p41"] = "horizontally" in region_phrase(
        [0.0, 0.0, 0.282, 0.2], 0.05)
    checks["vertically_extended_at_0p69"] = "vertically" in region_phrase(
        [0.0, 0.0, 0.138, 0.2], 0.02)
    checks["square_at_1p0"] = "broad as it is tall" in region_phrase(
        [0.0, 0.0, 0.2, 0.2], 0.04)
    checks["degenerate_bbox_raises"] = _raises(
        lambda: region_phrase([0.5, 0.5, 0.5, 0.6], 0.01))
    checks["default_suppresses_large_span"] = "not informative" in region_phrase(
        [0.0, 0.0, 1.0, 1.0], 0.658)

    # Every real rank-1 region. `require_all_evidence=False` on purpose: the
    # phrase needs only the bbox, which comes from Phase 3, so this sweep must
    # not depend on Phase 4 having finished.
    n = digit = unstable = 0
    n_suppressed = 0
    for category in ("bottle", "cable"):
        loader = BundleLoader(FusionConfig(
            reference_mode=REFERENCE_MODES[0], phase2_map_root=DEFAULT_PHASE2_MAP_ROOT,
            phase3_root=DEFAULT_PHASE3_ROOT, phase4_root=DEFAULT_PHASE4_ROOT,
            data_root=DEFAULT_DATA_ROOT, require_all_evidence=False))
        for stem in loader.keys(category):
            bundle = loader.load(category, stem)
            phrase = bundle.region_text(0.50)
            n += 1
            digit += bool(re.search(r"\d", phrase))
            unstable += phrase != bundle.region_text(0.50)
            n_suppressed += "not informative" in phrase
    checks["all_real_regions_digit_free"] = digit == 0
    checks["all_real_regions_stable"] = unstable == 0
    checks["n_real_regions"] = n
    checks["n_position_suppressed"] = n_suppressed
    return checks


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


def check_schema_validator() -> dict:
    """Accept well-formed replies; reject malformed and measurement-bearing ones."""
    good = json.dumps({
        "result": "anomalous",
        "reason": "there is a dark irregular mark on the surface",
        "supporting_evidence": ["a dark mark is visible"],
        "contradicting_evidence": ["the surrounding surface is uniform"],
        "remaining_uncertainty": "",
    })

    def variant(**over):
        obj = json.loads(good)
        obj.update(over)
        return json.dumps(obj)

    cases = (
        ("plain", good, True),
        ("fenced", f"```json\n{good}\n```", True),
        ("prose_wrapped", f"Here is my judgment:\n{good}\nDone.", True),
        # The two false positives that killed Phase 4's guard patterns.
        ("two_number_prose_allowed",
         variant(reason="two dark spots are visible (1, 2) on the rim"), True),
        ("score_concept_allowed",
         variant(reason="the PatchCore anomaly score is not proof of an anomaly"), True),
        # Genuine schema failures.
        ("bad_result_enum", variant(result="defective"), False),
        ("missing_key", json.dumps({"result": "normal"}), False),
        ("null_uncertainty", variant(remaining_uncertainty=None), False),
        ("empty_reason", variant(reason=""), False),
        ("blank_evidence_entry", variant(supporting_evidence=[""]), False),
        ("evidence_not_a_list", variant(supporting_evidence="a mark"), False),
        ("empty_contradicting_allowed", variant(contradicting_evidence=[]), True),
        ("null_contradicting", variant(contradicting_evidence=None), False),
        ("not_json", "The object looks normal to me.", False),
        ("empty", "", False),
        # Measurement-bearing content.
        ("bbox_in_reason", variant(reason="a mark at bbox [10, 20, 30, 40]"), False),
        ("coordinates_in_reason", variant(reason="a mark at coordinates 12, 40"), False),
        ("normalised_tuple", variant(reason="the region is [0.12, 0.34, 0.56, 0.78]"),
         False),
        ("four_number_tuple", variant(reason="the region is (10, 20, 30, 40)"), False),
        ("percent_value", variant(reason="the region covers 20 percent of the object"),
         False),
        ("pixel_units", variant(reason="the mark is about 120 px wide"), False),
        ("score_value", variant(reason="the anomaly score is 0.8"), False),
        ("decision_threshold", variant(reason="this is below the decision threshold"),
         False),
        ("patchcore_numeric", variant(reason="the patchcore score is elevated"), False),
        ("bbox_key", variant(bbox=[0.1, 0.2, 0.3, 0.4]), False),
    )
    checks = {label: (parse_fusion_output(text)[0] is not None) == want
              for label, text, want in cases}

    # Advisory rules must be recorded but must NOT discard a valid answer.
    parsed, violations = parse_fusion_output(variant(supporting_evidence=[]))
    checks["advisory_empty_evidence_recorded"] = (
        parsed is not None and any(v.startswith("advisory:") for v in violations))
    parsed, violations = parse_fusion_output(
        variant(result="uncertain", remaining_uncertainty=""))
    checks["advisory_uncertain_empty_note_recorded"] = (
        parsed is not None and any(v.startswith("advisory:") for v in violations))
    # No advisory message may collide with a fatal prefix.
    checks["advisory_never_fatal"] = not E.is_fatal([
        "advisory: supporting_evidence is empty while result is 'normal'"])
    checks["fatal_prefixes_disjoint_from_advisory"] = not any(
        p.startswith("advisory") for p in E.FATAL_VIOLATION_PREFIXES)
    # Stripping is applied on the happy path.
    parsed, _ = parse_fusion_output(variant(result="  ANOMALOUS  "))
    checks["verdict_lowercased_and_stripped"] = (
        parsed is not None and parsed["result"] == "anomalous")
    return checks


def check_bundle_loader() -> dict:
    """Every (key, mode) must resolve with complete evidence."""
    checks: dict = {}
    for mode in REFERENCE_MODES:
        loader = BundleLoader(FusionConfig(
            reference_mode=mode, phase2_map_root=DEFAULT_PHASE2_MAP_ROOT,
            phase3_root=DEFAULT_PHASE3_ROOT, phase4_root=DEFAULT_PHASE4_ROOT,
            data_root=DEFAULT_DATA_ROOT))
        n = incomplete = bad_crop = not_test = raised = 0
        first_error: str | None = None
        for category in ("bottle", "cable"):
            for stem in loader.keys(category):
                n += 1
                try:
                    bundle = loader.load(category, stem)
                except FusionBundleError as exc:
                    # Phase 4 has not covered this key yet. Report it as a
                    # failure with the reason rather than a traceback.
                    raised += 1
                    if first_error is None:
                        first_error = str(exc)
                    continue
                incomplete += bool(bundle.missing)
                bad_crop += not bundle.patch_path.is_file()
                not_test += "train" in bundle.query_image.parts
        checks[f"{mode}_n_keys"] = n
        checks[f"{mode}_all_complete"] = incomplete == 0 and raised == 0
        checks[f"{mode}_n_incomplete"] = incomplete + raised
        checks[f"{mode}_first_error"] = first_error
        checks[f"{mode}_all_crops_present"] = bad_crop == 0
        checks[f"{mode}_no_train_image"] = not_test == 0

    # Negative tests.
    strict = FusionConfig(
        reference_mode=REFERENCE_MODES[0], phase2_map_root=DEFAULT_PHASE2_MAP_ROOT,
        phase3_root=DEFAULT_PHASE3_ROOT, phase4_root=DEFAULT_PHASE4_ROOT,
        data_root=DEFAULT_DATA_ROOT)
    loader = BundleLoader(strict)
    checks["unknown_stem_raises"] = _raises(lambda: loader.load("bottle", "no_such_000"))
    checks["path_traversal_stem_raises"] = _raises(
        lambda: loader.load("bottle", "../../etc/passwd"))
    checks["unknown_category_raises"] = _raises(lambda: loader.keys("no_such_cat"))

    # A wrong reference mode must be rejected, not silently accepted.
    mismatch = FusionConfig(
        reference_mode=REFERENCE_MODES[1], phase2_map_root=DEFAULT_PHASE2_MAP_ROOT,
        phase3_root=DEFAULT_PHASE3_ROOT, phase4_root=DEFAULT_PHASE4_ROOT,
        data_root=DEFAULT_DATA_ROOT)
    mismatch_loader = BundleLoader(mismatch)
    checks["mode_mismatch_detected"] = _raises(
        lambda: _load_with_foreign_record(mismatch_loader))
    return checks


def _load_with_foreign_record(loader: BundleLoader) -> None:
    """Load a key while its Phase-4 record claims the *other* mode."""
    stem = loader.keys("bottle")[0]
    real_read = E._read_json

    def fake_read(path: Path) -> dict:
        record = real_read(path)
        if "phase4" in str(path):
            record = {**record, "mode_name": "top1_normal_reference",
                      "reference_k": 1}
        return record

    E._read_json = fake_read
    try:
        loader.load("bottle", stem)
    finally:
        E._read_json = real_read


def check_runtime_no_gt_read(categories: list[str],
                             keys: dict[str, list[str]]) -> dict:
    """Prove at the I/O level that nothing under train/ or ground_truth/ is read.

    A stub model replays a canned reply, so this runs without the GPU. It is the
    check that matters most for this module: Phase 5 consumes Phase 4's *text*,
    so opening a reference image or any ``train/`` path would mean the design has
    quietly changed.
    """
    import builtins

    touched: list[str] = []
    real_open, real_path_open = builtins.open, Path.open
    real_glob, real_rglob = Path.glob, Path.rglob

    def open_wrap(file, *a, **kw):
        touched.append(str(file))
        return real_open(file, *a, **kw)

    def path_open_wrap(self, *a, **kw):
        touched.append(str(self))
        return real_path_open(self, *a, **kw)

    def glob_wrap(self, pattern):
        touched.append(str(self))
        return real_glob(self, pattern)

    def rglob_wrap(self, pattern):
        touched.append(str(self))
        return real_rglob(self, pattern)

    builtins.open, Path.open = open_wrap, path_open_wrap
    Path.glob, Path.rglob = glob_wrap, rglob_wrap

    try:
        results: dict = {}
        for category in categories:
            stub = _StubChat(_STUB_REPLY)
            config = FusionConfig(
                reference_mode=REFERENCE_MODES[0], data_root=DEFAULT_DATA_ROOT,
                phase2_map_root=DEFAULT_PHASE2_MAP_ROOT,
                phase3_root=DEFAULT_PHASE3_ROOT, phase4_root=DEFAULT_PHASE4_ROOT,
                max_parse_retries=0)
            agent = EvidenceFusionAgent(config, chat=stub)
            stems = keys.get(category, [])[:3]
            records = asyncio.run(agent.fuse_many(
                [{"category": category, "stem": s} for s in stems]))
            results[category] = {
                "n_queries": len(stems),
                "stub_calls": stub.calls,
                "all_parse_ok": all(r["parse_ok"] for r in records),
                "leakage_ok": all(leakage_ok(r["leakage"]) for r in records),
            }
    finally:
        builtins.open, Path.open = real_open, real_path_open
        Path.glob, Path.rglob = real_glob, real_rglob

    offenders = sorted({
        p for p in touched if any(frag in p for frag in FORBIDDEN_PATH_FRAGMENTS)})
    train_touched = sorted({
        p for p in touched
        if re.search(r"/MVTec-AD/[^/]+/(?:train|ground_truth)(?:/|$)", p)})
    return {
        "no_annotation_file_touched": not offenders,
        "offending_paths": offenders,
        "no_reference_or_train_image_touched": not train_touched,
        "train_or_gt_paths": train_touched[:5],
        "n_paths_touched": len(touched),
        "sample_paths_touched": sorted({
            p for p in touched if "AgentIAD" in p or "MVTec" in p})[:6],
        "per_category": results,
    }


# -------------------------------------------------------------------- model
def run_model_checks(args, agents: dict[str, EvidenceFusionAgent],
                     keys: dict[str, list[str]]) -> tuple[dict, dict[str, list[dict]]]:
    checks: dict = {}
    records: dict[str, list[dict]] = {}

    async def run() -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for mode, agent in agents.items():
            requests = [{"category": c, "stem": s}
                        for c, stems in keys.items() for s in stems]
            out[mode] = await agent.fuse_many(requests)

        # determinism: one key, twice, each a lone batch of one
        mode = next(iter(agents))
        category, stems = next(iter(keys.items()))
        first = await agents[mode].fuse(category, stems[0])
        second = await agents[mode].fuse(category, stems[0])
        return {"records": out, "first": first, "second": second}

    out = asyncio.run(run())
    records = out["records"]

    for mode, recs in records.items():
        checks[f"{mode}_n_records"] = len(recs)
        checks[f"{mode}_all_parse_ok"] = all(r["parse_ok"] for r in recs)
        checks[f"{mode}_all_vlm_called_once"] = all(r["vlm_calls"] == 1 for r in recs)
        checks[f"{mode}_no_violations"] = all(not r["violations"] for r in recs)
        checks[f"{mode}_schema_valid"] = all(
            _schema_valid(r["parsed_output"]) for r in recs if r["parse_ok"])
        checks[f"{mode}_two_images"] = all(
            r["leakage"]["images_attached"] == 2 for r in recs)
        checks[f"{mode}_leakage_ok"] = all(leakage_ok(r["leakage"]) for r in recs)
        checks[f"{mode}_system_prompt_saved"] = all(
            r["system_prompt"] == E.FUSION_SYSTEM_PROMPT for r in recs)
        checks[f"{mode}_reference_k_matches"] = all(
            r["reference_k"] == E.REFERENCE_K_BY_MODE[mode] for r in recs)
        checks[f"{mode}_all_evidence_available"] = all(
            all(r["evidence_available"].values()) for r in recs)
        checks[f"{mode}_prompt_deterministic"] = all(
            r["source_provenance"]["prompt_sha256"]
            == E.hashlib.sha256(r["user_turn_text"].encode()).hexdigest()
            for r in recs)
        checks[f"{mode}_failures"] = [
            {"stem": r["stem"], "violations": r["violations"]}
            for r in recs if not r["parse_ok"]
        ]

    keysets = {mode: {(r["category"], r["stem"]) for r in recs}
               for mode, recs in records.items()}
    modes = sorted(records)
    checks["both_modes_same_key_set"] = (
        keysets[modes[0]] == keysets[modes[1]] and len(keysets[modes[0]]) > 0)

    sigs = {mode: agents[mode].config.decoding_signature() for mode in modes}
    checks["identical_decoding_settings"] = sigs[modes[0]] == sigs[modes[1]]
    checks["decoding_signature"] = {k: v for k, v in sigs[modes[0]].items()
                                    if k != "system_prompt_sha256"}
    checks["decoding_signature_differs_only_by_reference_mode"] = (
        agents[modes[0]].config.to_dict()["reference_mode"]
        != agents[modes[1]].config.to_dict()["reference_mode"])

    checks["reproducible_serialized_single_item_batches"] = (
        out["first"]["raw_model_output"] == out["second"]["raw_model_output"])
    checks["reproducible_parsed_output"] = (
        out["first"]["parsed_output"] == out["second"]["parsed_output"])

    # Informational only — see the driver docstring. `out["first"]` was a lone
    # batch of one; `records[modes[0]][0]` is the same key inside a full batch.
    checks["raw_bytes_equal_under_different_batch_composition"] = (
        out["first"]["raw_model_output"] == records[modes[0]][0]["raw_model_output"])
    checks["same_key_in_both"] = (
        out["first"]["stem"] == records[modes[0]][0]["stem"])

    checks["sample_outputs"] = [
        {
            "mode": mode,
            "stem": r["stem"],
            "result": r["result"],
            "n_supporting": len((r["parsed_output"] or {}).get("supporting_evidence", [])),
            "n_contradicting": len(
                (r["parsed_output"] or {}).get("contradicting_evidence", [])),
            "uncertainty": (r["parsed_output"] or {}).get("remaining_uncertainty"),
            "reason": (r["parsed_output"] or {}).get("reason", "")[:150],
        }
        for mode in modes for r in records[mode][:3]
    ]
    return checks, records


def _schema_valid(parsed: dict | None) -> bool:
    if not isinstance(parsed, dict) or set(parsed) != set(OUTPUT_KEYS):
        return False
    if parsed["result"] not in RESULT_VALUES:
        return False
    if not parsed["reason"]:
        return False
    for key in ("supporting_evidence", "contradicting_evidence"):
        if not isinstance(parsed[key], list):
            return False
        if any(not isinstance(e, str) or not e for e in parsed[key]):
            return False
    if not isinstance(parsed["remaining_uncertainty"], str):
        return False
    return True


def posthoc_report(records_by_mode: dict[str, list[dict]],
                   defect_of: dict[tuple[str, str], str]) -> dict:
    """Confusion against the defect folder — read by the HARNESS, not the module.

    The module never sees these labels (see ``leakage.defect_type_token_in_prompt``
    and ``no_reference_or_train_image_touched``). This block exists so the
    report can show accuracy; it is not part of ``required``.
    """
    report: dict = {"note": "harness-only; the module is provably label-free"}
    for mode, records in records_by_mode.items():
        tab = {d: {"normal": 0, "anomalous": 0, "uncertain": 0, "n": 0}
               for d in sorted(set(defect_of.values()))}
        for r in records:
            defect = defect_of.get((r["category"], r["stem"]))
            if defect is None:
                continue
            tab[defect]["n"] += 1
            if r["result"] in tab[defect]:
                tab[defect][r["result"]] += 1
        report[mode] = tab
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 evidence fusion smoke test")
    parser.add_argument("--categories", default="bottle,cable")
    parser.add_argument("--per-category", type=int, default=2,
                        help="query images per category for the model checks")
    parser.add_argument("--phase2-map-root", default=DEFAULT_PHASE2_MAP_ROOT)
    parser.add_argument("--phase3-root", default=DEFAULT_PHASE3_ROOT)
    parser.add_argument("--phase4-root", default=DEFAULT_PHASE4_ROOT)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-parse-retries", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*Kwargs passed to.*")

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    before = {name: manifest(root) for name, root in (
        ("phase2", PHASE2_ROOT), ("phase3", PHASE3_ROOT), ("phase4", PHASE4_ROOT))}
    checks: dict = {}

    common = dict(
        phase2_map_root=args.phase2_map_root, phase3_root=args.phase3_root,
        phase4_root=args.phase4_root, data_root=args.data_root,
        model_path=args.model_path, max_new_tokens=args.max_new_tokens,
        max_parse_retries=args.max_parse_retries, concurrency=args.concurrency)
    agents = {mode: EvidenceFusionAgent(FusionConfig(reference_mode=mode, **common))
              for mode in REFERENCE_MODES}

    # Resolve, per category, the keys whose evidence is complete — once, and
    # reused by every check below. A key that Phase 4 has not covered is skipped
    # rather than crashing the structural half of the run; the bundle-loader
    # check is what reports the gap.
    def complete_keys(category: str, limit: int | None) -> list[str]:
        loader = BundleLoader(agents[REFERENCE_MODES[0]].config)
        out: list[str] = []
        for stem in loader.keys(category):
            try:
                if loader.load(category, stem).missing:
                    continue
            except FusionBundleError:
                continue
            out.append(stem)
            if limit is not None and len(out) >= limit:
                break
        return out

    keys = {c: complete_keys(c, args.per_category) for c in categories}
    all_complete = {c: complete_keys(c, None) for c in categories}
    n_complete = sum(len(v) for v in all_complete.values())
    print(f"[smoke] {n_complete} keys with complete evidence; "
          f"{sum(len(v) for v in keys.values())} used for the model checks")

    # ------------------------------------------------------------ structural
    checks["static_module_gt_clean"] = check_static_gt_clean()
    checks["system_prompt_verbatim"] = check_system_prompt()
    checks["schema_validator"] = check_schema_validator()
    checks["region_phrase"] = check_region_phrase()
    checks["bundle_loader"] = check_bundle_loader()
    checks["runtime_no_gt_read"] = check_runtime_no_gt_read(categories, keys)

    demo_category = next((c for c in categories if keys[c]), None)
    if demo_category:
        demo_stem = keys[demo_category][0]
        checks["demo_key"] = f"{demo_category}/{demo_stem}"
        checks["prompt_assembly"] = check_prompt_assembly(
            agents[REFERENCE_MODES[1]].config, demo_category, demo_stem)
        checks["no_crop_mode"] = check_no_crop_mode(demo_category, demo_stem)
    else:
        checks["demo_key"] = None
        checks["prompt_assembly"] = {"no_complete_key": False}
        checks["no_crop_mode"] = {"no_complete_key": False}

    records: dict[str, list[dict]] = {}
    if not args.skip_model:
        checks["model"], records = run_model_checks(args, agents, keys)

    # -------------------------------------------------------- post-hoc only
    # `require_all_evidence=False`: this map needs only the folder name, which
    # comes from the query path, so it must not depend on Phase 4 coverage.
    defect_of: dict[tuple[str, str], str] = {}
    relaxed = BundleLoader(FusionConfig(
        reference_mode=REFERENCE_MODES[0], phase2_map_root=args.phase2_map_root,
        phase3_root=args.phase3_root, phase4_root=args.phase4_root,
        data_root=args.data_root, require_all_evidence=False))
    for category in categories:
        for stem in relaxed.keys(category):
            defect_of[(category, stem)] = relaxed.load(category, stem).defect_type
    checks["posthoc_report"] = (
        posthoc_report(records, defect_of) if records else {"note": "skipped"})

    # ------------------------------------------------------- untouched trees
    after = {name: manifest(root) for name, root in (
        ("phase2", PHASE2_ROOT), ("phase3", PHASE3_ROOT), ("phase4", PHASE4_ROOT))}
    for name in ("phase2", "phase3", "phase4"):
        checks[f"{name}_untouched"] = {
            "n_files": len(before[name]),
            "unchanged": before[name] == after[name],
            "added": sorted(set(after[name]) - set(before[name]))[:5],
            "removed": sorted(set(before[name]) - set(after[name]))[:5],
            "modified": sorted(k for k in set(before[name]) & set(after[name])
                               if before[name][k] != after[name][k])[:5],
        }

    print(json.dumps(checks, indent=2, default=str))

    # Labelled, so a failure names itself. `all()` over an unlabelled bool list
    # reports a count and nothing else, which is useless when one check out of
    # a hundred flips.
    def bools(section: str, skip: tuple[str, ...] = ()) -> list[tuple[str, object]]:
        return [(f"{section}.{k}", v) for k, v in checks[section].items()
                if isinstance(v, bool) and k not in skip]

    required: list[tuple[str, object]] = [
        ("static_module_gt_clean.clean", checks["static_module_gt_clean"]["clean"]),
        *bools("system_prompt_verbatim"),
        *bools("schema_validator"),
        *bools("region_phrase"),
        *bools("bundle_loader"),
        *bools("prompt_assembly", skip=("leakage_flags",)),
        # Compared against EXPECTED_LEAKAGE rather than asserted True, so the
        # safe value lives in exactly one place: several of these flags are
        # correct precisely when they are False.
        *[(f"prompt_assembly.leakage_flags.{k}",
           checks["prompt_assembly"].get("leakage_flags", {}).get(k)
           is EXPECTED_LEAKAGE[k])
          for k in ("no_numeric_value_in_prompt", "category_named_by_harness",
                    "region_phrase_is_digit_free", "authored_text_is_digit_free")],
        ("no_crop_mode.single_image_when_crop_disabled",
         checks["no_crop_mode"].get("single_image_when_crop_disabled")),
        ("runtime_no_gt_read.no_annotation_file_touched",
         checks["runtime_no_gt_read"]["no_annotation_file_touched"]),
        ("runtime_no_gt_read.no_reference_or_train_image_touched",
         checks["runtime_no_gt_read"]["no_reference_or_train_image_touched"]),
        ("runtime_no_gt_read.per_category_safe", all(
            v["all_parse_ok"] and v["leakage_ok"]
            for v in checks["runtime_no_gt_read"]["per_category"].values())),
        ("phase2_untouched.unchanged", checks["phase2_untouched"]["unchanged"]),
        ("phase3_untouched.unchanged", checks["phase3_untouched"]["unchanged"]),
        ("phase4_untouched.unchanged", checks["phase4_untouched"]["unchanged"]),
    ]

    if not args.skip_model:
        model = checks["model"]
        for mode in REFERENCE_MODES:
            for suffix in ("all_parse_ok", "no_violations", "schema_valid",
                           "all_vlm_called_once", "two_images", "leakage_ok",
                           "system_prompt_saved", "reference_k_matches",
                           "all_evidence_available", "prompt_deterministic"):
                key = f"{mode}_{suffix}"
                required.append((f"model.{key}", model.get(key)))
        for key in ("both_modes_same_key_set", "identical_decoding_settings",
                    "reproducible_serialized_single_item_batches",
                    "reproducible_parsed_output"):
            required.append((f"model.{key}", model.get(key)))

    failed = [name for name, value in required if value is not True]
    passed = not failed
    print("=" * 78)
    print("PHASE 5 SMOKE TEST", "PASSED" if passed else "FAILED")
    if failed:
        print(f"  {len(failed)} of {len(required)} required checks failed:")
        for name in failed:
            print(f"    - {name}")
    print("=" * 78)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
