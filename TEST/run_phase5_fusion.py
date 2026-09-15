"""Phase 5 driver: Evidence Fusion over every cached test image.

For every test image of the requested categories this script runs the Evidence
Fusion Agent in each requested reference mode and writes one JSON record per
(query, mode) under ``results/phase5/``.

Three things this driver exists to guarantee:

  1. **Preflight.** ``--preflight-only`` resolves all (key, mode) bundles without
     touching the GPU and aborts non-zero on any gap. Without it a run whose
     Phase-4 records are missing would emit hundreds of records whose normal-
     reference block reads "Not available" and whose verdicts were driven by the
     local observation alone — a plausible-looking and entirely wrong experiment.

  2. **Fusion runs in its own process.** ``load_vl_model(use_singleton=True)``
     keys on the model path alone and the Qwen batcher is first-load-wins, so a
     Phase-4 and a Phase-5 config in one process would silently share the first
     config's batcher settings. Never import the Phase-4 driver here.

  3. **Determinism claims are bounded.** ``prompt_assembly_deterministic`` is a
     real guarantee; ``reproducible_serialized_single_item_batches`` is measured
     and required; agreement across *different* batch compositions is recorded
     as an empirical observation, never asserted as a guarantee — QwenBatcher
     feeds ``gen_kwargs_list[0]`` to the whole batch and left-pads.

Usage::

    .../python run_phase5_fusion.py --preflight-only
    .../python run_phase5_fusion.py --limit-per-category 10
    .../python run_phase5_fusion.py --skip-existing
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "TEST")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evidence_fusion import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL_PATH,
    DEFAULT_OBSERVATION_MODE,
    DEFAULT_OUT_ROOT,
    DEFAULT_PHASE2_MAP_ROOT,
    DEFAULT_PHASE3_ROOT,
    DEFAULT_PHASE4_ROOT,
    EvidenceFusionAgent,
    FusionBundleError,
    FusionConfig,
    REFERENCE_K_BY_MODE,
    write_fusion_record,
)
from phase_common import defect_types_of, limit_round_robin, list_entries  # noqa: E402

REFERENCE_MODES = ",".join(sorted(REFERENCE_K_BY_MODE))


# --------------------------------------------------------------------- resume
def already_done(out_root: Path, reference_mode: str, category: str, stem: str) -> bool:
    """True only for a record that exists **and** produced a valid verdict.

    Resuming on mere existence would let a run that crashed mid-way, or a record
    whose ``parsed_output`` is ``None``, be skipped forever.
    """
    path = out_root / reference_mode / category / f"{stem}.json"
    if not path.is_file():
        return False
    try:
        with open(path) as f:
            rec = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return rec.get("parsed_output") is not None


# ----------------------------------------------------------------- preflight
def preflight(
    agents: dict[str, EvidenceFusionAgent],
    keys_by_category: dict[str, list[str]],
) -> tuple[dict, list[str]]:
    """Resolve every (key, mode) bundle without loading the model.

    Returns ``(report, problems)``. ``problems`` is non-empty exactly when some
    (key, mode) cannot produce a complete record, which is what gates the run.
    """
    report: dict = {"per_category": {}, "per_mode": {}}
    problems: list[str] = []

    for mode, agent in agents.items():
        n_ok = 0
        n_bad = 0
        missing_counter: Counter = Counter()
        for category, stems in keys_by_category.items():
            for stem in stems:
                try:
                    bundle = agent.loader.load(category, stem)
                except FusionBundleError as exc:
                    n_bad += 1
                    missing_counter[str(exc).split("incomplete evidence")[-1].strip()] += 1
                    if len(problems) < 25:
                        problems.append(f"[{mode}] {exc}")
                    continue
                if bundle.missing:
                    n_bad += 1
                    missing_counter[",".join(bundle.missing)] += 1
                    if len(problems) < 25:
                        problems.append(
                            f"[{mode}] {category}/{stem} missing {list(bundle.missing)}")
                    continue
                if not bundle.patch_path.is_file():
                    n_bad += 1
                    if len(problems) < 25:
                        problems.append(f"[{mode}] {category}/{stem} no crop on disk")
                    continue
                n_ok += 1
        report["per_mode"][mode] = {
            "n_keys": sum(len(v) for v in keys_by_category.values()),
            "n_complete": n_ok,
            "n_incomplete": n_bad,
            "missing_histogram": dict(missing_counter),
        }

    for category, stems in keys_by_category.items():
        block: dict = {"n_keys": len(stems), "defect_types": None, "per_mode": {}}
        for mode, agent in agents.items():
            counts = {"observation": 0, "reference": 0, "crop": 0, "query": 0}
            for stem in stems:
                try:
                    bundle = agent.loader.load(category, stem)
                except FusionBundleError:
                    continue
                avail = bundle.evidence_available()
                counts["observation"] += avail["local_observation"]
                counts["reference"] += avail["normal_reference"]
                counts["crop"] += avail["region_crop"]
                counts["query"] += avail["query_image"]
            block["per_mode"][mode] = counts
        report["per_category"][category] = block

    return report, problems


def print_preflight(report: dict, keys_by_category: dict[str, list[str]]) -> None:
    print("=" * 78)
    print("PREFLIGHT — bundle resolution (no model loaded)")
    print("=" * 78)
    for category, stems in keys_by_category.items():
        block = report["per_category"][category]
        print(f"\n[{category}] {block['n_keys']} keys")
        for mode, counts in block["per_mode"].items():
            n = block["n_keys"]
            print(f"  {mode:<26} query {counts['query']}/{n}  crop {counts['crop']}/{n}  "
                  f"observation {counts['observation']}/{n}  "
                  f"reference {counts['reference']}/{n}")
    print("\nper-mode totals:")
    for mode, stats in report["per_mode"].items():
        print(f"  {mode:<26} complete {stats['n_complete']}/{stats['n_keys']}"
              + (f"  incomplete {stats['n_incomplete']} "
                 f"{stats['missing_histogram']}" if stats["n_incomplete"] else ""))


# --------------------------------------------------------------- determinism
def determinism_checks(agent: EvidenceFusionAgent, category: str, stem: str) -> dict:
    """Bound every determinism claim to what is actually true.

    Runs in one event loop:
      * the same bundle is assembled twice -> prompts must be byte-identical;
      * one call as a lone batch of one, twice, sequentially -> raw output must
        be byte-identical (greedy; the batch composition is identical);
      * one call alone vs. batched with others -> recorded, not asserted.
    """
    checks: dict = {}
    bundle = agent.loader.load(category, stem)

    first_msgs = agent.build_prompt(bundle)
    second_msgs = agent.build_prompt(bundle)
    checks["prompt_assembly_deterministic"] = (
        json.dumps(first_msgs, sort_keys=True) == json.dumps(second_msgs, sort_keys=True))

    async def run() -> dict:
        alone_a = await agent.fuse(category, stem)
        alone_b = await agent.fuse(category, stem)

        others = [s for s in agent.loader.keys(category) if s != stem][:8]
        batched = await agent.fuse_many(
            [{"category": category, "stem": stem}]
            + [{"category": category, "stem": s} for s in others])
        return {"alone_a": alone_a, "alone_b": alone_b,
                "batched": batched[0], "n_batched_with": len(others)}

    out = asyncio.run(run())

    checks["reproducible_serialized_single_item_batches"] = (
        out["alone_a"]["raw_model_output"] == out["alone_b"]["raw_model_output"])
    checks["reproducible_parsed_output"] = (
        out["alone_a"]["parsed_output"] == out["alone_b"]["parsed_output"])
    checks["alone_batch_composition_is_single_item"] = (
        out["alone_a"]["vlm_calls"] == 1 and out["alone_b"]["vlm_calls"] == 1)

    # Empirical, one image. NOT a guarantee — see the module docstring.
    checks["result_agrees_under_different_batch_composition"] = (
        out["alone_a"]["parsed_output"] is not None
        and out["batched"]["parsed_output"] is not None
        and out["alone_a"]["parsed_output"]["result"]
        == out["batched"]["parsed_output"]["result"])
    # Informational: expected to be False, and that is not a bug.
    checks["raw_bytes_equal_under_different_batch_composition"] = (
        out["alone_a"]["raw_model_output"] == out["batched"]["raw_model_output"])
    checks["n_batched_with"] = out["n_batched_with"]

    a, b = out["alone_a"], out["batched"]
    checks["drift_example"] = {
        "alone_result": (a["parsed_output"] or {}).get("result"),
        "batched_result": (b["parsed_output"] or {}).get("result"),
        "alone_output_tokens": a["output_tokens"],
        "batched_output_tokens": b["output_tokens"],
        "raw_prefix_alone": a["raw_model_output"][:160],
        "raw_prefix_batched": b["raw_model_output"][:160],
    }
    return checks


# -------------------------------------------------------------------- report
def summarize(records_by_mode: dict[str, list[dict]]) -> dict:
    """Result distribution, violation histogram, evidence provenance."""
    out: dict = {"per_mode": {}}
    for mode, records in records_by_mode.items():
        results = Counter(r["result"] for r in records)
        violations = Counter(
            v for r in records for v in r["violations"])
        n = len(records)
        out["per_mode"][mode] = {
            "n_records": n,
            "n_parse_ok": sum(1 for r in records if r["parse_ok"]),
            "n_retried": sum(1 for r in records if r["vlm_calls"] > 1),
            "n_hit_token_cap": sum(1 for r in records if r["hit_token_cap"]),
            "mean_elapsed_s": round(sum(r["elapsed_s"] for r in records) / n, 4) if n else None,
            "result_distribution": {
                k: {"n": results.get(k, 0),
                    "pct": round(100.0 * results.get(k, 0) / n, 2) if n else None}
                for k in ("normal", "anomalous", "uncertain", None)
                if results.get(k)
            },
            "n_distinct_results": len([k for k in results if k is not None]),
            "violation_histogram": dict(violations),
            "n_numeric_value_in_prompt": sum(
                1 for r in records
                if not r["leakage"]["no_numeric_value_in_prompt"]),
            "n_harness_names_category": sum(
                1 for r in records if r["leakage"]["category_named_by_harness"]),
            "n_upstream_text_has_digit": sum(
                1 for r in records if not r["leakage"]["upstream_text_digit_free"]),
            "n_upstream_text_names_category": sum(
                1 for r in records if r["leakage"]["upstream_text_names_category"]),
            "n_withheld_leak": sum(
                1 for r in records if r["leakage"]["withheld_values_found_in_prompt"]),
            "n_ground_truth_read": sum(
                1 for r in records if r["leakage"]["ground_truth_read"]),
            "n_reference_image_opened": sum(
                1 for r in records if r["leakage"]["reference_image_opened"]),
            "images_attached": dict(Counter(
                r["leakage"]["images_attached"] for r in records)),
            "region_phrase_histogram": dict(Counter(
                r["region_phrase"] for r in records)),
        }

    modes = sorted(records_by_mode)
    if len(modes) == 2:
        a, b = modes
        by_key = {
            mode: {(r["category"], r["stem"]): r for r in records_by_mode[mode]}
            for mode in modes
        }
        shared = sorted(set(by_key[a]) & set(by_key[b]))
        disagree = [
            {"category": c, "stem": s,
             a: (by_key[a][(c, s)]["parsed_output"] or {}).get("result"),
             b: (by_key[b][(c, s)]["parsed_output"] or {}).get("result")}
            for c, s in shared
            if (by_key[a][(c, s)]["parsed_output"] or {}).get("result")
            != (by_key[b][(c, s)]["parsed_output"] or {}).get("result")
        ]
        out["mode_comparison"] = {
            "modes": modes,
            "n_shared_keys": len(shared),
            "n_disagree": len(disagree),
            "disagreement_rate": round(len(disagree) / len(shared), 4) if shared else None,
            "disagreements": disagree[:40],
        }
    return out


def cross_tabs(records_by_mode: dict[str, list[dict]]) -> dict:
    """Result vs. upstream evidence, per mode. This is evidence, not ground truth.

    The ``result x phase3 visual_irregularity`` and ``result x phase4
    global_consistency`` tabs only arrange what the model was *shown*; they
    compare the verdict against its own inputs, not against a label.
    """
    tabs: dict = {}
    for mode, records in records_by_mode.items():
        for name, key_fn, values in (
            ("phase3_visual_irregularity",
             lambda r: (r.get("_upstream") or {}).get("visual_irregularity"),
             ("present", "absent", "unclear", None)),
            ("phase4_global_consistency",
             lambda r: (r.get("_upstream") or {}).get("global_consistency"),
             ("high", "medium", "low", None)),
        ):
            tab = {v: Counter() for v in values}
            for r in records:
                tab.setdefault(key_fn(r), Counter())[r["result"]] += 1
            tabs[f"{mode}::{name}"] = {
                str(k): dict(sorted(v.items())) for k, v in sorted(
                    tab.items(), key=lambda kv: str(kv[0]))}
    return tabs


# ---------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 evidence fusion run")
    parser.add_argument("--map-root", default=DEFAULT_PHASE2_MAP_ROOT)
    parser.add_argument("--phase3-root", default=DEFAULT_PHASE3_ROOT)
    parser.add_argument("--phase4-root", default=DEFAULT_PHASE4_ROOT)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--categories", default="bottle,cable")
    parser.add_argument("--reference-modes", default=REFERENCE_MODES,
                        help="comma-separated: " + REFERENCE_MODES)
    parser.add_argument("--observation-mode", default=DEFAULT_OBSERVATION_MODE)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--limit-per-category", type=int, default=None)
    parser.add_argument("--per-type", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-missing-evidence", action="store_true",
                        help="degrade a missing upstream record to 'not available' "
                             "instead of failing. Off by default: a silent degrade "
                             "produces a plausible and wrong experiment")
    parser.add_argument("--no-region-crop", action="store_true")
    parser.add_argument("--state-reference-count", action="store_true")
    parser.add_argument("--determinism-check", action="store_true",
                        help="run the reproducibility checks (costs a few extra calls)")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-parse-retries", type=int, default=2)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--max-wait-ms", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*Kwargs passed to.*")

    map_root = Path(args.map_root)
    out_root = Path(args.out_root)
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    modes = [m.strip() for m in args.reference_modes.split(",") if m.strip()]
    for mode in modes:
        if mode not in REFERENCE_K_BY_MODE:
            raise SystemExit(
                f"unknown reference mode {mode!r}; expected one of "
                f"{sorted(REFERENCE_K_BY_MODE)}")

    attn = None if args.attn_implementation in ("", "None", "none") else args.attn_implementation
    configs = {
        mode: FusionConfig(
            observation_mode=args.observation_mode,
            reference_mode=mode,
            rank=args.rank,
            data_root=args.data_root,
            phase2_map_root=args.map_root,
            phase3_root=args.phase3_root,
            phase4_root=args.phase4_root,
            out_root=args.out_root,
            attach_region_crop=not args.no_region_crop,
            require_all_evidence=not args.allow_missing_evidence,
            state_reference_count=args.state_reference_count,
            model_path=args.model_path,
            attn_implementation=attn,
            max_new_tokens=args.max_new_tokens,
            max_parse_retries=args.max_parse_retries,
            max_batch_size=args.max_batch_size,
            max_wait_ms=args.max_wait_ms,
            concurrency=args.concurrency,
        )
        for mode in modes
    }
    agents = {mode: EvidenceFusionAgent(cfg) for mode, cfg in configs.items()}

    print(f"Phase 5 evidence fusion | modes={modes} | model={args.model_path}")
    print(f"  observation_mode: {args.observation_mode}  rank: {args.rank}")
    print(f"  out_root: {out_root}")

    # ------------------------------------------------------------ enumerate
    keys_by_category: dict[str, list[str]] = {}
    defect_types: dict[str, list[str]] = {}
    for category in categories:
        print(f"\n[{category}] enumerating cached Phase-2 maps ...", flush=True)
        entries = list_entries(map_root, category)
        defect_types[category] = defect_types_of(entries)
        if args.per_type:
            entries = limit_round_robin(entries, args.per_type * len(defect_types[category]))
        elif args.limit_per_category:
            entries = limit_round_robin(entries, args.limit_per_category)
        keys_by_category[category] = [e["stem"] for e in entries]
        print(f"[{category}] {len(keys_by_category[category])} keys "
              f"({', '.join(defect_types[category])})", flush=True)

    # ------------------------------------------------------------ preflight
    t0 = time.time()
    report, problems = preflight(agents, keys_by_category)
    print_preflight(report, keys_by_category)
    print(f"\npreflight resolved "
          f"{sum(len(v) for v in keys_by_category.values()) * len(modes)} "
          f"(key, mode) bundles in {time.time() - t0:.1f}s")

    if problems:
        print(f"\nPREFLIGHT FAILED — {len(problems)} problem(s), first few:")
        for p in problems[:25]:
            print(f"  {p}")
        if not args.allow_missing_evidence:
            print("\nRefusing to run: incomplete evidence would make the normal-"
                  "reference block read 'Not available' and let the verdict be "
                  "driven by the local observation alone.")
            print("Run Phase 4 first, or pass --allow-missing-evidence to accept a "
                  "degraded run.")
            sys.exit(2)
        print("\n--allow-missing-evidence is set: continuing with a DEGRADED run. "
              "This will be reported as such.")

    if args.preflight_only:
        print("\n--preflight-only: stopping before any model call.")
        sys.exit(0)

    # ------------------------------------------------------------- run
    manifest: dict = {
        "model_path": args.model_path,
        "observation_mode": args.observation_mode,
        "rank": args.rank,
        "degraded_run": bool(problems),
        "allow_missing_evidence": bool(args.allow_missing_evidence),
        "reference_modes": modes,
        "configs": {mode: configs[mode].to_dict() for mode in modes},
        "decoding_signature": {
            mode: configs[mode].decoding_signature() for mode in modes},
        "preflight": report,
        "categories": {},
    }

    records_by_mode: dict[str, list[dict]] = {mode: [] for mode in modes}

    for mode in modes:
        agent, cfg = agents[mode], configs[mode]
        mode_t0 = time.time()
        for category in categories:
            stems = keys_by_category[category]
            n_skipped = 0
            if args.skip_existing:
                kept = [s for s in stems
                        if not already_done(out_root, mode, category, s)]
                n_skipped = len(stems) - len(kept)
                stems = kept

            print(f"\n[{mode}][{category}] {len(stems)} to run"
                  + (f" ({n_skipped} already complete)" if n_skipped else ""),
                  flush=True)
            if stems:
                requests = [{"category": category, "stem": s} for s in stems]
                records = asyncio.run(agent.fuse_many(requests))
                for record in records:
                    write_fusion_record(record, out_root, category, record["stem"])
                records_by_mode[mode].extend(records)
                n_failed = sum(1 for r in records if not r["parse_ok"])
                print(f"  done in {time.time() - mode_t0:.1f}s "
                      f"({n_failed} failed to validate)", flush=True)

        manifest["categories"][mode] = {
            "n_run": len(records_by_mode[mode]),
            "n_parse_ok": sum(1 for r in records_by_mode[mode] if r["parse_ok"]),
        }

    # ------------------------------------------------------------- report
    manifest["summary"] = summarize(records_by_mode)
    manifest["cross_tabs"] = cross_tabs(records_by_mode)

    if args.determinism_check:
        mode = modes[0]
        category = categories[0]
        stem = keys_by_category[category][0]
        print(f"\n[determinism] {mode} / {category} / {stem}", flush=True)
        manifest["determinism"] = determinism_checks(agents[mode], category, stem)

    manifest["wall_time_s"] = round(time.time() - t0, 1)

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 78)
    for mode, stats in manifest["summary"]["per_mode"].items():
        print(f"{mode}: {stats['n_records']} records, ok={stats['n_parse_ok']}, "
              f"retried={stats['n_retried']}, capped={stats['n_hit_token_cap']}, "
              f"mean={stats['mean_elapsed_s']}s")
        print(f"  results: {stats['result_distribution']}")
        if stats["violation_histogram"]:
            print(f"  violations: {stats['violation_histogram']}")
    if "mode_comparison" in manifest["summary"]:
        mc = manifest["summary"]["mode_comparison"]
        print(f"\nmode disagreement: {mc['n_disagree']}/{mc['n_shared_keys']} "
              f"({mc['disagreement_rate']})")
    if manifest.get("determinism"):
        print(f"\ndeterminism: {json.dumps(manifest['determinism'], indent=2, default=str)}")
    print(f"\nwritten: {out_root}")


if __name__ == "__main__":
    main()
