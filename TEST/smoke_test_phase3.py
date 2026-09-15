"""Phase 3 smoke test: local visual observation via Qwen3-VL.

Structural checks (no model needed):

  1. the observation module contains no ground-truth dependency (static);
  2. the two input modes differ only in the ``You are given:`` bullet block,
     and attach exactly 2 images (image_and_patch) / 1 image (patch_only);
  3. nothing derived from the anomaly map reaches the model: the outgoing prompt
     text is exactly the system prompt plus the two image labels, with no score
     and no bbox anywhere in it;
  4. the schema validator accepts well-formed replies (plain / fenced / with
     surrounding prose) and rejects malformed ones;

Model checks (a few images per category, both modes):

  5. every record validates against the four-key schema, with enums in range;
  6. both modes produce a record for the same set of (query, rank) pairs;
  7. re-running one request reproduces the identical observation (greedy decoding);
  8. ``results/phase2/`` is byte-for-byte untouched (size + mtime_ns of every file).

Usage::

    .../python smoke_test_phase3.py                 # full check, 4 images/category
    .../python smoke_test_phase3.py --skip-model    # structural checks only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "TEST") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "TEST"))

from local_visual_observation import (  # noqa: E402
    CONFIDENCE_VALUES,
    IMAGE_AND_PATCH,
    IRREGULARITY_VALUES,
    OBSERVATION_KEYS,
    PATCH_ONLY,
    LocalVisualObserver,
    ObservationConfig,
    build_messages,
    parse_observation,
    prompt_text_of,
    system_prompt_for,
)
from patchcore_regions import RegionConfig  # noqa: E402
import run_phase3_observation as driver  # noqa: E402

PHASE2_ROOT = Path("/data/pfy/AgentIAD/results/phase2")


def manifest(root: Path) -> dict:
    """``{relpath: [size, mtime_ns]}`` for every file under ``root``."""
    if not root.is_dir():
        return {}
    return {
        str(p.relative_to(root)): [p.stat().st_size, p.stat().st_mtime_ns]
        for p in sorted(root.rglob("*")) if p.is_file()
    }


# --------------------------------------------------------------- structural
def check_static_no_gt_leakage() -> dict:
    """The observation module must not mention or read ground truth at all."""
    src = (_REPO_ROOT / "local_visual_observation.py").read_text()
    forbidden = ["ground_truth", "gt_mask", "mask.png"]
    hits = [t for t in forbidden if t in src]
    return {"clean": not hits, "forbidden_tokens_found": hits}


def check_prompt_modes() -> dict:
    """The two modes must differ only in the 'You are given:' bullet block."""
    import local_visual_observation as L

    both = system_prompt_for(IMAGE_AND_PATCH)
    patch = system_prompt_for(PATCH_ONLY)

    identical_body = (
        both.replace(L._GIVEN_BLOCK_BOTH, "")
        == patch.replace(L._GIVEN_BLOCK_PATCH_ONLY, "")
    )
    return {
        "verbatim_for_image_and_patch": both == L.LOCAL_INSPECTION_SYSTEM_PROMPT,
        "identical_outside_given_block": identical_body,
        "both_has_two_bullets": L._GIVEN_BLOCK_BOTH in both,
        "patch_only_has_one_bullet": L._GIVEN_BLOCK_PATCH_ONLY in patch,
        "modes_differ": both != patch,
    }


def check_prompt_shape_and_no_score(request: dict) -> dict:
    """The outgoing text must be the system prompt + two labels, and no scores."""
    checks = {}
    for mode, expected_images in ((IMAGE_AND_PATCH, 2), (PATCH_ONLY, 1)):
        messages = build_messages("data:image/png;base64,QQ==", "data:image/png;base64,QQ==", mode)
        n_images = sum(
            1 for item in messages[1]["content"] if item.get("type") == "image_url")
        text = prompt_text_of(messages)
        checks[f"{mode}_n_images"] = n_images == expected_images
        checks[f"{mode}_text_is_prompt_plus_labels"] = text == "\n".join(
            [system_prompt_for(mode), "Original query image:", "Local image patch:"]
            if mode == IMAGE_AND_PATCH
            else [system_prompt_for(mode), "Local image patch:"])

        # No anomaly score, bbox or rank may appear in the conversation.
        leaks = []
        for key in ("region_score", "image_score", "region_area_ratio"):
            value = request.get(key)
            if value is None:
                continue
            if f"{value}" in text or f"{float(value):.6f}" in text:
                leaks.append(key)
        if str(request.get("bbox")) in text or any(
                f"{v:.6f}" in text for v in request.get("bbox", [])):
            leaks.append("bbox")
        checks[f"{mode}_no_score_or_bbox_in_prompt"] = not leaks
        checks[f"{mode}_leaked_fields"] = leaks
        # Note: no separate "rank not in prompt" check — rank is a bare integer
        # that collides with the prompt's own numbered lists, so a substring test
        # would be meaningless. The exact-equality check above is the real
        # guarantee: the conversation is the system prompt plus the image labels
        # and nothing else, so no per-request field can have been added.
    return checks


def check_schema_validator() -> dict:
    good = json.dumps({
        "observation": "a thin scratch along the rim",
        "visual_irregularity": "present",
        "evidence": ["thin bright line"],
        "confidence": "high",
    })
    results = {}
    for label, text, want_ok in (
        ("plain", good, True),
        ("fenced", f"```json\n{good}\n```", True),
        ("prose_wrapped", f"Sure, here it is:\n{good}\nHope that helps.", True),
        ("bad_enum", good.replace('"present"', '"maybe"'), False),
        # an empty evidence list is legitimate for a region that looks normal,
        # but not for one claimed to be irregular
        ("present_with_empty_evidence",
         good.replace('["thin bright line"]', "[]"), False),
        ("absent_with_empty_evidence",
         good.replace('"present"', '"absent"').replace('["thin bright line"]', "[]"), True),
        ("uncertain_with_empty_evidence",
         good.replace('"present"', '"uncertain"').replace('["thin bright line"]', "[]"), True),
        ("missing_key", '{"observation":"x","confidence":"low"}', False),
        ("not_json", "The patch looks fine to me.", False),
        ("empty", "", False),
    ):
        observation, _ = parse_observation(text)
        results[label] = (observation is not None) == want_ok
    return results


# -------------------------------------------------------------------- model
def run_model_checks(args) -> dict:
    observer = LocalVisualObserver(ObservationConfig(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        max_parse_retries=args.max_parse_retries,
        concurrency=args.concurrency,
    ))

    category = args.categories.split(",")[0].strip()
    entries = driver.limit_round_robin(
        driver.list_entries(Path(args.map_root), category), args.limit_per_category)
    requests, _ = driver.build_requests(
        entries, category, RegionConfig(), Path(args.patch_root))
    print(f"[smoke] {category}: {len(requests)} patches from {len(entries)} images")

    async def run() -> dict:
        records: dict[str, list[dict]] = {}
        for mode in (IMAGE_AND_PATCH, PATCH_ONLY):
            cfg = ObservationConfig(**{**observer.config.to_dict(), "input_mode": mode})
            records[mode] = await observer.observe_many(requests, cfg)
        # reproducibility: single request, twice, sequentially (no batch padding)
        cfg = ObservationConfig(**{**observer.config.to_dict(), "input_mode": IMAGE_AND_PATCH})
        first = await observer.observe_many(requests[:1], cfg)
        second = await observer.observe_many(requests[:1], cfg)
        return {"records": records, "first": first[0], "second": second[0]}

    out = asyncio.run(run())
    records = out["records"]

    checks: dict = {}
    for mode, recs in records.items():
        checks[f"{mode}_all_parse_ok"] = all(r["parse_ok"] for r in recs)
        checks[f"{mode}_n_records"] = len(recs)
        checks[f"{mode}_enums_valid"] = all(
            r["observation"]
            and r["observation"]["visual_irregularity"] in IRREGULARITY_VALUES
            and r["observation"]["confidence"] in CONFIDENCE_VALUES
            and set(r["observation"]) == set(OBSERVATION_KEYS)
            for r in recs if r["parse_ok"]
        )
        checks[f"{mode}_score_flag_false"] = all(
            r["anomaly_score_used_in_prompt"] is False for r in recs)
        checks[f"{mode}_system_prompt_saved"] = all(
            r["system_prompt"] == system_prompt_for(mode) for r in recs)
        checks[f"{mode}_failures"] = [
            {"patch": r["patch_path"], "violations": r["violations"]}
            for r in recs if not r["parse_ok"]
        ]

    keys = {
        mode: {(r["query_image"], r["rank"]) for r in recs}
        for mode, recs in records.items()
    }
    checks["both_modes_same_query_rank_pairs"] = (
        keys[IMAGE_AND_PATCH] == keys[PATCH_ONLY] and len(keys[IMAGE_AND_PATCH]) > 0)
    checks["patch_only_keeps_query_traceability"] = all(
        r["query_image"] for r in records[PATCH_ONLY])

    checks["reproducible"] = (
        out["first"]["observation"] == out["second"]["observation"]
        and out["first"]["parse_ok"] == out["second"]["parse_ok"])
    checks["reproducible_raw"] = out["first"]["observation_raw"] == out["second"]["observation_raw"]
    checks["sample_observations"] = [
        {
            "mode": r["input_mode"],
            "patch": Path(r["patch_path"]).name,
            "observation": (r["observation"] or {}).get("observation"),
            "visual_irregularity": (r["observation"] or {}).get("visual_irregularity"),
            "confidence": (r["observation"] or {}).get("confidence"),
        }
        for r in records[IMAGE_AND_PATCH][:3]
    ]
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 local observation smoke test")
    parser.add_argument("--map-root", default=driver.DEFAULT_MAP_ROOT)
    parser.add_argument("--patch-root", default=driver.DEFAULT_PATCH_ROOT)
    parser.add_argument("--categories", default="bottle")
    parser.add_argument("--limit-per-category", type=int, default=4)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-parse-retries", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    if args.model_path is None:
        from local_visual_observation import DEFAULT_MODEL_PATH
        args.model_path = DEFAULT_MODEL_PATH

    before = manifest(PHASE2_ROOT)
    checks: dict = {}
    checks["static_module_gt_clean"] = check_static_no_gt_leakage()
    checks["prompt_modes"] = check_prompt_modes()
    checks["schema_validator"] = check_schema_validator()

    # a real request, so the no-score check runs against actual numbers
    entries = driver.limit_round_robin(
        driver.list_entries(Path(args.map_root), args.categories.split(",")[0].strip()), 1)
    requests, _ = driver.build_requests(
        entries, args.categories.split(",")[0].strip(), RegionConfig(), Path(args.patch_root))
    checks["prompt_shape_and_no_score"] = check_prompt_shape_and_no_score(requests[0])

    if not args.skip_model:
        checks["model"] = run_model_checks(args)

    after = manifest(PHASE2_ROOT)
    checks["phase2_untouched"] = {
        "n_files": len(before),
        "unchanged": before == after,
        "added": sorted(set(after) - set(before)),
        "removed": sorted(set(before) - set(after)),
        "modified": sorted(k for k in set(before) & set(after) if before[k] != after[k]),
    }

    print(json.dumps(checks, indent=2, default=str))

    required = [
        checks["static_module_gt_clean"]["clean"],
        all(checks["prompt_modes"].values()),
        all(checks["schema_validator"].values()),
        all(v for k, v in checks["prompt_shape_and_no_score"].items()
            if isinstance(v, bool)),
        checks["phase2_untouched"]["unchanged"],
    ]
    if not args.skip_model:
        model = checks["model"]
        required += [
            model[f"{IMAGE_AND_PATCH}_all_parse_ok"],
            model[f"{PATCH_ONLY}_all_parse_ok"],
            model[f"{IMAGE_AND_PATCH}_enums_valid"],
            model[f"{PATCH_ONLY}_enums_valid"],
            model[f"{IMAGE_AND_PATCH}_score_flag_false"],
            model[f"{PATCH_ONLY}_score_flag_false"],
            model["both_modes_same_query_rank_pairs"],
            model["patch_only_keeps_query_traceability"],
            model["reproducible"],
        ]

    passed = all(required)
    print("=" * 78)
    print("PHASE 3 SMOKE TEST", "PASSED" if passed else "FAILED")
    print("=" * 78)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
