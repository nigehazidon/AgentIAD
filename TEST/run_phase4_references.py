"""Phase 4 driver: normal-reference evidence over every cached test image.

For every test image of the requested categories this script runs the unchanged
Phase-4 Normal Reference Evidence Tool in each requested reference mode and
writes one JSON record per (query, mode) under ``results/phase4/``.

The tool itself is not modified: this driver only enumerates query paths, calls
``compare_many`` and persists what comes back.  No ground truth is read
anywhere, and no test image is used to build a reference pool.

Usage::

    # regression baseline against the 13 smoke records (~2 min)
    .../python run_phase4_references.py --per-type 1 \
        --out-root /data/pfy/AgentIAD/tmp/phase4_regression

    # full split (233 images: bottle 83, cable 150)
    .../python run_phase4_references.py --skip-existing
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "TEST")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from normal_reference_tool import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUT_ROOT,
    MODE_NAMES,
    NormalReferenceEvidenceTool,
    ReferenceToolConfig,
    write_reference_record,
)
from phase_common import defect_types_of, limit_round_robin, list_entries  # noqa: E402

DEFAULT_MAP_ROOT = "/data/pfy/AgentIAD/results/phase2/maps"
DEFAULT_MODE_NAMES = ",".join(MODE_NAMES[k] for k in sorted(MODE_NAMES))
K_BY_MODE_NAME = {name: k for k, name in MODE_NAMES.items()}


# --------------------------------------------------------------------- resume
def already_done(out_root: Path, mode_name: str, category: str, stem: str) -> bool:
    """True only for a record that exists **and** carries a parsed output.

    Resuming on mere existence would let a run that crashed mid-way (or a
    record whose ``parsed_output`` is ``None``) be skipped forever.
    """
    path = out_root / mode_name / category / f"{stem}.json"
    if not path.is_file():
        return False
    try:
        with open(path) as f:
            rec = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return rec.get("parsed_output") is not None


# ----------------------------------------------------------------- execution
async def run_modes(
    tools: dict[str, NormalReferenceEvidenceTool],
    requests: list[dict],
    configs: dict[str, ReferenceToolConfig],
) -> dict[str, list[dict]]:
    """Run every mode sequentially inside one event loop (model loads once).

    Sequential rather than concurrent: each ``compare_many`` opens its own
    ``concurrency``-wide semaphore, so overlapping them would multiply the
    in-flight image count.  Phase 3 established this pattern.
    """
    results: dict[str, list[dict]] = {}
    for mode_name, tool in tools.items():
        t0 = time.time()
        records = await tool.compare_many(requests, configs[mode_name])
        results[mode_name] = records
        n_failed = sum(1 for r in records if not r["parse_ok"])
        n_retried = sum(1 for r in records if r["vlm_calls"] > 1)
        print(f"  [{mode_name}] {len(records)} requests in {time.time() - t0:.1f}s "
              f"({n_failed} failed to validate, {n_retried} retried)", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 full normal-reference run")
    parser.add_argument("--map-root", default=DEFAULT_MAP_ROOT)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--categories", default="bottle,cable")
    parser.add_argument("--modes", default=DEFAULT_MODE_NAMES,
                        help="comma-separated mode names")
    parser.add_argument("--per-type", type=int, default=None,
                        help="smoke: N images from each test defect folder")
    parser.add_argument("--limit-per-category", type=int, default=None,
                        help="smoke: at most N images per category, spread over defect types")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip a (query, mode) whose record already parses")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--max-new-tokens", type=int, default=512)
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
    mode_names = [m.strip() for m in args.modes.split(",") if m.strip()]
    for name in mode_names:
        if name not in K_BY_MODE_NAME:
            raise SystemExit(f"unknown mode {name!r}; expected one of {list(K_BY_MODE_NAME)}")

    attn = None if args.attn_implementation in ("", "None", "none") else args.attn_implementation
    common = dict(
        model_path=args.model_path, data_root=args.data_root,
        attn_implementation=attn, max_new_tokens=args.max_new_tokens,
        max_parse_retries=args.max_parse_retries, max_batch_size=args.max_batch_size,
        max_wait_ms=args.max_wait_ms, concurrency=args.concurrency,
    )
    configs = {
        name: ReferenceToolConfig(reference_k=K_BY_MODE_NAME[name], **common)
        for name in mode_names
    }
    tools = {name: NormalReferenceEvidenceTool(cfg) for name, cfg in configs.items()}
    # Modes must share one embedder instance so retrieval is provably identical.
    first = mode_names[0]
    for name in mode_names[1:]:
        tools[name]._embedder = tools[first].embedder

    print(f"Phase 4 normal-reference evidence | modes={mode_names} | model={args.model_path}")
    print(f"  out_root: {out_root}")
    if args.per_type:
        print(f"  SMOKE: {args.per_type} image(s) per defect type")
    elif args.limit_per_category:
        print(f"  SMOKE: at most {args.limit_per_category} images per category")

    manifest = {
        "model_path": args.model_path,
        "modes": {name: configs[name].to_dict() for name in mode_names},
        "decoding_signature": {name: configs[name].decoding_signature() for name in mode_names},
        "source_root": str(map_root),
        "categories": {},
    }

    for category in categories:
        print(f"\n[{category}] enumerating cached Phase-2 maps ...", flush=True)
        entries = list_entries(map_root, category)
        defect_types = defect_types_of(entries)
        if args.per_type:
            entries = limit_round_robin(entries, args.per_type * len(defect_types))
        elif args.limit_per_category:
            entries = limit_round_robin(entries, args.limit_per_category)

        n_selected = len(entries)
        n_skipped = 0
        if args.skip_existing:
            kept = [e for e in entries
                    if not all(already_done(out_root, m, category, e["stem"])
                               for m in mode_names)]
            n_skipped = n_selected - len(kept)
            entries = kept

        print(f"[{category}] {n_selected} images ({', '.join(defect_types)})"
              + (f", {n_skipped} already complete" if n_skipped else "")
              + f" -> {len(entries)} to run", flush=True)

        requests = [{"query_image": e["query_image"], "category": category,
                     "stem": e["stem"]} for e in entries]

        results: dict[str, list[dict]] = {name: [] for name in mode_names}
        if requests:
            results = asyncio.run(run_modes(tools, requests, configs))
            for name, records in results.items():
                for record in records:
                    write_reference_record(record, out_root, category, record["stem"])

        manifest["categories"][category] = {
            "n_images": n_selected,
            "n_run": len(requests),
            "n_skipped_existing": n_skipped,
            "defect_types": defect_types,
            "per_mode": {
                name: {
                    "n_records": len(records),
                    "n_parse_ok": sum(1 for r in records if r["parse_ok"]),
                    "n_retried": sum(1 for r in records if r["vlm_calls"] > 1),
                    "mean_elapsed_s": (round(sum(r["elapsed_s"] for r in records)
                                             / len(records), 4) if records else None),
                }
                for name, records in results.items()
            },
        }

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 70)
    for category, block in manifest["categories"].items():
        print(f"{category}: {block['n_images']} images, {block['n_run']} run, "
              f"{block['n_skipped_existing']} skipped")
        for name, stats in block["per_mode"].items():
            print(f"  {name:<26} ok={stats['n_parse_ok']}/{stats['n_records']} "
                  f"retried={stats['n_retried']} mean_elapsed={stats['mean_elapsed_s']}s")
    print(f"\nwritten: {out_root}")


if __name__ == "__main__":
    main()
