"""Phase 3 driver: local visual observation over the Phase-2 candidate cache.

For every test image of the requested categories this script

  1. reads the **cached** Phase-2 anomaly map (``results/phase2/maps/...``),
  2. re-derives the candidate regions from it with the unchanged Phase-2
     ``extract_regions`` (no PatchCore fit, no PatchCore inference),
  3. crops each candidate from the **original query image** and saves the patch
     under ``results/phase3/patches/``,
  4. asks the local Qwen3-VL for a local visual observation of that patch, in
     both input modes (``image_and_patch`` and ``patch_only``),
  5. writes one JSON record per (query, rank, mode) under
     ``results/phase3/observations/``.

No anomaly score is ever placed in the prompt; no ground truth is read anywhere.

Usage::

    # smoke first
    .../python run_phase3_observation.py --limit-per-category 10

    # then the full split (233 images: bottle 83, cable 150)
    .../python run_phase3_observation.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from local_visual_observation import (
    DEFAULT_MODEL_PATH,
    INPUT_MODES,
    LocalVisualObserver,
    ObservationConfig,
    write_observation,
)
from patchcore_regions import (
    RegionConfig,
    extract_regions,
    read_query_image_bgr,
    write_crop,
)

DEFAULT_MAP_ROOT = "/data/pfy/AgentIAD/results/phase2/maps"
DEFAULT_PATCH_ROOT = "/data/pfy/AgentIAD/results/phase3/patches"
DEFAULT_OBS_ROOT = "/data/pfy/AgentIAD/results/phase3/observations"
PHASE2_ROOT = Path("/data/pfy/AgentIAD/results/phase2")


# --------------------------------------------------------------------- inputs
def list_entries(map_root: Path, category: str) -> list[dict]:
    """Every cached test image of ``category``, from the Phase-2 score files."""
    cat_dir = map_root / category
    if not cat_dir.is_dir():
        raise RuntimeError(f"[{category}] no cached anomaly maps under {cat_dir}")

    entries: list[dict] = []
    for score_file in sorted(cat_dir.glob("*__score.json")):
        stem = score_file.name[: -len("__score.json")]
        map_path = cat_dir / f"{stem}__anomaly_map.npy"
        if not map_path.is_file():
            continue
        with open(score_file) as f:
            rec = json.load(f)
        entries.append({
            "stem": stem,
            "query_image": rec["query_image"],
            "image_score": rec.get("image_score"),
            "map_path": map_path,
            # authoritative, taken from the query path rather than the stem
            "defect_type": Path(rec["query_image"]).parent.name,
        })
    if not entries:
        raise RuntimeError(f"[{category}] no cached anomaly maps in {cat_dir}")
    return entries


def limit_round_robin(entries: list[dict], n: int | None) -> list[dict]:
    """Take ``n`` entries spread across the defect types.

    Round-robin rather than a prefix, so a smoke run always includes the
    ``good`` (normal) test images and not only the first defect type.
    """
    if n is None or n >= len(entries):
        return entries
    groups: dict[str, list[dict]] = {}
    for entry in entries:
        groups.setdefault(entry["defect_type"], []).append(entry)

    out: list[dict] = []
    depth = 0
    while len(out) < n:
        added = False
        for name in sorted(groups):
            if depth < len(groups[name]):
                out.append(groups[name][depth])
                added = True
                if len(out) >= n:
                    break
        if not added:
            break
        depth += 1
    return out


def build_requests(
    entries: list[dict],
    category: str,
    cfg: RegionConfig,
    patch_root: Path,
) -> tuple[list[dict], list[dict]]:
    """Regionise from the cached maps and crop the patches from the originals."""
    requests: list[dict] = []
    image_rows: list[dict] = []

    for entry in entries:
        amap = np.load(entry["map_path"])
        regions = extract_regions(amap, cfg)          # anomaly map only, no GT
        image = read_query_image_bgr(entry["query_image"])
        if image.shape[:2] != amap.shape:
            raise RuntimeError(
                f"[{category}] anomaly map {amap.shape} does not match original "
                f"image {image.shape[:2]} for {entry['query_image']}")

        for r in regions:
            patch_path = patch_root / category / f"{entry['stem']}__rank{r.rank}.png"
            write_crop(image, r, patch_path)
            requests.append({
                "query_image": entry["query_image"],
                "patch_path": str(patch_path),
                # traceability only - never sent to the model
                "category": category,
                "rank": int(r.rank),
                "bbox": [round(float(v), 6) for v in r.bbox_norm],
                "region_score": round(float(r.region_score), 6),
                "region_area_ratio": round(float(r.area_ratio), 6),
                "image_score": (round(float(entry["image_score"]), 6)
                                if entry["image_score"] is not None else None),
            })

        image_rows.append({
            "stem": entry["stem"],
            "query_image": entry["query_image"],
            "defect_type": entry["defect_type"],
            "n_candidates": len(regions),
        })
    return requests, image_rows


# ----------------------------------------------------------------- execution
async def observe_all_modes(
    observer: LocalVisualObserver,
    requests: list[dict],
    modes: list[str],
    base_cfg: ObservationConfig,
) -> dict[str, list[dict]]:
    """Run every mode inside one event loop (so the model loads only once)."""
    results: dict[str, list[dict]] = {}
    for mode in modes:
        cfg = ObservationConfig(**{**base_cfg.to_dict(), "input_mode": mode})
        t0 = time.time()
        results[mode] = await observer.observe_many(requests, cfg)
        n_failed = sum(1 for r in results[mode] if not r["parse_ok"])
        print(f"  [{mode}] {len(requests)} requests in {time.time() - t0:.1f}s "
              f"({n_failed} failed to validate)", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 local visual observation")
    parser.add_argument("--map-root", default=DEFAULT_MAP_ROOT)
    parser.add_argument("--patch-root", default=DEFAULT_PATCH_ROOT)
    parser.add_argument("--obs-root", default=DEFAULT_OBS_ROOT)
    parser.add_argument("--categories", default="bottle,cable")
    parser.add_argument("--limit-per-category", type=int, default=None,
                        help="smoke: take only N images per category, spread over defect types")
    parser.add_argument("--modes", default=",".join(INPUT_MODES))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-parse-retries", type=int, default=2)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--max-wait-ms", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=32)
    # Phase-2 region parameters: the defaults are unchanged from Phase 2.
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area-ratio", type=float, default=0.001)
    parser.add_argument("--max-regions", type=int, default=3)
    args = parser.parse_args()

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    map_root = Path(args.map_root)
    patch_root = Path(args.patch_root)
    obs_root = Path(args.obs_root)
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for mode in modes:
        if mode not in INPUT_MODES:
            raise SystemExit(f"unknown mode {mode!r}; expected one of {INPUT_MODES}")

    region_cfg = RegionConfig(threshold=args.threshold,
                              min_area_ratio=args.min_area_ratio,
                              max_regions=args.max_regions)
    attn = None if args.attn_implementation in ("", "None", "none") else args.attn_implementation

    print(f"Phase 3 local visual observation | modes={modes} | model={args.model_path}")
    print(f"  region config (unchanged from Phase 2): {region_cfg.to_dict()}")
    if args.limit_per_category:
        print(f"  SMOKE: {args.limit_per_category} images per category")

    observer = LocalVisualObserver(ObservationConfig(
        model_path=args.model_path,
        attn_implementation=attn,
        max_new_tokens=args.max_new_tokens,
        max_parse_retries=args.max_parse_retries,
        max_batch_size=args.max_batch_size,
        max_wait_ms=args.max_wait_ms,
        concurrency=args.concurrency,
    ))

    manifest = {
        "model_path": args.model_path,
        "modes": modes,
        "region_config": region_cfg.to_dict(),
        "categories": {},
    }

    for category in categories:
        print(f"\n[{category}] enumerating cached Phase-2 maps ...", flush=True)
        entries = list_entries(map_root, category)
        entries = limit_round_robin(entries, args.limit_per_category)
        requests, image_rows = build_requests(entries, category, region_cfg, patch_root)
        defect_types = sorted({e["defect_type"] for e in entries})
        print(f"[{category}] {len(entries)} images ({', '.join(defect_types)}) "
              f"-> {len(requests)} candidate patches", flush=True)

        results = asyncio.run(observe_all_modes(observer, requests, modes, observer.config))

        for mode, records in results.items():
            for record in records:
                write_observation(record, obs_root, category, _stem_of(record), record["rank"])

        manifest["categories"][category] = {
            "n_images": len(entries),
            "n_patches": len(requests),
            "defect_types": defect_types,
            "images": image_rows,
            "per_mode": {
                mode: {
                    "n_records": len(records),
                    "n_parse_ok": sum(1 for r in records if r["parse_ok"]),
                    "n_retried": sum(1 for r in records if r["attempts"] > 1),
                    "mean_attempts": round(
                        float(np.mean([r["attempts"] for r in records])), 4),
                    "mean_elapsed_s": round(
                        float(np.mean([r["elapsed_s"] for r in records])), 4),
                }
                for mode, records in results.items()
            },
        }

    obs_root.mkdir(parents=True, exist_ok=True)
    with open(obs_root / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 70)
    for category, block in manifest["categories"].items():
        print(f"{category}: {block['n_images']} images, {block['n_patches']} patches")
        for mode, stats in block["per_mode"].items():
            print(f"  {mode:<18} ok={stats['n_parse_ok']}/{stats['n_records']} "
                  f"retried={stats['n_retried']} "
                  f"mean_attempts={stats['mean_attempts']} "
                  f"mean_elapsed={stats['mean_elapsed_s']}s")
    print(f"\nwritten: {obs_root}")
    if PHASE2_ROOT.is_dir():
        print(f"Phase 2 results were only read (maps/crops), never written: {PHASE2_ROOT}")


def _stem_of(record: dict) -> str:
    """``<defect_type>_<image>`` from the patch filename."""
    return Path(record["patch_path"]).name.split("__rank")[0]


if __name__ == "__main__":
    main()
