"""Phase 2 smoke test: anomaly_map -> candidate regions -> bbox -> crop.

Verifies, on 1-2 MVTec-AD categories, the Phase-2 acceptance criteria:

  1. every returned candidate bbox is legal (``0 <= x1 < x2 <= 1``,
     ``0 <= y1 < y2 <= 1``, normalised coordinates, top-3 default);
  2. crops are really generated (files exist, non-empty) and are taken from the
     *original* query image;
  3. the anomaly_map -> bbox pipeline is complete and reproducible (a full
     re-run through PatchCore, and a re-derivation from the persisted anomaly
     map, both reproduce the identical candidate list);
  4. Top-1 / Top-3 anomaly coverage can be computed against GT masks;
  5. no ground-truth leakage: the regionisation module never reads GT (checked
     statically), and the query is not a memory (training) image;
  6. the emitted JSON is parseable and follows the Phase-2 schema;
  7. the stage is deterministic (same map + config -> same regions).

Ground-truth masks are read **here only**, after the regions have been produced,
and only to score coverage - never to build them.

Usage::

    /user/pfy/anaconda3/envs/anomalyagent/bin/python smoke_test_phase2.py \
        --categories bottle,cable --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np

from patchcore_predictor import PatchcorePredictor
from patchcore_regions import (
    RegionConfig,
    RegionPredictor,
    extract_regions,
    is_valid_bbox,
)

DEFAULT_DATA_ROOT = "/data/pfy/dataset/MVTec-AD"
DEFAULT_CACHE_ROOT = "/data/pfy/AgentIAD/results/phase1/cache"   # reuse Phase-1 memory
DEFAULT_MAP_ROOT = "/data/pfy/AgentIAD/results/phase2/maps"
DEFAULT_CROP_ROOT = "/data/pfy/AgentIAD/results/phase2/crops"
DEFAULT_REGION_ROOT = "/data/pfy/AgentIAD/results/phase2/regions"

REQUIRED_TOP_KEYS = {"query_image", "image_score", "candidate_regions", "postprocess"}
REQUIRED_REGION_KEYS = {"rank", "bbox", "region_score", "area_ratio", "crop_path"}
REQUIRED_POST_KEYS = {"threshold", "min_area_ratio", "max_regions"}


def category_test_split(data_root: Path, category: str) -> tuple[list[Path], list[Path]]:
    """Return (defective test images that have a GT mask, good test images)."""
    test_dir = data_root / category / "test"
    gt_dir = data_root / category / "ground_truth"

    defective: list[Path] = []
    for dtype_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        if dtype_dir.name == "good":
            continue
        for img in sorted(dtype_dir.glob("*.png")):
            if (gt_dir / dtype_dir.name / f"{img.stem}_mask.png").is_file():
                defective.append(img)

    good_dir = test_dir / "good"
    good = sorted(good_dir.glob("*.png")) if good_dir.is_dir() else []
    return defective, good


def gt_mask_path(data_root: Path, category: str, query: Path) -> Path:
    rel = query.resolve().relative_to((data_root / category / "test").resolve())
    return data_root / category / "ground_truth" / rel.parts[0] / f"{Path(rel.parts[1]).stem}_mask.png"


def load_gt_mask(data_root: Path, category: str, query: Path) -> np.ndarray:
    """Load the GT anomaly mask (offline evaluation only)."""
    path = gt_mask_path(data_root, category, query)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"GT mask not readable: {path}")
    return mask > 0


def coverage_and_iou(regions, gt_mask: np.ndarray, k: int) -> tuple[float, float]:
    """Recall of GT anomaly pixels covered by top-k regions, and pixel IoU."""
    union = np.zeros(gt_mask.shape, dtype=bool)
    for r in regions[:k]:
        union |= r.mask
    gt_px = int(gt_mask.sum())
    inter = int(np.count_nonzero(gt_mask & union))
    union_px = int(np.count_nonzero(gt_mask | union))
    coverage = inter / gt_px if gt_px > 0 else float("nan")
    iou = inter / union_px if union_px > 0 else float("nan")
    return coverage, iou


def check_static_no_gt_leakage() -> dict:
    """The regionisation module must not mention/read ground truth at all."""
    src = (_REPO_ROOT / "patchcore_regions.py").read_text()
    forbidden = ["ground_truth", "gt_mask", "mask.png"]
    hits = [token for token in forbidden if token in src]
    return {"clean": not hits, "forbidden_tokens_found": hits}


def run_category(args, category: str) -> dict:
    data_root = Path(args.data_root)
    cfg = RegionConfig(threshold=args.threshold,
                       min_area_ratio=args.min_area_ratio,
                       max_regions=args.max_regions)

    predictor = PatchcorePredictor(
        data_root=data_root,
        cache_root=Path(args.cache_root),
        output_root=Path(args.map_root),
        seed=args.seed,
        memory_size=args.memory_size,
        device=args.device,
    )
    region_predictor = RegionPredictor(
        patchcore=predictor,
        crop_root=Path(args.crop_root),
        region_root=Path(args.region_root),
        config=cfg,
    )

    memory_meta = predictor.get_memory(category)
    defective, good = category_test_split(data_root, category)
    if not defective:
        raise RuntimeError(f"[{category}] no defective test image with GT mask found")
    query = defective[0]

    checks: dict = {"category": category,
                    "memory_loaded_from_cache": memory_meta.get("loaded_from_cache"),
                    "query": str(query)}

    # ---- run 1: full pipeline (PatchCore -> map -> regions -> crops -> JSON) --
    out1 = region_predictor.predict_regions(query, category, cfg)

    json_path = Path(args.region_root) / category / f"{query.parent.name}_{query.stem}__regions.json"
    with open(json_path) as f:                       # check 6: parseable JSON
        reloaded = json.load(f)
    checks["json_path"] = str(json_path)
    checks["json_parseable"] = True
    checks["json_schema_ok"] = (
        REQUIRED_TOP_KEYS.issubset(reloaded)
        and all(REQUIRED_REGION_KEYS.issubset(r) for r in reloaded["candidate_regions"])
        and REQUIRED_POST_KEYS == set(reloaded["postprocess"])
    )
    checks["n_candidates"] = len(reloaded["candidate_regions"])
    checks["max_regions_respected"] = checks["n_candidates"] <= cfg.max_regions

    # ---- run 2: full re-run, must reproduce the identical candidate list -----
    out2 = region_predictor.predict_regions(query, category, cfg)
    checks["reproducible_full_rerun"] = (
        out1["candidate_regions"] == out2["candidate_regions"]
        and out1["image_score"] == out2["image_score"]
    )

    # ---- check 1: bboxes legal ----------------------------------------------
    bboxes = [r["bbox"] for r in reloaded["candidate_regions"]]
    checks["all_bbox_valid"] = all(is_valid_bbox(b) for b in bboxes)
    checks["bboxes"] = bboxes

    # ---- check 2: crops exist, non-empty, and come from the original image ---
    crop_ok = True
    crop_info = []
    for r in reloaded["candidate_regions"]:
        p = Path(r["crop_path"])
        exists = p.is_file() and p.stat().st_size > 0
        crop_ok = crop_ok and exists
        crop_info.append({"rank": r["rank"], "crop_path": str(p), "exists": exists,
                          "bytes": p.stat().st_size if p.is_file() else 0})
    checks["crops_ok"] = crop_ok
    checks["crops"] = crop_info

    # crops really are sub-images of the original query, at the bbox pixels
    orig = cv2.imread(str(query))
    same_pixels = True
    for r in reloaded["candidate_regions"]:
        crop = cv2.imread(r["crop_path"])
        x1, y1 = int(round(r["bbox"][0] * orig.shape[1])), int(round(r["bbox"][1] * orig.shape[0]))
        x2, y2 = int(round(r["bbox"][2] * orig.shape[1])), int(round(r["bbox"][3] * orig.shape[0]))
        ref = orig[y1:y2, x1:x2]
        if crop is None or ref.shape != crop.shape or not np.array_equal(ref, crop):
            same_pixels = False
    checks["crops_match_original_pixels"] = same_pixels
    checks["source_of_crop"] = "original query image (verified pixel-identical)"

    # ---- check 4: coverage computable (GT used here, offline only) -----------
    amap = np.load(out1["anomaly_map_raw_path"])
    regions = extract_regions(amap, cfg)
    checks["rederived_from_persisted_map_matches"] = (
        [r.to_dict() for r in regions] ==
        [{k: v for k, v in r.items() if k != "crop_path"} for r in reloaded["candidate_regions"]]
    )
    gt = load_gt_mask(data_root, category, query)
    cov1, iou1 = coverage_and_iou(regions, gt, 1)
    cov3, iou3 = coverage_and_iou(regions, gt, 3)
    checks["coverage_top1"] = round(float(cov1), 4)
    checks["coverage_top3"] = round(float(cov3), 4)
    checks["iou_top1"] = round(float(iou1), 4)
    checks["iou_top3"] = round(float(iou3), 4)
    checks["gt_pixels"] = int(gt.sum())

    # ---- check 5: no leakage -------------------------------------------------
    mem_abs = {str(Path(p).resolve()) for p in memory_meta["memory_images"]}
    checks["query_not_in_memory"] = str(query.resolve()) not in mem_abs
    checks["region_module_gt_clean"] = check_static_no_gt_leakage()

    # ---- a good (normal) query must not crash the pipeline -------------------
    if good:
        good_out = region_predictor.predict_regions(good[0], category, cfg)
        checks["good_query_ok"] = isinstance(good_out["image_score"], float)
        checks["good_query_n_candidates"] = len(good_out["candidate_regions"])

    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 candidate-region smoke test")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--map-root", default=DEFAULT_MAP_ROOT)
    parser.add_argument("--crop-root", default=DEFAULT_CROP_ROOT)
    parser.add_argument("--region-root", default=DEFAULT_REGION_ROOT)
    parser.add_argument("--categories", default="bottle,cable")
    parser.add_argument("--memory-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area-ratio", type=float, default=0.001)
    parser.add_argument("--max-regions", type=int, default=3)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    all_checks = []
    for cat in categories:
        print("=" * 78)
        print(f"[{cat}] Phase 2 smoke")
        print("=" * 78)
        checks = run_category(args, cat)
        print(json.dumps(checks, indent=2))
        all_checks.append(checks)

    required_true = [
        "json_parseable", "json_schema_ok", "max_regions_respected",
        "reproducible_full_rerun", "all_bbox_valid", "crops_ok",
        "crops_match_original_pixels", "rederived_from_persisted_map_matches",
        "query_not_in_memory", "good_query_ok",
    ]
    passed = all(
        all(c.get(k) is True for k in required_true) and c["region_module_gt_clean"]["clean"]
        and np.isfinite(c["coverage_top1"]) and np.isfinite(c["coverage_top3"])
        for c in all_checks
    )
    print("=" * 78)
    print("PHASE 2 SMOKE TEST", "PASSED" if passed else "FAILED")
    print("=" * 78)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
