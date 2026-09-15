"""Phase 2 offline evaluation: candidate regions vs MVTec-AD ground-truth masks.

Runs the Phase-2 pipeline (PatchCore anomaly map -> normalize -> threshold ->
connected components -> bbox -> crop) over the test split of 1-2 categories and
scores the produced candidate regions against the MVTec-AD ground-truth masks.

IMPORTANT — ground-truth usage: the GT mask is read in this script *only*, in a
separate step that runs **after** the candidate regions have been produced, and
is used exclusively to compute evaluation metrics.  It is never passed into the
regionisation code, so it cannot influence the bboxes.  The threshold and all
other region parameters are plain config values, never derived from labels.

Metrics (evaluation population = defective test images, i.e. those that have a
GT mask):

  * ``valid_bbox_rate``        fraction of produced bboxes that are legal
                               (``0 <= x1 < x2 <= 1``, ``0 <= y1 < y2 <= 1``);
  * ``anomaly_coverage``       mean fraction of GT anomaly pixels covered by the
                               top-1 / top-3 candidate regions;
  * ``iou``                    mean pixel IoU between the union of the top-1 /
                               top-3 candidate region masks and the GT mask;
  * ``empty_candidate_rate``   fraction of defective images for which the
                               pipeline returned zero candidates;
  * ``average_candidate_count`` mean number of candidate regions per image.

Usage::

    /user/pfy/anaconda3/envs/anomalyagent/bin/python eval_phase2_regions.py \
        --categories bottle,cable --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np

from patchcore_predictor import PatchcorePredictor
from patchcore_regions import RegionConfig, extract_regions, is_valid_bbox

DEFAULT_DATA_ROOT = "/data/pfy/dataset/MVTec-AD"
DEFAULT_CACHE_ROOT = "/data/pfy/AgentIAD/results/phase1/cache"   # reuse Phase-1 memory
DEFAULT_MAP_ROOT = "/data/pfy/AgentIAD/results/phase2/maps"
DEFAULT_RESULT_ROOT = "/data/pfy/AgentIAD/results/phase2/eval"


# --------------------------------------------------------------------- data
def defective_test_images(data_root: Path, category: str) -> list[tuple[Path, Path]]:
    """(query image, GT mask) for every defective test image of ``category``."""
    test_dir = data_root / category / "test"
    gt_dir = data_root / category / "ground_truth"
    pairs: list[tuple[Path, Path]] = []
    for dtype_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        if dtype_dir.name == "good":
            continue
        for img in sorted(dtype_dir.glob("*.png")):
            mask = gt_dir / dtype_dir.name / f"{img.stem}_mask.png"
            if mask.is_file():
                pairs.append((img, mask))
    return pairs


def good_test_images(data_root: Path, category: str) -> list[Path]:
    d = data_root / category / "test" / "good"
    return sorted(d.glob("*.png")) if d.is_dir() else []


# ---------------------------------------------------------------- inference
def ensure_prediction(predictor: PatchcorePredictor, query: Path, category: str) -> dict:
    """Reuse the persisted anomaly map/score when present, else run PatchCore."""
    cat_dir = Path(predictor.output_root) / category
    stem = f"{query.parent.name}_{query.stem}"
    npy = cat_dir / f"{stem}__anomaly_map.npy"
    js = cat_dir / f"{stem}__score.json"
    if npy.is_file() and js.is_file():
        with open(js) as f:
            rec = json.load(f)
        if "image_score" in rec:
            return rec
    return predictor.predict(query, category)


# ------------------------------------------------------------ GT comparison
def coverage_and_iou(regions, gt_mask: np.ndarray, k: int) -> tuple[float, float]:
    union = np.zeros(gt_mask.shape, dtype=bool)
    for r in regions[:k]:
        union |= r.mask
    gt_px = int(gt_mask.sum())
    inter = int(np.count_nonzero(gt_mask & union))
    union_px = int(np.count_nonzero(gt_mask | union))
    coverage = inter / gt_px if gt_px > 0 else float("nan")
    iou = inter / union_px if union_px > 0 else float("nan")
    return coverage, iou


def evaluate_category(args, category: str, cfg: RegionConfig) -> tuple[dict, list[dict]]:
    """Regionise every defective test image and score it against its GT mask."""
    data_root = Path(args.data_root)
    predictor = PatchcorePredictor(
        data_root=data_root,
        cache_root=Path(args.cache_root),
        output_root=Path(args.map_root),
        seed=args.seed,
        memory_size=args.memory_size,
        device=args.device,
    )
    memory_meta = predictor.get_memory(category)

    pairs = defective_test_images(data_root, category)
    if not pairs:
        raise RuntimeError(f"[{category}] no defective test image with GT mask")

    records: list[dict] = []
    n_bbox = n_bbox_valid = 0
    for img, mask_path in pairs:
        # --- 1) regionisation: anomaly map in, regions out (no GT involved) ---
        pred = ensure_prediction(predictor, img, category)
        amap = np.load(pred["anomaly_map_raw_path"])
        regions = extract_regions(amap, cfg)

        # --- 2) offline scoring against the GT mask (never fed back upstream) -
        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
        if gt.shape != amap.shape:
            raise RuntimeError(
                f"[{category}] GT mask {gt.shape} != anomaly map {amap.shape} for {img}")

        for r in regions:
            n_bbox += 1
            n_bbox_valid += int(is_valid_bbox(r.bbox_norm))

        cov1, iou1 = coverage_and_iou(regions, gt, 1)
        cov3, iou3 = coverage_and_iou(regions, gt, 3)

        records.append({
            "category": category,
            "query_image": str(img),
            "defect_type": img.parent.name,
            "image_score": float(pred["image_score"]),
            "n_candidates": len(regions),
            "bboxes": [r.to_dict()["bbox"] for r in regions],
            "valid_bbox": [is_valid_bbox(r.bbox_norm) for r in regions],
            "empty": len(regions) == 0,
            "gt_pixels": int(gt.sum()),
            "coverage_top1": round(float(cov1), 6),
            "coverage_top3": round(float(cov3), 6),
            "iou_top1": round(float(iou1), 6),
            "iou_top3": round(float(iou3), 6),
        })

    n = len(records)
    metrics = {
        "category": category,
        "n_defective_test_images": n,
        "valid_bbox_rate": round(n_bbox_valid / n_bbox, 6) if n_bbox else float("nan"),
        "n_bbox_total": n_bbox,
        "anomaly_coverage_top1": round(float(np.mean([r["coverage_top1"] for r in records])), 6),
        "anomaly_coverage_top3": round(float(np.mean([r["coverage_top3"] for r in records])), 6),
        "iou_top1": round(float(np.mean([r["iou_top1"] for r in records])), 6),
        "iou_top3": round(float(np.mean([r["iou_top3"] for r in records])), 6),
        "empty_candidate_rate": round(float(np.mean([r["empty"] for r in records])), 6),
        "average_candidate_count": round(float(np.mean([r["n_candidates"] for r in records])), 6),
        "postprocess": cfg.to_dict(),
        "memory_images": memory_meta["memory_images"],
        "memory_size": memory_meta["memory_size"],
        "seed": memory_meta["seed"],
        "memory_loaded_from_cache": memory_meta.get("loaded_from_cache"),
    }

    # Extra context (not part of the headline metrics): candidate counts on the
    # category's normal test images. No coverage is possible there (no anomaly).
    if args.include_good:
        counts = []
        for img in good_test_images(data_root, category):
            pred = ensure_prediction(predictor, img, category)
            amap = np.load(pred["anomaly_map_raw_path"])
            counts.append(len(extract_regions(amap, cfg)))
        metrics["n_good_test_images"] = len(counts)
        metrics["average_candidate_count_on_good"] = (
            round(float(np.mean(counts)), 6) if counts else float("nan"))
        metrics["empty_candidate_rate_on_good"] = (
            round(float(np.mean([c == 0 for c in counts])), 6) if counts else float("nan"))

    return metrics, records


# ------------------------------------------------------------------ output
def print_table(rows: list[dict], macro: dict) -> None:
    header = (f"{'category':<12} {'n':>4} {'valid_bbox_rate':>16} "
              f"{'cov_top1':>9} {'cov_top3':>9} {'iou_top1':>9} {'iou_top3':>9} "
              f"{'empty_rate':>11} {'avg_count':>10}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['category']:<12} {r['n_defective_test_images']:>4} "
              f"{r['valid_bbox_rate']:>16.4f} {r['anomaly_coverage_top1']:>9.4f} "
              f"{r['anomaly_coverage_top3']:>9.4f} {r['iou_top1']:>9.4f} "
              f"{r['iou_top3']:>9.4f} {r['empty_candidate_rate']:>11.4f} "
              f"{r['average_candidate_count']:>10.4f}")
    print("-" * len(header))
    print(f"{'MACRO':<12} {macro['n_defective_test_images']:>4} "
          f"{macro['valid_bbox_rate']:>16.4f} {macro['anomaly_coverage_top1']:>9.4f} "
          f"{macro['anomaly_coverage_top3']:>9.4f} {macro['iou_top1']:>9.4f} "
          f"{macro['iou_top3']:>9.4f} {macro['empty_candidate_rate']:>11.4f} "
          f"{macro['average_candidate_count']:>10.4f}")


MACRO_KEYS = ["valid_bbox_rate", "anomaly_coverage_top1", "anomaly_coverage_top3",
              "iou_top1", "iou_top3", "empty_candidate_rate", "average_candidate_count"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 candidate-region offline evaluation")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--map-root", default=DEFAULT_MAP_ROOT)
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--categories", default="bottle,cable")
    parser.add_argument("--memory-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area-ratio", type=float, default=0.001)
    parser.add_argument("--max-regions", type=int, default=3)
    parser.add_argument("--include-good", action="store_true", default=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    cfg = RegionConfig(threshold=args.threshold,
                       min_area_ratio=args.min_area_ratio,
                       max_regions=args.max_regions)
    result_root = Path(args.result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    rows: list[dict] = []
    for cat in categories:
        print(f"[{cat}] regionising test split ...", flush=True)
        metrics, records = evaluate_category(args, cat, cfg)
        cat_dir = result_root / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        with open(cat_dir / "region_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(cat_dir / "per_image_regions.json", "w") as f:
            json.dump(records, f, indent=2)
        rows.append(metrics)

    macro = {key: round(float(np.mean([r[key] for r in rows])), 6) for key in MACRO_KEYS}
    macro["n_defective_test_images"] = int(sum(r["n_defective_test_images"] for r in rows))
    summary = {
        "postprocess": cfg.to_dict(),
        "evaluation_population": "defective test images (GT mask available)",
        "metric_definitions": {
            "valid_bbox_rate": "legal normalised bboxes / all produced bboxes",
            "anomaly_coverage_topK": "mean fraction of GT anomaly pixels covered by top-K candidate region masks",
            "iou_topK": "mean pixel IoU between union of top-K candidate region masks and GT mask",
            "empty_candidate_rate": "fraction of defective images with zero candidates",
            "average_candidate_count": "mean candidates per defective image",
        },
        "per_category": rows,
        "macro_mean": macro,
    }
    with open(result_root / "phase2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    fields = ["category", "n_defective_test_images", *MACRO_KEYS, "n_bbox_total",
              "memory_loaded_from_cache", "n_good_test_images",
              "average_candidate_count_on_good", "empty_candidate_rate_on_good"]
    with open(result_root / "phase2_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"evaluation population: defective test images | postprocess: {cfg.to_dict()}")
    print_table(rows, macro)
    print(f"\nwritten: {result_root}")


if __name__ == "__main__":
    main()
