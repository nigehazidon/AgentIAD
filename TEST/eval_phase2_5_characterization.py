"""Phase 2.5: offline characterization + sensitivity analysis of Phase 2 regions.

This stage is **characterization, not optimization**. It answers:

  1. are candidate regions too large?           -> area-ratio distributions
  2. do normal test images flood us with candidates? -> normal candidate stats
  3. is Top-1 already enough?                   -> Top-1 vs Top-3 comparison
  4. is threshold=0.5 fragile?                  -> threshold sensitivity 0.3..0.7

It is a **pure post-processing analysis**: it reads the anomaly maps already
persisted by Phase 2 (``results/phase2/maps/<category>/*__anomaly_map.npy``) and
re-runs only the regionisation stage (``patchcore_regions.extract_regions``) at
several thresholds. No PatchCore fit and no model inference is performed — the
script never instantiates a predictor.

Ground-truth usage: GT masks are read **only** in the scoring step, after the
candidate regions for that image have already been produced, and are used
exclusively to compute coverage/precision/IoU. They never enter regionisation,
threshold selection, connected components, cropping or any inference decision.
The default threshold is never modified from test labels: every threshold in the
sweep is a fixed config value, reported as-is.

Outputs (independent of `results/phase2/`, which is never written):
    results/phase2_5/summary.json
    results/phase2_5/summary.csv
    results/phase2_5/threshold_sensitivity.csv
    results/phase2_5/category_metrics/<category>.json

Usage::

    /user/pfy/anaconda3/envs/anomalyagent/bin/python eval_phase2_5_characterization.py
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

from patchcore_regions import RegionConfig, extract_regions, is_valid_bbox

DEFAULT_DATA_ROOT = "/data/pfy/dataset/MVTec-AD"
DEFAULT_MAP_ROOT = "/data/pfy/AgentIAD/results/phase2/maps"      # read-only reuse
DEFAULT_RESULT_ROOT = "/data/pfy/AgentIAD/results/phase2_5"      # new, independent
PHASE2_ROOT = Path("/data/pfy/AgentIAD/results/phase2")          # immutable

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
BASE_THRESHOLD = 0.5
MIN_AREA_RATIO = 0.001
MAX_REGIONS = 3

MACRO = "MACRO"


# ------------------------------------------------------------- provenance
def manifest(root: Path) -> dict[str, list[int]]:
    """{relative path: [size, mtime_ns]} for immutability checking."""
    out: dict[str, list[int]] = {}
    if not root.exists():
        return out
    for p in sorted(root.rglob("*")):
        if p.is_file():
            st = p.stat()
            out[str(p.relative_to(root))] = [st.st_size, st.st_mtime_ns]
    return out


def static_no_gt_leakage_check() -> dict:
    """The candidate-extraction module must not depend on ground truth."""
    src = (_REPO_ROOT / "patchcore_regions.py").read_text()
    forbidden = ["ground_truth", "gt_mask", "mask.png"]
    hits = [t for t in forbidden if t in src]
    return {"module": "patchcore_regions.py", "clean": not hits,
            "forbidden_tokens_found": hits}


# ------------------------------------------------------------------ data
def split_images(data_root: Path, category: str) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """(defective [(query, gt_mask)], normal [query]) for one category."""
    test_dir = data_root / category / "test"
    gt_dir = data_root / category / "ground_truth"
    defective: list[tuple[Path, Path]] = []
    for dtype_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        if dtype_dir.name == "good":
            continue
        for img in sorted(dtype_dir.glob("*.png")):
            mask = gt_dir / dtype_dir.name / f"{img.stem}_mask.png"
            if mask.is_file():
                defective.append((img, mask))
    good_dir = test_dir / "good"
    normal = sorted(good_dir.glob("*.png")) if good_dir.is_dir() else []
    return defective, normal


def map_path(map_root: Path, category: str, query: Path) -> Path:
    return map_root / category / f"{query.parent.name}_{query.stem}__anomaly_map.npy"


def score_path(map_root: Path, category: str, query: Path) -> Path:
    return map_root / category / f"{query.parent.name}_{query.stem}__score.json"


# --------------------------------------------------------------- scoring
def aggregate(values: list[float]) -> dict[str, float]:
    """mean / median / P25 / P75 over finite values (NaN when no data)."""
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return {"mean": float("nan"), "median": float("nan"),
                "p25": float("nan"), "p75": float("nan"), "n": 0}
    a = np.asarray(vals, dtype=np.float64)
    return {
        "mean": round(float(a.mean()), 6),
        "median": round(float(np.median(a)), 6),
        "p25": round(float(np.percentile(a, 25)), 6),
        "p75": round(float(np.percentile(a, 75)), 6),
        "n": int(a.size),
    }


def region_metrics(regions, gt: np.ndarray | None, shape: tuple[int, int]) -> dict:
    """Top-1 / Top-3-union coverage, precision, IoU and area ratios for one image.

    ``regions`` come from ``extract_regions`` (anomaly map only). ``gt`` is only
    consulted here, for scoring.
    """
    H, W = shape
    total = float(H * W)
    out: dict = {"n_candidates": len(regions), "bboxes_valid": None}
    if regions:
        out["bboxes_valid"] = all(is_valid_bbox(r.bbox_norm) for r in regions)

    for k in (1, 3):
        if regions:
            union = np.zeros((H, W), dtype=bool)
            for r in regions[:k]:
                union |= r.mask
        else:
            union = np.zeros((H, W), dtype=bool)
        area = int(np.count_nonzero(union))
        out[f"top{k}_mask_area_ratio"] = area / total

        if k == 1:
            if regions:
                b = regions[0].bbox_norm
                out["top1_bbox_area_ratio"] = (b[2] - b[0]) * (b[3] - b[1])
                out["top1_region_score"] = float(regions[0].region_score)
            else:
                out["top1_bbox_area_ratio"] = float("nan")
                out["top1_region_score"] = float("nan")

        if gt is None:  # normal image: no GT, coverage/precision undefined
            out[f"top{k}_coverage"] = float("nan")
            out[f"top{k}_precision"] = float("nan")
            out[f"top{k}_iou"] = float("nan")
            continue

        gt_px = int(np.count_nonzero(gt))
        inter = int(np.count_nonzero(gt & union))
        union_px = int(np.count_nonzero(gt | union))
        out[f"top{k}_coverage"] = (inter / gt_px) if gt_px > 0 else float("nan")
        out[f"top{k}_precision"] = (inter / area) if area > 0 else float("nan")
        out[f"top{k}_iou"] = (inter / union_px) if union_px > 0 else float("nan")
    return out


def summarize_records(records: list[dict]) -> dict:
    """Aggregate per-image records into the reported metric block."""
    def col(key):
        return [r[key] for r in records if key in r]

    summary = {
        "n_images": len(records),
        "candidate_rate": round(float(np.mean([r["n_candidates"] > 0 for r in records])), 6),
        "average_candidate_count": round(float(np.mean(col("n_candidates"))), 6),
        "empty_candidate_rate": round(float(np.mean([r["n_candidates"] == 0 for r in records])), 6),
        "all_bboxes_valid": bool(all(r["bboxes_valid"] for r in records if r["bboxes_valid"] is not None)),
    }
    for k in (1, 3):
        summary[f"top{k}"] = {
            "coverage": aggregate(col(f"top{k}_coverage"))["mean"],
            "precision": aggregate(col(f"top{k}_precision"))["mean"],
            "iou": aggregate(col(f"top{k}_iou"))["mean"],
            "area_ratio_mean": aggregate(col(f"top{k}_mask_area_ratio"))["mean"],
        }
    summary["top1_area_ratio"] = aggregate(col("top1_mask_area_ratio"))
    summary["top3_area_ratio"] = aggregate(col("top3_mask_area_ratio"))
    summary["top1_bbox_area_ratio"] = aggregate(col("top1_bbox_area_ratio"))
    summary["top1_region_score"] = aggregate(col("top1_region_score"))
    return summary


# ------------------------------------------------------------------- main
def analyze_category(args, category: str) -> tuple[dict, list[dict]]:
    data_root = Path(args.data_root)
    map_root = Path(args.map_root)
    defective, normal = split_images(data_root, category)

    per_threshold: dict[str, dict] = {}
    base_records: dict[str, list[dict]] = {"defective": [], "normal": []}

    for thr in THRESHOLDS:
        cfg = RegionConfig(threshold=thr, min_area_ratio=MIN_AREA_RATIO,
                           max_regions=MAX_REGIONS)
        recs: dict[str, list[dict]] = {"defective": [], "normal": []}

        # --- defective: regionise from cached map, then score against GT ------
        for img, mask_path in defective:
            amap = np.load(map_path(map_root, category, img))
            regions = extract_regions(amap, cfg)          # anomaly map only
            gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
            if gt.shape != amap.shape:
                raise RuntimeError(f"[{category}] GT {gt.shape} != map {amap.shape} for {img}")
            rec = region_metrics(regions, gt, amap.shape)
            rec.update({"query_image": str(img), "split": "defective",
                        "defect_type": img.parent.name})
            sp = score_path(map_root, category, img)
            rec["image_score"] = json.loads(sp.read_text())["image_score"] if sp.is_file() else None
            recs["defective"].append(rec)

        # --- normal: regionise only; there is no GT for normal images ---------
        for img in normal:
            amap = np.load(map_path(map_root, category, img))
            regions = extract_regions(amap, cfg)
            rec = region_metrics(regions, None, amap.shape)
            rec["split"] = "normal"
            sp = score_path(map_root, category, img)
            rec["image_score"] = json.loads(sp.read_text())["image_score"] if sp.is_file() else None
            recs["normal"].append(rec)

        per_threshold[str(thr)] = {split: summarize_records(recs[split])
                                   for split in ("defective", "normal")}
        if thr == BASE_THRESHOLD:
            base_records = recs

    # ---- Top-1 vs Top-3 comparison at the base threshold ---------------------
    d = per_threshold[str(BASE_THRESHOLD)]["defective"]
    comparison = {
        "coverage_gain": round(d["top3"]["coverage"] - d["top1"]["coverage"], 6),
        "precision_change": round(d["top3"]["precision"] - d["top1"]["precision"], 6),
        "iou_change": round(d["top3"]["iou"] - d["top1"]["iou"], 6),
        "area_increase": round(d["top3"]["area_ratio_mean"] - d["top1"]["area_ratio_mean"], 6),
        "area_ratio_ratio": round(d["top3"]["area_ratio_mean"] / d["top1"]["area_ratio_mean"], 6)
        if d["top1"]["area_ratio_mean"] else float("nan"),
    }

    metrics = {
        "category": category,
        "dataset": "MVTec-AD",
        "memory_size": args.memory_size,
        "base_postprocess": {"threshold": BASE_THRESHOLD,
                             "min_area_ratio": MIN_AREA_RATIO,
                             "max_regions": MAX_REGIONS},
        "n_defective_test_images": len(defective),
        "n_normal_test_images": len(normal),
        "defective": d,
        "normal": per_threshold[str(BASE_THRESHOLD)]["normal"],
        "top1_vs_top3": comparison,
        "threshold_sensitivity": per_threshold,
        "images": base_records["defective"] + base_records["normal"],
    }
    return metrics, base_records["defective"] + base_records["normal"]


def macro_of(category_metrics: list[dict]) -> dict:
    """Macro-average the per-category summaries (equal weight per category)."""
    def avg(path_keys, key):
        vals = []
        for m in category_metrics:
            node = m
            for p in path_keys:
                node = node[p]
            v = node[key]
            if np.isfinite(v):
                vals.append(float(v))
        return round(float(np.mean(vals)), 6) if vals else float("nan")

    def avg_scalar(path_keys, key):
        return avg(path_keys, key)

    out = {
        "n_categories": len(category_metrics),
        "n_defective_test_images": int(sum(m["n_defective_test_images"] for m in category_metrics)),
        "n_normal_test_images": int(sum(m["n_normal_test_images"] for m in category_metrics)),
        "defective": {
            "n_images": int(sum(m["n_defective_test_images"] for m in category_metrics)),
            **{k: avg_scalar(["defective"], k) for k in
               ("candidate_rate", "average_candidate_count", "empty_candidate_rate")},
            **{f"top{k}": {key: avg(["defective", f"top{k}"], key)
                           for key in ("coverage", "precision", "iou", "area_ratio_mean")}
               for k in (1, 3)},
            **{f"top{k}_area_ratio": {key: avg(["defective", f"top{k}_area_ratio"], key)
                                      for key in ("mean", "median", "p25", "p75")}
               for k in (1, 3)},
            "top1_bbox_area_ratio": {key: avg(["defective", "top1_bbox_area_ratio"], key)
                                     for key in ("mean", "median", "p25", "p75")},
            "top1_region_score": {key: avg(["defective", "top1_region_score"], key)
                                  for key in ("mean", "median", "p25", "p75")},
            "all_bboxes_valid": bool(all(m["defective"]["all_bboxes_valid"] for m in category_metrics)),
        },
        "normal": {
            "n_images": int(sum(m["n_normal_test_images"] for m in category_metrics)),
            **{k: avg_scalar(["normal"], k) for k in
               ("candidate_rate", "average_candidate_count", "empty_candidate_rate")},
            "top1_area_ratio_mean": avg(["normal", "top1_area_ratio"], "mean"),
            "top1_region_score_mean": avg(["normal", "top1_region_score"], "mean"),
            # nested forms so the CSV flattener can report the distribution too
            **{f"top{k}_area_ratio": {key: avg(["normal", f"top{k}_area_ratio"], key)
                                      for key in ("mean", "median", "p25", "p75")}
               for k in (1, 3)},
            "top1_region_score": {key: avg(["normal", "top1_region_score"], key)
                                  for key in ("mean", "median", "p25", "p75")},
            "top1_bbox_area_ratio": {key: avg(["normal", "top1_bbox_area_ratio"], key)
                                     for key in ("mean", "median", "p25", "p75")},
            "all_bboxes_valid": bool(all(m["normal"]["all_bboxes_valid"] for m in category_metrics)),
        },
        "top1_vs_top3": {key: round(float(np.mean([m["top1_vs_top3"][key] for m in category_metrics
                                                   if np.isfinite(m["top1_vs_top3"][key])])), 6)
                         for key in ("coverage_gain", "precision_change", "iou_change",
                                     "area_increase", "area_ratio_ratio")},
    }
    return out


def macro_block(blocks: list[dict]) -> dict:
    """Field-wise macro average of per-category metric blocks (recursive).

    Floats -> mean over categories (NaN excluded); ints -> sum; bools -> all();
    nested dicts -> recursed. Keeps percentiles as the mean of per-category
    percentiles, which is the natural macro counterpart.
    """
    out: dict = {}
    for key in blocks[0]:
        values = [b[key] for b in blocks if key in b]
        if not values:
            continue
        sample = values[0]
        if isinstance(sample, dict):
            out[key] = macro_block(values)
        elif isinstance(sample, bool):
            out[key] = bool(all(values))
        elif isinstance(sample, int):
            out[key] = int(sum(values))
        elif isinstance(sample, (float, np.floating)):
            finite = [float(v) for v in values if np.isfinite(v)]
            out[key] = round(float(np.mean(finite)), 6) if finite else float("nan")
        else:
            out[key] = sample
    return out


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


NAN = float("nan")


def flat_row(name: str, split: str, m: dict) -> dict:
    """Flatten a metric block into one CSV row (missing keys -> NaN)."""
    def area(node: str, stat: str = "mean"):
        return m.get(node, {}).get(stat, NAN)

    def top(k: int, key: str):
        return m.get(f"top{k}", {}).get(key, NAN)

    return {
        "category": name, "split": split, "n_images": m.get("n_images"),
        "top1_coverage": top(1, "coverage"), "top1_precision": top(1, "precision"),
        "top1_iou": top(1, "iou"),
        "top1_area_ratio_mean": area("top1_area_ratio", "mean"),
        "top1_area_ratio_median": area("top1_area_ratio", "median"),
        "top1_area_ratio_p25": area("top1_area_ratio", "p25"),
        "top1_area_ratio_p75": area("top1_area_ratio", "p75"),
        "top3_coverage": top(3, "coverage"), "top3_precision": top(3, "precision"),
        "top3_iou": top(3, "iou"),
        "top3_area_ratio_mean": area("top3_area_ratio", "mean"),
        "top3_area_ratio_median": area("top3_area_ratio", "median"),
        "candidate_rate": m.get("candidate_rate", NAN),
        "average_candidate_count": m.get("average_candidate_count", NAN),
        "empty_candidate_rate": m.get("empty_candidate_rate", NAN),
        "top1_region_score_mean": area("top1_region_score"),
        "top1_bbox_area_ratio_mean": area("top1_bbox_area_ratio"),
        "all_bboxes_valid": m.get("all_bboxes_valid"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2.5 offline characterization")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--map-root", default=DEFAULT_MAP_ROOT,
                        help="cached Phase-2 anomaly maps (read-only)")
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--categories", default="",
                        help="comma list; default = every category with cached maps")
    parser.add_argument("--memory-size", type=int, default=24)
    args = parser.parse_args()

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    map_root = Path(args.map_root)
    if args.categories.strip():
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    else:
        categories = sorted(p.name for p in map_root.iterdir() if p.is_dir())
    if not categories:
        raise SystemExit(f"no cached anomaly maps under {map_root}")

    # ---- immutability: snapshot Phase 2 outputs before the analysis ----------
    before = manifest(PHASE2_ROOT)
    leakage = static_no_gt_leakage_check()

    result_root = Path(args.result_root)
    (result_root / "category_metrics").mkdir(parents=True, exist_ok=True)

    per_category: list[dict] = []
    sens_rows: list[dict] = []
    summary_rows: list[dict] = []

    for cat in categories:
        print(f"[{cat}] characterizing cached anomaly maps ...", flush=True)
        metrics, _ = analyze_category(args, cat)
        per_category.append(metrics)
        with open(result_root / "category_metrics" / f"{cat}.json", "w") as f:
            json.dump(metrics, f, indent=2)

        for split in ("defective", "normal"):
            summary_rows.append(flat_row(cat, split, metrics[split]))
            for thr in THRESHOLDS:
                block = metrics["threshold_sensitivity"][str(thr)][split]
                sens_rows.append(flat_row(cat, split, block) | {"threshold": thr})

    macro = macro_of(per_category)
    macro_sens: dict[str, dict[str, dict]] = {}
    for split in ("defective", "normal"):
        summary_rows.append(flat_row(MACRO, split, macro[split]))
        macro_sens[split] = {}
        for thr in THRESHOLDS:
            block = macro_block([m["threshold_sensitivity"][str(thr)][split] for m in per_category])
            macro_sens[split][str(thr)] = block
            sens_rows.append(flat_row(MACRO, split, block) | {"threshold": thr})

    # ---- immutability check: Phase 2 outputs must be byte-for-byte untouched -
    after = manifest(PHASE2_ROOT)
    unchanged = before == after

    summary = {
        "dataset": "MVTec-AD",
        "memory_size": args.memory_size,
        "base_postprocess": {"threshold": BASE_THRESHOLD,
                             "min_area_ratio": MIN_AREA_RATIO,
                             "max_regions": MAX_REGIONS},
        "thresholds_swept": list(THRESHOLDS),
        "categories": [m["category"] for m in per_category],
        "metric_definitions": {
            "coverage": "|candidate_mask ∩ gt_mask| / |gt_mask|",
            "precision": "|candidate_mask ∩ gt_mask| / |candidate_mask|",
            "iou": "|candidate_mask ∩ gt_mask| / |candidate_mask ∪ gt_mask|",
            "area_ratio": "candidate_mask_area / image_area",
            "top3": "union of the Top-3 candidate masks (fewer if fewer exist)",
            "aggregation": "mean over images, then equal-weight macro over categories",
            "normal": "normal images have no GT anomaly -> only proposal statistics reported",
        },
        "defective": macro["defective"],
        "normal": macro["normal"],
        "top1_vs_top3": macro["top1_vs_top3"],
        "threshold_sensitivity": {
            str(thr): {"defective": macro_sens["defective"][str(thr)],
                       "normal": macro_sens["normal"][str(thr)]}
            for thr in THRESHOLDS},
        "per_category": {m["category"]: {"defective": m["defective"], "normal": m["normal"],
                                         "top1_vs_top3": m["top1_vs_top3"]}
                         for m in per_category},
        "provenance": {
            "source_anomaly_maps": str(map_root),
            "patchcore_refit": False,
            "patchcore_inference": False,
            "note": "regionisation re-run offline from cached anomaly maps only",
            "phase2_outputs_unchanged": unchanged,
            "phase2_files_checked": len(before),
            "gt_leakage_static_check": leakage,
        },
    }
    with open(result_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    fields = ["category", "split", "n_images", "top1_coverage", "top1_precision", "top1_iou",
              "top1_area_ratio_mean", "top1_area_ratio_median", "top1_area_ratio_p25",
              "top1_area_ratio_p75", "top3_coverage", "top3_precision", "top3_iou",
              "top3_area_ratio_mean", "top3_area_ratio_median", "candidate_rate",
              "average_candidate_count", "empty_candidate_rate", "top1_region_score_mean",
              "top1_bbox_area_ratio_mean", "all_bboxes_valid"]
    write_csv(result_root / "summary.csv", summary_rows, fields)
    write_csv(result_root / "threshold_sensitivity.csv",
              sens_rows, ["threshold", *fields])

    print(f"\nPhase 2 outputs unchanged: {unchanged} ({len(before)} files compared)")
    print(f"GT-leakage static check: {leakage}")
    print(f"written: {result_root}")


if __name__ == "__main__":
    main()
