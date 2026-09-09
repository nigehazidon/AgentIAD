"""PatchCore full test-set evaluation on MVTecAD (image-level + pixel-level).

For each MVTecAD category: fit PatchCore on the category's training split (builds
the memory bank and computes normalization / anomaly threshold from val = test
copy, anomalib's SAME_AS_TEST default), then evaluate over the ENTIRE test split.

Metrics (computed by anomalib's Evaluator during ``engine.test`` on every test
image, at the model's 256x256 resolution):

  image-level (all test images, good + defective):
    image_AUROC / image_AUPR   (threshold-free, pred_score vs gt_label)
    image_F1Score              (F1 at the validation-derived anomaly threshold)
    image_F1Max                (best F1 over thresholds on the test set)
  pixel-level (images with a GT mask; background of good images excluded):
    pixel_AUROC / pixel_AUPR / pixel_F1Max   (anomaly_map vs gt_mask)
    pixel_F1Score              (at the validation-derived threshold)
    pixel_AUPRO                (per-region overlap, FPR-limited at 0.3)

Outputs under <result_root>/<category>/metrics.json (per category) plus a
summary.csv and overall_metrics.json at result_root.

Run (HF mirror env is unset on purpose so timm weights hit huggingface.co):
    env -u HF_ENDPOINT /user/pfy/anaconda3/envs/anomalyagent/bin/python \
        eval_patchcore_full_mvtec.py --categories bottle --max-train-images 24
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

_ANOMALIB_SRC = Path("/data/pfy/AgentIAD/anomalib/src").resolve()
if str(_ANOMALIB_SRC) not in sys.path:
    sys.path.insert(0, str(_ANOMALIB_SRC))

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUPR, AUROC, Evaluator, F1Max, F1Score
from anomalib.models import Patchcore

from test_patchcore_mvtec import MVTecLimited, MVTEC_CATEGORIES


def build_evaluator() -> Evaluator:
    """Evaluator with image- and pixel-level metrics for the full test loop."""
    test_metrics = [
        AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
        AUPR(fields=["pred_score", "gt_label"], prefix="image_"),
        F1Score(fields=["pred_label", "gt_label"], prefix="image_"),
        F1Max(fields=["pred_score", "gt_label"], prefix="image_"),
        AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False),
        AUPR(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False),
        F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_", strict=False),
        F1Max(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False),
        AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False),
    ]
    return Evaluator(test_metrics=test_metrics)


def run_category(args, category: str) -> dict:
    """Fit PatchCore for one category and evaluate over the whole test split."""
    t_start = time.time()
    result_root = Path(args.result_root)
    cat_dir = result_root / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    dm_kwargs = dict(
        root=args.data_root,
        category=category,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    if args.max_train_images > 0:
        datamodule = MVTecLimited(max_train_images=args.max_train_images, **dm_kwargs)
    else:
        datamodule = MVTecAD(**dm_kwargs)

    model = Patchcore(
        backbone=args.backbone,
        layers=["layer2", "layer3"],
        pre_trained=True,
        num_neighbors=args.num_neighbors,
        evaluator=build_evaluator(),
        visualizer=False,
    )
    engine = Engine(
        max_epochs=1,
        devices=args.devices,
        accelerator="auto",
        barebones=True,  # skip writing large memory-bank checkpoints
        default_root_dir=str(result_root / "engine"),
    )
    print(f"[{category}] fit: building memory bank from train split ...", flush=True)
    engine.fit(model=model, datamodule=datamodule)

    samples = datamodule.test_data.samples
    mask_col = samples["mask_path"]
    n_test = int(len(samples))
    n_defective = int(mask_col.notna().sum())
    n_normal = n_test - n_defective

    print(f"[{category}] test: evaluating over all {n_test} test images "
          f"({n_normal} good, {n_defective} defective) ...", flush=True)
    results = engine.test(model=model, datamodule=datamodule, verbose=False)
    metrics: dict[str, float] = {}
    for out in results or []:
        for key, value in out.items():
            metrics[key] = float(value) if hasattr(value, "item") else float(value)

    record = {
        "category": category,
        "backbone": args.backbone,
        "num_neighbors": args.num_neighbors,
        "train_images_used": len(datamodule.train_data.samples),
        "n_test_images": n_test,
        "n_normal": n_normal,
        "n_defective": n_defective,
        **metrics,
    }
    with open(cat_dir / "metrics.json", "w") as f:
        json.dump(record, f, indent=2)
    print(f"[{category}] done in {time.time() - t_start:.1f}s", flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="PatchCore full test-set evaluation on MVTecAD")
    parser.add_argument("--data-root", default="/data/pfy/dataset/MVTec-AD")
    parser.add_argument("--result-root", default="/data/pfy/AgentIAD/results/patchcore/full_eval")
    parser.add_argument("--categories", default=",".join(MVTEC_CATEGORIES),
                        help="comma-separated category list")
    parser.add_argument("--max-train-images", type=int, default=0,
                        help="0 = use full training split for the memory bank; >0 limits to N (smoke)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--num-neighbors", type=int, default=9)
    parser.add_argument("--devices", type=int, default=1)
    args = parser.parse_args()

    result_root = Path(args.result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    records: list[dict] = []
    for cat in categories:
        try:
            run_category(args, cat)
        except Exception as exc:  # noqa: BLE001 - keep going across categories
            print(f"[{cat}] FAILED: {exc!r}", flush=True)
        if (result_root / cat / "metrics.json").exists():
            with open(result_root / cat / "metrics.json") as f:
                records.append(json.load(f))

    metric_keys = [
        "image_AUROC", "image_AUPR", "image_F1Score", "image_F1Max",
        "pixel_AUROC", "pixel_AUPR", "pixel_F1Score", "pixel_F1Max", "pixel_AUPRO",
    ]
    fieldnames = ["category", "train_images_used", "n_test_images", "n_normal", "n_defective", *metric_keys]
    with open(result_root / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({k: rec.get(k) for k in fieldnames} for rec in records)

    overall = {
        "n_categories": len(records),
        "per_category": records,
        "macro_mean": {
            key: round(float(sum(rec[key] for rec in records if key in rec)) / max(len(records), 1), 6)
            for key in metric_keys
        },
    }
    with open(result_root / "overall_metrics.json", "w") as f:
        json.dump(overall, f, indent=2)

    print("Macro-average over evaluated categories:")
    for key in metric_keys:
        vals = [rec[key] for rec in records if key in rec]
        if vals:
            print(f"  {key:16s} = {sum(vals) / len(vals):.4f}")
    print(f"Summary written to {result_root}")


if __name__ == "__main__":
    main()
