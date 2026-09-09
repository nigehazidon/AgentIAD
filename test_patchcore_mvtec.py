"""PatchCore smoke test on MVTecAD.

For each MVTecAD category: fit PatchCore on the category's training split (builds
the memory bank and computes normalization/thresholds from val = full test copy),
then predict on the first N defective test images (has GT mask). Saves per-image
artifacts (original, GT mask, anomaly heatmap, predicted mask, montage) plus a
metrics.json under <result_root>/<category>/, and a summary.csv at result_root.

Run (weight downloads need HF reachable; mirror env is unset on purpose):
    env -u HF_ENDPOINT /user/pfy/anaconda3/envs/anomalyagent/bin/python \
        test_patchcore_mvtec.py --categories bottle --max-train-images 24
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

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from anomalib.data import MVTecAD
from anomalib.data.dataclasses import ImageBatch
from anomalib.engine import Engine
from anomalib.models import Patchcore

MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


class MVTecLimited(MVTecAD):
    """MVTecAD datamodule that restricts the number of training samples.

    Slices the train samples after the per-category dataset is built (but before
    the base datamodule derives the val/test splits) so the full anomalib setup /
    model-resize injection still runs untouched.
    """

    def __init__(self, max_train_images: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_train_images = int(max_train_images)

    def _setup(self, stage: str | None = None) -> None:
        super()._setup(stage)
        if self.max_train_images > 0 and self.train_data is not None:
            n = min(self.max_train_images, len(self.train_data.samples))
            self.train_data.samples = self.train_data.samples.iloc[:n].reset_index(drop=True)


def _squeeze(t: torch.Tensor) -> torch.Tensor:
    """Drop leading singleton dims of a per-sample item field."""
    a = t.detach().cpu()
    while a.ndim > 2:
        a = a[0]
    return a


def _iou_dice(pred_bool: np.ndarray, gt_bool: np.ndarray) -> tuple[float, float]:
    inter = float(np.logical_and(pred_bool, gt_bool).sum())
    union = float(np.logical_or(pred_bool, gt_bool).sum())
    denom = float(pred_bool.sum()) + float(gt_bool.sum())
    iou = inter / union if union > 0 else float("nan")
    dice = 2.0 * inter / denom if denom > 0 else float("nan")
    return iou, dice


def save_artifacts(cat_dir: Path, stem: str, orig_bgr: np.ndarray, heat: np.ndarray,
                   gt_u8: np.ndarray, pred_u8: np.ndarray) -> Path:
    """Write per-image PNGs and a 4-panel montage into cat_dir. Return montage path."""
    image_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    H, W = image_rgb.shape[:2]
    gt_rgb = cv2.cvtColor(cv2.resize(gt_u8, (W, H), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2RGB)
    pred_rgb = cv2.cvtColor(cv2.resize(pred_u8, (W, H), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2RGB)
    heat_rgb = cv2.resize(heat, (W, H), interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(str(cat_dir / f"{stem}__image.png"), orig_bgr)
    cv2.imwrite(str(cat_dir / f"{stem}__gt_mask.png"), gt_u8)
    cv2.imwrite(str(cat_dir / f"{stem}__pred_mask.png"), pred_u8)
    cv2.imwrite(str(cat_dir / f"{stem}__heatmap.png"), cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR))

    labels = ["image", "GT mask", "heatmap", "pred mask"]
    cells = []
    max_h = max(im.shape[0] for im in (image_rgb, gt_rgb, heat_rgb, pred_rgb))
    for im, lab in zip((image_rgb, gt_rgb, heat_rgb, pred_rgb), labels):
        if im.shape[0] < max_h:
            scale = max_h / im.shape[0]
            im = cv2.resize(im, (int(round(im.shape[1] * scale)), max_h), interpolation=cv2.INTER_AREA)
        cv2.putText(im, lab, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cells.append(im)
    montage = cv2.hconcat(cells)
    montage_path = cat_dir / f"{stem}__montage.png"
    cv2.imwrite(str(montage_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    return montage_path


def run_category(args, category: str) -> None:
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
    defect_pos = [i for i in range(len(samples)) if samples.iloc[i]["mask_path"] is not None]
    picks = defect_pos[: args.num_test_images]
    if not picks:
        raise RuntimeError(f"[{category}] no defective test sample with GT mask found.")
    print(f"[{category}] predict on {len(picks)} test image(s): "
          f"{[samples.iloc[p]['image_path'] for p in picks]}", flush=True)
    predict_loader = DataLoader(
        Subset(datamodule.test_data, picks),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=datamodule.test_data.collate_fn,
    )
    raw_preds = engine.predict(model=model, dataloaders=[predict_loader], return_predictions=True)

    items = []
    for pred in raw_preds if raw_preds is not None else []:
        if isinstance(pred, ImageBatch):
            items.extend(pred)
        else:
            items.append(pred)
    if len(items) != len(picks):
        raise RuntimeError(f"[{category}] expected {len(picks)} predictions, got {len(items)}")

    records = []
    for item in items:
        image_path = Path(item.image_path)
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise RuntimeError(f"[{category}] failed to read {image_path}")
        anomaly_map = _squeeze(item.anomaly_map).float().numpy()  # ~[0,1]
        pred_mask = _squeeze(item.pred_mask).bool().numpy()
        gt_mask = _squeeze(item.gt_mask)
        gt_mask = gt_mask.bool().numpy() if gt_mask.dtype != torch.bool else gt_mask.numpy()
        if gt_mask.dtype != np.bool_:
            gt_mask = gt_mask > 0.5

        iou, dice = _iou_dice(pred_mask, gt_mask)
        stem = f"{image_path.parent.name}_{image_path.stem}"
        heat_u8 = np.clip(anomaly_map * 255.0, 0, 255).astype(np.uint8)
        heat_jet = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        heat_rgb = cv2.cvtColor(heat_jet, cv2.COLOR_BGR2RGB)
        gt_u8 = (gt_mask.astype(np.uint8)) * 255
        pred_u8 = (pred_mask.astype(np.uint8)) * 255
        montage_path = save_artifacts(cat_dir, stem, img_bgr, heat_rgb, gt_u8, pred_u8)

        gt_label = int(_squeeze(item.gt_label).item())
        pred_label = int(_squeeze(item.pred_label).item())
        rec = {
            "category": category,
            "image_path": str(image_path),
            "gt_label": gt_label,
            "pred_label": pred_label,
            "classification_correct": bool(gt_label == pred_label),
            "pred_score": float(_squeeze(item.pred_score).item()),
            "pixel_iou": round(iou, 4),
            "pixel_dice": round(dice, 4),
            "train_images_used": (
                len(datamodule.train_data.samples) if hasattr(datamodule, "train_data") else -1
            ),
            "montage": str(montage_path.relative_to(result_root)),
        }
        records.append(rec)

    with open(cat_dir / "metrics.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"[{category}] done in {time.time() - t_start:.1f}s -> {cat_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="PatchCore smoke test on MVTecAD")
    parser.add_argument("--data-root", default="/data/pfy/dataset/MVTec-AD")
    parser.add_argument("--result-root", default="/data/pfy/AgentIAD/results/patchcore")
    parser.add_argument("--categories", default=",".join(MVTEC_CATEGORIES),
                        help="comma-separated category list")
    parser.add_argument("--max-train-images", type=int, default=0,
                        help="0 = use full training split for the memory bank; >0 limits to N")
    parser.add_argument("--num-test-images", type=int, default=1,
                        help="number of defective test images per category to predict")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--num-neighbors", type=int, default=9)
    parser.add_argument("--devices", type=int, default=1)
    args = parser.parse_args()

    result_root = Path(args.result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    summary_path = result_root / "summary.csv"
    fields = ["category", "image_path", "gt_label", "pred_label", "classification_correct",
              "pred_score", "pixel_iou", "pixel_dice", "train_images_used", "montage"]
    summary = []
    for cat in categories:
        try:
            run_category(args, cat)
        except Exception as exc:  # noqa: BLE001 - keep going across categories
            print(f"[{cat}] FAILED: {exc!r}", flush=True)
        if (result_root / cat / "metrics.json").exists():
            with open(result_root / cat / "metrics.json") as f:
                summary.extend(json.load(f))

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
