"""Phase 2: turn PatchCore anomaly maps into candidate regions + crops.

Runs *after* the Phase-1 ``PatchcorePredictor`` (which produces, per query, an
image-level anomaly score and a native-resolution anomaly map ``.npy``) and
converts the anomaly map into a small, ranked set of local candidate regions:

    anomaly_map
      -> normalization (per-image min-max, map to [0,1])
      -> threshold
      -> connected components
      -> candidate regions (filtered by min area, ranked by region score)
      -> bbox (normalised coordinates)
      -> crop (taken from the *original* query image, never from the map)

Design guarantees:

  * every bbox is derived purely from the anomaly map by this code — no
    hand-authored bbox and no LLM involvement;
  * bboxes use normalised coordinates ``[x1, y1, x2, y2]`` with
    ``0 <= x1 < x2 <= 1`` and ``0 <= y1 < y2 <= 1``;
  * nothing in this module reads a ground-truth mask — the pipeline is a pure
    function of ``(anomaly_map, config, original_image)``;
  * defaults return at most ``max_regions`` (3) candidates sorted by region
    anomaly score;
  * the whole stage is deterministic given the same anomaly map and config, so
    results are reproducible.

This module intentionally does **not** emit a normal/anomalous label.

Minimal interface::

    from patchcore_regions import RegionConfig, extract_regions
    regions = extract_regions(anomaly_map, RegionConfig())   # ranked list
    for r in regions:
        print(r.rank, r.bbox, r.region_score, r.area_ratio)

Or, end-to-end through the Phase-1 predictor::

    from patchcore_regions import RegionPredictor
    rp = RegionPredictor(patchcore=patchcore_predictor, crop_root=..., region_root=...)
    out = rp.predict_regions("/.../test/broken_large/000.png", "bottle")
    # out == {query_image, image_score, candidate_regions:[...], postprocess:{...}}
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from patchcore_predictor import PatchcorePredictor

logger = logging.getLogger("phase2.regions")

#: Anomaly-map scaling applied before thresholding (per-image min-max).
NORMALIZATION_METHOD = "minmax"

# Bounding-box legality -------------------------------------------------

def is_valid_bbox(bbox: list[float] | tuple[float, ...]) -> bool:
    """True iff ``bbox == [x1, y1, x2, y2]`` lies in the open unit box.

    Rules (spec): ``0 <= x1 < x2 <= 1`` and ``0 <= y1 < y2 <= 1``.
    """
    if len(bbox) != 4:
        return False
    x1, y1, x2, y2 = (float(v) for v in bbox)
    ok = all(np.isfinite(v) for v in (x1, y1, x2, y2))
    return ok and 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0


# Configuration ----------------------------------------------------------

@dataclass(frozen=True)
class RegionConfig:
    """Independent, user-settable parameters of the regionisation stage.

    ``threshold`` is applied to the per-image min-max *normalised* anomaly map
    (so it is a comparable number in [0,1] independent of the absolute PatchCore
    distance scale) and is saved verbatim with every query record.
    """

    threshold: float = 0.5
    min_area_ratio: float = 0.001
    max_regions: int = 3

    def __post_init__(self) -> None:
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        if not (0.0 <= self.min_area_ratio < 1.0):
            raise ValueError(
                f"min_area_ratio must be in [0, 1), got {self.min_area_ratio}")
        if self.max_regions < 1:
            raise ValueError(f"max_regions must be >= 1, got {self.max_regions}")

    def to_dict(self) -> dict[str, float]:
        return {
            "threshold": float(self.threshold),
            "min_area_ratio": float(self.min_area_ratio),
            "max_regions": int(self.max_regions),
        }


DEFAULT_CONFIG = RegionConfig()


# Normalization ----------------------------------------------------------

def minmax_normalize(anomaly_map: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Map the anomaly map to [0,1] by its per-image min and max.

    Returns the scaled map plus the raw min/max so the scaling is auditable.
    A perfectly flat map becomes all zeros.
    """
    amap = np.asarray(anomaly_map, dtype=np.float64)
    lo = float(np.min(amap))
    hi = float(np.max(amap))
    span = hi - lo
    norm = np.zeros_like(amap)
    if span > 1e-12:
        norm = (amap - lo) / span
    return norm, {"method": NORMALIZATION_METHOD, "raw_min": lo, "raw_max": hi}


# Candidate regions -------------------------------------------------------

@dataclass
class CandidateRegion:
    """One detected anomaly region with its bounding box and mask.

    ``bbox_norm`` is the normalised ``[x1, y1, x2, y2]`` in the unit image box;
    ``bbox_px`` is the same box as pixel slice bounds ``[x1, y1, x2, y2)``
    (upper edge exclusive) used to crop the original image.
    ``mask`` is a bool (H, W) array of exactly the component's pixels.
    """

    rank: int
    bbox_norm: tuple[float, float, float, float]
    bbox_px: tuple[int, int, int, int]
    region_score: float
    area_ratio: float
    n_pixels: int
    mask: np.ndarray = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": int(self.rank),
            "bbox": [round(float(v), 6) for v in self.bbox_norm],
            "region_score": round(float(self.region_score), 6),
            "area_ratio": round(float(self.area_ratio), 6),
        }


def extract_regions(
    anomaly_map: np.ndarray,
    config: RegionConfig | None = None,
) -> list[CandidateRegion]:
    """Full regionisation pipeline on one (native-resolution) anomaly map.

    Implements: normalize -> threshold -> connected components -> filter by
    min area -> rank by mean region anomaly score -> keep top-``max_regions``.

    This is a pure function of the anomaly map and config: no ground truth, no
    labels, no randomness.
    """
    cfg = config or DEFAULT_CONFIG
    amap = np.asarray(anomaly_map, dtype=np.float64)
    if amap.ndim != 2 or amap.size == 0:
        raise ValueError(f"anomaly map must be a non-empty 2D array, got {amap.shape}")
    H, W = int(amap.shape[0]), int(amap.shape[1])
    if H < 1 or W < 1:
        return []

    norm, _ = minmax_normalize(amap)
    binary = norm >= cfg.threshold
    if not bool(binary.any()):
        return []

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8)

    min_pixels = cfg.min_area_ratio * float(H * W)
    found: list[CandidateRegion] = []

    for c in range(1, n_labels):  # 0 == background
        area = int(stats[c, cv2.CC_STAT_AREA])
        if area < min_pixels:
            continue
        comp_mask = labels == c
        x = int(stats[c, cv2.CC_STAT_LEFT])
        y = int(stats[c, cv2.CC_STAT_TOP])
        w = int(stats[c, cv2.CC_STAT_WIDTH])
        h = int(stats[c, cv2.CC_STAT_HEIGHT])
        x2 = min(W, x + w)
        y2 = min(H, y + h)
        if x2 <= x or y2 <= y:  # degenerate (should not happen with area>=1)
            continue

        region_score = float(np.mean(amap[comp_mask]))
        bbox_norm = (float(x) / W, float(y) / H, float(x2) / W, float(y2) / H)
        found.append(
            CandidateRegion(
                rank=-1,  # assigned after sorting
                bbox_norm=bbox_norm,
                bbox_px=(x, y, x2, y2),
                region_score=region_score,
                area_ratio=area / float(H * W),
                n_pixels=area,
                mask=comp_mask,
            )
        )

    if not found:
        return []

    # Rank by region anomaly score (descending); deterministic tie-break by
    # position so the ranking is reproducible.
    found.sort(key=lambda r: (-r.region_score, r.bbox_norm[0], r.bbox_norm[1]))
    found = found[: cfg.max_regions]
    for i, r in enumerate(found, start=1):
        r.rank = i
    return found


# Cropping ---------------------------------------------------------------

def read_query_image_bgr(image_path: str | Path) -> np.ndarray:
    """Read the original query image (BGR) for cropping."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"failed to read image for cropping: {image_path}")
    return img


def crop_region(image: np.ndarray, bbox_px: tuple[int, int, int, int]) -> np.ndarray:
    """Crop ``image`` (H,W[,C]) to a pixel box, clipped to image bounds."""
    H, W = image.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox_px)
    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(x1 + 1, min(x2, W))
    y2 = max(y1 + 1, min(y2, H))
    return image[y1:y2, x1:x2]


def write_crop(image: np.ndarray, region: CandidateRegion, out_path: str | Path) -> Path:
    """Save the crop for ``region`` and return the written path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop = crop_region(image, region.bbox_px)
    if not cv2.imwrite(str(out_path), crop):
        raise IOError(f"failed to write crop: {out_path}")
    return out_path


# End-to-end predictor ----------------------------------------------------

class RegionPredictor:
    """Compose the Phase-1 PatchCore predictor with Phase-2 regionisation.

    ``predict_regions`` runs PatchCore inference (reusing the cached memory
    bank), regionises the resulting anomaly map, crops the candidates from the
    original query image, persists crops + one JSON record per query, and
    returns the record.
    """

    def __init__(
        self,
        patchcore: PatchcorePredictor | None = None,
        *,
        crop_root: str | Path,
        region_root: str | Path,
        config: RegionConfig | None = None,
        **patchcore_kwargs: Any,
    ) -> None:
        if patchcore is None:
            patchcore = PatchcorePredictor(**patchcore_kwargs)
        self.patchcore = patchcore
        self.crop_root = Path(crop_root)
        self.region_root = Path(region_root)
        self.config = config or DEFAULT_CONFIG
        self.crop_root.mkdir(parents=True, exist_ok=True)
        self.region_root.mkdir(parents=True, exist_ok=True)

    def _stem(self, query: Path) -> str:
        return f"{query.parent.name}_{query.stem}"

    def predict_regions(
        self,
        image_path: str | Path,
        category: str,
        config: RegionConfig | None = None,
        *,
        save_crops: bool = True,
    ) -> dict[str, Any]:
        """Full Phase-2 record for one query image.

        Runs PatchCore (cached memory), derives candidate regions purely from the
        anomaly map, crops from the original image, and persists a crop PNG per
        candidate plus a per-query JSON record. The returned dict follows the
        Phase-2 output schema (plus a few traceability keys).
        """
        cfg = config or self.config
        query = Path(image_path)
        t0 = time.time()

        pc = self.patchcore.predict(query, category)  # image_score + anomaly map
        amap_path = Path(pc["anomaly_map_raw_path"])
        amap = np.load(amap_path)  # native-resolution float map
        regions = extract_regions(amap, cfg)

        image = read_query_image_bgr(query)
        if image.shape[:2] != amap.shape:
            raise RuntimeError(
                f"[{category}] anomaly map {amap.shape} does not match original "
                f"image {image.shape[:2]}; coordinates would be misaligned."
            )

        stem = self._stem(query)
        candidates: list[dict[str, Any]] = []
        if save_crops:
            for r in regions:
                crop_path = self.crop_root / category / f"{stem}__rank{r.rank}.png"
                write_crop(image, r, crop_path)
                rec = r.to_dict()
                rec["crop_path"] = str(crop_path)
                candidates.append(rec)
        else:
            for r in regions:
                rec = r.to_dict()
                rec["crop_path"] = ""
                candidates.append(rec)

        record = {
            "query_image": str(query.resolve()),
            "image_score": float(pc["image_score"]),
            "category": category,
            "candidate_regions": candidates,
            "postprocess": cfg.to_dict(),
            # --- traceability (Phase-1 artifacts) ---
            "anomaly_map_path": pc["anomaly_map_path"],
            "anomaly_map_raw_path": str(amap_path),
            "memory_loaded_from_cache": bool(pc.get("memory_loaded_from_cache", False)),
            "elapsed_s": round(time.time() - t0, 3),
        }

        json_path = self.region_root / category / f"{stem}__regions.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(record, f, indent=2)

        logger.info("[%s] %d candidate region(s) for %s -> %s (%.1fs)",
                    category, len(candidates), query.name, json_path.parent,
                    time.time() - t0)
        return record
