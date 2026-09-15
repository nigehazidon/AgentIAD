"""Phase 1: standalone Anomalib-PatchCore predictor for MVTec-AD (24-shot).

This module is deliberately decoupled from the Qwen Agent flow and from the
existing Direct-Qwen baseline. It implements, for *one* MVTec-AD category, a
fixed 24-shot normal-image PatchCore memory with:

  * deterministic 24-image selection from ``<root>/<category>/train/good``
    (fixed seed, list of used paths recorded),
  * a category-level PatchCore memory bank built **only** from those 24 normal
    images (test images never enter the bank),
  * an on-disk memory cache (``cache_root/<category>/memory.pt`` +
    ``metadata.json``) so a second run reuses the bank instead of re-fitting,
  * inference on a single query image -> image-level anomaly score + pixel-level
    anomaly map (both persisted), and
  * logging / error handling.

Minimal interface::

    predictor = PatchcorePredictor(data_root=..., cache_root=..., output_root=...)
    result = predictor.predict(image_path="/.../cable/test/.../000.png",
                               category="cable")
    # result == {"image_score": float, "anomaly_map_path": "...", ...}

No anomaly/normal label is ever returned.

The code reuses the real Anomalib ``PatchcoreModel`` (feature backbone, patch
pooling, nearest-neighbour search, image-score weighting and anomaly-map
generation) but runs it directly on tensors, bypassing Lightning/Engine so the
memory bank can be cached and reused per query.

Note on imports: the vendored Anomalib lives at ``<this repo>/anomalib/src``.
A sibling folder named ``anomalib/`` at the repo root shadows the package if the
working directory is the repo root, so we insert the *src* path at the very
front of ``sys.path`` before importing Anomalib (mirroring
``test_patchcore_mvtec.py``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

_ANOMALIB_SRC = Path(__file__).resolve().parent / "anomalib" / "src"
if str(_ANOMALIB_SRC) not in sys.path:
    sys.path.insert(0, str(_ANOMALIB_SRC))

import cv2
import numpy as np
import torch

from anomalib.data.utils import read_image
from anomalib.models.image.patchcore.lightning_model import Patchcore as _PatchcoreLightning
from anomalib.models.image.patchcore.torch_model import PatchcoreModel

logger = logging.getLogger("phase1.patchcore")

# Anomalib default MVTec categories (kept for validation of the ``category`` arg).
MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood",
    "zipper",
]

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _category_seed(base_seed: int, category: str) -> int:
    """Derive a deterministic per-category integer seed from ``base_seed``.

    Uses sha256 of the category string so the selection is stable across runs
    and independent of the order in which categories are processed.
    """
    digest = hashlib.sha256(category.encode("utf-8")).digest()
    return (base_seed ^ int.from_bytes(digest[:8], "little")) & ((1 << 64) - 1)


class PatchcorePredictor:
    """24-shot Anomalib-PatchCore predictor for one MVTec-AD category.

    Args:
        data_root: Path to the MVTec-AD dataset root.
        cache_root: Directory holding the per-category memory caches.
        output_root: Directory where per-query results (score + anomaly map)
            are written.
        seed: Base random seed used for the deterministic 24-image selection.
        memory_size: Number of normal training images used to build the memory.
        backbone: Timm backbone name used by PatchCore.
        layers: Backbone layers used by PatchCore.
        num_neighbors: Number of neighbours used by PatchCore for the weighted
            image-level score.
        image_size: (H, W) PatchCore input resolution.
        device: Torch device string. ``None`` -> cuda:0 when available else cpu.
    """

    def __init__(
        self,
        *,
        data_root: str | Path,
        cache_root: str | Path,
        output_root: str | Path,
        seed: int = 0,
        memory_size: int = 24,
        backbone: str = "wide_resnet50_2",
        layers: tuple[str, ...] = ("layer2", "layer3"),
        num_neighbors: int = 9,
        image_size: int | tuple[int, int] = 256,
        device: str | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.cache_root = Path(cache_root)
        self.output_root = Path(output_root)
        self.seed = int(seed)
        self.memory_size = int(memory_size)
        self.backbone = backbone
        self.layers = tuple(layers)
        self.num_neighbors = int(num_neighbors)
        self.image_size = (int(image_size), int(image_size)) if isinstance(image_size, int) else tuple(image_size)

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)

        # Pre-processor replicating Anomalib's default PatchCore pre-processing:
        # resize to image_size then ImageNet normalisation over a [0,1] image.
        self.pre_processor = _PatchcoreLightning.configure_pre_processor(image_size=self.image_size)

        self._model: PatchcoreModel | None = None  # lazily built backbone
        self._current_category: str | None = None

    # ------------------------------------------------------------------ config
    @property
    def config(self) -> dict[str, Any]:
        """The part of the config that the cache content depends on."""
        return {
            "backbone": self.backbone,
            "layers": list(self.layers),
            "num_neighbors": self.num_neighbors,
            "image_size": list(self.image_size),
            "memory_size": self.memory_size,
            "seed": self.seed,
            "data_root": str(self.data_root.resolve()),
        }

    # --------------------------------------------------------- image selection
    def category_train_good_paths(self, category: str) -> list[Path]:
        """Return all normal training images of ``category`` (sorted, absolute)."""
        good_dir = self.data_root / category / "train" / "good"
        if not good_dir.is_dir():
            raise FileNotFoundError(f"[{category}] train/good dir not found: {good_dir}")
        paths = sorted(p.resolve() for p in good_dir.glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"[{category}] no normal training images under {good_dir}")
        return paths

    def select_memory_images(self, category: str) -> list[Path]:
        """Deterministically pick ``memory_size`` normal images (seeded sampling).

        The candidate pool is the category's ``train/good`` split only, so by
        construction no test image can ever enter the memory bank.
        """
        pool = self.category_train_good_paths(category)
        if len(pool) < self.memory_size:
            raise RuntimeError(
                f"[{category}] only {len(pool)} normal training images available, "
                f"need {self.memory_size}"
            )
        rng = random.Random(_category_seed(self.seed, category))
        picked = rng.sample(pool, self.memory_size)
        return picked  # selection order == memory construction order

    # -------------------------------------------------------------- backbone
    @property
    def model(self) -> PatchcoreModel:
        """Lazily built PatchCore torch model (backbone weights loaded once)."""
        if self._model is None:
            t0 = time.time()
            logger.info("Loading PatchCore backbone %s (layers=%s) on %s ...",
                        self.backbone, list(self.layers), self.device)
            model = PatchcoreModel(
                backbone=self.backbone,
                layers=list(self.layers),
                pre_trained=True,
                num_neighbors=self.num_neighbors,
            ).to(self.device)
            model.eval()
            self._model = model
            logger.info("Backbone ready in %.1fs.", time.time() - t0)
        return self._model

    # ------------------------------------------------- memory build / load
    def _preprocess(self, image_path: str | Path) -> torch.Tensor:
        """Read one image and return a (1, 3, H, W) normalised tensor."""
        image = read_image(str(image_path), as_tensor=True)  # float [0,1] CHW
        tensor = self.pre_processor(image).unsqueeze(0)
        return tensor

    def _attach_memory(self, memory: torch.Tensor, category: str) -> None:
        """Attach ``memory`` to the shared model for inference on ``category``."""
        if self._current_category != category:
            mem = memory.float().to(self.device)
            self.model.memory_bank = mem
            self._current_category = category
            logger.debug("[%s] memory attached: %d patches x %d dims on %s.",
                         category, *mem.shape, self.device)

    def build_memory(self, category: str) -> dict[str, Any]:
        """Fit a PatchCore memory bank from the 24 selected normal images.

        Returns the category metadata (with the recorded 24 image paths).
        The bank is **not** cached here; use :meth:`get_memory` for caching.
        """
        t0 = time.time()
        images = self.select_memory_images(category)
        logger.info("[%s] building memory from %d normal train images (seed=%d) ...",
                    category, len(images), self.seed)

        model = self.model
        # Keep the pretrained backbone in eval mode (frozen BatchNorm), while the
        # PatchCoreModel wrapper must be in train mode so its forward() stores the
        # patch embeddings (memory collection path).
        model.train()
        model.feature_extractor.eval()
        model.embedding_store.clear()

        with torch.no_grad():
            for i, img_path in enumerate(images, start=1):
                x = self._preprocess(img_path).to(self.device)
                model(x)
                if i % 8 == 0 or i == len(images):
                    logger.info("  extracted %d/%d images", i, len(images))

        model.eval()
        if not model.embedding_store:
            raise RuntimeError(f"[{category}] no patch embeddings were collected.")
        memory = torch.vstack(model.embedding_store).float()
        model.embedding_store.clear()

        if not bool(torch.isfinite(memory).all()):
            raise RuntimeError(f"[{category}] memory bank contains non-finite values.")

        self._attach_memory(memory, category)

        rel = [str(Path(p).relative_to(self.data_root.resolve())) for p in images]
        metadata = self._make_metadata(category, images=images, memory_images_rel=rel)
        metadata["n_memory_patches"] = int(memory.shape[0])
        metadata["embedding_dim"] = int(memory.shape[1])
        logger.info("[%s] memory built in %.1fs: %d patches x %d dims.",
                    category, time.time() - t0, memory.shape[0], memory.shape[1])
        return metadata

    def _make_metadata(self, category: str, images: list[Path],
                       memory_images_rel: list[str]) -> dict[str, Any]:
        return {
            "dataset": "MVTec-AD",
            "category": category,
            "memory_size": len(images),
            "seed": self.seed,
            "category_seed": _category_seed(self.seed, category),
            "source_split": "train/good",
            "config": self.config,
            "memory_images": [str(p) for p in images],           # absolute paths
            "memory_images_rel": memory_images_rel,               # portable copy
        }

    # -------------------------------------------------------------- caching
    def _cache_paths(self, category: str) -> tuple[Path, Path]:
        cat_dir = self.cache_root / category
        return cat_dir / "memory.pt", cat_dir / "metadata.json"

    def _save_cache(self, category: str, metadata: dict[str, Any]) -> None:
        mem_path, meta_path = self._cache_paths(category)
        mem_path.parent.mkdir(parents=True, exist_ok=True)
        memory = self.model.memory_bank.detach().float().cpu()
        torch.save(memory, mem_path)  # write tensor first, metadata second
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _load_cache(self, category: str) -> dict[str, Any] | None:
        """Load a cached memory bank if present and consistent with config."""
        mem_path, meta_path = self._cache_paths(category)
        if not (mem_path.exists() and meta_path.exists()):
            return None
        with open(meta_path) as f:
            metadata = json.load(f)
        if metadata.get("config") != self.config:
            logger.warning("[%s] cache config mismatch -> will rebuild memory.", category)
            return None
        memory = torch.load(mem_path, map_location=self.device)
        if memory.dim() != 2 or memory.shape[0] == 0:
            logger.warning("[%s] cached memory bank is empty/corrupt -> rebuild.", category)
            return None
        if not bool(torch.isfinite(memory).all()):
            logger.warning("[%s] cached memory bank is non-finite -> rebuild.", category)
            return None
        self._attach_memory(memory, category)
        return metadata

    def get_memory(self, category: str) -> dict[str, Any]:
        """Return the category memory metadata, building or loading as needed.

        The returned dict has an extra key ``"loaded_from_cache"`` reporting
        whether an existing cache was reused (False => freshly built).
        """
        metadata = self._load_cache(category)
        if metadata is not None:
            logger.info("[%s] memory cache HIT -> reuse %s (no re-fit).",
                        category, self._cache_paths(category)[1])
            metadata = dict(metadata)
            metadata["loaded_from_cache"] = True
            return metadata

        logger.info("[%s] memory cache MISS -> fitting memory now.", category)
        metadata = self.build_memory(category)
        self._save_cache(category, metadata)
        metadata["loaded_from_cache"] = False
        logger.info("[%s] memory cached under %s", category, self.cache_root / category)
        return metadata

    # ------------------------------------------------------------- inference
    def _result_paths(self, category: str, query_image: str | Path) -> tuple[Path, Path, Path]:
        q = Path(query_image)
        stem = f"{q.parent.name}_{q.stem}"
        cat_dir = self.output_root / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        return cat_dir / f"{stem}__anomaly_map.npy", \
            cat_dir / f"{stem}__anomaly_map.png", \
            cat_dir / f"{stem}__score.json"

    def predict(self, image_path: str | Path, category: str) -> dict[str, Any]:
        """Run PatchCore inference on ``image_path`` for ``category``.

        Builds/loads the category memory (cached), then returns at least::

            {
                "image_score": float,
                "anomaly_map_path": "...",
                "category": "...",
                "query_image": "...",
            }

        Artifacts (raw float anomaly map ``.npy``, visual ``.png`` and a ``.json``
        score record) are persisted under ``output_root/<category>/``.
        """
        t0 = time.time()
        query = Path(image_path)
        if not query.is_file():
            raise FileNotFoundError(f"[{category}] query image not found: {query}")

        memory_meta = self.get_memory(category)

        # Hard no-leakage guard: the query must not be one of the memory images.
        mem_abs = {Path(p) for p in memory_meta["memory_images"]}
        if query.resolve() in mem_abs:
            raise RuntimeError(
                f"[{category}] leakage guard: query image {query} is itself a memory "
                f"(training) image; refusing to score it."
            )

        logger.info("[%s] predicting on %s ...", category, query)
        model = self.model
        model.eval()
        x = self._preprocess(query).to(self.device)
        with torch.no_grad():
            out = model(x)  # InferenceBatch(pred_score, anomaly_map)

        image_score = float(out.pred_score.detach().cpu().flatten()[0])
        anomaly_map = out.anomaly_map.detach().cpu().squeeze(0).squeeze(0)  # (256, 256)
        if not (np.isfinite(image_score) and bool(torch.isfinite(anomaly_map).all())):
            raise RuntimeError(f"[{category}] non-finite prediction for {query}")

        # Upscale the 256x256 map to the query's original resolution (pixel-aligned).
        orig_h, orig_w = int(x.shape[-2]), int(x.shape[-1])  # resized size, safer fallback
        try:
            img = read_image(str(query), as_tensor=True)  # raw [0,1] CHW at native size
            orig_h, orig_w = int(img.shape[-2]), int(img.shape[-1])
        except Exception:  # noqa: BLE001
            pass
        map_np = anomaly_map.float().numpy()
        if (orig_h, orig_w) != map_np.shape:
            map_full = cv2.resize(map_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        else:
            map_full = map_np

        npy_path, png_path, json_path = self._result_paths(category, query)
        np.save(npy_path, map_full.astype(np.float32))

        # Visual PNG (min-max normalised heatmap over the query's native size).
        lo, hi = float(map_full.min()), float(map_full.max())
        if hi - lo > 1e-9:
            vis = ((map_full - lo) / (hi - lo) * 255.0).astype(np.uint8)
        else:
            vis = np.zeros_like(map_full, dtype=np.uint8)
        heat = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        cv2.imwrite(str(png_path), heat)

        record = {
            "image_score": float(image_score),
            "anomaly_map_path": str(png_path),
            "anomaly_map_raw_path": str(npy_path),
            "category": category,
            "query_image": str(query),
            "seed": self.seed,
            "memory_size": self.memory_size,
            "memory_loaded_from_cache": bool(memory_meta.get("loaded_from_cache", False)),
            "memory_metadata": str(self._cache_paths(category)[1]),
            "elapsed_s": round(time.time() - t0, 3),
        }
        with open(json_path, "w") as f:
            json.dump(record, f, indent=2)

        logger.info("[%s] image_score=%.6g -> %s (%.1fs)", category, image_score,
                    npy_path.parent, time.time() - t0)
        return record
