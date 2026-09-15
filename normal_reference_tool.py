"""Phase 4: Normal Reference Evidence Tool.

Given a query image and its category, this tool retrieves the most similar
**known normal** reference images and asks the local Qwen3-VL to compare the
query against them. It produces *normality evidence* only.

    query image + category
        -> pretrained image embedding (DINO ViT-S/16)
        -> cosine similarity against the 24 frozen normal training images
        -> Top-K normal references (K = 1 for Mode A, K = 3 for Mode B)
        -> Qwen3-VL comparison
        -> normality-evidence JSON

Design guarantees:

  * this tool is **not an anomaly classifier.** It never returns a final
    NORMAL/ANOMALOUS label, an anomaly probability, a bounding box, or
    coordinates — see :data:`FORBIDDEN_PATTERNS`, which rejects such content;
  * the reference similarity is used **only to rank and select** references.
    It is never placed in the prompt and never turned into a decision
    (``similarity_used_as_decision`` / ``similarity_used_in_prompt`` are
    recorded as ``False`` on every record);
  * the reference pool is the category's ``train/good`` split only, selected by
    the **frozen Phase-2 24-shot routine** (imported, never re-implemented, so
    the sample cannot drift). Test images can never enter the pool, and a query
    that is itself a pool member is rejected outright;
  * no ground-truth annotation and no anomaly label is read anywhere, and no
    PatchCore artifact is consumed — retrieval is a separate pretrained
    embedding that is independent of the PatchCore backbone.

The prompt is fixed; the two modes share it byte-for-byte and differ **only**
in the ``You are given:`` reference-count line, so the Mode A / Mode B
comparison isolates the effect of the number of references.

Minimal interface::

    import asyncio
    from normal_reference_tool import NormalReferenceEvidenceTool, ReferenceToolConfig

    tool = NormalReferenceEvidenceTool(ReferenceToolConfig(reference_k=3))
    record = asyncio.run(tool.compare("/.../test/broken_large/000.png", "bottle"))
    print(record["parsed_output"])

Usage note: the vendored Anomalib shadows the ``anomalib/`` folder at the repo
root, so importing :mod:`patchcore_predictor` (which puts ``anomalib/src`` first
on ``sys.path``) must happen before anything else imports anomalib.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "AnomalyAgent" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_MODEL_PATH = "/data/pfy/MLLMs/Qwen3-VL-4B-Instruct"
DEFAULT_DATA_ROOT = "/data/pfy/dataset/MVTec-AD"
DEFAULT_OUT_ROOT = "/data/pfy/AgentIAD/results/phase4"
#: Phase-2 memory cache; its ``metadata.json`` records the frozen 24-shot list.
DEFAULT_PHASE2_MEMORY_ROOT = "/data/pfy/AgentIAD/results/phase1/cache"

#: Retrieval embedding. DINO ViT-S/16 is self-supervised and independent of the
#: supervised ImageNet ResNet-50 PatchCore uses, so the references are not
#: selected by the anomaly detector's own features.
DEFAULT_EMBEDDER = "dino_vits16"
_DINO_CKPT = "dino_deitsmall16_pretrain.pth"
_EMBED_IMAGE_SIZE = 224
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

#: The frozen Phase-2 24-shot selection (``PatchcorePredictor`` defaults).
PHASE2_SEED = 0
PHASE2_MEMORY_SIZE = 24

#: Modes. Both run the identical prompt, query set and decoding settings; only
#: the number of attached references differs.
MODE_A_K = 1
MODE_B_K = 3
REFERENCE_K_VALUES = (MODE_A_K, MODE_B_K)
MODE_NAMES = {MODE_A_K: "top1_normal_reference", MODE_B_K: "top3_normal_references"}

#: Allowed values of the enum fields.
GLOBAL_CONSISTENCY_VALUES = ("high", "medium", "low")
ABNORMALITY_VALUES = ("likely", "uncertain", "unlikely")

#: The four keys the model must return, in schema order.
OUTPUT_KEYS = ("global_consistency", "meaningful_differences",
               "normality_evidence", "comparison_summary")
#: The three keys of every entry of ``meaningful_differences``.
DIFFERENCE_KEYS = ("description", "abnormality", "evidence")


class ReferenceToolError(RuntimeError):
    """Raised when the reference stage cannot be configured or executed."""


# --------------------------------------------------------------------------
# System prompt (verbatim) and its per-mode variants
# --------------------------------------------------------------------------

#: Text before the ``You are given:`` reference-count line.
_PROMPT_HEADER = """You are a visual comparison module for industrial anomaly detection.

You are given:

1. A query image of an object that is being inspected.
"""

#: The single line that distinguishes Mode A from Mode B. Everything else in
#: the prompt is byte-identical between the two modes.
_REFERENCE_LINE_BY_K = {
    1: "2. One (1) reference image of the same object category, which is KNOWN TO BE NORMAL.",
    3: "2. Three (3) reference images of the same object category, which are all KNOWN TO BE NORMAL.",
}

#: Text after the reference-count line; byte-identical across modes.
_PROMPT_TAIL = """
The reference images come from the normal training set of the same category, so they
show the natural variation of a normal item: viewpoint, illumination, pose, orientation,
position, and normal manufacturing variation.

Your task is ONLY to compare the query image against the normal references and to report
normality evidence.

Do NOT make the final NORMAL/ANOMALOUS decision.

Do NOT output bounding boxes or coordinates.

Do NOT output an anomaly probability, an anomaly score, or a numerical decision threshold.

Do NOT treat every difference as an anomaly. Normal intra-class variation, viewpoint,
illumination and pose differences are expected and are NOT defects.

You must:

1. Identify the meaningful differences between the query image and the normal references.
2. For each difference, judge whether it is likely a real abnormality, uncertain, or
   unlikely (that is, plausibly explained by normal variation).
3. Account for normal intra-class variation.
4. Account for viewpoint, illumination and pose differences.
5. Do not invent defects.
6. Report what is consistent with normal appearance as normality evidence.
7. State how consistent the query is with the normal references overall.

Return ONLY valid JSON:

{
"global_consistency": "high | medium | low",
"meaningful_differences": [
{
"description": "objective description of the difference",
"abnormality": "likely | uncertain | unlikely",
"evidence": "specific visible observation supporting the judgement"
}
],
"normality_evidence": [
"specific visible observation consistent with normal appearance"
],
"comparison_summary": "short overall comparison of the query against the references"
}
"""


def system_prompt_for(k: int) -> str:
    """The system prompt for ``k`` references.

    The two modes share :data:`_PROMPT_HEADER` and :data:`_PROMPT_TAIL`
    byte-for-byte; only :data:`_REFERENCE_LINE_BY_K` differs.
    """
    if k not in REFERENCE_K_VALUES:
        raise ReferenceToolError(
            f"reference_k must be one of {REFERENCE_K_VALUES}, got {k!r}")
    return _PROMPT_HEADER + _REFERENCE_LINE_BY_K[k] + "\n" + _PROMPT_TAIL


#: The Mode B prompt (the maximal-reference variant).
NORMAL_REFERENCE_SYSTEM_PROMPT = system_prompt_for(MODE_B_K)

CORRECTION_INSTRUCTION = (
    "Your previous response was not valid JSON in the required schema. "
    "Return ONLY the JSON object, with exactly these four keys: "
    '"global_consistency" (one of: high, medium, low), '
    '"meaningful_differences" (array of objects, each with exactly the keys '
    '"description", "abnormality" (one of: likely, uncertain, unlikely) and "evidence"), '
    '"normality_evidence" (array of short strings), '
    '"comparison_summary" (string). '
    "Do not output bounding boxes, coordinates, anomaly probabilities or anomaly scores. "
    "No prose, no markdown fence."
)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ReferenceToolConfig:
    """Independent, user-settable parameters of the reference-evidence stage."""

    reference_k: int = MODE_B_K
    model_path: str = DEFAULT_MODEL_PATH
    data_root: str = DEFAULT_DATA_ROOT
    phase2_memory_root: str = DEFAULT_PHASE2_MEMORY_ROOT
    embedder: str = DEFAULT_EMBEDDER
    device: str | None = None
    attn_implementation: str | None = "flash_attention_2"
    max_new_tokens: int = 512
    #: Extra attempts after the first one when the reply fails to validate.
    max_parse_retries: int = 2
    #: Qwen micro-batcher settings (forwarded to ``load_vl_model``).
    max_batch_size: int = 16
    max_wait_ms: int = 50
    concurrency: int = 32

    def __post_init__(self) -> None:
        if self.reference_k not in REFERENCE_K_VALUES:
            raise ValueError(
                f"reference_k must be one of {REFERENCE_K_VALUES}, got {self.reference_k!r}")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if self.max_parse_retries < 0:
            raise ValueError(
                f"max_parse_retries must be >= 0, got {self.max_parse_retries}")
        if self.max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {self.max_batch_size}")
        if self.concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {self.concurrency}")

    @property
    def mode_name(self) -> str:
        return MODE_NAMES[self.reference_k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_k": int(self.reference_k),
            "mode_name": self.mode_name,
            "model_path": self.model_path,
            "data_root": self.data_root,
            "phase2_memory_root": self.phase2_memory_root,
            "embedder": self.embedder,
            "device": self.device,
            "attn_implementation": self.attn_implementation,
            "max_new_tokens": int(self.max_new_tokens),
            "max_parse_retries": int(self.max_parse_retries),
            "max_batch_size": int(self.max_batch_size),
            "max_wait_ms": int(self.max_wait_ms),
            "concurrency": int(self.concurrency),
        }

    def decoding_signature(self) -> dict[str, Any]:
        """The settings that must match across modes for a fair comparison.

        Excludes ``reference_k`` — the number of references is the variable
        under study, everything else is held constant.
        """
        return {
            "model_path": self.model_path,
            "attn_implementation": self.attn_implementation,
            "max_new_tokens": int(self.max_new_tokens),
            "do_sample": False,
            "embedder": self.embedder,
            "data_root": self.data_root,
            "phase2_memory_root": self.phase2_memory_root,
            "system_prompt": system_prompt_for(self.reference_k),
        }


DEFAULT_CONFIG = ReferenceToolConfig()


# --------------------------------------------------------------------------
# Message construction
# --------------------------------------------------------------------------

def build_user_content(
    query_data_url: str,
    reference_data_urls: Sequence[str],
    k: int,
) -> list[dict[str, Any]]:
    """The user turn: the query image, then the ``k`` known-normal references.

    Mirrors the shape ``agent_v1_27.py`` / ``local_visual_observation`` use:
    ``{"type": "image_url", "image_url": {"url": <data URL>}}``.

    Only images and literal labels appear here — never a similarity value, an
    anomaly score, or a label.
    """
    if k not in REFERENCE_K_VALUES:
        raise ReferenceToolError(f"reference_k must be one of {REFERENCE_K_VALUES}, got {k!r}")
    if len(reference_data_urls) != k:
        raise ReferenceToolError(
            f"expected exactly {k} reference image(s), got {len(reference_data_urls)}")

    content: list[dict[str, Any]] = [
        {"type": "text", "text": "Query image:"},
        {"type": "image_url", "image_url": {"url": query_data_url}},
    ]
    for i, url in enumerate(reference_data_urls, start=1):
        content.append({"type": "text", "text": f"Known NORMAL reference image {i} of {k}:"})
        content.append({"type": "image_url", "image_url": {"url": url}})
    return content


def build_messages(
    query_data_url: str,
    reference_data_urls: Sequence[str],
    k: int,
) -> list[dict[str, Any]]:
    """Full chat message list (system + user) for one comparison."""
    return [
        {"role": "system", "content": system_prompt_for(k)},
        {"role": "user",
         "content": build_user_content(query_data_url, reference_data_urls, k)},
    ]


def prompt_text_of(messages: Sequence[dict[str, Any]]) -> str:
    """All text carried by ``messages``, with image payloads dropped.

    Used to assert that no similarity, score, bbox or anomaly label reaches the
    model.
    """
    chunks: list[str] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
    return "\n".join(chunks)


# --------------------------------------------------------------------------
# Parsing & schema validation
# --------------------------------------------------------------------------

_FENCED_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.I)

#: Content the tool must never emit. Every match is a fatal violation: it
#: triggers the correction retry, and if it survives, the record is stored with
#: ``parse_ok: false`` rather than silently accepted.
FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("bbox", re.compile(r"\bbbox\b|\bbounding\s+box", re.I)),
    ("coordinates", re.compile(r"\bcoordinate", re.I)),
    ("xy_keys", re.compile(r"\b(?:x1|y1|x2|y2)\b", re.I)),
    # a bracketed/parenthesised numeric tuple, i.e. a coordinate pair
    ("coordinate_tuple", re.compile(r"[\[(]\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*[\])]")),
    ("pixel_units", re.compile(r"\b\d+(?:\.\d+)?\s*(?:px|pixels?)\b", re.I)),
    ("anomaly_probability", re.compile(r"\banomaly\s+(?:probability|score)", re.I)),
    ("decision_threshold", re.compile(r"\b(?:threshold|cut-?off)\b", re.I)),
)

#: Keys that would carry a localisation or a decision if the model added them.
FORBIDDEN_KEYS = frozenset({
    "bbox", "box", "boxes", "coordinates", "coord", "x", "y", "x1", "y1", "x2",
    "y2", "width", "height", "area", "score", "probability", "label",
    "is_anomaly", "anomaly_score", "prediction",
})


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of one JSON object from a model reply.

    Tries, in order: the whole reply, a ```json fenced block, the repo's shared
    ``_parse_json_from_text`` helper, then the first brace-delimited object that
    decodes.
    """
    if not text or not text.strip():
        return None
    stripped = text.strip()

    for candidate in (stripped, *(m.group(1) for m in _FENCED_RE.finditer(stripped))):
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj

    try:  # the shared helper used across the AnomalyAgent code base
        from react_agent.utils import _parse_json_from_text
        obj = _parse_json_from_text(stripped)
        if isinstance(obj, dict) and obj:
            return obj
    except Exception:
        pass

    start = stripped.find("{")
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(stripped[start:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def _iter_strings(node: Any):
    """Yield every string anywhere inside ``node``."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield str(key)
            yield from _iter_strings(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_strings(item)


def find_forbidden_content(obj: Any) -> list[str]:
    """Names of the forbidden patterns present in ``obj`` (empty means clean)."""
    found: list[str] = []
    for name, pattern in FORBIDDEN_PATTERNS:
        if any(pattern.search(s) for s in _iter_strings(obj)):
            found.append(name)
    return found


def validate_reference_output(obj: Any) -> list[str]:
    """Schema violations of ``obj``; empty list means the output is valid.

    Only violations listed in :data:`FATAL_VIOLATION_PREFIXES` cause a retry;
    extra keys are recorded but stripped, since the contract is about the
    required fields rather than about forbidding additions.
    """
    if not isinstance(obj, dict):
        return ["not a JSON object"]

    problems: list[str] = []

    missing = [k for k in OUTPUT_KEYS if k not in obj]
    if missing:
        problems.append(f"missing keys: {missing}")
    extra = sorted(set(obj) - set(OUTPUT_KEYS))
    if extra:
        problems.append(f"extra keys (ignored): {extra}")

    consistency = obj.get("global_consistency")
    if not isinstance(consistency, str) or consistency.strip().lower() not in GLOBAL_CONSISTENCY_VALUES:
        problems.append(
            f"global_consistency must be one of {list(GLOBAL_CONSISTENCY_VALUES)}, "
            f"got {consistency!r}")

    differences = obj.get("meaningful_differences")
    if not isinstance(differences, list):
        problems.append("meaningful_differences must be a list of objects")
    else:
        for i, diff in enumerate(differences):
            if not isinstance(diff, dict):
                problems.append(f"meaningful_differences[{i}] must be an object")
                continue
            miss = [k for k in DIFFERENCE_KEYS if k not in diff]
            if miss:
                problems.append(f"meaningful_differences[{i}] missing keys: {miss}")
                continue
            if not isinstance(diff["description"], str) or not diff["description"].strip():
                problems.append(
                    f"meaningful_differences[{i}].description must be a non-empty string")
            if not isinstance(diff["evidence"], str) or not diff["evidence"].strip():
                problems.append(
                    f"meaningful_differences[{i}].evidence must be a non-empty string")
            abnormality = diff["abnormality"]
            if (not isinstance(abnormality, str)
                    or abnormality.strip().lower() not in ABNORMALITY_VALUES):
                problems.append(
                    f"meaningful_differences[{i}].abnormality must be one of "
                    f"{list(ABNORMALITY_VALUES)}, got {abnormality!r}")

    normality = obj.get("normality_evidence")
    if not isinstance(normality, list):
        problems.append("normality_evidence must be a list of strings")
    elif not all(isinstance(e, str) and e.strip() for e in normality):
        problems.append("every entry of normality_evidence must be a non-empty string")

    summary = obj.get("comparison_summary")
    if not isinstance(summary, str) or not summary.strip():
        problems.append("comparison_summary must be a non-empty string")

    forbidden = find_forbidden_content(obj)
    if forbidden:
        problems.append(f"forbidden content (no bbox/coordinates/score): {forbidden}")

    bad_keys = sorted(
        {k for diff in differences or [] if isinstance(diff, dict)
         for k in diff if k in FORBIDDEN_KEYS}
    )
    if bad_keys:
        problems.append(f"forbidden localisation/decision keys: {bad_keys}")

    return problems


#: A violation with one of these prefixes is not repairable by normalisation.
FATAL_VIOLATION_PREFIXES = (
    "not a JSON object", "missing keys", "global_consistency must",
    "meaningful_differences", "normality_evidence must",
    "every entry of normality_evidence", "comparison_summary must",
    "forbidden content", "forbidden localisation/decision keys",
)


def is_fatal(violations: Sequence[str]) -> bool:
    return any(v.startswith(FATAL_VIOLATION_PREFIXES) for v in violations)


def normalize_reference_output(obj: dict[str, Any]) -> dict[str, Any]:
    """Keep exactly the schema keys, with enums lower-cased and strings stripped."""
    return {
        "global_consistency": obj["global_consistency"].strip().lower(),
        "meaningful_differences": [
            {
                "description": d["description"].strip(),
                "abnormality": d["abnormality"].strip().lower(),
                "evidence": d["evidence"].strip(),
            }
            for d in obj["meaningful_differences"]
        ],
        "normality_evidence": [e.strip() for e in obj["normality_evidence"]],
        "comparison_summary": obj["comparison_summary"].strip(),
    }


def parse_reference_output(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Parse ``text`` into a valid normality-evidence output.

    Returns ``(output, violations)``. ``output`` is ``None`` unless the reply
    decoded *and* had no fatal violation; ``violations`` always describes
    everything that was wrong (empty when the reply was fully valid).
    """
    obj = extract_json_object(text)
    if obj is None:
        return None, ["not a JSON object"]
    violations = validate_reference_output(obj)
    if is_fatal(violations):
        return None, violations
    return normalize_reference_output(obj), violations


# --------------------------------------------------------------------------
# Image encoding
# --------------------------------------------------------------------------

def load_image_data_url(image_path: str | Path) -> str:
    """Read an image and return it as a PNG data URL."""
    from PIL import Image
    from react_agent.tools import _encode_image_to_base64

    path = Path(image_path)
    if not path.is_file():
        raise ReferenceToolError(f"image not found: {path}")
    with Image.open(path) as im:
        return _encode_image_to_base64(im.convert("RGB"), fmt="PNG")


# --------------------------------------------------------------------------
# Reference retrieval
# --------------------------------------------------------------------------

class ImageEmbedder:
    """Pretrained image embedding used to rank normal references.

    Deliberately *not* PatchCore: the retrieval signal is an independent
    pretrained network, so reference selection cannot inherit the anomaly
    detector's biases. Embeddings are L2-normalised, so a dot product is the
    cosine similarity.
    """

    def __init__(self, name: str = DEFAULT_EMBEDDER, device: str | None = None) -> None:
        self.name = name
        if device is None:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._model: Any = None
        self._lock = threading.Lock()
        self._cache: dict[str, Any] = {}

    @property
    def model_id(self) -> str:
        return f"{self.name}@{_EMBED_IMAGE_SIZE}"

    # -- model ------------------------------------------------------------
    def _build(self) -> Any:
        import timm
        import torch

        if self.name == DEFAULT_EMBEDDER:
            ckpt = Path(torch.hub.get_dir()) / "checkpoints" / _DINO_CKPT
            if not ckpt.is_file():
                raise ReferenceToolError(
                    f"DINO checkpoint not found at {ckpt}; the default embedder is "
                    f"offline-only. Pass --embedder with a timm model name to use a "
                    f"downloadable backbone instead.")
            model = timm.create_model(
                "vit_small_patch16_224", pretrained=False, num_classes=0)
            state = torch.load(ckpt, map_location="cpu")
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise ReferenceToolError(
                    f"DINO checkpoint does not match vit_small_patch16_224: "
                    f"missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}")
        else:
            model = timm.create_model(self.name, pretrained=True, num_classes=0)

        model.eval().to(self.device)
        return model

    @property
    def model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._build()
        return self._model

    # -- preprocessing ----------------------------------------------------
    def _transform(self) -> Any:
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(_EMBED_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    # -- embedding --------------------------------------------------------
    def embed_paths(self, paths: Sequence[str | Path], batch_size: int = 32) -> Any:
        """Return L2-normalised embeddings ``(N, D)`` for ``paths``."""
        import numpy as np
        import torch
        from PIL import Image

        paths = [Path(p) for p in paths]
        assert paths, "embed_paths requires at least one path"

        tf = self._transform()
        batches = []
        with torch.no_grad():
            for start in range(0, len(paths), batch_size):
                chunk = paths[start:start + batch_size]
                tensors = []
                for path in chunk:
                    if not path.is_file():
                        raise ReferenceToolError(f"reference image not found: {path}")
                    with Image.open(path) as im:
                        tensors.append(tf(im.convert("RGB")))
                batch = torch.stack(tensors).to(self.device)
                feats = self.model(batch).float()
                feats = torch.nn.functional.normalize(feats, dim=-1)
                batches.append(feats.cpu().numpy())

        emb = np.concatenate(batches, axis=0).astype(np.float32)
        if not np.isfinite(emb).all():
            raise ReferenceToolError("embedding contains non-finite values")
        return emb

    def embed_cached(self, key: str, paths: Sequence[str | Path]) -> Any:
        """``embed_paths`` memoised on ``key`` (the pool identity)."""
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        emb = self.embed_paths(paths)
        with self._lock:
            self._cache[key] = emb
        return emb


def rank_references(query_emb: Any, pool_paths: Sequence[Path], pool_embs: Any,
                    k: int) -> list[dict[str, Any]]:
    """Top-``k`` pool entries by cosine similarity to ``query_emb``.

    Ties are broken by image path so the ranking is fully deterministic. The
    similarity is a ranking key only — it is never a decision.
    """
    import numpy as np

    if k < 1:
        raise ReferenceToolError(f"k must be >= 1, got {k}")
    if len(pool_paths) < k:
        raise ReferenceToolError(
            f"reference pool has {len(pool_paths)} images, cannot take top-{k}")

    sims = pool_embs @ np.asarray(query_emb, dtype=np.float32).reshape(-1)
    order = sorted(range(len(pool_paths)), key=lambda i: (-float(sims[i]), str(pool_paths[i])))
    picked = order[:k]
    if len(set(picked)) != k:
        raise ReferenceToolError("retrieval returned duplicate references")
    return [
        {"image_path": str(pool_paths[i]), "similarity": round(float(sims[i]), 6)}
        for i in picked
    ]


# --------------------------------------------------------------------------
# Tool
# --------------------------------------------------------------------------

def _message_text(response: Any) -> str:
    """Plain text of a LangChain message (``content`` may be str or blocks)."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item if isinstance(item, str) else str(item.get("text", ""))
            for item in content
        )
    return str(content)


class NormalReferenceEvidenceTool:
    """Retrieve known-normal references and compare a query against them.

    The Qwen model is loaded lazily on first use, so importing this module (and
    running its structural checks) does not touch the GPU. ``chat`` and
    ``embedder`` may be injected to run the stage against stubs.
    """

    def __init__(self, config: ReferenceToolConfig | None = None, *,
                 chat: Any = None, embedder: ImageEmbedder | None = None) -> None:
        self.config = config or DEFAULT_CONFIG
        self.data_root = Path(self.config.data_root)
        self._chat = chat
        self._embedder = embedder
        self._load_lock = threading.Lock()
        self._pool_cache: dict[str, list[Path]] = {}

    # -- model ------------------------------------------------------------
    @property
    def chat(self) -> Any:
        if self._chat is None:
            with self._load_lock:
                if self._chat is None:
                    from react_agent.utils import load_vl_model

                    self._chat = load_vl_model(
                        self.config.model_path,
                        attn_implementation=self.config.attn_implementation,
                        max_batch_size=self.config.max_batch_size,
                        max_wait_ms=self.config.max_wait_ms,
                    )
        return self._chat

    @property
    def embedder(self) -> ImageEmbedder:
        if self._embedder is None:
            self._embedder = ImageEmbedder(
                self.config.embedder, device=self.config.device)
        return self._embedder

    # -- reference pool ----------------------------------------------------
    def reference_pool(self, category: str) -> list[Path]:
        """The frozen 24 normal training images of ``category``.

        The selection is **imported** from the Phase-2 routine rather than
        re-implemented, so the sample cannot silently drift. When the Phase-2
        memory record exists it is cross-checked against the recomputation and a
        mismatch is a hard error.
        """
        if category in self._pool_cache:
            return self._pool_cache[category]

        from patchcore_predictor import PatchcorePredictor

        # Only ``select_memory_images`` is used — no memory is built and no
        # inference runs — but the constructor creates its output directory, so
        # point it at a scratch dir rather than anywhere near the Phase-1/2/3
        # artifacts.
        scratch = Path(tempfile.gettempdir()) / "phase4_reference_tool"
        predictor = PatchcorePredictor(
            data_root=self.data_root,
            cache_root=self.config.phase2_memory_root,
            output_root=scratch,
            seed=PHASE2_SEED,
            memory_size=PHASE2_MEMORY_SIZE,
        )
        selected = [Path(p).resolve() for p in predictor.select_memory_images(category)]
        if len(selected) != PHASE2_MEMORY_SIZE:
            raise ReferenceToolError(
                f"[{category}] expected {PHASE2_MEMORY_SIZE} references, got {len(selected)}")

        # Guard: every reference must live in the normal training split.
        good_dir = (self.data_root.resolve() / category / "train" / "good")
        outside = [str(p) for p in selected if not p.is_relative_to(good_dir)]
        if outside:
            raise ReferenceToolError(
                f"[{category}] leakage guard: references outside train/good: {outside}")

        # Cross-check against the record Phase 2 actually used, when present.
        recorded = self._recorded_phase2_pool(category)
        if recorded is not None and set(recorded) != set(selected):
            raise ReferenceToolError(
                f"[{category}] Phase-2 24-shot list does not match the recorded memory "
                f"metadata; refusing to guess which one is authoritative.")

        self._pool_cache[category] = selected
        return selected

    def _recorded_phase2_pool(self, category: str) -> list[Path] | None:
        """The 24 paths recorded by the Phase-2 memory cache, if it exists."""
        meta_path = Path(self.config.phase2_memory_root) / category / "metadata.json"
        if not meta_path.is_file():
            return None
        with open(meta_path) as f:
            metadata = json.load(f)
        images = metadata.get("memory_images")
        if not isinstance(images, list) or not images:
            return None
        cfg = metadata.get("config", {})
        if (cfg.get("seed") != PHASE2_SEED
                or cfg.get("memory_size") != PHASE2_MEMORY_SIZE):
            raise ReferenceToolError(
                f"[{category}] Phase-2 memory metadata was built with "
                f"seed={cfg.get('seed')} memory_size={cfg.get('memory_size')}, "
                f"expected seed={PHASE2_SEED} memory_size={PHASE2_MEMORY_SIZE}")
        return [Path(p).resolve() for p in images]

    # -- retrieval ---------------------------------------------------------
    def guard_query(self, category: str, query_image: str | Path) -> Path:
        """Reject a query that is not a test image or is itself a reference."""
        query = Path(query_image).resolve()
        if not query.is_file():
            raise ReferenceToolError(f"[{category}] query image not found: {query}")

        pool = {p.resolve() for p in self.reference_pool(category)}
        if query in pool:
            raise ReferenceToolError(
                f"[{category}] leakage guard: query {query} is itself a normal reference.")

        test_dir = (self.data_root.resolve() / category / "test")
        if not query.is_relative_to(test_dir):
            raise ReferenceToolError(
                f"[{category}] leakage guard: query {query} is not under {test_dir}; "
                f"only test images may be queried.")
        return query

    def retrieve(self, query_image: str | Path, category: str,
                 k: int | None = None) -> list[dict[str, Any]]:
        """Top-``k`` known-normal references for ``query_image``."""
        k = self.config.reference_k if k is None else int(k)
        query = self.guard_query(category, query_image)

        pool = self.reference_pool(category)
        embedder = self.embedder
        key = f"{category}|{embedder.model_id}|" + hashlib.sha256(
            "\n".join(str(p) for p in pool).encode()).hexdigest()[:16]
        pool_embs = embedder.embed_cached(key, pool)
        # The query embedding is a pure function of the image, so memoise it on
        # the path: repeat retrievals of the same query are free.
        query_emb = embedder.embed_cached(
            f"query|{embedder.model_id}|{query}", [query])[0]

        references = rank_references(query_emb, pool, pool_embs, k)

        paths = [r["image_path"] for r in references]
        if len(set(paths)) != k:
            raise ReferenceToolError(f"[{category}] duplicate references: {paths}")
        if str(query) in set(paths):
            raise ReferenceToolError(
                f"[{category}] leakage guard: query appears among the references.")
        return references

    def leakage_audit(self, category: str, query_image: str | Path,
                      references: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """The explicit no-leakage statement recorded with every query."""
        query = Path(query_image).resolve()
        good_dir = (self.data_root.resolve() / category / "train" / "good")
        pool = {p.resolve() for p in self.reference_pool(category)}
        ref_paths = [Path(r["image_path"]).resolve() for r in references]

        return {
            "reference_source_split": "train/good",
            "references_all_from_train_good": all(
                p.is_relative_to(good_dir) for p in ref_paths),
            "references_in_frozen_phase2_pool": all(p in pool for p in ref_paths),
            "query_is_test_image": query.is_relative_to(
                self.data_root.resolve() / category / "test"),
            "query_in_reference_pool": query in pool,
            "query_among_references": query in set(ref_paths),
            "reference_pool_size": len(pool),
            "test_images_in_reference_pool": False,
            "ground_truth_used_in_retrieval": False,
            "ground_truth_used_in_prompt": False,
            "ground_truth_used_in_decision": False,
            "anomaly_label_used": False,
            "patchcore_artifact_used": False,
            "similarity_used_as_decision": False,
            "similarity_used_in_prompt": False,
        }

    # -- one comparison ----------------------------------------------------
    async def compare(
        self,
        query_image: str | Path,
        category: str,
        references: Sequence[dict[str, Any]] | None = None,
        config: ReferenceToolConfig | None = None,
    ) -> dict[str, Any]:
        """Compare one query against its normal references.

        ``references`` may be supplied explicitly (the tool's documented input);
        when omitted they are retrieved here. Either way the same guards run.
        """
        cfg = config or self.config
        k = cfg.reference_k
        t0 = time.time()

        if references is None:
            references = self.retrieve(query_image, category, k)
        references = [dict(r) for r in references]
        if len(references) != k:
            raise ReferenceToolError(
                f"[{category}] expected {k} references, got {len(references)}")

        query = self.guard_query(category, query_image)
        reference_paths = [Path(r["image_path"]) for r in references]
        if query in {p.resolve() for p in reference_paths}:
            raise ReferenceToolError(
                f"[{category}] leakage guard: query appears among the references.")

        leakage = self.leakage_audit(category, query, references)

        query_url = load_image_data_url(query)
        reference_urls = [load_image_data_url(p) for p in reference_paths]

        messages = build_messages(query_url, reference_urls, k)
        system_prompt = messages[0]["content"]

        attempts: list[str] = []
        violations: list[str] = []
        parsed: dict[str, Any] | None = None

        for attempt in range(cfg.max_parse_retries + 1):
            response = await self.chat.ainvoke(
                messages, max_new_tokens=cfg.max_new_tokens, do_sample=False)
            text = _message_text(response)
            attempts.append(text)

            parsed, violations = parse_reference_output(text)
            if parsed is not None:
                break
            if attempt < cfg.max_parse_retries:
                # Greedy decoding reproduces an identical request verbatim, so
                # feed the failed answer back instead of re-asking blindly.
                messages = [
                    *messages,
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": CORRECTION_INSTRUCTION},
                ]

        return {
            # ---- the record contract required for every query
            "dataset": "MVTec-AD",
            "category": category,
            "query_image": str(query),
            "reference_k": int(k),
            "references": references,
            "raw_model_output": attempts[-1] if attempts else "",
            "parsed_output": parsed,
            "parse_ok": parsed is not None,
            "vlm_calls": len(attempts),
            "elapsed_s": round(time.time() - t0, 3),
            # ---- traceability
            "mode": "A" if k == MODE_A_K else "B",
            "mode_name": cfg.mode_name,
            "model_path": cfg.model_path,
            "embedder": self.embedder.model_id,
            "system_prompt": system_prompt,
            "violations": violations,
            "raw_attempts": attempts,
            "leakage": leakage,
        }

    async def evidence(
        self,
        query_image: str | Path,
        category: str,
        references: Sequence[dict[str, Any]] | None = None,
        config: ReferenceToolConfig | None = None,
    ) -> dict[str, Any]:
        """The tool's output contract: exactly the normality-evidence schema.

        Raises :class:`ReferenceToolError` when the model did not return a valid
        object, so a caller cannot mistake a failed comparison for evidence.
        The full auditable record is available from :meth:`compare`.
        """
        record = await self.compare(query_image, category, references, config)
        if record["parsed_output"] is None:
            raise ReferenceToolError(
                f"[{category}] no valid normality evidence for {query_image}: "
                f"{record['violations']}")
        return record["parsed_output"]

    # -- many comparisons --------------------------------------------------
    async def compare_many(
        self,
        requests: Sequence[dict[str, Any]],
        config: ReferenceToolConfig | None = None,
    ) -> list[dict[str, Any]]:
        """Compare many queries concurrently.

        Each request needs ``query_image`` and ``category``, and may carry
        ``references`` and any other keys, which are copied onto the record for
        traceability only. Concurrency is bounded by ``config.concurrency``.
        """
        cfg = config or self.config
        # Created per call: an asyncio primitive binds to the loop that first
        # awaits it, so a cached one would break a second asyncio.run().
        semaphore = asyncio.Semaphore(cfg.concurrency)
        reserved = {"query_image", "category", "references"}

        async def run_one(request: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                record = await self.compare(
                    request["query_image"], request["category"],
                    request.get("references"), cfg)
            for key, value in request.items():
                if key not in reserved:
                    record[key] = value
            return record

        return list(await asyncio.gather(*(run_one(r) for r in requests)))


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def write_reference_record(
    record: dict[str, Any],
    out_root: str | Path,
    category: str,
    stem: str,
) -> Path:
    """Persist one record as ``<out_root>/<mode>/<category>/<stem>.json``."""
    out_dir = Path(out_root) / record["mode_name"] / category
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path
