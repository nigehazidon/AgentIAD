"""Phase 3: local visual observation of a candidate patch with Qwen3-VL.

Phase 1 (``patchcore_predictor``) produces an image-level score and a
native-resolution anomaly map. Phase 2 (``patchcore_regions``) turns that map
into ranked candidate regions and crops. This module consumes one such crop and
asks a local vision-language model to **describe what is visibly there**:

    (original query image) + (local patch) -> Qwen3-VL -> observation JSON

Design guarantees:

  * the module never emits a NORMAL / ANOMALOUS decision — it returns only an
    observation, leaving the verdict to a later Planner/Reflector stage;
  * it never outputs bounding boxes or coordinates, even though it is *given* a
    bbox by its caller (the bbox is used to crop, never passed to the model);
  * **no anomaly score ever enters the prompt.**  ``image_score`` and
    ``region_score`` are relative per-image quantities; the system prompt
    explicitly forbids treating an anomaly score as proof of an anomaly, so the
    score is deliberately kept out of the conversation and only stored next to
    the record for traceability (``anomaly_score_used_in_prompt: false``);
  * nothing here reads a ground-truth mask.

The system prompt is fixed verbatim (``LOCAL_INSPECTION_SYSTEM_PROMPT``). Two
input modes are supported so they can be compared:

  * ``image_and_patch`` — the original query image *and* the patch (the literal
    reading of the prompt);
  * ``patch_only``      — the patch alone.

The two modes differ **only** in what is attached; only the ``You are given:``
bullet list is rewritten for ``patch_only``, and the exact system prompt used is
saved with every record so the comparison stays auditable.

Minimal interface::

    import asyncio
    from local_visual_observation import LocalVisualObserver, ObservationConfig

    obs = LocalVisualObserver(ObservationConfig(input_mode="image_and_patch"))
    rec = asyncio.run(obs.observe("/.../test/broken_large/000.png",
                                  "/.../crops/broken_large_000__rank1.png"))
    print(rec["observation"])

Batching many queries (the Qwen micro-batcher groups the concurrent calls)::

    recs = asyncio.run(obs.observe_many([
        {"query_image": q, "patch_path": p, "category": "bottle", "rank": 1},
        ...
    ]))
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parent
_AGENT_SRC = _REPO_ROOT / "AnomalyAgent" / "src"
for _p in (str(_AGENT_SRC), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_MODEL_PATH = "/data/pfy/MLLMs/Qwen3-VL-4B-Instruct"

#: Input modes (what is attached to the user turn).
PATCH_ONLY = "patch_only"
IMAGE_AND_PATCH = "image_and_patch"
INPUT_MODES = (IMAGE_AND_PATCH, PATCH_ONLY)

#: Allowed values of the two enum fields.
IRREGULARITY_VALUES = ("present", "absent", "uncertain")
CONFIDENCE_VALUES = ("low", "medium", "high")
ENUM_VALUES = {
    "visual_irregularity": IRREGULARITY_VALUES,
    "confidence": CONFIDENCE_VALUES,
}

#: The four keys the model must return, in schema order.
OBSERVATION_KEYS = ("observation", "visual_irregularity", "evidence", "confidence")


class ObservationError(RuntimeError):
    """Raised when the observation stage cannot be configured or executed."""


# --------------------------------------------------------------------------
# System prompt (verbatim) and its per-mode variants
# --------------------------------------------------------------------------

#: The system prompt, exactly as specified for this module.
LOCAL_INSPECTION_SYSTEM_PROMPT = """You are a local visual inspection module for industrial anomaly detection.

You are given:

1. The original query image.
2. A local image patch selected by an external anomaly localization system.

The patch was selected because it contains visual patterns that may be inconsistent with normal appearance.

Your task is ONLY to inspect and describe the local visual evidence.

Do NOT make the final NORMAL/ANOMALOUS decision.

Do NOT output bounding boxes or coordinates.

Do NOT assume that the patch is defective just because it was selected by the external system.

You must:

1. Describe what is visibly present in the patch.
2. Identify scratches, cracks, dents, stains, contamination, deformation, missing material, irregular texture, broken parts, or other visible irregularities when they are actually present.
3. State explicitly when the region appears visually normal.
4. Distinguish clear visual evidence from uncertain interpretation.
5. Do not invent defects.
6. Do not use the anomaly score as proof of an anomaly.

Return ONLY valid JSON:

{
"observation": "objective description of what is visible",
"visual_irregularity": "present | absent | uncertain",
"evidence": [
"specific visible observation"
],
"confidence": "low | medium | high"
}
"""

#: The "You are given:" block, in both of its variants.  ``patch_only``
#: rewrites *only* this block; everything else (the task, the prohibitions, the
#: JSON contract) stays byte-identical between the two modes.
_GIVEN_BLOCK_BOTH = (
    "1. The original query image.\n"
    "2. A local image patch selected by an external anomaly localization system."
)
_GIVEN_BLOCK_PATCH_ONLY = (
    "1. A local image patch selected by an external anomaly localization system."
)

#: Sent as a corrective user turn when a response failed to validate.  Greedy
#: decoding would reproduce the same output for an identical request, so the
#: retry appends the failed answer and this instruction instead.
CORRECTION_INSTRUCTION = (
    "Your previous response was not valid JSON in the required schema. "
    "Return ONLY the JSON object, with exactly these four keys: "
    '"observation" (string), "visual_irregularity" (one of: present, absent, uncertain), '
    '"evidence" (array of short strings), "confidence" (one of: low, medium, high). '
    'If "visual_irregularity" is "present" then "evidence" must list at least one '
    'specific visible observation; for "absent" or "uncertain" an empty array is allowed. '
    "No prose, no markdown fence."
)


def system_prompt_for(mode: str) -> str:
    """The system prompt for ``mode``.

    ``image_and_patch`` returns :data:`LOCAL_INSPECTION_SYSTEM_PROMPT` verbatim.
    ``patch_only`` rewrites only the ``You are given:`` bullet list.
    """
    if mode == IMAGE_AND_PATCH:
        return LOCAL_INSPECTION_SYSTEM_PROMPT
    if mode == PATCH_ONLY:
        prompt = LOCAL_INSPECTION_SYSTEM_PROMPT.replace(
            _GIVEN_BLOCK_BOTH, _GIVEN_BLOCK_PATCH_ONLY)
        if prompt == LOCAL_INSPECTION_SYSTEM_PROMPT:
            raise ObservationError(
                "patch_only prompt rewrite did not apply: the 'You are given:' "
                "block in LOCAL_INSPECTION_SYSTEM_PROMPT has changed")
        return prompt
    raise ObservationError(f"unknown input mode: {mode!r} (expected one of {INPUT_MODES})")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ObservationConfig:
    """Independent, user-settable parameters of the observation stage."""

    input_mode: str = IMAGE_AND_PATCH
    model_path: str = DEFAULT_MODEL_PATH
    attn_implementation: str | None = "flash_attention_2"
    max_new_tokens: int = 512
    #: Extra attempts after the first one when the reply fails to validate.
    max_parse_retries: int = 2
    #: Qwen micro-batcher settings (forwarded to ``load_vl_model``).
    max_batch_size: int = 16
    max_wait_ms: int = 50
    #: Maximum in-flight requests. Kept above ``max_batch_size`` so the batcher
    #: still has calls to group, and bounded so the encoded images in flight do
    #: not grow without limit.
    concurrency: int = 32

    def __post_init__(self) -> None:
        if self.input_mode not in INPUT_MODES:
            raise ValueError(
                f"input_mode must be one of {INPUT_MODES}, got {self.input_mode!r}")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if self.max_parse_retries < 0:
            raise ValueError(
                f"max_parse_retries must be >= 0, got {self.max_parse_retries}")
        if self.max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {self.max_batch_size}")
        if self.concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {self.concurrency}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_mode": self.input_mode,
            "model_path": self.model_path,
            "attn_implementation": self.attn_implementation,
            "max_new_tokens": int(self.max_new_tokens),
            "max_parse_retries": int(self.max_parse_retries),
            "max_batch_size": int(self.max_batch_size),
            "max_wait_ms": int(self.max_wait_ms),
            "concurrency": int(self.concurrency),
        }


DEFAULT_CONFIG = ObservationConfig()


# --------------------------------------------------------------------------
# Message construction
# --------------------------------------------------------------------------

def build_user_content(
    query_data_url: str | None,
    patch_data_url: str,
    mode: str,
) -> list[dict[str, Any]]:
    """The user turn: the attached image(s), each preceded by a short label.

    Mirrors the shape ``agent_v1_27.py`` uses for its reasoner/reflector calls:
    ``{"type": "image_url", "image_url": {"url": <data URL>}}``.
    """
    if mode not in INPUT_MODES:
        raise ObservationError(f"unknown input mode: {mode!r}")
    if mode == IMAGE_AND_PATCH and not query_data_url:
        raise ObservationError("image_and_patch mode requires the original query image")

    content: list[dict[str, Any]] = []
    if mode == IMAGE_AND_PATCH:
        content.append({"type": "text", "text": "Original query image:"})
        content.append({"type": "image_url", "image_url": {"url": query_data_url}})
    content.append({"type": "text", "text": "Local image patch:"})
    content.append({"type": "image_url", "image_url": {"url": patch_data_url}})
    return content


def build_messages(
    query_data_url: str | None,
    patch_data_url: str,
    mode: str,
) -> list[dict[str, Any]]:
    """Full chat message list (system + user) for one observation."""
    return [
        {"role": "system", "content": system_prompt_for(mode)},
        {"role": "user", "content": build_user_content(query_data_url, patch_data_url, mode)},
    ]


def prompt_text_of(messages: Sequence[dict[str, Any]]) -> str:
    """All text carried by ``messages``, with image payloads dropped.

    Used by the smoke test to assert that no anomaly score reaches the model.
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


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of one JSON object from a model reply.

    Tries, in order: the whole reply, a ```json fenced block, the repo's shared
    ``_parse_json_from_text`` helper, then the first brace-delimited object that
    decodes (``raw_decode``, which stops at the object's end rather than at the
    last ``}`` in the text).
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


def validate_observation(obj: Any) -> list[str]:
    """Schema violations of ``obj``; empty list means the observation is valid.

    Only violations listed in :data:`FATAL_VIOLATION_PREFIXES` cause a retry;
    extra keys are recorded but stripped, since the contract is about the four
    required fields rather than about forbidding additions.
    """
    if not isinstance(obj, dict):
        return ["not a JSON object"]

    problems: list[str] = []
    missing = [k for k in OBSERVATION_KEYS if k not in obj]
    if missing:
        problems.append(f"missing keys: {missing}")

    extra = sorted(set(obj) - set(OBSERVATION_KEYS))
    if extra:
        problems.append(f"extra keys (ignored): {extra}")

    obs = obj.get("observation")
    if not isinstance(obs, str) or not obs.strip():
        problems.append("observation must be a non-empty string")

    for key, allowed in ENUM_VALUES.items():
        if key not in obj:
            continue
        value = obj[key]
        if not isinstance(value, str) or value.lower() not in allowed:
            problems.append(f"{key} must be one of {list(allowed)}, got {value!r}")

    evidence = obj.get("evidence")
    if not isinstance(evidence, list):
        problems.append("evidence must be a list of strings")
    elif not all(isinstance(e, str) and e.strip() for e in evidence):
        problems.append("every entry of evidence must be a non-empty string")
    elif not evidence and str(obj.get("visual_irregularity", "")).strip().lower() == "present":
        # Claiming an irregularity is *present* requires at least one visible
        # observation behind it. An empty list is allowed for "absent" and
        # "uncertain": demanding evidence for a region that looks normal would
        # only push the model to invent some, which the prompt forbids.
        problems.append("evidence must not be empty when visual_irregularity is 'present'")

    return problems


#: A violation with one of these prefixes is not repairable by normalisation.
FATAL_VIOLATION_PREFIXES = ("not a JSON object", "missing keys", "observation must",
                            "visual_irregularity must", "confidence must",
                            "evidence must", "every entry of evidence")


def is_fatal(violations: Sequence[str]) -> bool:
    return any(v.startswith(FATAL_VIOLATION_PREFIXES) for v in violations)


def normalize_observation(obj: dict[str, Any]) -> dict[str, Any]:
    """Keep exactly the four schema keys, with enums lower-cased."""
    return {
        "observation": obj["observation"].strip(),
        "visual_irregularity": obj["visual_irregularity"].strip().lower(),
        "evidence": [e.strip() for e in obj["evidence"]],
        "confidence": obj["confidence"].strip().lower(),
    }


def parse_observation(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Parse ``text`` into a valid observation.

    Returns ``(observation, violations)``. ``observation`` is ``None`` unless the
    reply decoded *and* had no fatal violation; ``violations`` always describes
    everything that was wrong (empty when the reply was fully valid).
    """
    obj = extract_json_object(text)
    if obj is None:
        return None, ["not a JSON object"]
    violations = validate_observation(obj)
    if is_fatal(violations):
        return None, violations
    return normalize_observation(obj), violations


# --------------------------------------------------------------------------
# Image encoding
# --------------------------------------------------------------------------

def load_image_data_url(image_path: str | Path) -> str:
    """Read an image and return it as a PNG data URL.

    Uses the AnomalyAgent helper so the payload matches what the agent sends.
    """
    from PIL import Image
    from react_agent.tools import _encode_image_to_base64

    path = Path(image_path)
    if not path.is_file():
        raise ObservationError(f"image not found: {path}")
    with Image.open(path) as im:
        return _encode_image_to_base64(im.convert("RGB"), fmt="PNG")


# --------------------------------------------------------------------------
# Observer
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


class LocalVisualObserver:
    """Run the local visual observation prompt against a Qwen-VL chat model.

    The model is loaded lazily on first use, so importing this module (and
    running its unit-level checks) does not touch the GPU. ``chat`` may be
    injected to run the stage against a stub.
    """

    def __init__(self, config: ObservationConfig | None = None, *, chat: Any = None) -> None:
        self.config = config or DEFAULT_CONFIG
        self._chat = chat
        self._load_lock = threading.Lock()

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

    # -- one observation ---------------------------------------------------
    async def observe(
        self,
        query_image: str | Path | None,
        patch_path: str | Path,
        config: ObservationConfig | None = None,
    ) -> dict[str, Any]:
        """Observe one patch and return the record (see :meth:`observe_many`).

        ``query_image`` may be ``None`` in ``patch_only`` mode. In that mode the
        original image is neither encoded nor attached; its path is still carried
        on the record so the observation stays traceable back to its query.
        """
        cfg = config or self.config
        attach_query = cfg.input_mode == IMAGE_AND_PATCH
        if attach_query and not query_image:
            raise ObservationError("image_and_patch mode requires a query image")

        t0 = time.time()
        query_url = load_image_data_url(query_image) if attach_query else None
        patch_url = load_image_data_url(patch_path)

        messages = build_messages(query_url, patch_url, cfg.input_mode)
        # No anomaly score is ever placed in the conversation.
        system_prompt = messages[0]["content"]

        attempts: list[str] = []
        violations: list[str] = []
        observation: dict[str, Any] | None = None

        for attempt in range(cfg.max_parse_retries + 1):
            response = await self.chat.ainvoke(
                messages, max_new_tokens=cfg.max_new_tokens, do_sample=False)
            text = _message_text(response)
            attempts.append(text)

            observation, violations = parse_observation(text)
            if observation is not None:
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
            "query_image": str(query_image) if query_image else None,
            "patch_path": str(patch_path),
            "input_mode": cfg.input_mode,
            "model_path": cfg.model_path,
            "system_prompt": system_prompt,
            "observation": observation,
            "observation_raw": attempts[-1] if attempts else "",
            "attempts": len(attempts),
            "parse_ok": observation is not None,
            "violations": violations,
            "raw_attempts": attempts,
            "elapsed_s": round(time.time() - t0, 3),
            # Traceability: the score is never sent to the model.
            "anomaly_score_used_in_prompt": False,
        }

    # -- many observations -------------------------------------------------
    async def observe_many(
        self,
        requests: Sequence[dict[str, Any]],
        config: ObservationConfig | None = None,
    ) -> list[dict[str, Any]]:
        """Observe many patches concurrently.

        Each request is a dict with ``query_image`` and ``patch_path``; any other
        keys (``category``, ``rank``, ``bbox``, ``region_score``, ...) are copied
        onto the returned record for traceability only — none of them reach the
        model.

        Concurrency is bounded by ``config.concurrency``; the Qwen micro-batcher
        groups the in-flight calls into forward passes.
        """
        cfg = config or self.config
        # Created per call: an asyncio primitive binds to the loop that first
        # awaits it, so a cached one would break a second asyncio.run().
        semaphore = asyncio.Semaphore(cfg.concurrency)

        reserved = {"query_image", "patch_path"}

        async def run_one(request: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                record = await self.observe(
                    request.get("query_image"), request["patch_path"], cfg)
            for key, value in request.items():
                if key not in reserved:
                    record[key] = value
            return record

        return list(await asyncio.gather(*(run_one(r) for r in requests)))


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def write_observation(
    record: dict[str, Any],
    out_root: str | Path,
    category: str,
    stem: str,
    rank: int,
) -> Path:
    """Persist one record as ``<out_root>/<mode>/<category>/<stem>__rank<k>.json``."""
    out_dir = Path(out_root) / record["input_mode"] / category
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}__rank{rank}.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path
