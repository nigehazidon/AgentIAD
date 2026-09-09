"""Tool definitions, image utility functions, and heuristic decision tools for AnomalyAgent."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from PIL import Image as PILImage

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .context import Context

# ================================================================
# Image utility functions
# ================================================================


def _encode_image_to_base64(pil_img: PILImage.Image, fmt: str = "PNG") -> str:
    """Convert PIL Image to base64 data URL.

    Args:
        pil_img: PIL Image to encode.
        fmt: Image format (PNG, JPEG, etc.). Default: PNG.

    Returns:
        Base64 data URL string, e.g. "data:image/png;base64,..."
    """
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _decode_base64_image(b64_str: str) -> PILImage.Image:
    """Decode a base64 string or data URL to a PIL Image.

    Args:
        b64_str: Base64 string or data URL (e.g. "data:image/png;base64,...").

    Returns:
        PIL Image.
    """
    if b64_str.startswith("data:image"):
        b64_str = b64_str.split(",", 1)[1]
    b64_str = b64_str.strip()
    b64_str += "=" * (-len(b64_str) % 4)
    img_bytes = base64.b64decode(b64_str)
    return PILImage.open(io.BytesIO(img_bytes)).convert("RGB")


def _normalize_to_data_url(image_str: Optional[str]) -> Optional[str]:
    """Normalize a base64 string or data URL to a proper 'data:image/...' format.

    Args:
        image_str: Raw base64 string or data URL.

    Returns:
        Normalized data URL, or None if input is invalid.
    """
    if not image_str or not isinstance(image_str, str):
        return None
    image_str = image_str.strip()
    if image_str.startswith("data:image"):
        return image_str
    if image_str.startswith("data:"):
        return image_str
    try:
        test = image_str.replace("-", "+").replace("_", "/")
        test += "=" * (-len(test) % 4)
        base64.b64decode(test, validate=True)
        return f"data:image/png;base64,{image_str}"
    except Exception:
        pass
    try:
        PILImage.open(io.BytesIO(image_str.encode()))
        return f"data:image/png;base64,{image_str}"
    except Exception:
        pass
    return None


def to_pure_base64(data_url: str) -> str:
    """Extract the pure base64 payload from a data URL.

    Args:
        data_url: A data URL like "data:image/png;base64,ABCD...".

    Returns:
        Pure base64 string.
    """
    if data_url.startswith("data:"):
        return data_url.split(",", 1)[-1]
    return data_url


def _pil_to_cv2(pil_img: PILImage.Image) -> np.ndarray:
    """Convert PIL Image (RGB) to OpenCV image (BGR).

    Args:
        pil_img: PIL Image in RGB.

    Returns:
        numpy array in BGR format (H, W, 3).
    """
    return np.array(pil_img)[:, :, ::-1].copy()


def _cv2_to_pil(cv2_img: np.ndarray) -> PILImage.Image:
    """Convert OpenCV image (BGR) to PIL Image (RGB).

    Args:
        cv2_img: numpy array in BGR format.

    Returns:
        PIL Image in RGB.
    """
    return PILImage.fromarray(cv2_img[:, :, ::-1])


def _extract_image_from_messages(messages: List) -> Optional[str]:
    """Extract the last image data URL from ToolMessages in the message list.

    Scans messages in reverse, looking for ToolMessages whose content contains
    image_url entries. Returns the last such image URL found.

    Args:
        messages: List of LangChain message objects.

    Returns:
        Data URL string of the last found image, or None.
    """
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            content = getattr(msg, "content", None)
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url:
                            return _normalize_to_data_url(url)
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = getattr(msg, "content", None)
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url:
                            return _normalize_to_data_url(url)
    return None


# ================================================================
# PromptOrder sentinel
# ================================================================


class PromptOrder:
    """Sentinel class for prompt ordering (placeholder)."""
    pass


# ================================================================
# Image processing tools
# ================================================================

IMAGE_PROCESSING_AVAILABLE = False
cv2 = None
try:
    import cv2 as _cv2
    cv2 = _cv2
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError:
    pass


# --- Tool input schemas ---


class ImageDenoisingInput(BaseModel):
    image_base64: str = Field(description="Base64-encoded image data URL.")
    strength: int = Field(default=10, description="Denoising strength (1-20). Higher = stronger denoising.")


class ImageDeblurringInput(BaseModel):
    image_base64: str = Field(description="Base64-encoded image data URL.")
    sigma: float = Field(default=1.5, description="Blur sigma for unsharp masking (0.5-5.0).")


class ImageSuperResolutionInput(BaseModel):
    image_base64: str = Field(description="Base64-encoded image data URL.")
    scale: float = Field(default=2.0, description="Upscale factor (2.0 or 4.0).")


class ImageZoomingInput(BaseModel):
    image_base64: str = Field(description="Base64-encoded image data URL.")
    zoom_factor: float = Field(default=2.0, description="Zoom/magnification factor (1.5-8.0).")
    region: str = Field(default="center", description="Region to zoom: 'center' or 'top_left', 'top_right', 'bottom_left', 'bottom_right'.")


class ImageBrightnessInput(BaseModel):
    image_base64: str = Field(description="Base64-encoded image data URL.")
    clip_limit: float = Field(default=2.0, description="CLAHE clip limit (1.0-5.0). Higher = more contrast.")


# --- Tool implementations ---


def _image_denoising(image_base64: str, strength: int = 10) -> str:
    """Apply fast non-local means denoising to reduce noise/grain."""
    global cv2
    if cv2 is None:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except ImportError:
            return "Error: OpenCV (cv2) is not available. Install with: pip install opencv-python"

    try:
        pil_img = _decode_base64_image(image_base64)
        cv_img = _pil_to_cv2(pil_img)
        strength = max(1, min(20, strength))
        denoised = cv2.fastNlMeansDenoisingColored(cv_img, None, strength, strength, 7, 21)
        result_pil = _cv2_to_pil(denoised)
        b64 = _encode_image_to_base64(result_pil, fmt="JPEG")
        return json.dumps({"processed_image_base64": b64, "text": f"Denoising applied (strength={strength})."})
    except Exception as e:
        return f"Error: image_denoising failed: {e}"


def _image_deblurring(image_base64: str, sigma: float = 1.5) -> str:
    """Sharpen image using unsharp masking with Gaussian blur subtraction."""
    global cv2
    if cv2 is None:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except ImportError:
            return "Error: OpenCV (cv2) is not available."

    try:
        pil_img = _decode_base64_image(image_base64)
        cv_img = _pil_to_cv2(pil_img)
        sigma = max(0.5, min(5.0, sigma))
        ksize = int(2 * round(3 * sigma) + 1)
        ksize = max(3, ksize)
        blurred = cv2.GaussianBlur(cv_img, (ksize, ksize), sigma)
        sharpened = cv2.addWeighted(cv_img, 1.5, blurred, -0.5, 0)
        result_pil = _cv2_to_pil(sharpened)
        b64 = _encode_image_to_base64(result_pil, fmt="JPEG")
        return json.dumps({"processed_image_base64": b64, "text": f"Unsharp masking applied (sigma={sigma})."})
    except Exception as e:
        return f"Error: image_deblurring failed: {e}"


def _image_super_resolution(image_base64: str, scale: float = 2.0) -> str:
    """Enhance image resolution using bicubic upscaling.

    Note: This is a simple interpolation-based upscaling. When realesrgan is
    available, the actual Real-ESRGAN model is used instead.
    """
    try:
        pil_img = _decode_base64_image(image_base64)
        w, h = pil_img.size
        scale = 4.0 if scale >= 3.0 else 2.0
        new_w, new_h = int(w * scale), int(h * scale)
        # Cap maximum dimension to prevent OOM
        if max(new_w, new_h) > 4096:
            scale = 4096 / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)

        result = pil_img.resize((new_w, new_h), PILImage.Resampling.BICUBIC)
        b64 = _encode_image_to_base64(result, fmt="JPEG")
        return json.dumps({"processed_image_base64": b64, "text": f"Super-resolution applied (scale={scale:.1f}x, {w}x{h} -> {new_w}x{new_h})."})
    except Exception as e:
        return f"Error: image_super_resolution failed: {e}"


def _image_zooming(image_base64: str, zoom_factor: float = 2.0, region: str = "center") -> str:
    """Magnify a specific region of the image."""
    global cv2
    if cv2 is None:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except ImportError:
            return "Error: OpenCV (cv2) is not available."

    try:
        pil_img = _decode_base64_image(image_base64)
        w, h = pil_img.size
        zoom_factor = max(1.1, min(8.0, zoom_factor))
        crop_w, crop_h = int(w / zoom_factor), int(h / zoom_factor)
        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)

        region = (region or "center").lower().strip()
        if region == "center" or region not in ("top_left", "top_right", "bottom_left", "bottom_right"):
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2
        elif region == "top_left":
            left, top = 0, 0
        elif region == "top_right":
            left, top = w - crop_w, 0
        elif region == "bottom_left":
            left, top = 0, h - crop_h
        else:  # bottom_right
            left, top = w - crop_w, h - crop_h

        cropped = pil_img.crop((left, top, left + crop_w, top + crop_h))
        zoomed = cropped.resize((w, h), PILImage.Resampling.LANCZOS)
        b64 = _encode_image_to_base64(zoomed, fmt="JPEG")
        return json.dumps({"processed_image_base64": b64, "text": f"Zoom applied (factor={zoom_factor}x, region={region})."})
    except Exception as e:
        return f"Error: image_zooming failed: {e}"


def _image_brightness_enhancement(image_base64: str, clip_limit: float = 2.0) -> str:
    """Enhance brightness/contrast using CLAHE on LAB L-channel."""
    global cv2
    if cv2 is None:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except ImportError:
            return "Error: OpenCV (cv2) is not available."

    try:
        pil_img = _decode_base64_image(image_base64)
        cv_img = _pil_to_cv2(pil_img)
        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clip_limit = max(0.5, min(5.0, clip_limit))
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        eq_lab = cv2.merge([l_eq, a, b])
        eq_bgr = cv2.cvtColor(eq_lab, cv2.COLOR_LAB2BGR)
        result_pil = _cv2_to_pil(eq_bgr)
        b64 = _encode_image_to_base64(result_pil, fmt="JPEG")
        return json.dumps({"processed_image_base64": b64, "text": f"CLAHE brightness enhancement applied (clip_limit={clip_limit})."})
    except Exception as e:
        return f"Error: image_brightness_enhancement failed: {e}"


# --- Build TOOLS list ---

TOOLS = [
    StructuredTool(
        name="image_denoising",
        description="Reduce noise or grain in the image using fast non-local means denoising.",
        args_schema=ImageDenoisingInput,
        func=_image_denoising,
    ),
    StructuredTool(
        name="image_deblurring",
        description="Sharpen blurry or out-of-focus images using unsharp masking.",
        args_schema=ImageDeblurringInput,
        func=_image_deblurring,
    ),
    StructuredTool(
        name="image_super_resolution",
        description="Enhance fine-grained details by upscaling image resolution.",
        args_schema=ImageSuperResolutionInput,
        func=_image_super_resolution,
    ),
    StructuredTool(
        name="image_zooming",
        description="Magnify a specific region of the image for closer inspection.",
        args_schema=ImageZoomingInput,
        func=_image_zooming,
    ),
    StructuredTool(
        name="image_brightness_enhancement",
        description="Enhance brightness and contrast for dimly lit or overexposed images using CLAHE.",
        args_schema=ImageBrightnessInput,
        func=_image_brightness_enhancement,
    ),
]


# ================================================================
# WinCLIP-style text templates for general template analysis
# ================================================================

DEFAULT_NORMAL_TEMPLATES = [
    "a normal {cls}",
    "a good {cls}",
    "a perfect {cls}",
    "a clean {cls}",
    "an intact {cls}",
    "a {cls} in good condition",
    "a {cls} without defects",
    "a flawless {cls}",
]

DEFAULT_ANOMALY_TEMPLATES = [
    "a damaged {cls}",
    "a broken {cls}",
    "a defective {cls}",
    "a faulty {cls}",
    "a {cls} with a defect",
    "a {cls} that is damaged",
    "a {cls} with scratch",
    "a {cls} with crack",
    "a {cls} with dent",
    "a {cls} with stain",
    "a {cls} with missing part",
    "a {cls} with deformation",
]


# ================================================================
# Helper: get text from message content (duplicated from utils for isolation)
# ================================================================


def _get_text(content: Any) -> str:
    """Normalize AIMessage.content into string."""
    if isinstance(content, str):
        return content

    def _collect(v: Any) -> List[str]:
        frags: List[str] = []
        if isinstance(v, dict):
            tv = v.get("text")
            if isinstance(tv, str) and tv:
                frags.append(tv)
            for child in v.values():
                frags.extend(_collect(child))
        elif isinstance(v, list):
            for child in v:
                frags.extend(_collect(child))
        return frags

    frags = _collect(content)
    if frags:
        return "\n".join(f for f in frags if f)
    return ""


def _parse_json_from_text(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from model output."""
    if not text:
        return {}
    pattern = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```|(\{[\s\S]*\})", re.IGNORECASE)
    match = pattern.search(text)
    candidate = (match.group(1) or match.group(2)) if match else text.strip()
    if not candidate:
        return {}
    try:
        return json.loads(candidate)
    except Exception:
        try:
            return json.loads(candidate.strip("` \n\t"))
        except Exception:
            return {}


# ================================================================
# Template analysis: compute prototype embeddings and scores
# ================================================================


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


async def _compute_prototypes(
    embedding_model: Any,
    class_name: str,
    normal_templates: Optional[List[str]] = None,
    anomaly_templates: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute averaged normal and anomaly prototype embeddings for a class.

    Returns:
        Tuple of (normal_prototype, anomaly_prototype) as numpy arrays.
    """
    if normal_templates is None:
        normal_templates = [t.format(cls=class_name) for t in DEFAULT_NORMAL_TEMPLATES]
    else:
        normal_templates = [t.format(cls=class_name) for t in normal_templates]

    if anomaly_templates is None:
        anomaly_templates = [t.format(cls=class_name) for t in DEFAULT_ANOMALY_TEMPLATES]
    else:
        anomaly_templates = [t.format(cls=class_name) for t in anomaly_templates]

    all_templates = normal_templates + anomaly_templates
    has_async = hasattr(embedding_model, "aembed_documents") and callable(embedding_model.aembed_documents)

    if has_async:
        embs = await embedding_model.aembed_documents(all_templates)
    else:
        embs = embedding_model.embed_documents(all_templates)

    embs = [np.array(e) for e in embs]
    n = len(normal_templates)
    normal_proto = np.mean(embs[:n], axis=0)
    anomaly_proto = np.mean(embs[n:], axis=0)
    return normal_proto, anomaly_proto


async def _compute_caption_scores(
    captions: List[str],
    embedding_model: Any,
    normal_proto: np.ndarray,
    anomaly_proto: np.ndarray,
    tau: float = 0.3,
) -> List[Dict[str, Any]]:
    """Compute anomaly scores and margins for each caption.

    Returns:
        List of dicts with keys: caption, sim_norm, sim_anom, margin, score.
    """
    has_async = hasattr(embedding_model, "aembed_documents") and callable(embedding_model.aembed_documents)
    if has_async:
        cap_embs = await embedding_model.aembed_documents(captions)
    else:
        cap_embs = embedding_model.embed_documents(captions)

    results = []
    for i, cap in enumerate(captions):
        emb = np.array(cap_embs[i])
        sim_norm = _cosine_similarity(emb, normal_proto)
        sim_anom = _cosine_similarity(emb, anomaly_proto)
        margin = sim_anom - sim_norm
        score = 1.0 / (1.0 + np.exp(-margin / max(tau, 0.01)))
        results.append({
            "caption": cap,
            "sim_norm": sim_norm,
            "sim_anom": sim_anom,
            "margin": margin,
            "score": float(score),
        })
    return results


def _format_scores(results: List[Dict[str, Any]]) -> Tuple[str, str, str, str, str, str]:
    """Format score and margin strings for reports."""
    score_strs = []
    margin_strs = []
    for r in results:
        score_strs.append(f"{r['score']:.3f} ({'anomaly' if r['score'] > 0.5 else 'normal'}-leaning)")
        margin_strs.append(f"{r['margin']:+.3f}")
    return (*score_strs, *margin_strs)


def _escape_caption(c: str) -> str:
    """Escape a caption for safe inclusion in a report."""
    return c.replace("{", "{{").replace("}", "}}")


# ================================================================
# Keyword heuristic decision
# ================================================================


async def _keyword_heuristic_decision(
    state: Any,
    runtime: Any,
    decision: bool = True,
) -> Tuple[str, float, float, float, str]:
    """Keyword heuristic decision using caption-prototype comparison.

    Generates 3 captions from the image using the image description model,
    compares their embeddings against normal/anomaly text prototypes
    (WinCLIP-style ensembles), and produces a structured evidence report.

    Args:
        state: Agent State object.
        runtime: LangGraph Runtime providing context.
        decision: If True, use hard decision threshold; if False, return a
            detailed report even for borderline cases.

    Returns:
        Tuple of (predicted_label, score1, score2, score3, report_text).
        predicted_label: "anomalous" or "normal".
        score1-3: Anomaly scores for each of the 3 captions.
        report_text: Formatted evidence report.
    """
    from .prompts import PROMPT4DESCRIPTION

    ctx: Context = runtime.context
    class_name = getattr(state, "class_name", None) or "object"

    # 1. Extract raw image
    raw_image_url = getattr(state, "raw_image_url", None)
    if not raw_image_url:
        from .utils import extract_images_for_reasoner
        raw_image_url, _ = extract_images_for_reasoner(state.messages)
    if not raw_image_url:
        norm = _normalize_to_data_url(raw_image_url)
    else:
        norm = _normalize_to_data_url(raw_image_url)

    if not norm:
        return "normal", 0.5, 0.5, 0.5, "No valid image available for keyword heuristic."

    raw_image_url = norm

    # 2. Generate 3 captions using image description model
    model = ctx.image_description_model
    sys_time = datetime.now(tz=UTC).isoformat()
    system_prompt = ctx.system_prompt.format(system_time=sys_time)

    content = [
        {"type": "image_url", "image_url": {"url": raw_image_url}},
        {"type": "text", "text": PROMPT4DESCRIPTION},
    ]
    msg_list = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=content),
    ]

    try:
        resp = await model.ainvoke(msg_list)
        resp_text = _get_text(resp.content)
        parsed = _parse_json_from_text(resp_text)
        captions = parsed.get("captions", [])
        if not isinstance(captions, list):
            captions = list(captions) if isinstance(captions, (list, tuple)) else [str(captions)]
    except Exception as e:
        captions = []

    # Ensure exactly 3 captions
    while len(captions) < 3:
        captions.append(f"General visual inspection of {class_name}.")
    captions = [str(c) for c in captions[:3]]

    # 3. Compute prototypes
    embedding_model = ctx.embedding_model
    if embedding_model is None:
        return "normal", 0.5, 0.5, 0.5, "No embedding model available."

    try:
        normal_proto, anomaly_proto = await _compute_prototypes(
            embedding_model, class_name
        )
    except Exception as e:
        return "normal", 0.5, 0.5, 0.5, f"Prototype computation failed: {e}"

    # 4. Compute scores
    tau = getattr(ctx, "tau", 0.3)
    try:
        results = await _compute_caption_scores(captions, embedding_model, normal_proto, anomaly_proto, tau)
    except Exception as e:
        return "normal", 0.5, 0.5, 0.5, f"Score computation failed: {e}"

    if not results:
        return "normal", 0.5, 0.5, 0.5, "No valid scores computed."

    # Pad to 3 results
    while len(results) < 3:
        results.append({"caption": "N/A", "sim_norm": 0.0, "sim_anom": 0.0, "margin": 0.0, "score": 0.5})

    scores = [r["score"] for r in results[:3]]
    margins = [r["margin"] for r in results[:3]]

    # 5. Format report
    from .prompts import format_keyword_heuristic_report

    escaped_captions = [_escape_caption(r["caption"]) for r in results[:3]]
    s1_s, s2_s, s3_s, m1_s, m2_s, m3_s = _format_scores(results[:3])

    # Determine status per caption
    caption_status = []
    for r in results[:3]:
        if abs(r["margin"]) < 0.01:
            caption_status.append("low-confidence")
        elif r["score"] > 0.5:
            caption_status.append("anomaly-leaning")
        else:
            caption_status.append("normal-leaning")

    # Agreement pattern
    above_half = sum(1 for s in scores if s > 0.5)
    if above_half == 3:
        agreement = "All 3 perspectives lean toward ANOMALY"
    elif above_half == 0:
        agreement = "All 3 perspectives lean toward NORMAL"
    elif above_half >= 2:
        agreement = "Majority (2/3) lean toward ANOMALY"
    else:
        agreement = "Majority (2/3) lean toward NORMAL"

    # Top perspectives
    score_arr = np.array(scores)
    high_idx = np.argsort(score_arr)[-2:]
    top_persps = f"perspectives {[i+1 for i in high_idx]}"

    score_spread = float(np.ptp(score_arr))
    margin_spread = float(np.ptp(margins))

    # Uncertainty note
    if score_spread > 0.3:
        unc_note = f"High disagreement among perspectives (spread={score_spread:.3f})."
    elif max(scores) < 0.6 and min(scores) > 0.4:
        unc_note = "All scores near 0.5 boundary — low confidence."
    else:
        unc_note = "Reasonable agreement across perspectives."

    status_text = f"{sum(1 for s in caption_status if 'normal' in s)} normal, {sum(1 for s in caption_status if 'anomaly' in s)} anomaly"

    report = format_keyword_heuristic_report(
        tool_name="keyword_heuristic",
        class_name=class_name,
        escaped_captions=escaped_captions,
        score1_s=s1_s, score2_s=s2_s, score3_s=s3_s,
        margin1_s=m1_s, margin2_s=m2_s, margin3_s=m3_s,
        caption_status=caption_status,
        agreement_pattern=agreement,
        top_perspectives=top_persps,
        score_spread=score_spread,
        margin_spread=margin_spread,
        uncertainty_note=unc_note,
        status_text=status_text,
    )

    # 6. Hard decision
    avg_score = float(np.mean(scores))
    if decision:
        label = "anomalous" if avg_score > 0.5 else "normal"
    else:
        label = "anomalous" if avg_score > 0.65 else "normal"

    return label, float(scores[0]), float(scores[1]), float(scores[2]), report


# ================================================================
# Counterfactual atomic candidate tool
# ================================================================


async def counterfactual_atomic_candidate_tool(
    state: Any,
    runtime: Any,
    k: int = 3,
) -> str:
    """Counterfactual atomic candidate matching for anomaly detection.

    Uses class-conditioned atomic candidates (anomaly & normal textual descriptions)
    and matches image captions against them using embedding similarity.

    Args:
        state: Agent State object (must have class_name, optional image_captions).
        runtime: LangGraph Runtime providing context.
        k: Number of top-K matches to retrieve per caption.

    Returns:
        Formatted counterfactual analysis report string.
    """
    from .prompts import PROMPT4DESCRIPTION, PROMPT4CANDIDATE_GENERATION

    ctx: Context = runtime.context
    class_name = getattr(state, "class_name", None) or "object"

    if not class_name or class_name == "object":
        return "[counterfactual_atomic] No class name available for candidate generation."

    # 1. Get image captions (from state or generate)
    captions = getattr(state, "image_captions", None)
    if not captions or not isinstance(captions, list) or len(captions) < 3:
        raw_image_url = getattr(state, "raw_image_url", None)
        if not raw_image_url:
            from .utils import extract_images_for_reasoner
            raw_image_url, _ = extract_images_for_reasoner(state.messages)
        if raw_image_url:
            raw_image_url = _normalize_to_data_url(raw_image_url)

        if raw_image_url:
            model = ctx.image_description_model
            sys_time = datetime.now(tz=UTC).isoformat()
            system_prompt = ctx.system_prompt.format(system_time=sys_time)
            content = [
                {"type": "image_url", "image_url": {"url": raw_image_url}},
                {"type": "text", "text": PROMPT4DESCRIPTION},
            ]
            msg_list = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=content),
            ]
            try:
                resp = await model.ainvoke(msg_list)
                parsed = _parse_json_from_text(_get_text(resp.content))
                captions = parsed.get("captions", [])
            except Exception:
                captions = []

        if not captions:
            captions = [f"Visual inspection of {class_name}."] * 3

    captions = [str(c) for c in captions[:3]]
    while len(captions) < 3:
        captions.append(f"General inspection of {class_name}.")

    # 2. Load or generate atomic candidates
    anomaly_candidates = []
    normal_candidates = []
    candidate_version = "unknown"

    # Check cache in context
    if not hasattr(ctx, '_atomic_candidate_cache'):
        ctx._atomic_candidate_cache = {}

    prompt_hash = hashlib.md5(PROMPT4CANDIDATE_GENERATION.encode()).hexdigest()[:8]
    try:
        model_fingerprint = getattr(ctx.atomic_candidate_llm_model, 'model_name', None) or type(ctx.atomic_candidate_llm_model).__name__
    except Exception:
        model_fingerprint = "unknown"
    model_hash = hashlib.md5(str(model_fingerprint).encode()).hexdigest()[:8]
    cache_key = f"{class_name.strip().lower()}_{prompt_hash}_{model_hash}"

    if cache_key in ctx._atomic_candidate_cache:
        anomaly_candidates, normal_candidates, candidate_version = ctx._atomic_candidate_cache[cache_key]
    else:
        # Check atomic_candidates_file first
        candidate_file = getattr(ctx, "atomic_candidates_file", None)
        if candidate_file and os.path.exists(candidate_file):
            try:
                with open(candidate_file, "r") as f:
                    all_candidates = json.load(f)
                if class_name in all_candidates:
                    entry = all_candidates[class_name]
                    anomaly_candidates = entry.get("anomaly_candidates", [])
                    normal_candidates = entry.get("normal_candidates", [])
                    candidate_version = "file"
            except Exception:
                pass

        # Generate if still not found
        if not anomaly_candidates or not normal_candidates:
            try:
                llm_model = ctx.atomic_candidate_llm_model
                prompt = f"{PROMPT4CANDIDATE_GENERATION}\n\nClass name: {class_name}"
                msg_list = [{"role": "user", "content": prompt}]
                resp = await llm_model.ainvoke(msg_list)
                parsed = _parse_json_from_text(_get_text(resp.content))
                anomaly_candidates = parsed.get("anomaly_candidates", [])
                normal_candidates = parsed.get("normal_candidates", [])
                if not isinstance(anomaly_candidates, list):
                    anomaly_candidates = []
                if not isinstance(normal_candidates, list):
                    normal_candidates = []
                # Filter and version
                anomaly_candidates = [c for c in anomaly_candidates if c and isinstance(c, str) and c.strip()]
                normal_candidates = [c for c in normal_candidates if c and isinstance(c, str) and c.strip()]
                content_for_hash = json.dumps([anomaly_candidates, normal_candidates], sort_keys=True)
                candidate_version = hashlib.md5(content_for_hash.encode()).hexdigest()[:8]
                # Cache it
                if hasattr(ctx, '_atomic_candidate_cache'):
                    ctx._atomic_candidate_cache[cache_key] = (anomaly_candidates, normal_candidates, candidate_version)
            except Exception as e:
                return f"[counterfactual_atomic] Candidate generation failed: {e}"

    if not anomaly_candidates or not normal_candidates:
        return f"[counterfactual_atomic] No candidates available for class '{class_name}'."

    # 3. Compute embeddings for captions and candidates
    embedding_model = ctx.embedding_model
    if embedding_model is None:
        return "[counterfactual_atomic] No embedding model available."

    try:
        has_async = hasattr(embedding_model, "aembed_documents") and callable(embedding_model.aembed_documents)
        if has_async:
            cap_embs = await embedding_model.aembed_documents(captions)
            anomaly_embs = await embedding_model.aembed_documents(anomaly_candidates)
            normal_embs = await embedding_model.aembed_documents(normal_candidates)
        else:
            cap_embs = embedding_model.embed_documents(captions)
            anomaly_embs = embedding_model.embed_documents(anomaly_candidates)
            normal_embs = embedding_model.embed_documents(normal_candidates)
    except Exception as e:
        return f"[counterfactual_atomic] Embedding computation failed: {e}"

    cap_embs = [np.array(e) for e in cap_embs]
    anomaly_embs = [np.array(e) for e in anomaly_embs]
    normal_embs = [np.array(e) for e in normal_embs]

    # 4. For each caption, find top-K matches
    tau_cand = getattr(ctx, "tau_cand", 0.1)
    all_caption_results = []
    overall_scores = []

    for ci, cap_emb in enumerate(cap_embs):
        # Similarities to anomaly candidates
        anom_sims = [_cosine_similarity(cap_emb, ae) for ae in anomaly_embs]
        # Similarities to normal candidates
        norm_sims = [_cosine_similarity(cap_emb, ne) for ne in normal_embs]

        # Top-K anomaly matches
        anom_top_k = sorted(range(len(anom_sims)), key=lambda i: anom_sims[i], reverse=True)[:k]
        # Top-K normal matches
        norm_top_k = sorted(range(len(norm_sims)), key=lambda i: norm_sims[i], reverse=True)[:k]

        # Margin = max sim to anomaly - max sim to normal
        max_anom_sim = max(anom_sims) if anom_sims else 0.0
        max_norm_sim = max(norm_sims) if norm_sims else 0.0
        margin = max_anom_sim - max_norm_sim

        # Evidence strength (sigmoid of margin)
        evidence_strength = 1.0 / (1.0 + np.exp(-margin / max(tau_cand, 0.01)))

        top_anom = [{"candidate": anomaly_candidates[i], "similarity": float(anom_sims[i])} for i in anom_top_k]
        top_norm = [{"candidate": normal_candidates[i], "similarity": float(norm_sims[i])} for i in norm_top_k]

        all_caption_results.append({
            "caption": captions[ci],
            "top_anomaly": top_anom,
            "top_normal": top_norm,
            "max_anom_sim": float(max_anom_sim),
            "max_norm_sim": float(max_norm_sim),
            "margin": float(margin),
            "evidence_strength": float(evidence_strength),
        })
        overall_scores.append(float(evidence_strength))

    # 5. Format report
    valid_results = [r for r in all_caption_results if r["evidence_strength"] > 0.0]
    num_valid = len(valid_results)
    num_total = len(all_caption_results)
    avg_margin = float(np.mean([r["margin"] for r in all_caption_results])) if all_caption_results else 0.0
    avg_evidence = float(np.mean([r["evidence_strength"] for r in all_caption_results])) if all_caption_results else 0.5
    avg_max_sim_anom = float(np.mean([r["max_anom_sim"] for r in all_caption_results])) if all_caption_results else 0.0
    avg_max_sim_norm = float(np.mean([r["max_norm_sim"] for r in all_caption_results])) if all_caption_results else 0.0

    # Build counterfactual_stats and attach to state for printing
    stats = {
        "num_valid": num_valid,
        "num_total": num_total,
        "avg_margin": avg_margin,
        "avg_evidence_strength": avg_evidence,
        "avg_max_sim_anom": avg_max_sim_anom,
        "avg_max_sim_norm": avg_max_sim_norm,
    }
    # Attach to state for logging in template_tools
    if hasattr(state, "__dict__"):
        state.counterfactual_stats = stats

    lines = [
        f"[counterfactual_atomic] Counterfactual template analysis for class=\"{class_name}\" (v={candidate_version}):",
        f"",
        f"Candidate counts: {len(anomaly_candidates)} anomaly, {len(normal_candidates)} normal",
        f"",
    ]
    for ci, r in enumerate(all_caption_results):
        lines.append(f"--- Perspective {ci + 1}: {_escape_caption(r['caption'])} ---")
        lines.append(f"  Top anomaly candidates:")
        for t in r["top_anomaly"]:
            lines.append(f"    - {_escape_caption(t['candidate'])} (sim={t['similarity']:.3f})")
        lines.append(f"  Top normal candidates:")
        for t in r["top_normal"]:
            lines.append(f"    - {_escape_caption(t['candidate'])} (sim={t['similarity']:.3f})")
        lines.append(f"  Max anomaly sim: {r['max_anom_sim']:.3f}, Max normal sim: {r['max_norm_sim']:.3f}")
        lines.append(f"  Margin: {r['margin']:+.3f}, Evidence strength: {r['evidence_strength']:.3f}")
        lines.append(f"")

    lines.append(f"--- Summary ---")
    lines.append(f"Valid perspectives: {num_valid}/{num_total}")
    lines.append(f"Average margin: {avg_margin:+.3f}")
    lines.append(f"Average evidence strength: {avg_evidence:.3f}")
    lines.append(f"Average max anomaly sim: {avg_max_sim_anom:.3f}")
    lines.append(f"Average max normal sim: {avg_max_sim_norm:.3f}")

    # Store captions in state for later use
    if hasattr(state, "__dict__") and not getattr(state, "image_captions", None):
        state.image_captions = captions

    return "\n".join(lines)
