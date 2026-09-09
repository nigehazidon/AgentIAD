"""Tool definitions for v1.27 agent (re-exports from tools.py).

v1.27 adds counterfactual template support for MVTec LOCO logical anomaly detection,
including configurable strictness levels (soft/hard), matching rules (textual_matching
/visual_check), and hard rule violation thresholds. These are handled by the agent
graph state and context; the tool functions remain the same.
"""

from .tools import (
    TOOLS,
    _extract_image_from_messages,
    _encode_image_to_base64,
    _decode_base64_image,
    _normalize_to_data_url,
    to_pure_base64,
    _pil_to_cv2,
    _cv2_to_pil,
    IMAGE_PROCESSING_AVAILABLE,
    PromptOrder,
    _keyword_heuristic_decision,
    counterfactual_atomic_candidate_tool,
)

__all__ = [
    "TOOLS",
    "_extract_image_from_messages",
    "_encode_image_to_base64",
    "_decode_base64_image",
    "_normalize_to_data_url",
    "to_pure_base64",
    "_pil_to_cv2",
    "_cv2_to_pil",
    "IMAGE_PROCESSING_AVAILABLE",
    "PromptOrder",
    "_keyword_heuristic_decision",
    "counterfactual_atomic_candidate_tool",
]
