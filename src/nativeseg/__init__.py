"""NativeSeg: Text Prompted 3D Computed Tomography Segmentation via Spatial Visual Tokens and Verified Local Zoom."""

from .agent import Request, apply_zoom, propose_free_roi, run_agent
from .data import ORGANS, PROMPTS, BalancedRequests, SliceBank, pack
from .model import (
    ACTIONS,
    CANVAS,
    CELLS,
    FULL,
    GRID,
    TOKENS,
    AgentSeg,
    HighRes25DBranch,
    MaskDecoder,
    NativeSeg,
    attach_lora,
    letterbox_params,
    overlay,
    render,
    to_native,
    to_native_feathered,
)

__version__ = "1.0.0"

__all__ = [
    "ACTIONS",
    "CANVAS",
    "CELLS",
    "FULL",
    "GRID",
    "ORGANS",
    "PROMPTS",
    "TOKENS",
    "AgentSeg",
    "BalancedRequests",
    "HighRes25DBranch",
    "MaskDecoder",
    "NativeSeg",
    "Request",
    "SliceBank",
    "apply_zoom",
    "attach_lora",
    "letterbox_params",
    "overlay",
    "pack",
    "propose_free_roi",
    "render",
    "run_agent",
    "to_native",
    "to_native_feathered",
]
