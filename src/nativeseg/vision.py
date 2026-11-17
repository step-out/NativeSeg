"""Batched Qwen3.8 vision encoding of letterboxed CT views, matching the official processor."""

from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

PATCH, MERGE, TEMPORAL = 16, 2, 2


def letterbox_batch(images, size=512):
    """Fit (B,1,H,W) same-size images into (B,1,size,size); returns canvas and mapping."""
    h, w = images.shape[-2:]
    nh, nw = max(1, round(h * size / max(h, w))), max(1, round(w * size / max(h, w)))
    resized = F.interpolate(
        images.float(), (nh, nw), mode="bilinear", align_corners=False, antialias=True
    )
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = F.pad(resized, (left, size - nw - left, top, size - nh - top))
    return canvas, {
        "native_hw": [h, w],
        "resized_hw": [nh, nw],
        "padding_top_left": [top, left],
        "canvas_hw": [size, size],
    }


def patchify(canvas):
    """(B,1,S,S) in [0,1] -> (B*gh*gw, 3*2*16*16) in exact Qwen-VL patch order."""
    b, c, s1, s2 = canvas.shape
    if c != 1 or s1 % (PATCH * MERGE) or s2 % (PATCH * MERGE):
        raise ValueError("Expected single-channel canvases divisible by 32")
    x = (canvas.repeat(1, 3, 1, 1) - 0.5) / 0.5
    gh, gw = s1 // PATCH, s2 // PATCH
    x = x.reshape(b, 3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    x = x.permute(0, 2, 5, 3, 6, 1, 4, 7)
    x = x.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, TEMPORAL, -1, -1)
    return x.reshape(b * gh * gw, 3 * TEMPORAL * PATCH * PATCH), (gh, gw)


class QwenVision(nn.Module):
    """Frozen vision tower; returns pre-merge (B,1152,gh,gw) and merged (B,gh*gw/4,5120) tokens."""

    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    @torch.no_grad()
    def forward(self, canvas):
        dtype = next(self.visual.parameters()).dtype
        patches, (gh, gw) = patchify(canvas)
        b = canvas.shape[0]
        grid = torch.tensor([[1, gh, gw]] * b, device=canvas.device)
        out = self.visual(patches.to(dtype), grid_thw=grid, return_dict=True)
        pre = out.last_hidden_state.reshape(b, gh // MERGE, gw // MERGE, MERGE, MERGE, -1)
        pre = pre.permute(0, 5, 1, 3, 2, 4).reshape(b, -1, gh, gw)
        merged = out.pooler_output.reshape(b, (gh // MERGE) * (gw // MERGE), -1)
        return pre, merged


def load_vision(model_dir, device, dtype=torch.bfloat16):
    """Load only the vision tower weights (shard 1) from a local Qwen3.8-27B folder."""
    from safetensors import safe_open
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    local_only = Path(model_dir).exists()
    config = Qwen3_5Config.from_pretrained(
        str(model_dir), local_files_only=local_only
    ).vision_config
    config._attn_implementation = "sdpa"
    model = Qwen3_5VisionModel(config)
    prefix = "model.visual."
    with safe_open(
        Path(model_dir) / "model-00001-of-00018.safetensors", framework="pt", device="cpu"
    ) as file:
        weights = {
            k.removeprefix(prefix): file.get_tensor(k)
            for k in file.keys()
            if k.startswith(prefix)
        }
    model.load_state_dict(weights, strict=True)
    model.eval().requires_grad_(False)
    frequencies = model.rotary_pos_emb.inv_freq.detach().float().clone()
    model.to(device=device, dtype=dtype)
    model.rotary_pos_emb.inv_freq = frequencies.to(device=device)
    return QwenVision(model)
