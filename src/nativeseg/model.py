"""NativeSeg model: Round-0 5-slice spatial-token readout, multiscale error gate, and verified zoom.

All step types share one frozen-vision, LoRA-adapted Qwen3.8-27B language model:
  initial: [text][global view*][<pres>]             -> full-view logits, target presence
  decide:  [text][global overlay][<err>]            -> predicted error areas across whole view + 20 cells
  zoom:    [text][global overlay][zoomed overlay*]  -> revised local logits inside shifted ROI
(* = 2D spatial visual tokens that also receive learnable spatial query offsets at the first LoRA layer)
"""

from functools import lru_cache
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

from .vision import MERGE, PATCH, TEMPORAL, patchify

IM_START, IM_END = 248045, 248046
VISION_START, VISION_END, IMAGE_PAD, PAD = 248053, 248054, 248056, 248044
GRID, TOKENS = 32, 256          # 512x512 canvas: 32x32 patches, 16x16 merged spatial tokens
CANVAS = 512
# (top, left, size) of each candidate zoom cell in canvas pixels; index 0 is whole canvas.
CELLS = (
    [(r * 256, c * 256, 256) for r in range(2) for c in range(2)]
    + [(r * 128, c * 128, 128) for r in range(4) for c in range(4)]
)
ACTIONS = 1 + len(CELLS)
FULL = (0, 0, CANVAS)


# ----------------------------------------------------------------------------- geometry

def letterbox_params(h, w, size=CANVAS):
    nh, nw = max(1, round(h * size / max(h, w))), max(1, round(w * size / max(h, w)))
    return {
        "hw": (h, w),
        "resized": (nh, nw),
        "top_left": ((size - nh) // 2, (size - nw) // 2),
    }


@lru_cache(maxsize=256)
def _view_grid_cached(hw, resized, top_left, cell, out):
    (h, w), (nh, nw), (top, left) = hw, resized, top_left
    v0, u0, s = cell
    centers = (torch.arange(out, dtype=torch.float32) + 0.5) * s / out
    y = (v0 + centers - top) * h / nh
    x = (u0 + centers - left) * w / nw
    gy, gx = torch.meshgrid(2 * y / h - 1, 2 * x / w - 1, indexing="ij")
    grid = torch.stack((gx, gy), -1)
    valid = (grid.abs() <= 1).all(-1)
    return grid, valid


def view_grid(params, cell, out):
    """Sampling grid mapping out x out view pixels to normalized native coordinates."""
    return _view_grid_cached(
        params["hw"], params["resized"], params["top_left"], cell, out
    )[0]


def render(native, params, cell, out=CANVAS, mode="bilinear"):
    """native (C,H,W) -> (C,out,out) view; edge-clamped inside, 0 outside the image."""
    grid, valid = _view_grid_cached(
        params["hw"], params["resized"], params["top_left"], cell, out
    )
    grid = grid.to(native.device)
    values = F.grid_sample(
        native[None].float(),
        grid[None],
        mode=mode,
        padding_mode="border",
        align_corners=False,
    )[0]
    return values * valid.to(native.device)


@lru_cache(maxsize=256)
def _native_grid_cached(hw, resized, top_left, cell):
    (h, w), (nh, nw), (top, left) = hw, resized, top_left
    v0, u0, s = cell
    v = top + (torch.arange(h, dtype=torch.float32) + 0.5) * nh / h
    u = left + (torch.arange(w, dtype=torch.float32) + 0.5) * nw / w
    iy, ix = (v - v0) / s, (u - u0) / s
    inside = ((iy >= 0) & (iy < 1))[:, None] & ((ix >= 0) & (ix < 1))[None, :]
    gy, gx = torch.meshgrid(2 * iy - 1, 2 * ix - 1, indexing="ij")
    return torch.stack((gx, gy), -1), inside


def native_grid(params, cell):
    """Grid mapping native pixels to normalized view coordinates, plus inside-cell mask."""
    return _native_grid_cached(
        params["hw"], params["resized"], params["top_left"], cell
    )


def to_native(view_logits, params, cell):
    """view logits (1,R,R) -> native logits (H,W) and native pixels covered by cell."""
    grid, inside = native_grid(params, cell)
    values = F.grid_sample(
        view_logits[None].float(),
        grid[None].to(view_logits.device),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )[0, 0]
    return values, inside.to(view_logits.device)


def to_native_feathered(view_logits, params, cell, taper=0.12):
    """Map view logits (1,R,R) of any continuous ROI cell=(v0, u0, s) to native (H,W)
    with a separable raised-cosine feathering window in [0, 1] to avoid crop-border seams."""
    grid, inside = native_grid(params, cell)
    device = view_logits.device
    grid_d = grid.to(device)
    values = F.grid_sample(
        view_logits[None].float(),
        grid_d[None],
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )[0, 0]
    ux = (grid_d[..., 0] + 1.0) * 0.5
    uy = (grid_d[..., 1] + 1.0) * 0.5
    if taper > 0:
        wx = torch.clamp(torch.minimum(ux, 1.0 - ux) / taper, 0.0, 1.0)
        wy = torch.clamp(torch.minimum(uy, 1.0 - uy) / taper, 0.0, 1.0)
        wx = 0.5 * (1.0 - torch.cos(torch.pi * wx))
        wy = 0.5 * (1.0 - torch.cos(torch.pi * wy))
        weight = (wx * wy) * inside.to(device).float()
    else:
        weight = inside.to(device).float()
    return values, inside.to(device), weight


def overlay(gray, mask):
    """Red translucent fill (alpha=0.35) plus solid contour of mask over a grayscale view; (3,S,S)."""
    m = (mask > 0.5).float()
    edge = (m - (-F.max_pool2d(-m[None], 3, 1, 1)[0])).clamp(0, 1)
    red = torch.maximum(gray * (1 - 0.35 * m) + 0.35 * m, edge)
    other = gray * (1 - 0.35 * m) * (1 - edge)
    return torch.cat((red, other, other), 0)


def patchify_rgb(canvas):
    """(B,3,S,S) in [0,1] -> official Qwen-VL patch order."""
    b, _, s1, s2 = canvas.shape
    x = (canvas - 0.5) / 0.5
    gh, gw = s1 // PATCH, s2 // PATCH
    x = x.reshape(b, 3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    x = x.permute(0, 2, 5, 3, 6, 1, 4, 7)
    x = x.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, TEMPORAL, -1, -1)
    return x.reshape(b * gh * gw, 3 * TEMPORAL * PATCH * PATCH)


# ----------------------------------------------------------------------------- modules

def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(8, cout),
        nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(8, cout),
        nn.GELU(),
    )


class ResFiLMBlock(nn.Module):
    """Residual conv block with VLM language-context FiLM modulation."""

    def __init__(self, cin, cout, hidden=None):
        super().__init__()
        self.proj = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, cout)
        self.act = nn.GELU()
        self.film = nn.Linear(hidden, 2 * cout) if hidden is not None else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, x, ctx=None):
        res = self.proj(x)
        h = self.act(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        if self.film is not None and ctx is not None:
            gb = self.film(ctx).unsqueeze(-1).unsqueeze(-1)
            gamma, beta = gb.chunk(2, dim=1)
            h = h * (1.0 + gamma) + beta
        return self.act(h + res)


class HighRes25DBranch(nn.Module):
    """Five-slice 512x512 residual U-Net decoder branch with narrow soft-tissue window [-60, 140] HU,
    FiLM-modulated by the VLM hidden state and fused with multiscale spatial visual tokens."""

    def __init__(self, hidden, width=256):
        super().__init__()
        self.ctx_norm = nn.LayerNorm(hidden)
        # Input channels: 5 axial slices [z-2..z+2] + 1 narrow window [-60,140] HU + 1 edge mag + 1 state = 8
        self.enc512 = ResFiLMBlock(8, 48, hidden)
        self.down256 = nn.Conv2d(48, 64, 3, stride=2, padding=1)
        self.enc256 = ResFiLMBlock(64, 64, hidden)
        self.down128 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.enc128 = ResFiLMBlock(128, 128, hidden)
        self.down64 = nn.Conv2d(128, 192, 3, stride=2, padding=1)
        self.enc64 = ResFiLMBlock(192, 192, hidden)
        self.down32 = nn.Conv2d(192, width, 3, stride=2, padding=1)
        self.enc32 = ResFiLMBlock(width, width, hidden)

        self.fuse32_unet = ResFiLMBlock(width + width, width, hidden)
        self.dec64 = ResFiLMBlock(width + 192 + 128, 192, hidden)
        self.dec128 = ResFiLMBlock(192 + 128 + 64, 128, hidden)
        self.dec256 = ResFiLMBlock(128 + 64 + 32, 64, hidden)
        self.dec512 = ResFiLMBlock(64 + 48, 48, hidden)
        self.out512 = nn.Conv2d(48, 1, 1)
        nn.init.zeros_(self.out512.weight)
        nn.init.zeros_(self.out512.bias)

    def forward(self, gray, state, ctx_vec, x32, x64, x128, x256):
        if gray.shape[1] == 1:
            stack5 = gray.repeat(1, 5, 1, 1)
            center = gray
        elif gray.shape[1] == 3:
            stack5 = torch.cat((gray[:, :1], gray, gray[:, -1:]), 1)
            center = gray[:, 1:2]
        else:
            stack5 = gray[:, :5]
            center = gray[:, 2:3]
        narrow = ((center - 0.25) / 0.50).clamp(0.0, 1.0)
        gx = F.pad(center[:, :, :, 1:] - center[:, :, :, :-1], (0, 1, 0, 0))
        gy = F.pad(center[:, :, 1:, :] - center[:, :, :-1, :], (0, 0, 0, 1))
        edge_mag = (gx.abs() + gy.abs()).clamp(0.0, 1.0)
        inp = torch.cat((stack5, narrow, edge_mag, state), 1).float()
        ctx = self.ctx_norm(ctx_vec.float()) if ctx_vec is not None else None

        e512 = self.enc512(inp, ctx)
        e256 = self.enc256(self.down256(e512), ctx)
        e128 = self.enc128(self.down128(e256), ctx)
        e64 = self.enc64(self.down64(e128), ctx)
        e32 = self.enc32(self.down32(e64), ctx)

        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
        d32 = self.fuse32_unet(torch.cat((x32, e32), 1), ctx)
        d64 = self.dec64(torch.cat((up(d32), e64, x64), 1), ctx)
        d128 = self.dec128(torch.cat((up(d64), e128, x128), 1), ctx)
        d256 = self.dec256(torch.cat((up(d128), e256, x256), 1), ctx)
        d512 = self.dec512(torch.cat((up(d256), e512), 1), ctx)
        return self.out512(d512)


class MaskDecoder(nn.Module):
    """Spatial visual token hidden states (16x16) + pre-merge vision patch features (32x32)
    + five-slice U-Net branch (decoder_type='five_slice') -> view logits."""

    def __init__(self, hidden, vision=1152, width=256, decoder_type="five_slice"):
        super().__init__()
        self.decoder_type = decoder_type
        self.query = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, width))
        self.features = nn.Sequential(
            nn.GroupNorm(1, vision, affine=False), nn.Conv2d(vision, width, 1)
        )
        self.fuse32 = block(2 * width, width)
        self.raw = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(2, 8, 3, 2, 1), nn.GELU()),     # 256
                nn.Sequential(nn.Conv2d(8, 16, 3, 2, 1), nn.GELU()),    # 128
                nn.Sequential(nn.Conv2d(16, 32, 3, 2, 1), nn.GELU()),   # 64
            ]
        )
        self.fuse64 = block(width + 32, 128)
        self.fuse128 = block(128 + 16, 64)
        self.fuse256 = block(64 + 8, 32)
        self.out = nn.Conv2d(32, 1, 1)
        self.five_slice_unet = (
            HighRes25DBranch(hidden, width) if decoder_type == "five_slice" else None
        )

    def forward(self, hidden, features, gray, state, ctx_vec=None):
        b = hidden.shape[0]
        gray_center = (
            gray
            if gray.shape[1] == 1
            else gray[:, gray.shape[1] // 2 : gray.shape[1] // 2 + 1]
        )
        q = self.query(hidden.float()).transpose(1, 2).reshape(b, -1, 16, 16)
        q = F.interpolate(q, scale_factor=2, mode="bilinear", align_corners=False)
        x32 = self.fuse32(torch.cat((q, self.features(features.float())), 1))
        r256 = self.raw[0](torch.cat((gray_center.float(), state.float()), 1))
        r128 = self.raw[1](r256)
        r64 = self.raw[2](r128)
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
        x64 = self.fuse64(torch.cat((up(x32), r64), 1))
        x128 = self.fuse128(torch.cat((up(x64), r128), 1))
        x256 = self.fuse256(torch.cat((up(x128), r256), 1))
        base256 = self.out(x256)
        if self.five_slice_unet is None:
            return base256
        if ctx_vec is None:
            ctx_vec = hidden.float().mean(dim=1)
        delta512 = self.five_slice_unet(gray, state.float(), ctx_vec, x32, x64, x128, x256)
        return up(base256) + delta512


class NativeSeg(nn.Module):
    """Native VLM multiround segmentation framework built on Qwen3.8-27B."""

    def __init__(
        self,
        qwen,
        tokenizer,
        inject_layer,
        readout="fused",
        state_residual=0.0,
        decoder_type="five_slice",
        freeze_qwen=False,
    ):
        super().__init__()
        self.readout = readout
        self.state_residual = float(state_residual)
        self.decoder_type = decoder_type
        self.freeze_qwen = freeze_qwen
        self.qwen = qwen
        base = qwen.base_model.model if hasattr(qwen, "base_model") else qwen
        self.core = base.model
        self.tokenizer = tokenizer
        hidden = self.core.config.text_config.hidden_size
        self.query_offset = nn.Parameter(torch.zeros(TOKENS, hidden))
        self.query_type = nn.Parameter(torch.randn(2, hidden) * 0.02)
        self.presence_token = nn.Parameter(torch.randn(hidden) * 0.02)
        self.decide_token = nn.Parameter(torch.randn(hidden) * 0.02)
        self.presence = nn.Linear(hidden, 1)
        self.decide = nn.Linear(hidden, ACTIONS)
        self.decoder = MaskDecoder(
            hidden,
            self.core.config.vision_config.hidden_size,
            decoder_type=decoder_type,
        )
        self._prefix = {}
        self._inject = None
        self._hook_handle = self.core.language_model.layers[
            inject_layer
        ].register_forward_pre_hook(self._add_injection, with_kwargs=True)

    def close(self):
        """Remove registered forward pre-hook so the model can be garbage-collected cleanly."""
        if getattr(self, "_hook_handle", None) is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def _add_injection(self, module, args, kwargs):
        if self._inject is None:
            return None
        if args:
            return (args[0] + self._inject.to(args[0].dtype), *args[1:]), kwargs
        kwargs["hidden_states"] = kwargs["hidden_states"] + self._inject.to(
            kwargs["hidden_states"].dtype
        )
        return args, kwargs

    @torch.no_grad()
    def encode(self, canvas):
        """(B,1 or 3 or 5,S,S) -> pre-merge (B,1152,32,32), merged (B,256,H)."""
        visual = self.core.visual
        dtype = next(visual.parameters()).dtype
        b = canvas.shape[0]
        if canvas.shape[1] == 1:
            patches = patchify(canvas)[0]
        elif canvas.shape[1] == 3:
            patches = patchify_rgb(canvas)
        else:
            patches = patchify_rgb(canvas[:, 1:4])
        grid = torch.tensor([[1, GRID, GRID]] * b, device=canvas.device)
        out = visual(patches.to(dtype), grid_thw=grid, return_dict=True)
        pre = out.last_hidden_state.reshape(b, GRID // 2, GRID // 2, 2, 2, -1)
        pre = pre.permute(0, 5, 1, 3, 2, 4).reshape(b, -1, GRID, GRID)
        return pre, out.pooler_output.reshape(b, TOKENS, -1)

    def prompt_ids(self, text, images):
        key = (text, images)
        if key not in self._prefix:
            body = "<|vision_start|><|image_pad|><|vision_end|>" * images
            rendered = (
                f"<|im_start|>user\n{text}{body}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
            ids = self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
            expanded = []
            for i in ids:
                expanded.extend([IMAGE_PAD] * TOKENS if i == IMAGE_PAD else [i])
            if expanded.count(IMAGE_PAD) != TOKENS * images:
                raise ValueError("Unexpected prompt tokenization")
            self._prefix[key] = expanded
        return self._prefix[key]

    def run(self, texts, images, offsets, tail):
        b = len(texts)
        device = images[0].device
        dtype = self.core.language_model.embed_tokens.weight.dtype
        t = 0 if tail is None else tail.shape[1]
        rows = [self.prompt_ids(text, len(images)) + [PAD] * t for text in texts]
        length = max(map(len, rows))
        ids = torch.full((b, length), PAD, dtype=torch.long)
        attention = torch.zeros((b, length), dtype=torch.long)
        for i, row in enumerate(rows):
            ids[i, length - len(row) :] = torch.tensor(row)
            attention[i, length - len(row) :] = 1
        ids, attention = ids.to(device), attention.to(device)
        types = (ids == IMAGE_PAD).int()
        grid = torch.tensor([[1, GRID, GRID]] * (b * len(images)), device=device)
        position, _ = self.core.get_rope_index(
            ids, types, image_grid_thw=grid, attention_mask=attention
        )
        embeds = self.core.language_model.embed_tokens(ids)
        image_positions = (ids == IMAGE_PAD).nonzero()[:, 1].reshape(
            b, len(images), TOKENS
        )
        rows_index = torch.arange(b, device=device)[:, None]
        for k, segment in enumerate(images):
            embeds[rows_index, image_positions[:, k]] = segment.to(dtype)
        inject = torch.zeros(
            b, length, embeds.shape[-1], device=device, dtype=torch.float32
        )
        for k, offset in offsets.items():
            inject = inject.index_put(
                (rows_index.expand(-1, TOKENS), image_positions[:, k]),
                offset.float().expand(b, TOKENS, -1),
                accumulate=True,
            )
        if t:
            inject = torch.cat((inject[:, : length - t], tail.float()), 1)
        self._inject = inject
        try:
            out = self.core.language_model(
                inputs_embeds=embeds,
                attention_mask=attention,
                position_ids=position,
                use_cache=False,
            ).last_hidden_state
        finally:
            self._inject = None
        segments = [out[rows_index, image_positions[:, k]] for k in range(len(images))]
        return segments, (out[:, length - t :] if t else None)

    def initial(self, texts, gray, state=None, encoded=None):
        pre, merged = self.encode(gray) if encoded is None else encoded
        b = gray.shape[0]
        tail = self.presence_token.expand(b, 1, -1)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_qwen):
            (hq,), th = self.run(
                texts, [merged], {0: self.query_offset + self.query_type[0]}, tail
            )
        if state is None:
            state = torch.zeros_like(gray[:, :1])
        if self.readout == "llm":
            pre = torch.zeros_like(pre)
        elif self.readout == "token":
            hq = th[:, :1].expand(-1, hq.shape[1], -1)
        logits = self.decoder(hq, pre, gray, state, ctx_vec=th[:, 0])
        return logits, self.presence(th[:, 0].float())[:, 0]

    def zoom(self, texts, full_overlay, cell_gray, cell_overlay, cell_state):
        _, full_tokens = self.encode(full_overlay)
        _, cell_tokens = self.encode(cell_overlay)
        pre, _ = self.encode(cell_gray)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_qwen):
            (_, hq), _ = self.run(
                texts,
                [full_tokens, cell_tokens],
                {1: self.query_offset + self.query_type[1]},
                None,
            )
        logits = self.decoder(hq, pre, cell_gray, cell_state, ctx_vec=hq.mean(dim=1))
        if self.state_residual:
            if logits.shape[-1] == cell_state.shape[-1]:
                s_res = cell_state.float().to(logits.dtype)
            else:
                s_res = F.avg_pool2d(cell_state.float(), 2).to(logits.dtype)
            logits = logits + self.state_residual * (2 * s_res - 1)
        return logits

    def decide_logits(self, texts, full_overlay):
        _, tokens = self.encode(full_overlay)
        b = full_overlay.shape[0]
        _, th = self.run(texts, [tokens], {}, self.decide_token.expand(b, 1, -1))
        return self.decide(th[:, 0].float())

    def trainable_state(self):
        return {n: p.detach().cpu() for n, p in self.named_parameters() if p.requires_grad}


# Backward-compatible class alias
AgentSeg = NativeSeg


def attach_lora(qwen, last_layers=16, rank=16):
    from peft import LoraConfig, get_peft_model

    qwen.requires_grad_(False)
    depth = qwen.config.text_config.num_hidden_layers
    names = [
        n
        for n, m in qwen.named_modules()
        if ".language_model.layers." in n
        and isinstance(m, nn.Linear)
        and int(n.split(".layers.")[1].split(".")[0]) >= depth - last_layers
    ]
    if not names:
        raise ValueError("No language-model linear layers selected for LoRA")
    wrapped = get_peft_model(
        qwen,
        LoraConfig(
            r=rank,
            lora_alpha=2 * rank,
            lora_dropout=0.0,
            target_modules=names,
            bias="none",
        ),
    )
    base = wrapped.base_model.model
    base.model.visual.eval().requires_grad_(False)
    return wrapped


def resolve_checkpoint_path(checkpoint):
    p = Path(checkpoint)
    if p.is_dir():
        pt = p / "trainable.pt"
        cfg_file = p / "checkpoint_config.json"
        cfg = json.loads(cfg_file.read_text()) if cfg_file.exists() else {}
        return pt, cfg
    cfg_file = p.parent / "checkpoint_config.json"
    cfg = json.loads(cfg_file.read_text()) if cfg_file.exists() else {}
    return p, cfg


def build_model(
    model_dir,
    last_layers=16,
    rank=16,
    device=torch.device("cuda"),
    readout="fused",
    state_residual=0.0,
    decoder_type="five_slice",
    freeze_qwen=False,
):
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    local_only = Path(model_dir).exists()
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    qwen = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(model_dir),
        local_files_only=local_only,
        dtype=dtype,
        device_map={"": device},
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=local_only)
    depth = qwen.config.text_config.num_hidden_layers
    model = NativeSeg(
        attach_lora(qwen, last_layers, rank),
        tokenizer,
        inject_layer=depth - last_layers,
        readout=readout,
        state_residual=state_residual,
        decoder_type=decoder_type,
        freeze_qwen=freeze_qwen,
    )
    for _, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.decoder.to(device)
    for module in (model.presence, model.decide):
        module.to(device)
    for name in ("query_offset", "query_type", "presence_token", "decide_token"):
        getattr(model, name).data = getattr(model, name).data.to(device)
    return model


def load_checkpoint(model, checkpoint):
    pt_path, _ = resolve_checkpoint_path(checkpoint)
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or set(model.trainable_state()) != set(state):
        raise ValueError(
            f"Checkpoint {pt_path} does not match trainable parameters "
            f"(decoder={model.decoder_type}, unexpected={result.unexpected_keys[:3]})"
        )
    return model
