"""CPU verification suite for NativeSeg geometry, 5-slice U-Net forward/backward, multiround loop, LRN†, and checkpoints."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from .agent import Request, perturb, propose_free_roi, run_agent
from .evaluate import validate_npz_alignment
from .lrn_baseline import CNNEditor
from .model import (
    CELLS,
    FULL,
    NativeSeg,
    attach_lora,
    letterbox_params,
    overlay,
    render,
    to_native,
    to_native_feathered,
)
from .predict import apply_slice_continuity


def _tiny_qwen():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    text = dict(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        full_attention_interval=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        vocab_size=248320,
        attn_output_gate=True,
        partial_rotary_factor=0.25,
        rope_parameters={
            "mrope_interleaved": True,
            "mrope_section": [1, 1, 2],
            "partial_rotary_factor": 0.25,
            "rope_theta": 1e7,
            "rope_type": "default",
        },
        mtp_num_hidden_layers=0,
    )
    vision = dict(
        depth=1,
        hidden_size=64,
        intermediate_size=128,
        num_heads=2,
        out_hidden_size=128,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        in_channels=3,
        num_position_embeddings=2304,
        deepstack_visual_indexes=[],
    )
    config = Qwen3_5Config(
        text_config=text,
        vision_config=vision,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
    )
    config.text_config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation = "sdpa"
    return Qwen3_5ForConditionalGeneration(config).float()


def main():
    parser = argparse.ArgumentParser(description="Run NativeSeg CPU smoke tests.")
    parser.add_argument("--model-dir", required=True, help="Local Qwen3.8-27B tokenizer dir or HF ID")
    parser.add_argument("--weights-dir", default=None, help="Optional release_weights dir to verify parameter keys")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    torch.manual_seed(0)
    local_only = Path(args.model_dir).exists()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=local_only)

    # 1. Geometry & feathering tests
    params = letterbox_params(188, 249)
    covered2 = sum(to_native(torch.zeros(1, 64, 64), params, c)[1].int() for c in CELLS[:4])
    covered4 = sum(to_native(torch.zeros(1, 64, 64), params, c)[1].int() for c in CELLS[4:])
    assert bool((covered2 == 1).all()), "2x2 cells must partition native image"
    assert bool((covered4 == 1).all()), "4x4 cells must partition native image"
    vals, inside, weight = to_native_feathered(torch.ones(1, 512, 512), params, (100.0, 120.0, 128.0), taper=0.06)
    assert vals.shape == (188, 249) and bool((weight[~inside] == 0).all()) and float(weight[inside].max()) <= 1.0

    # 2. Five-slice U-Net and single-slice skip decoder forward/backward & frozen lower layers
    for decoder_type in ("five_slice", "single_slice"):
        model = NativeSeg(
            attach_lora(_tiny_qwen(), last_layers=2, rank=4),
            tokenizer,
            inject_layer=2,
            state_residual=1.0,
            decoder_type=decoder_type,
        )
        model.qwen.eval()
        texts = ["Segment the gallbladder.", "Segment the pancreas."]
        gray = torch.rand(2, 5 if decoder_type == "five_slice" else 1, 512, 512)
        state = (torch.rand(2, 1, 512, 512) > 0.7).float()
        over = torch.stack([overlay(gray[i, gray.shape[1] // 2 : gray.shape[1] // 2 + 1], state[i]) for i in range(2)])
        logits, presence = model.initial(texts, gray)
        cell_logits = model.zoom(texts, over, gray, over, state)
        decide = model.decide_logits(texts, over)
        expected_res = 512 if decoder_type == "five_slice" else 256
        assert logits.shape == (2, 1, expected_res, expected_res), f"Unexpected initial shape {logits.shape}"
        assert cell_logits.shape == (2, 1, expected_res, expected_res), f"Unexpected zoom shape {cell_logits.shape}"
        assert decide.shape == (2, 21), f"Unexpected decide shape {decide.shape}"

        loss = logits.mean() + presence.mean() + cell_logits.mean() + decide.mean()
        loss.backward()
        lower = [p for n, p in model.qwen.named_parameters() if ".layers.0." in n or ".layers.1." in n]
        assert all(p.grad is None for p in lower), "Frozen lower layers must have no gradient"

        with torch.no_grad():
            _, merged = model.encode(gray)
            off = {0: model.query_offset + 1}
            segs, tail = model.run(texts, [merged], off, model.decide_token.expand(2, 1, -1))
            for i in range(2):
                s1, t1 = model.run([texts[i]], [merged[i : i + 1]], off, model.decide_token.expand(1, 1, -1))
                assert float((segs[0][i] - s1[0][0]).abs().max()) < 1e-5
                assert float((tail[i] - t1[0]).abs().max()) < 1e-5

        stacks = [np.random.rand(5, 120, 140).astype(np.float32), np.random.rand(5, 150, 100).astype(np.float32)]
        reqs = [
            Request(im if decoder_type == "five_slice" else im[2], t, torch.device("cpu"))
            for im, t in zip(stacks, texts)
        ]
        errs = torch.linspace(0, 500, 21)
        roi = propose_free_roi(reqs[0], errs, 10)
        assert len(roi) == 3 and 0.0 <= roi[0] <= 512.0 - roi[2]
        run_agent(model, reqs, "learned", max_steps=2, tau=0.0, verify=True, free_roi=True, logit_memory=True)
        model.close()
        assert model._hook_handle is None

    # 3. LRN† CNNEditor forward/backward, perturbation, & axial continuity smoke checks
    lrn = CNNEditor()
    lrn_out = lrn(torch.rand(2, 1, 64, 64), torch.rand(2, 1, 64, 64), torch.tensor([0, 3]))
    assert lrn_out.shape == (2, 1, 64, 64)
    lrn_out.mean().backward()

    import random as _random

    rng_p = _random.Random(42)
    empty_t = torch.zeros(64, 64, dtype=torch.bool)
    pos_t = torch.zeros(64, 64, dtype=torch.bool)
    pos_t[20:44, 18:46] = True
    for _ in range(40):
        pe = perturb(empty_t, rng_p)
        pp = perturb(pos_t, rng_p)
        assert pe.shape == (64, 64) and pp.shape == (64, 64)

    base = np.zeros((5, 7, 32, 32), bool)
    raw = np.zeros((5, 7, 32, 32), bool)
    prob = np.full((5, 7), 0.05, np.float32)
    base[1:3, 0, 8:16, 8:16] = True
    raw[1:4, 0, 8:16, 8:16] = True
    prob[1:3, 0] = 0.9
    prob[3, 0] = 0.35
    base[4, 1, 2:5, 2:5] = True
    raw[4, 1, 2:5, 2:5] = True
    prob[4, 1] = 0.52
    cont = apply_slice_continuity(base, raw, prob)
    assert cont[3, 0].any() and not cont[4, 1].any()

    dummy_npz = {
        "shape": np.array([3, 7, 16, 16]),
        "z": np.array([0, 1, 2]),
        "logits_mask": np.packbits(np.zeros((3, 7, 16, 16), bool), axis=-1),
        "presence": np.zeros((3, 7), np.float16),
    }
    assert validate_npz_alignment(dummy_npz, "s_test", [0, 1, 2], (16, 16)) == (3, 7, 16, 16)
    try:
        validate_npz_alignment(dummy_npz, "s_test", [0, 2, 1], (16, 16))
        raise AssertionError("Expected ValueError for mismatched z-indices")
    except ValueError:
        pass

    # 4. Optional checkpoint structure & tensor name verification (using os.walk for FUSE/HDFS compatibility)
    verified_ckpts = []
    if args.weights_dir:
        w_dir = Path(args.weights_dir)
        if not w_dir.is_dir():
            raise FileNotFoundError(f"--weights-dir does not exist: {w_dir}")
        from transformers import AutoConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

        cfg = AutoConfig.from_pretrained(str(args.model_dir), local_files_only=local_only)
        ckpt_configs = []
        for dirpath, _, filenames in os.walk(w_dir):
            if "checkpoint_config.json" in filenames:
                ckpt_configs.append(Path(dirpath) / "checkpoint_config.json")
        ckpt_configs.sort()
        for c_json in ckpt_configs:
            sub_d = c_json.parent
            pt = sub_d / "trainable.pt"
            if not pt.exists():
                raise FileNotFoundError(f"Incomplete checkpoint folder {sub_d} (missing trainable.pt)")
            meta = json.loads(c_json.read_text())
            saved = torch.load(pt, map_location="cpu", weights_only=True)
            saved_keys = {n: tuple(t.shape) for n, t in saved.items()}
            if meta.get("architecture") == "CNNEditor":
                lrn_ref = CNNEditor(widths=tuple(meta.get("widths", [32, 64, 128, 256, 320, 320])))
                expected_keys = {n: tuple(p.shape) for n, p in lrn_ref.state_dict().items()}
                arch_label = "CNNEditor"
            else:
                with torch.device("meta"):
                    q_meta = Qwen3_5ForConditionalGeneration(cfg)
                    m_meta = NativeSeg(
                        attach_lora(q_meta, last_layers=meta["last_layers"], rank=meta["rank"]),
                        tokenizer,
                        inject_layer=cfg.text_config.num_hidden_layers - meta["last_layers"],
                        decoder_type=meta["decoder"],
                    )
                expected_keys = {n: tuple(p.shape) for n, p in m_meta.named_parameters() if p.requires_grad}
                arch_label = meta["decoder"]
                m_meta.close()
            assert expected_keys == saved_keys, f"Key/shape mismatch in {sub_d}"
            has_nonfinite = any(not torch.isfinite(t).all().item() for t in saved.values() if t.is_floating_point())
            assert not has_nonfinite, f"Non-finite weights detected in {sub_d}"
            total_params = sum(t.numel() for t in saved.values())
            verified_ckpts.append(
                {
                    "name": str(sub_d.relative_to(w_dir)),
                    "tensors": len(saved_keys),
                    "trainable_params_m": round(total_params / 1e6, 2),
                    "architecture": arch_label,
                }
            )
        if not verified_ckpts:
            raise ValueError(f"No checkpoints found under --weights-dir {w_dir}")

    print(
        json.dumps(
            {
                "status": "PASSED",
                "geometry_partitions": True,
                "decoders_tested": ["five_slice", "single_slice", "CNNEditor"],
                "verified_checkpoints": verified_ckpts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
