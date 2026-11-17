"""Direct end-to-end 3D NIfTI (.nii.gz) inference supporting Single-Checkpoint Stage B, Benchmark Standalone, and Plug-and-Play Refiner modes."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from .agent import Request, run_agent
from .data import ORGANS, PROMPTS
from .geometry import load_native_image, load_native_mask
from .model import (
    FULL,
    build_model,
    letterbox_params,
    load_checkpoint,
    render,
    resolve_checkpoint_path,
    to_native,
)
from .predict import apply_slice_continuity


def _slice_stack(norm_slices, z, deltas=(-2, -1, 0, 1, 2)):
    s_len = len(norm_slices)
    return np.stack(
        [norm_slices[min(max(z + dz, 0), s_len - 1)] for dz in deltas], axis=0
    )


def _gc_cuda(device):
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def _exec_round0_checkpoint(model_dir, checkpoint, norm_slices, organ_list, device, quantize_fp16=True):
    """Load a Round-0 checkpoint inside an isolated scope, run all slices, close hook, and return arrays."""
    _, cfg = resolve_checkpoint_path(checkpoint)
    model = build_model(
        model_dir,
        last_layers=cfg.get("last_layers", 16),
        rank=cfg.get("rank", 16),
        device=device,
        readout=cfg.get("readout", "fused"),
        decoder_type=cfg.get("decoder", "five_slice"),
    )
    try:
        load_checkpoint(model, checkpoint).eval()
        s_len, h, w = norm_slices.shape
        params = letterbox_params(h, w)
        use_stack = model.decoder_type == "five_slice"
        amp = torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        )
        masks = np.zeros((s_len, len(organ_list), h, w), bool)
        probs = np.zeros((s_len, len(organ_list)), np.float32)
        texts = [PROMPTS[org] for org in organ_list]
        k_o = len(organ_list)
        rep = lambda t: t.repeat_interleave(k_o, 0)
        for z0 in range(0, s_len, 4):
            zs = list(range(z0, min(z0 + 4, s_len)))
            gray = torch.stack(
                [
                    render(
                        torch.from_numpy(
                            _slice_stack(norm_slices, z)
                            if use_stack
                            else norm_slices[z][None]
                        ).to(device),
                        params,
                        FULL,
                    )
                    for z in zs
                ]
            )
            pre, merged = model.encode(gray)
            with amp:
                lg, pr = model.initial(
                    texts * len(zs), rep(gray), encoded=(rep(pre), rep(merged))
                )
            pr_np = pr.float().sigmoid().cpu().numpy()
            if quantize_fp16:
                pr_np = pr_np.astype(np.float16).astype(np.float32)
            for j, z in enumerate(zs):
                for o_i in range(k_o):
                    nat, _ = to_native(lg[j * k_o + o_i].float(), params, FULL)
                    masks[z, o_i] = (nat > 0).cpu().numpy()
                    probs[z, o_i] = pr_np[j * k_o + o_i]
        return masks, probs
    finally:
        model.close()


@torch.no_grad()
def _exec_agent_checkpoint(
    model_dir,
    checkpoint,
    norm_slices,
    organ_list,
    device,
    batch_size,
    tau,
    max_steps,
    state_residual,
    init_masks_zyx=None,
    free_roi=False,
    logit_memory=False,
    quantize_fp16=False,
):
    """Load an agent checkpoint inside an isolated scope, run multiround zoom, close hook, and return arrays."""
    _, cfg = resolve_checkpoint_path(checkpoint)
    decoder_type = cfg.get("decoder", "five_slice")
    model = build_model(
        model_dir,
        last_layers=cfg.get("last_layers", 16),
        rank=cfg.get("rank", 16),
        device=device,
        readout=cfg.get("readout", "fused"),
        state_residual=state_residual,
        decoder_type=decoder_type,
    )
    try:
        load_checkpoint(model, checkpoint).eval()
        use_stack = decoder_type == "five_slice"
        s_len, h, w = norm_slices.shape
        raw_init = np.zeros((s_len, len(organ_list), h, w), bool)
        presence = np.zeros((s_len, len(organ_list)), np.float32)
        final = np.zeros((s_len, len(organ_list), h, w), bool)
        zoom_steps = np.zeros((s_len, len(organ_list)), np.int16)
        jobs = [(z, o_i) for z in range(s_len) for o_i in range(len(organ_list))]
        for start in range(0, len(jobs), batch_size):
            chunk = jobs[start : start + batch_size]
            reqs = [
                Request(
                    _slice_stack(norm_slices, z) if use_stack else norm_slices[z],
                    PROMPTS[organ_list[o_i]],
                    device,
                )
                for z, o_i in chunk
            ]
            init_m = (
                [init_masks_zyx[z, o_i] for z, o_i in chunk]
                if init_masks_zyx is not None
                else None
            )
            run_agent(
                model,
                reqs,
                policy="learned",
                max_steps=max_steps,
                tau=tau,
                initial_masks=init_m,
                verify=True,
                absent_exit=0.1,
                free_roi=free_roi,
                logit_memory=logit_memory,
            )
            for (z, o_i), r in zip(chunk, reqs):
                raw_init[z, o_i] = r.initial_raw.cpu().numpy()
                p_val = float(np.float16(r.presence)) if quantize_fp16 else float(r.presence)
                presence[z, o_i] = p_val
                final[z, o_i] = r.mask.cpu().numpy()
                zoom_steps[z, o_i] = len(r.visited)
        return raw_init, presence, final, zoom_steps
    finally:
        model.close()


@torch.no_grad()
def infer_volume(
    ct_path,
    output_dir,
    model_dir="Qwen/Qwen3.8-27B",
    mode="stage-b",
    checkpoint=None,
    round0_checkpoint=None,
    init2d_checkpoint=None,
    zoom_checkpoint=None,
    consensus_checkpoint=None,
    allow_no_consensus=False,
    organs=ORGANS,
    specialist_masks=None,
    device=None,
    batch_size=16,
    tau=None,
    max_steps=None,
):
    if specialist_masks is not None:
        mode = "refiner"
    if mode == "refiner" and not specialist_masks:
        raise ValueError("--specialist-mask is required when --mode refiner is selected")
    if mode == "refiner" and consensus_checkpoint is None and not allow_no_consensus:
        raise ValueError(
            "--mode refiner requires --consensus-checkpoint (e.g. checkpoints/nativeseg_stage_a) to match the paper's "
            "cross-model consensus protocol (Tables 1b, 2b, 4b), or pass --no-consensus explicitly for raw zoom edits."
        )
    if mode == "benchmark-standalone" and not (
        round0_checkpoint and init2d_checkpoint and zoom_checkpoint
    ):
        raise ValueError(
            "--mode benchmark-standalone requires --round0-checkpoint, --init2d-checkpoint, and --zoom-checkpoint"
        )
    if mode in ("stage-b", "refiner") and checkpoint is None:
        raise ValueError(f"--checkpoint is required for --mode {mode}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vol = load_native_image(ct_path)
    ct_zyx = np.clip(np.round(vol.data), -160.0, 240.0).transpose(2, 1, 0)
    norm_slices = ((ct_zyx + 160.0) / 400.0).astype(np.float32)
    organ_list = [organs] if isinstance(organs, str) else list(organs)

    if mode == "benchmark-standalone":
        tau = 0.05 if tau is None else tau
        max_steps = 3 if max_steps is None else max_steps

        raw_r0, prob_r0 = _exec_round0_checkpoint(
            model_dir, round0_checkpoint, norm_slices, organ_list, device, quantize_fp16=True
        )
        _gc_cuda(device)

        raw_i2, prob_i2 = _exec_round0_checkpoint(
            model_dir, init2d_checkpoint, norm_slices, organ_list, device, quantize_fp16=True
        )
        init_2d = raw_i2 & (prob_i2 >= 0.5)[..., None, None]
        _gc_cuda(device)

        _, prob_zm, ag_zm, zoom_steps = _exec_agent_checkpoint(
            model_dir,
            zoom_checkpoint,
            norm_slices,
            organ_list,
            device,
            batch_size=batch_size,
            tau=tau,
            max_steps=max_steps,
            state_residual=1.0,
            init_masks_zyx=init_2d,
            free_roi=False,
            logit_memory=False,
            quantize_fp16=True,
        )
        _gc_cuda(device)

        prob_ens = 0.65 * prob_r0 + 0.35 * prob_zm
        base_m = (raw_r0 & (prob_r0 >= 0.5)[..., None, None]) | (ag_zm & raw_r0)
        delivered = apply_slice_continuity(
            base_m, raw_r0 | ag_zm, prob_ens, p_cont=0.25, p_iso=0.62, cc_3d=True
        )
    elif mode == "stage-b":
        tau = 0.05 if tau is None else tau
        max_steps = 3 if max_steps is None else max_steps
        raw_init, presence, final, zoom_steps = _exec_agent_checkpoint(
            model_dir,
            checkpoint,
            norm_slices,
            organ_list,
            device,
            batch_size=batch_size,
            tau=tau,
            max_steps=max_steps,
            state_residual=1.0,
            init_masks_zyx=None,
            free_roi=True,
            logit_memory=True,
            quantize_fp16=False,   # Matches predict.py online float32 continuity
        )
        _gc_cuda(device)
        delivered = apply_slice_continuity(final, raw_init, presence)
    else:
        # mode == "refiner"
        tau = 0.10 if tau is None else tau
        max_steps = 2 if max_steps is None else max_steps
        m_sp = np.stack(
            [
                load_native_mask(specialist_masks[org], vol).transpose(2, 1, 0)
                for org in organ_list
            ],
            axis=1,
        )
        _, _, final, zoom_steps = _exec_agent_checkpoint(
            model_dir,
            checkpoint,
            norm_slices,
            organ_list,
            device,
            batch_size=batch_size,
            tau=tau,
            max_steps=max_steps,
            state_residual=1.5,
            init_masks_zyx=m_sp,
            free_roi=False,
            logit_memory=False,
            quantize_fp16=True,
        )
        _gc_cuda(device)

        if consensus_checkpoint is not None:
            m_r0, p_r0 = _exec_round0_checkpoint(
                model_dir, consensus_checkpoint, norm_slices, organ_list, device, quantize_fp16=True
            )
            _gc_cuda(device)
            diff = final != m_sp
            agree = (m_r0 == final) & ((p_r0[..., None, None] >= 0.5) | (~final))
            delivered = np.where(diff & agree, final, m_sp)
            has_sp = m_sp.any(axis=(-2, -1))
            prev_sp = np.pad(has_sp[:-1], ((1, 0), (0, 0)), constant_values=False)
            next_sp = np.pad(has_sp[1:], ((0, 1), (0, 0)), constant_values=False)
            iso_fp = (
                has_sp
                & (~prev_sp)
                & (~next_sp)
                & (p_r0 < 0.03)
                & (~m_r0.any(axis=(-2, -1)))
            )
            delivered = delivered & (~iso_fp[..., None, None])
        else:
            delivered = final

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for o_i, org in enumerate(organ_list):
        mask_ras = delivered[:, o_i].transpose(2, 1, 0)
        out_file = output_dir / f"{org}.nii.gz"
        vol.save_prediction(mask_ras, out_file)
        summary[org] = {
            "output": str(out_file),
            "mode": mode,
            "voxels": int(mask_ras.sum()),
            "mean_zoom_rounds_per_slice": round(float(zoom_steps[:, o_i].mean()), 4),
        }
    (output_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run NativeSeg directly on a 3D NIfTI CT scan.")
    parser.add_argument("--ct", required=True)
    parser.add_argument(
        "--mode",
        default="stage-b",
        choices=("stage-b", "benchmark-standalone", "refiner"),
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--round0-checkpoint", default=None)
    parser.add_argument("--init2d-checkpoint", default=None)
    parser.add_argument("--zoom-checkpoint", default=None)
    parser.add_argument("--consensus-checkpoint", default=None)
    parser.add_argument(
        "--no-consensus",
        action="store_true",
        help="Run raw zoom edits in --mode refiner without Stage A cross-model consensus",
    )
    parser.add_argument("--model-dir", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--organ", default="all", choices=["all", *ORGANS])
    parser.add_argument("--specialist-mask", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    organs = list(ORGANS) if args.organ == "all" else [args.organ]
    spec_map = None
    if args.specialist_mask:
        sp = Path(args.specialist_mask)
        if sp.is_dir():
            spec_map = {org: sp / f"{org}.nii.gz" for org in organs}
        else:
            if len(organs) != 1:
                raise ValueError("Pass --organ <name> when --specialist-mask is a single NIfTI file")
            spec_map = {organs[0]: sp}

    res = infer_volume(
        ct_path=args.ct,
        output_dir=args.output,
        model_dir=args.model_dir,
        mode=args.mode,
        checkpoint=args.checkpoint,
        round0_checkpoint=args.round0_checkpoint,
        init2d_checkpoint=args.init2d_checkpoint,
        zoom_checkpoint=args.zoom_checkpoint,
        consensus_checkpoint=args.consensus_checkpoint,
        allow_no_consensus=args.no_consensus,
        organs=organs,
        specialist_masks=spec_map,
        batch_size=args.batch_size,
        tau=args.tau,
        max_steps=args.max_steps,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
