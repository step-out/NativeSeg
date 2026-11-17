"""Full-split volumetric prediction (supports single-GPU `nativeseg-predict` or multi-GPU `torchrun`)."""

import argparse
import datetime
import io
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import label
import torch
from torch import distributed as dist

from .agent import Request, run_agent
from .data import ORGANS, PROMPTS, SliceBank
from .model import (
    FULL,
    build_model,
    letterbox_params,
    load_checkpoint,
    render,
    resolve_checkpoint_path,
    to_native,
)


@torch.no_grad()
def predict_patient(model, bank, rows, device, slices_per_batch):
    h, w = bank.slices[rows[0]]["h"], bank.slices[rows[0]]["w"]
    params = letterbox_params(h, w)
    masks = np.zeros((len(rows), len(ORGANS), h, w), bool)
    presence = np.zeros((len(rows), len(ORGANS)), np.float32)
    texts = [PROMPTS[o] for o in ORGANS]
    use_stack = getattr(model, "decoder_type", "five_slice") == "five_slice"
    amp = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    for start in range(0, len(rows), slices_per_batch):
        chunk = rows[start : start + slices_per_batch]
        gray = torch.stack(
            [
                render(
                    torch.from_numpy(
                        bank.image_stack(i) if use_stack else bank.image(i)[None]
                    ).to(device),
                    params,
                    FULL,
                )
                for i in chunk
            ]
        )
        pre, merged = model.encode(gray)
        k = len(ORGANS)
        rep = lambda t: t.repeat_interleave(k, 0)
        with amp:
            logits, prob = model.initial(
                texts * len(chunk), rep(gray), encoded=(rep(pre), rep(merged))
            )
        prob = prob.float().sigmoid().cpu().numpy()
        for j in range(len(chunk)):
            for o in range(k):
                native, _ = to_native(logits[j * k + o].float(), params, FULL)
                masks[start + j, o] = (native > 0).cpu().numpy()
                presence[start + j, o] = prob[j * k + o]
    return masks, presence


def apply_slice_continuity(base_m, raw_m, prob, p_cont=0.25, p_iso=0.62, cc_3d=True):
    """Deterministic two-threshold axial continuity: recover adjacent organ-tip slices with
    p_z >= p_cont that overlap active neighbors, prune isolated 1-slice FPs, and filter tiny 3D islands."""
    out_m = base_m.copy()
    s_len, n_org = out_m.shape[:2]
    for o in range(n_org):
        for z in range(s_len):
            if out_m[z, o].any() and prob[z, o] < p_iso:
                prev_ok = z > 0 and (out_m[z - 1, o].any() or prob[z - 1, o] >= 0.35)
                next_ok = z + 1 < s_len and (
                    out_m[z + 1, o].any() or prob[z + 1, o] >= 0.35
                )
                if not prev_ok and not next_ok:
                    out_m[z, o] = False
        for _ in range(2):
            for z in range(s_len):
                if not out_m[z, o].any() and prob[z, o] >= p_cont and raw_m[z, o].any():
                    cand = raw_m[z, o]
                    ov_prev = (
                        z > 0
                        and out_m[z - 1, o].any()
                        and (cand & out_m[z - 1, o]).sum() > 0
                    )
                    ov_next = (
                        z + 1 < s_len
                        and out_m[z + 1, o].any()
                        and (cand & out_m[z + 1, o]).sum() > 0
                    )
                    if ov_prev or ov_next:
                        out_m[z, o] = cand
        for z in range(1, s_len - 1):
            if (
                not out_m[z, o].any()
                and out_m[z - 1, o].any()
                and out_m[z + 1, o].any()
            ):
                if raw_m[z, o].any() and (
                    (raw_m[z, o] & out_m[z - 1, o]).sum() > 0
                    or (raw_m[z, o] & out_m[z + 1, o]).sum() > 0
                ):
                    out_m[z, o] = raw_m[z, o]
        if cc_3d and out_m[:, o].any():
            labeled, n_comp = label(out_m[:, o])
            if n_comp > 1:
                sizes = np.bincount(labeled.ravel())[1:]
                scores = []
                for k in range(1, n_comp + 1):
                    comp = labeled == k
                    z_idx = np.where(comp.any(axis=(1, 2)))[0]
                    scores.append(sizes[k - 1] * float(np.mean(prob[z_idx, o])))
                scores = np.array(scores)
                keep = np.where(scores >= scores.max() * 0.22)[0] + 1
                out_m[:, o] = np.isin(labeled, keep)
    return out_m


@torch.no_grad()
def agent_patient(
    model,
    bank,
    rows,
    device,
    policy,
    max_steps,
    tau,
    requests_per_batch,
    seed,
    external=None,
    verify=False,
    blank_state=False,
    no_overlay=False,
    absent_exit=None,
    record=False,
    revise="zoom",
    free_roi=False,
    logit_memory=False,
):
    h, w = bank.slices[rows[0]]["h"], bank.slices[rows[0]]["w"]
    raw = np.zeros((len(rows), len(ORGANS), h, w), bool)
    presence = np.zeros((len(rows), len(ORGANS)), np.float32)
    final = np.zeros((len(rows), len(ORGANS), h, w), bool)
    steps = np.zeros((len(rows), len(ORGANS)), np.int16)
    decides = np.zeros((len(rows), len(ORGANS)), np.int16)
    visited = [[None] * len(ORGANS) for _ in rows]
    states = {"idx": [], "best": [], "area": [], "masks": []}
    jobs = [(j, o) for j in range(len(rows)) for o in range(len(ORGANS))]
    use_stack = getattr(model, "decoder_type", "five_slice") == "five_slice"
    for start in range(0, len(jobs), requests_per_batch):
        chunk = jobs[start : start + requests_per_batch]
        requests = [
            Request(
                bank.image_stack(rows[j]) if use_stack else bank.image(rows[j]),
                PROMPTS[ORGANS[o]],
                device,
                blank_state=blank_state,
                no_overlay=no_overlay,
            )
            for j, o in chunk
        ]
        initial = None if external is None else [external[j, o] for j, o in chunk]
        truths = (
            None
            if policy != "oracle"
            else [
                torch.from_numpy(bank.mask(rows[j], ORGANS[o])).to(device)
                for j, o in chunk
            ]
        )
        run_agent(
            model,
            requests,
            policy,
            max_steps,
            tau=tau,
            seed=seed + start,
            initial_masks=initial,
            verify=verify,
            absent_exit=absent_exit,
            record=record,
            truths=truths,
            revise=revise,
            free_roi=free_roi,
            logit_memory=logit_memory,
        )
        for (j, o), r in zip(chunk, requests):
            raw[j, o] = r.initial_raw.cpu().numpy()
            presence[j, o] = r.presence
            final[j, o] = r.mask.cpu().numpy()
            steps[j, o] = len(r.visited)
            decides[j, o] = r.decide_calls
            visited[j][o] = {"visited": r.visited, "reverted": r.reverted}
            for step, mask, best, area in r.states:
                states["idx"].append((j, o, step))
                states["best"].append(best)
                states["area"].append(area)
                states["masks"].append(np.packbits(mask.cpu().numpy(), axis=-1))
    return raw, presence, final, steps, decides, visited, states


def main():
    parser = argparse.ArgumentParser(description="Run NativeSeg full-split volumetric inference.")
    parser.add_argument("--model-dir", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", required=True)
    parser.add_argument("--last-layers", type=int, default=16)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument(
        "--readout", default="auto", choices=("auto", "fused", "llm", "token")
    )
    parser.add_argument(
        "--decoder", default="auto", choices=("auto", "five_slice", "single_slice")
    )
    parser.add_argument("--state-residual", type=float, default=0.0)
    parser.add_argument("--slices-per-batch", type=int, default=4)
    parser.add_argument("--subjects", default=None)
    parser.add_argument(
        "--policy",
        default=None,
        choices=[
            None,
            "learned",
            "uncertainty",
            "random",
            "all2x2",
            "all4x4",
            "reread",
            "oracle",
        ],
    )
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--revise", default="zoom", choices=("zoom", "initial"))
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--requests-per-batch", type=int, default=28)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initial-from", default=None)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--blank-state", action="store_true", help="Zero both visual overlay and decoder prior channel")
    parser.add_argument("--no-overlay", action="store_true", help="Zero visual RGB mask overlay only (alpha_ov=0)")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--absent-exit", type=float, default=None)
    parser.add_argument("--free-roi", action="store_true")
    parser.add_argument("--logit-memory", action="store_true")
    parser.add_argument("--slice-continuity", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing prediction files in output directory")
    args = parser.parse_args()

    _, ckpt_cfg = resolve_checkpoint_path(args.checkpoint)
    decoder_type = (
        ckpt_cfg.get("decoder", "five_slice")
        if args.decoder == "auto"
        else args.decoder
    )
    readout = ckpt_cfg.get("readout", "fused") if args.readout == "auto" else args.readout

    use_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if use_ddp:
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=48))
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        rank, world = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output = Path(args.output)
    ckpt_pt, _ = resolve_checkpoint_path(args.checkpoint)
    ckpt_stat = ckpt_pt.stat()
    fp_payload = {
        "model_dir": str(args.model_dir),
        "data": str(Path(args.data).resolve()),
        "checkpoint": str(ckpt_pt.resolve()),
        "checkpoint_size": ckpt_stat.st_size,
        "checkpoint_mtime_ns": ckpt_stat.st_mtime_ns,
        "split": args.split,
        "decoder": decoder_type,
        "last_layers": args.last_layers,
        "rank": args.rank,
        "readout": readout,
        "state_residual": args.state_residual,
        "slices_per_batch": args.slices_per_batch,
        "subjects": (
            ",".join(sorted(s.strip() for s in args.subjects.split(",") if s.strip()))
            if args.subjects is not None
            else None
        ),
        "policy": args.policy,
        "max_steps": args.max_steps,
        "revise": args.revise,
        "tau": args.tau,
        "requests_per_batch": args.requests_per_batch,
        "seed": args.seed,
        "initial_from": str(Path(args.initial_from).resolve()) if args.initial_from else None,
        "verify": args.verify,
        "blank_state": args.blank_state,
        "no_overlay": args.no_overlay,
        "record": args.record,
        "absent_exit": args.absent_exit,
        "free_roi": args.free_roi,
        "logit_memory": args.logit_memory,
        "slice_continuity": args.slice_continuity,
    }
    import hashlib as _hashlib

    run_fp = _hashlib.sha256(
        json.dumps(fp_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    run_cfg = {
        **vars(args),
        "decoder": decoder_type,
        "readout": readout,
        "world": world,
        "run_fingerprint": run_fp,
    }
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        cfg_path = output / "config.json"
        existing_artifacts = (
            list(output.glob("*.npz")) + list(output.glob("*.visited.json"))
        )
        if args.overwrite:
            for stale in existing_artifacts:
                stale.unlink(missing_ok=True)
        elif existing_artifacts:
            if not cfg_path.exists():
                raise ValueError(
                    f"Output directory {output} contains existing prediction files without config.json provenance. "
                    "Pass --overwrite or use a clean --output directory."
                )
            old_cfg = json.loads(cfg_path.read_text())
            if old_cfg.get("run_fingerprint") != run_fp:
                raise ValueError(
                    f"Output directory {output} belongs to a different run configuration. "
                    "Pass --overwrite or use a clean --output directory."
                )
        cfg_tmp = cfg_path.with_suffix(".json.tmp")
        cfg_tmp.write_text(json.dumps(run_cfg, indent=1))
        cfg_tmp.replace(cfg_path)
    if use_ddp:
        dist.barrier()

    model = build_model(
        args.model_dir,
        args.last_layers,
        args.rank,
        device,
        readout,
        args.state_residual,
        decoder_type,
    )
    load_checkpoint(model, args.checkpoint)
    model.eval()

    bank = SliceBank(args.data, args.split)
    by_subject = {}
    for i, r in enumerate(bank.slices):
        by_subject.setdefault(r["subject_id"], []).append(i)
    subjects = sorted(by_subject)
    if args.subjects is not None:
        wanted = {s.strip() for s in args.subjects.split(",") if s.strip()}
        unknown = sorted(wanted - set(subjects))
        if not wanted or unknown:
            raise ValueError(f"Unknown or empty --subjects for split '{args.split}': {unknown}")
        subjects = [s for s in subjects if s in wanted]

    load = [0] * world
    owner = {}
    for sid in sorted(subjects, key=lambda x: -len(by_subject[x])):
        k = load.index(min(load))
        owner[sid] = k
        load[k] += len(by_subject[sid])
    mine = [sid for sid in subjects if owner[sid] == rank]

    started = time.perf_counter()
    for n, sid in enumerate(mine, 1):
        target = output / f"{sid}.npz"
        if target.exists() and not args.overwrite:
            continue
        tick = time.perf_counter()
        rows = sorted(by_subject[sid], key=lambda i: bank.slices[i]["z"])
        extra = {}
        if args.policy:
            external = None
            ext_raw, ext_prob = None, None
            if args.initial_from:
                ext = np.load(Path(args.initial_from) / f"{sid}.npz")
                shape = tuple(int(v) for v in ext["shape"])
                h_exp, w_exp = bank.slices[rows[0]]["h"], bank.slices[rows[0]]["w"]
                if shape != (len(rows), len(ORGANS), h_exp, w_exp):
                    raise ValueError(f"{sid}: unexpected external mask shape {shape}")
                if list(ext["z"]) != [bank.slices[i]["z"] for i in rows]:
                    raise ValueError(f"{sid}: external masks cover different slices")
                ext_raw = (
                    np.unpackbits(ext["logits_mask"], axis=-1, count=shape[-1])
                    .astype(bool)
                    .reshape(shape)
                )
                ext_prob = ext["presence"].astype(np.float32)
                external = ext_raw & (ext_prob >= 0.5)[..., None, None]
                if args.slice_continuity:
                    external = apply_slice_continuity(external, ext_raw, ext_prob)
            masks, presence, final, steps, decides, visited, states = agent_patient(
                model,
                bank,
                rows,
                device,
                args.policy,
                args.max_steps,
                args.tau,
                args.requests_per_batch,
                args.seed,
                external,
                args.verify,
                args.blank_state,
                args.no_overlay,
                args.absent_exit,
                args.record,
                args.revise,
                free_roi=args.free_roi,
                logit_memory=args.logit_memory,
            )
            if args.slice_continuity:
                final = apply_slice_continuity(
                    final,
                    ext_raw if ext_raw is not None else masks,
                    ext_prob if ext_prob is not None else presence,
                )
            if args.record and states["idx"]:
                sbuf = io.BytesIO()
                np.savez(
                    sbuf,
                    idx=np.array(states["idx"], np.int32),
                    best=np.array(states["best"], np.float32),
                    area=np.array(states["area"], np.float32),
                    masks=np.stack(states["masks"]),
                )
                st_tmp = output / f"{sid}.states.npz.tmp"
                st_tmp.write_bytes(sbuf.getvalue())
                st_tmp.replace(output / f"{sid}.states.npz")
            extra = {
                "final_mask": np.packbits(final, axis=-1),
                "steps": steps,
                "decides": decides,
            }
            vis_tmp = output / f"{sid}.visited.json.tmp"
            vis_tmp.write_text(json.dumps(visited))
            vis_tmp.replace(output / f"{sid}.visited.json")
        else:
            masks, presence = predict_patient(
                model, bank, rows, device, args.slices_per_batch
            )
        buffer = io.BytesIO()
        np.savez(
            buffer,
            logits_mask=np.packbits(masks, axis=-1),
            presence=presence.astype(np.float16),
            shape=np.array(masks.shape),
            z=np.array([bank.slices[i]["z"] for i in rows]),
            **extra,
        )
        target.with_suffix(".tmp").write_bytes(buffer.getvalue())
        target.with_suffix(".tmp").rename(target)
        print(
            json.dumps(
                {
                    "rank": rank,
                    "subject": sid,
                    "slices": len(rows),
                    "done": n,
                    "of": len(mine),
                    "seconds": round(time.perf_counter() - tick, 1),
                    "elapsed": round(time.perf_counter() - started, 1),
                }
            ),
            flush=True,
        )
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
