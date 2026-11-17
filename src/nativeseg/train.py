"""Two-stage curriculum training of NativeSeg on Qwen3.8-27B (single-GPU or distributed torchrun).

Stage A (`--stage initial`):
  Trains top-16-layer rank-16 LoRA adapters, 2D spatial query offsets, presence head (<pres>),
  and five-slice U-Net decoder (`--decoder five_slice`) on organ-balanced slices.

Stage B (`--stage agent`):
  Initializes from a pretrained Round-0 or single-slice local refiner checkpoint (`--init-from`)
  and jointly optimizes Round-0 readout, 21-region log-error regression (<err>), and local zoom
  refinement over the multi-state curriculum.
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

import torch
from torch import distributed as dist
from torch.nn import functional as F

from .agent import Request, apply_zoom, cell_box, decide_targets, perturb, run_agent
from .data import PROMPTS, BalancedRequests, SliceBank
from .model import FULL, build_model, letterbox_params, render, resolve_checkpoint_path, to_native


def git_commit(root):
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def views(batch, device, rng=None, scale_aug=0.0):
    canvases, params, cells = [], [], []
    for item in batch:
        h, w = item["image"].shape
        p = letterbox_params(h, w)
        cell = FULL
        if rng is not None and scale_aug and rng.random() < scale_aug:
            size = rng.uniform(0.75, 1.25) * 512
            top = 256 - size / 2 + rng.uniform(-0.1, 0.1) * size
            left = 256 - size / 2 + rng.uniform(-0.1, 0.1) * size
            cell = (top, left, size)
        raw_arr = item["stack"] if "stack" in item else item["image"][None]
        image = torch.from_numpy(raw_arr).to(device)
        canvases.append(render(image, p, cell))
        params.append(p)
        cells.append(cell)
    return torch.stack(canvases), params, cells


def mask_loss(logits, truth):
    bce = F.binary_cross_entropy_with_logits(logits, truth)
    if truth.sum() == 0:
        return bce, torch.zeros((), device=logits.device)
    prob = logits.sigmoid()
    dice = 1 - (2 * (prob * truth).sum() + 1) / (prob.sum() + truth.sum() + 1)
    return bce, dice


def initial_step(model, batch, device, rng=None, scale_aug=0.0):
    gray, params, cells = views(batch, device, rng, scale_aug)
    texts = [PROMPTS[item["organ"]] for item in batch]
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits, presence = model.initial(texts, gray)
    terms = {"bce": [], "dice": [], "presence": [], "correct": []}
    for i, item in enumerate(batch):
        native, inside = to_native(logits[i].float(), params[i], cells[i])
        truth = torch.from_numpy(item["mask"]).to(device).float()
        bce, dice = mask_loss(native[inside], truth[inside])
        target = torch.tensor(float(truth[inside].any()), device=device)
        terms["bce"].append(bce)
        terms["dice"].append(dice)
        terms["presence"].append(
            F.binary_cross_entropy_with_logits(presence[i].float(), target)
        )
        terms["correct"].append(((presence[i] > 0).float() == target).float())
    terms = {k: torch.stack(v).mean() for k, v in terms.items()}
    loss = terms["bce"] + terms["dice"] + 0.5 * terms["presence"]
    return loss, terms


def choose_cell(request, truth, rng):
    errors = request.cell_errors(truth)[1:]
    valid = [0] + [c + 1 for c, m in enumerate(request.inside) if bool(m.any())]
    if float(errors.max()) == 0:
        return rng.choice(valid)
    draw = rng.random()
    if draw < 0.15:
        return 0
    if draw < 0.55:
        return int(errors[:4].argmax()) + 1
    if draw < 0.85:
        return int(errors[4:].argmax()) + 5
    region = request.mask | truth
    relevant = [c + 1 for c, m in enumerate(request.inside) if bool((region & m).any())]
    return rng.choice(relevant or valid)


def amp(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def zoom_forward(model, requests, cells):
    c_views = [r.cell_views(c) for r, c in zip(requests, cells)]
    with amp(requests[0].image.device):
        return model.zoom(
            [r.text for r in requests],
            torch.stack([r.full_overlay() for r in requests]),
            torch.stack([v[0] for v in c_views]),
            torch.stack([v[1] for v in c_views]),
            torch.stack([v[2] for v in c_views]),
        )


def agent_step(
    model,
    batch,
    device,
    rng,
    synthetic=0.35,
    extra=0.4,
    zoom_weight=1.0,
    clean=0.15,
    focus_weight=2.0,
):
    requests = [
        Request(item.get("stack", item["image"]), PROMPTS[item["organ"]], device)
        for item in batch
    ]
    truths = [torch.from_numpy(item["mask"]).to(device) for item in batch]
    with torch.no_grad():
        run_agent(model, requests, "random", max_steps=0)
        for r, t in zip(requests, truths):
            u = rng.random()
            if u < clean:
                r.mask = t.clone()
            elif u < clean + synthetic:
                r.mask = perturb(t, rng)
        more = [i for i in range(len(requests)) if rng.random() < extra]
        if more:
            cells = [choose_cell(requests[i], truths[i], rng) for i in more]
            logits = zoom_forward(model, [requests[i] for i in more], cells)
            for j, i in enumerate(more):
                apply_zoom(requests[i], cells[j], logits[j])
                requests[i].visited.append(cells[j])
    with amp(device):
        pred = model.decide_logits(
            [r.text for r in requests],
            torch.stack([r.full_overlay() for r in requests]),
        ).float()
    target = decide_targets(requests, truths)
    decide = F.smooth_l1_loss(pred, target)
    cells = [choose_cell(r, t, rng) for r, t in zip(requests, truths)]
    logits = zoom_forward(model, requests, cells)
    bce, dice, gain = [], [], []
    for r, t, c, lg in zip(requests, truths, cells, logits):
        native, inside = to_native(lg.float(), r.params, cell_box(c))
        values, truth = native[inside], t[inside].float()
        b, d = mask_loss(values, truth)
        if focus_weight > 0:
            cur = r.mask[inside].float()
            w = 1.0 + focus_weight * (cur != truth).float()
            b = F.binary_cross_entropy_with_logits(values, truth, weight=w)
        bce.append(b)
        dice.append(d)
        before = (r.mask[inside] != t[inside]).sum()
        after = ((values > 0) != t[inside]).sum()
        gain.append((before - after).float())
    terms = {
        "zoom_bce": torch.stack(bce).mean(),
        "zoom_dice": torch.stack(dice).mean(),
        "decide": decide,
        "zoom_fixed_px": torch.stack(gain).mean().detach(),
    }
    return zoom_weight * (terms["zoom_bce"] + terms["zoom_dice"]) + terms["decide"], terms


def main():
    parser = argparse.ArgumentParser(description="Train NativeSeg Stage A or Stage B.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--model-dir", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=["initial", "agent"], default="initial")
    parser.add_argument("--agent-batch", type=int, default=3)
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--scale-aug", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--last-layers", type=int, default=16)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--readout", default="fused", choices=("fused", "llm", "token"))
    parser.add_argument(
        "--decoder", default="five_slice", choices=("five_slice", "single_slice")
    )
    parser.add_argument("--hard-organs", action="store_true")
    parser.add_argument("--freeze-qwen", action="store_true")
    parser.add_argument("--state-residual", type=float, default=0.0)
    parser.add_argument("--clean", type=float, default=0.15)
    parser.add_argument("--focus-weight", type=float, default=2.0)
    parser.add_argument("--synthetic", type=float, default=0.35)
    parser.add_argument("--extra", type=float, default=0.35)
    parser.add_argument("--zoom-weight", type=float, default=1.0)
    parser.add_argument("--lr-lora", type=float, default=1e-4)
    parser.add_argument("--lr-new", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args()

    use_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if use_ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        rank, world = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        config = {
            **vars(args),
            "world": world,
            "git_commit": git_commit(args.root),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (
            output
            / ("config.json" if args.resume is None else f"config_resume_{int(time.time())}.json")
        ).write_text(json.dumps(config, indent=1))
    if use_ddp:
        dist.barrier()

    model = build_model(
        args.model_dir,
        args.last_layers,
        args.rank,
        device,
        args.readout,
        args.state_residual,
        args.decoder,
        args.freeze_qwen,
    )
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    lora = [p for n, p in trainable if "lora_" in n]
    new = [p for n, p in trainable if "lora_" not in n]
    optimizer = torch.optim.AdamW(
        [{"params": lora, "lr": args.lr_lora}, {"params": new, "lr": args.lr_new}],
        weight_decay=0.01,
    )
    base_lrs = [args.lr_lora, args.lr_new]
    start = 0
    if args.resume:
        pt_path, _ = resolve_checkpoint_path(args.resume)
        resume_dir = pt_path.parent
        state = torch.load(pt_path, map_location="cpu", weights_only=True)
        missing = model.load_state_dict(state, strict=False)
        if missing.unexpected_keys:
            raise ValueError(f"Unexpected keys in checkpoint: {missing.unexpected_keys[:5]}")
        optimizer.load_state_dict(
            torch.load(resume_dir / "optimizer.pt", map_location="cpu", weights_only=True)
        )
        start = json.loads((resume_dir / "step.json").read_text())["step"]
    if args.init_from and not args.resume:
        pt_path, init_cfg = resolve_checkpoint_path(args.init_from)
        state = torch.load(pt_path, map_location="cpu", weights_only=True)
        if model.load_state_dict(state, strict=False).unexpected_keys:
            raise ValueError("Initial checkpoint has unexpected tensors")
        if args.stage == "agent" and (
            init_cfg.get("stage") == "initial" or "initial" in str(args.init_from) or "stage_a" in str(args.init_from)
        ):
            model.query_type.data[1].copy_(model.query_type.data[0])

    bank = SliceBank(args.data, "train")
    dataset = BalancedRequests(
        bank,
        args.seed + start,
        rank=rank,
        multi_slice=(args.decoder == "five_slice"),
        hard_organs=args.hard_organs,
    )
    loader = iter(
        torch.utils.data.DataLoader(
            dataset, batch_size=args.batch, num_workers=4, collate_fn=list, prefetch_factor=4
        )
    )
    agent_loader = None
    rng = None
    if args.stage == "agent":
        agent_data = BalancedRequests(
            bank,
            args.seed + start + 7,
            rank=rank,
            positive_fraction=0.7,
            multi_slice=(args.decoder == "five_slice"),
            hard_organs=args.hard_organs,
        )
        agent_loader = iter(
            torch.utils.data.DataLoader(
                agent_data,
                batch_size=args.agent_batch,
                num_workers=4,
                collate_fn=list,
                prefetch_factor=4,
            )
        )
        rng = random.Random(args.seed * 31 + rank + start)

    numel = sum(p.numel() for _, p in trainable)
    if rank == 0:
        print(
            json.dumps(
                {
                    "trainable_parameters": numel,
                    "lora_tensors": len(lora),
                    "new_tensors": len(new),
                }
            ),
            flush=True,
        )
    aug_rng = random.Random(args.seed * 17 + rank + start)
    log_dir = Path(args.log_dir) if args.log_dir else output
    if rank == 0:
        log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / "train_log.jsonl", "a") if rank == 0 else None
    model.qwen.train()
    model.core.visual.eval()
    tick = time.perf_counter()
    for step in range(start + 1, args.steps + 1):
        scale = (
            min(1.0, step / max(1, args.warmup))
            * 0.5
            * (1.0 + math.cos(math.pi * min(1.0, step / args.steps)))
        )
        for group, lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = lr * scale
        read = time.perf_counter()
        batch = next(loader)
        agent_batch = next(agent_loader) if agent_loader is not None else None
        read = time.perf_counter() - read
        optimizer.zero_grad(set_to_none=True)
        loss, terms = initial_step(model, batch, device, aug_rng, args.scale_aug)
        if agent_batch is not None:
            loss.backward()
            agent_loss, agent_terms = agent_step(
                model,
                agent_batch,
                device,
                rng,
                args.synthetic,
                args.extra,
                args.zoom_weight,
                args.clean,
                args.focus_weight,
            )
            agent_loss.backward()
            loss = loss.detach() + agent_loss.detach()
            terms = {**terms, **agent_terms}
        else:
            loss.backward()
        if use_ddp:
            used_mask = torch.tensor(
                [1.0 if p.grad is not None else 0.0 for _, p in trainable],
                device=device,
            )
            dist.all_reduce(used_mask)
            flat = torch.cat(
                [
                    (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                    for _, p in trainable
                ]
            )
            dist.all_reduce(flat)
            flat /= world
            offset = 0
            for idx_p, (_, p) in enumerate(trainable):
                n = p.numel()
                p.grad = (
                    flat[offset : offset + n].view_as(p)
                    if float(used_mask[idx_p]) > 0
                    else None
                )
                offset += n
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        optimizer.step()
        stats = torch.stack([loss.detach(), *[v.detach() for v in terms.values()], norm])
        if use_ddp:
            dist.all_reduce(stats)
            stats /= world
        if rank == 0:
            names = ["loss", *terms.keys(), "grad_norm"]
            mem_gb = (
                round(torch.cuda.max_memory_allocated() / 2**30, 1)
                if device.type == "cuda"
                else 0.0
            )
            record = {
                "step": step,
                **{k: round(float(v), 5) for k, v in zip(names, stats)},
                "lr_scale": round(scale, 5),
                "read_s": round(read, 3),
                "step_s": round(time.perf_counter() - tick, 3),
                "max_mem_gb": mem_gb,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step % 20 == 0 or step <= 5:
                print(json.dumps(record), flush=True)
        tick = time.perf_counter()
        if step % args.save_every == 0 or step == args.steps:
            if rank == 0:
                folder = output / f"step_{step:06d}"
                folder.mkdir(exist_ok=True)
                torch.save(model.trainable_state(), folder / "trainable.pt")
                torch.save(optimizer.state_dict(), folder / "optimizer.pt")
                (folder / "step.json").write_text(json.dumps({"step": step}))
                (folder / "checkpoint_config.json").write_text(
                    json.dumps(
                        {
                            "decoder": args.decoder,
                            "stage": args.stage,
                            "step": step,
                            "last_layers": args.last_layers,
                            "rank": args.rank,
                            "readout": args.readout,
                            "state_residual": args.state_residual,
                        },
                        indent=2,
                    )
                )
            if use_ddp:
                dist.barrier()
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
