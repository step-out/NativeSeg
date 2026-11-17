"""Perturbation-trained CNN mask refiner baseline (LRN†, following Chen et al., MedIA 2025).

Trains and evaluates an organ-conditioned 2D residual U-Net refiner on TotalSegmentator synthetic
mask perturbations (`nativeseg.agent.perturb`) over the identical 1,081-patient training split and
512x512 letterbox canvas without a vision-language backbone (reproduces the `+ LRN†` rows in
Tables 1b, 2b, and 4b).
"""

import argparse
import io
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .agent import perturb
from .data import ORGANS, BalancedRequests, SliceBank
from .evaluate import validate_npz_alignment
from .model import FULL, letterbox_params, render, to_native
from .train import mask_loss


class _LRNBlock(nn.Module):
    def __init__(self, cin, cout, organs):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(cout, affine=True)
        self.norm2 = nn.InstanceNorm2d(cout, affine=True)
        self.film = nn.Embedding(organs, 2 * cout)
        nn.init.zeros_(self.film.weight)

    def forward(self, x, organ):
        x = F.leaky_relu(self.norm1(self.conv1(x)), 0.01)
        gamma, beta = self.film(organ)[:, :, None, None].chunk(2, 1)
        x = x * (1.0 + gamma) + beta
        return F.leaky_relu(self.norm2(self.conv2(x)), 0.01)


class CNNEditor(nn.Module):
    """Organ-conditioned 6-stage U-Net mask editor (`10.05M` trainable parameters)."""

    def __init__(self, widths=(32, 64, 128, 256, 320, 320), organs=len(ORGANS)):
        super().__init__()
        self.down = nn.ModuleList()
        cin = 2
        for w in widths:
            self.down.append(_LRNBlock(cin, w, organs))
            cin = w
        self.up = nn.ModuleList()
        self.reduce = nn.ModuleList()
        for skip, w in zip(reversed(widths[:-1]), reversed(widths[1:])):
            self.reduce.append(nn.Conv2d(w, skip, 1))
            self.up.append(_LRNBlock(2 * skip, skip, organs))
        self.out = nn.Conv2d(widths[0], 1, 1)

    def forward(self, gray, state, organ):
        x = torch.cat((gray, state), 1)
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else F.max_pool2d(x, 2), organ)
            skips.append(x)
        for reduce, block, skip in zip(self.reduce, self.up, reversed(skips[:-1])):
            x = F.interpolate(reduce(x), scale_factor=2, mode="bilinear", align_corners=False)
            x = block(torch.cat((x, skip), 1), organ)
        return self.out(x)


def resolve_lrn_checkpoint(path_or_dir):
    p = Path(path_or_dir)
    if p.is_dir():
        for cand in ("trainable.pt", "editor.pt"):
            if (p / cand).is_file():
                return p / cand
        raise FileNotFoundError(f"No trainable.pt or editor.pt found inside {p}")
    if not p.is_file():
        raise FileNotFoundError(f"LRN checkpoint not found: {p}")
    return p


def _canvas_pair(image, mask, device, cell=FULL):
    h, w = image.shape
    p = letterbox_params(h, w)
    gray = render(torch.as_tensor(image, device=device).float()[None], p, cell)
    state = render(torch.as_tensor(mask, device=device).float()[None], p, cell)
    return gray, state, p


def train_lrn(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoint_config.json").write_text(
        json.dumps(
            {
                "architecture": "CNNEditor",
                "widths": [32, 64, 128, 256, 320, 320],
                "steps": args.steps,
                "batch": args.batch,
                "lr": args.lr,
                "scale_aug": args.scale_aug,
                "seed": args.seed,
            },
            indent=2,
        )
    )
    bank = SliceBank(args.data, "train")
    loader = iter(
        torch.utils.data.DataLoader(
            BalancedRequests(bank, args.seed, positive_fraction=0.7),
            batch_size=args.batch,
            num_workers=args.workers,
            collate_fn=list,
        )
    )
    model = CNNEditor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    amp = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    tick = time.perf_counter()
    with (output / "train_log.jsonl").open("a") as log:
        for step in range(1, args.steps + 1):
            lr_now = (
                args.lr
                * min(1.0, step / 500.0)
                * 0.5
                * (1.0 + np.cos(np.pi * step / args.steps))
            )
            for g in optimizer.param_groups:
                g["lr"] = lr_now
            batch = next(loader)
            grays, states, params, cells, truths, organs = [], [], [], [], [], []
            for item in batch:
                truth = torch.from_numpy(item["mask"]).to(device)
                state = perturb(truth, rng)
                cell = FULL
                if rng.random() < args.scale_aug:
                    size = rng.uniform(0.75, 1.25) * 512.0
                    cell = (
                        256.0 - size / 2.0 + rng.uniform(-0.1, 0.1) * size,
                        256.0 - size / 2.0 + rng.uniform(-0.1, 0.1) * size,
                        size,
                    )
                g, s, p = _canvas_pair(item["image"], state, device, cell)
                grays.append(g)
                states.append(s)
                params.append(p)
                cells.append(cell)
                truths.append(truth)
                organs.append(ORGANS.index(item["organ"]))
            with amp:
                logits = model(
                    torch.stack(grays),
                    torch.stack(states),
                    torch.tensor(organs, device=device),
                )
            losses = []
            for i in range(len(batch)):
                native, inside = to_native(logits[i].float(), params[i], cells[i])
                b, d = mask_loss(native[inside], truths[i][inside].float())
                losses.append(b + d)
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 50 == 0:
                rec = {
                    "step": step,
                    "loss": round(float(loss), 4),
                    "seconds": round(time.perf_counter() - tick, 1),
                }
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if step % 500 == 0:
                    print(json.dumps(rec), flush=True)
            if step % 5000 == 0 or step == args.steps:
                torch.save(model.state_dict(), output / "trainable.pt")


@torch.no_grad()
def predict_lrn(args):
    if not args.checkpoint or not args.initial_from:
        raise ValueError("predict mode requires both --checkpoint and --initial-from")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_pt = resolve_lrn_checkpoint(args.checkpoint)
    model = CNNEditor().to(device)
    model.load_state_dict(torch.load(ckpt_pt, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    amp = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    bank = SliceBank(args.data, args.split)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    by_subject = {}
    for i, r in enumerate(bank.slices):
        by_subject.setdefault(r["subject_id"], []).append(i)
    subjects = sorted(by_subject)[args.shard :: args.shards]
    for sid in subjects:
        target = output / f"{sid}.npz"
        if target.exists() and not args.overwrite:
            continue
        rows = sorted(by_subject[sid], key=lambda i: bank.slices[i]["z"])
        exp_zs = [bank.slices[i]["z"] for i in rows]
        exp_hw = (bank.slices[rows[0]]["h"], bank.slices[rows[0]]["w"])
        ext = np.load(Path(args.initial_from) / f"{sid}.npz")
        shape = validate_npz_alignment(ext, sid, exp_zs, exp_hw, required_key="logits_mask")
        initial = (
            np.unpackbits(ext["logits_mask"], axis=-1, count=shape[-1])
            .astype(bool)
            .reshape(shape)
        )
        if args.presence is not None:
            initial &= (ext["presence"].astype(np.float32) >= args.presence)[..., None, None]
        final = np.zeros_like(initial)
        organ_ids = torch.arange(len(ORGANS), device=device)
        for j, i in enumerate(rows):
            image = bank.image(i)
            grays, states, params = [], [], []
            for o in range(len(ORGANS)):
                g, s, p = _canvas_pair(image, initial[j, o], device)
                grays.append(g)
                states.append(s)
                params.append(p)
            with amp:
                logits = model(torch.stack(grays), torch.stack(states), organ_ids)
            for o in range(len(ORGANS)):
                native, _ = to_native(logits[o].float(), params[o], FULL)
                final[j, o] = (native > 0).cpu().numpy()
        buf = io.BytesIO()
        np.savez(
            buf,
            logits_mask=np.packbits(final, axis=-1),
            presence=np.ones(final.shape[:2], np.float16),
            shape=np.array(final.shape),
            z=np.asarray(exp_zs),
        )
        target.with_suffix(".tmp").write_bytes(buf.getvalue())
        target.with_suffix(".tmp").rename(target)
        print(json.dumps({"subject": sid, "slices": len(rows)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate LRN† CNN mask refiner baseline.")
    parser.add_argument("mode", choices=["train", "predict"])
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scale-aug", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--split", default="val")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--initial-from", default=None)
    parser.add_argument("--presence", type=float, default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.mode == "train":
        train_lrn(args)
    else:
        predict_lrn(args)


if __name__ == "__main__":
    main()
