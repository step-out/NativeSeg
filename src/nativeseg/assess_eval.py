"""Test-time calibration evaluation of the multiscale error gate <err> (AUROC, Hit Rate, 20-cell Spearman rho)."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import torch

from .agent import Request, entropy_scores, run_agent
from .data import ORGANS, PROMPTS, SliceBank
from .evaluate import validate_npz_alignment
from .model import FULL, build_model, load_checkpoint, resolve_checkpoint_path, to_native


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Evaluate <err> slice detection and 20-cell error ranking.")
    parser.add_argument("--model-dir", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--subjects", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial-from", default=None)
    parser.add_argument(
        "--decoder", default="auto", choices=("auto", "five_slice", "single_slice")
    )
    parser.add_argument("--batch", type=int, default=28)
    args = parser.parse_args()

    _, ckpt_cfg = resolve_checkpoint_path(args.checkpoint)
    decoder_type = (
        ckpt_cfg.get("decoder", "five_slice")
        if args.decoder == "auto"
        else args.decoder
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_dir, 16, 16, device, decoder_type=decoder_type)
    load_checkpoint(model, args.checkpoint)
    model.eval()
    use_stack = decoder_type == "five_slice"

    bank = SliceBank(args.data, args.split)
    wanted = [s.strip() for s in args.subjects.split(",") if s.strip()]
    if not wanted:
        raise ValueError("--subjects must contain at least one subject ID")
    by_sub = {}
    for i, r in enumerate(bank.slices):
        by_sub.setdefault(r["subject_id"], []).append(i)
    missing_s = [s for s in wanted if s not in by_sub]
    if missing_s:
        raise ValueError(f"Requested subjects not found in split '{args.split}': {missing_s}")
    rows = [i for i, r in enumerate(bank.slices) if r["subject_id"] in set(wanted)]
    external = {}
    if args.initial_from:
        for sid in sorted(wanted):
            s_rows = sorted(by_sub[sid], key=lambda idx: bank.slices[idx]["z"])
            exp_zs = [bank.slices[idx]["z"] for idx in s_rows]
            exp_hw = (bank.slices[s_rows[0]]["h"], bank.slices[s_rows[0]]["w"])
            ext = np.load(Path(args.initial_from) / f"{sid}.npz")
            shape = validate_npz_alignment(ext, sid, exp_zs, exp_hw, required_key="logits_mask")
            m = (
                np.unpackbits(ext["logits_mask"], axis=-1, count=shape[-1])
                .astype(bool)
                .reshape(shape)
            )
            m &= (ext["presence"].astype(np.float32) >= 0.5)[..., None, None]
            for k, z in enumerate(ext["z"]):
                external[(sid, int(z))] = m[k]
    jobs = [(i, o) for i in rows for o in range(len(ORGANS))]
    records = []
    autocast = torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    )
    for start in range(0, len(jobs), args.batch):
        chunk = jobs[start : start + args.batch]
        requests = [
            Request(
                bank.image_stack(i) if use_stack else bank.image(i),
                PROMPTS[ORGANS[o]],
                device,
            )
            for i, o in chunk
        ]
        truths = [
            torch.from_numpy(bank.mask(i, ORGANS[o])).to(device) for i, o in chunk
        ]
        initial = None
        if external:
            initial = [
                external[(bank.slices[i]["subject_id"], bank.slices[i]["z"])][o]
                for i, o in chunk
            ]
        run_agent(model, requests, "random", max_steps=0, initial_masks=initial)
        entropy = None
        if not external:
            with autocast:
                logits, _ = model.initial(
                    [r.text for r in requests], torch.stack([r.gray for r in requests])
                )
            entropy = [
                entropy_scores(
                    to_native(logits[k].float(), r.params, FULL)[0], r
                ).cpu().numpy()
                for k, r in enumerate(requests)
            ]
        with autocast:
            pred = model.decide_logits(
                [r.text for r in requests],
                torch.stack([r.full_overlay() for r in requests]),
            ).float()
        pred = torch.expm1(pred.clamp(min=0)).cpu().numpy()
        for k, ((i, o), r, t) in enumerate(zip(chunk, requests, truths)):
            sid_r = bank.slices[i]["subject_id"]
            z_r = int(bank.slices[i]["z"])
            actual = (r.cell_errors(t) * r.scale2).cpu().numpy()
            if actual[0] == 0 and pred[k, 0] < 1:
                records.append(
                    {
                        "subject_id": sid_r,
                        "z": z_r,
                        "organ": ORGANS[o],
                        "trivial": True,
                        "pred_total": float(pred[k, 0]),
                        "actual_total": 0.0,
                    }
                )
                continue
            rec = {
                "subject_id": sid_r,
                "z": z_r,
                "trivial": False,
                "organ": ORGANS[o],
                "pred_total": float(pred[k, 0]),
                "actual_total": float(actual[0]),
                "area": float(t.sum().item() * r.scale2),
                "top1_learned": (
                    int(np.argmax(pred[k, 1:5]) == np.argmax(actual[1:5]))
                    if actual[1:5].max() > 0
                    else None
                ),
                "spearman_learned": (
                    float(spearmanr(pred[k, 1:], actual[1:]).correlation)
                    if np.ptp(actual[1:]) > 0 and np.ptp(pred[k, 1:]) > 0
                    else None
                ),
            }
            if entropy is not None:
                e = entropy[k]
                rec["top1_entropy"] = (
                    int(np.argmax(e[:4]) == np.argmax(actual[1:5]))
                    if actual[1:5].max() > 0
                    else None
                )
                rec["spearman_entropy"] = (
                    float(spearmanr(e, actual[1:]).correlation)
                    if np.ptp(actual[1:]) > 0 and np.ptp(e) > 0
                    else None
                )
            records.append(rec)
    nontrivial = [r for r in records if not r["trivial"]]
    labels = np.array(
        [r["actual_total"] > 0.05 * max(r.get("area", 0), 200) for r in records]
    )
    scores = np.array([r["pred_total"] for r in records])
    summary = {
        "requests": len(records),
        "nontrivial": len(nontrivial),
        "auroc_error_detection": (
            float(roc_auc_score(labels, scores))
            if 0 < labels.sum() < len(labels)
            else None
        ),
    }
    for name in ("learned", "entropy"):
        top1 = [
            r[f"top1_{name}"] for r in nontrivial if r.get(f"top1_{name}") is not None
        ]
        rho = [
            r[f"spearman_{name}"]
            for r in nontrivial
            if r.get(f"spearman_{name}") is not None
        ]
        if top1:
            summary[f"top1_2x2_{name}"] = float(np.mean(top1))
            summary[f"spearman_{name}"] = float(np.nanmean(rho))
    print(json.dumps(summary, indent=2))
    Path(args.output).write_text(json.dumps({"summary": summary, "records": records}))


if __name__ == "__main__":
    main()
