"""Offline sweep of tolerance tau and max zoom budget K from one recorded trajectory run (`--record --tau 0`)."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from .data import ORGANS, SliceBank
from .evaluate import validate_npz_alignment


def simulate(job):
    sid, pred_dir, data_dir, split, taus, budgets, min_area, external_init = job
    bank = SliceBank(data_dir, split)
    rows = sorted(
        [i for i, r in enumerate(bank.slices) if r["subject_id"] == sid],
        key=lambda i: bank.slices[i]["z"],
    )
    expected_zs = [bank.slices[i]["z"] for i in rows]
    expected_hw = (bank.slices[rows[0]]["h"], bank.slices[rows[0]]["w"])
    data = np.load(Path(pred_dir) / f"{sid}.npz")
    shape = validate_npz_alignment(data, sid, expected_zs, expected_hw, required_key="logits_mask")
    raw_init = (
        np.unpackbits(data["logits_mask"], axis=-1, count=shape[-1])
        .astype(bool)
        .reshape(shape)
    )
    # External initialization masks stored in `logits_mask` are already gated at load time;
    # only self-initialized runs gate raw Round-0 logits by `presence >= 0.5`.
    initial = (
        raw_init
        if external_init
        else (raw_init & (data["presence"].astype(np.float32) >= 0.5)[..., None, None])
    )
    traj = {}
    path = Path(pred_dir) / f"{sid}.states.npz"
    masks = None
    if path.exists():
        st = np.load(path)
        for k, (j, o, step) in enumerate(st["idx"]):
            traj.setdefault((int(j), int(o)), []).append(
                (int(step), float(st["best"][k]), float(st["area"][k]), k)
            )
        masks = st["masks"]
    truths = {
        (j, o): bank.mask(i, organ)
        for j, i in enumerate(rows)
        for o, organ in enumerate(ORGANS)
    }
    results = {}
    for tau in taus:
        for K in budgets:
            tp = np.zeros(len(ORGANS))
            fp = np.zeros(len(ORGANS))
            fn = np.zeros(len(ORGANS))
            empty_fp = missed = edits = assessments = requests = 0
            for (j, o), truth in truths.items():
                requests += 1
                states = traj.get((j, o))
                if K == 0:
                    mask = initial[j, o]
                elif not states:
                    mask = initial[j, o]
                else:
                    chosen = None
                    for step, best, area, k in sorted(states):
                        if step >= K or best < tau * max(area, min_area):
                            chosen = (step, k)
                            break
                    if chosen is None:
                        step, _, _, k = sorted(states)[-1]
                        chosen = (step, k)
                    step, k = chosen
                    mask = np.unpackbits(
                        masks[k], axis=-1, count=shape[-1]
                    ).astype(bool)
                    edits += step
                    assessments += step + 1
                tp[o] += (mask & truth).sum()
                fp[o] += (mask & ~truth).sum()
                fn[o] += (~mask & truth).sum()
                if truth.any() and not mask.any():
                    missed += 1
                if not truth.any() and mask.any():
                    empty_fp += 1
            results[(tau, K)] = (
                tp,
                fp,
                fn,
                empty_fp,
                missed,
                edits,
                assessments,
                requests,
            )
    return sid, results


def main():
    parser = argparse.ArgumentParser(description="Offline (tau, K) granularity sweep.")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--subjects", default=None)
    parser.add_argument("--taus", default="0,0.02,0.05,0.1,0.2")
    parser.add_argument("--budgets", default="0,1,2,3,4")
    parser.add_argument("--min-area", type=float, default=800)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    pred_dir = Path(args.pred)
    cfg_path = pred_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}; run nativeseg-predict with --record --tau 0 first")
    run_cfg = json.loads(cfg_path.read_text())
    if run_cfg.get("policy") != "learned" or not run_cfg.get("record"):
        raise ValueError("Offline sweep requires a learned-policy run recorded with --record")
    if float(run_cfg.get("tau", 1.0)) > 0.0:
        raise ValueError(f"Recorded trajectory must use --tau 0, got tau={run_cfg.get('tau')}")

    taus = [float(t) for t in args.taus.split(",")]
    budgets = [int(b) for b in args.budgets.split(",")]
    max_rec = int(run_cfg.get("max_steps", 0))
    if max(budgets) > max_rec:
        raise ValueError(f"Requested budget {max(budgets)} exceeds recorded max_steps={max_rec}")
    external_init = bool(run_cfg.get("initial_from"))

    bank = SliceBank(args.data, args.split)
    all_subjects = sorted({r["subject_id"] for r in bank.slices})
    if args.subjects is not None:
        wanted = {s.strip() for s in args.subjects.split(",") if s.strip()}
    elif run_cfg.get("subjects"):
        wanted = {s.strip() for s in str(run_cfg["subjects"]).split(",") if s.strip()}
    else:
        wanted = set(all_subjects)
    unknown = sorted(wanted - set(all_subjects))
    if not wanted or unknown:
        raise ValueError(f"Unknown or empty subject selection for split '{args.split}': {unknown}")
    subjects = [s for s in all_subjects if s in wanted]
    missing_npz = [s for s in subjects if not (pred_dir / f"{s}.npz").exists()]
    if missing_npz:
        raise FileNotFoundError(f"Missing prediction files for {len(missing_npz)} subjects: {missing_npz[:5]}")

    with ProcessPoolExecutor(args.workers) as pool:
        per_subject = dict(
            pool.map(
                simulate,
                [
                    (
                        s,
                        args.pred,
                        args.data,
                        args.split,
                        taus,
                        budgets,
                        args.min_area,
                        external_init,
                    )
                    for s in subjects
                ],
            )
        )
    rows = []
    for tau in taus:
        for K in budgets:
            per_organ = [[] for _ in ORGANS]
            totals = np.zeros(5)
            for sid in subjects:
                tp, fp, fn, empty_fp, missed, edits, assessments, requests = per_subject[
                    sid
                ][(tau, K)]
                for o in range(len(ORGANS)):
                    if tp[o] + fn[o] > 0:
                        per_organ[o].append(
                            2 * tp[o] / (2 * tp[o] + fp[o] + fn[o] + 1e-6)
                        )
                totals += [empty_fp, missed, edits, assessments, requests]
            row = {
                "tau": tau,
                "budget": K,
                "volume_positive_dice_macro": float(
                    np.mean([np.mean(d) for d in per_organ if d])
                ),
                "per_organ": {
                    o: float(np.mean(d)) for o, d in zip(ORGANS, per_organ) if d
                },
                "empty_fp_requests": int(totals[0]),
                "missed_positive_requests": int(totals[1]),
                "mean_edits": float(totals[2] / totals[4]),
                "mean_assessments": float(totals[3] / totals[4]),
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in row.items()
                        if k != "per_organ"
                    }
                )
            )
    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
