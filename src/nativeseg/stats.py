"""Paired patient-level statistical testing (Wilcoxon signed-rank test + 10,000-rep bootstrap 95% CI)."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def per_patient(score, metric):
    rows = score["volume_aggregate"]["records"]
    table = {}
    for r in rows:
        if r["truth_empty"] or r.get(metric) is None:
            continue
        table.setdefault(r["subject_id"], {})[r["target"]] = r[metric]
    return table


def bootstrap(diff, reps=10000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, (reps, len(diff)), replace=True).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    parser = argparse.ArgumentParser(description="Paired patient-level Wilcoxon & bootstrap test.")
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--names", nargs=2, default=["A", "B"])
    parser.add_argument("--metric", default="publisher_dice")
    args = parser.parse_args()
    raw_a = json.loads(Path(args.a).read_text())
    raw_b = json.loads(Path(args.b).read_text())
    a = per_patient(raw_a, args.metric)
    b = per_patient(raw_b, args.metric)
    if not a or set(a) != set(b):
        raise SystemExit("The two score files cover different patients or are empty")
    subjects = sorted(a)
    for s in subjects:
        if set(a[s]) != set(b[s]):
            raise SystemExit(
                f"Patient {s} has different applicable target sets between A ({sorted(a[s])}) and B ({sorted(b[s])})"
            )
    organs = sorted({o for s in subjects for o in a[s]})
    pa = np.array([np.mean([a[s][o] for o in sorted(a[s])]) for s in subjects])
    pb = np.array([np.mean([b[s][o] for o in sorted(b[s])]) for s in subjects])
    diff = pb - pa
    per_organ = {}
    for o in organs:
        ss = [s for s in subjects if o in a[s]]
        va = np.array([a[s][o] for s in ss])
        vb = np.array([b[s][o] for s in ss])
        d = vb - va
        per_organ[o] = {
            "patients": len(ss),
            f"mean_{args.names[0]}": float(va.mean()),
            f"mean_{args.names[1]}": float(vb.mean()),
            "mean_difference": float(d.mean()),
            "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
        }
    out = {
        "metric": args.metric,
        "patients": len(subjects),
        "organ_macro_table_estimand": {
            args.names[0]: float(np.mean([per_organ[o][f"mean_{args.names[0]}"] for o in organs])),
            args.names[1]: float(np.mean([per_organ[o][f"mean_{args.names[1]}"] for o in organs])),
        },
        "patient_macro_paired_estimand": {
            args.names[0]: float(pa.mean()),
            args.names[1]: float(pb.mean()),
            "mean_difference": float(diff.mean()),
            "bootstrap_95ci": bootstrap(diff),
            "patients_improved": int((diff > 0).sum()),
            "patients_worsened": int((diff < 0).sum()),
            "wilcoxon_p": float(wilcoxon(pb, pa).pvalue) if np.any(diff != 0) else 1.0,
        },
        "per_organ": per_organ,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
