"""2D slice-level and 3D physical-spacing volumetric segmentation metrics (DSC, NSD, HD95)."""

from collections import defaultdict
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

try:
    from surface_distance import metrics as _surface
except ImportError:
    _surface = None


def _compute_nsd_and_hd95(
    truth: np.ndarray,
    pred: np.ndarray,
    spacing: np.ndarray,
    tol_mm: float,
    allow_scipy_approx: bool = False,
):
    if _surface is not None:
        distances = _surface.compute_surface_distances(truth, pred, spacing)
        nsd = float(_surface.compute_surface_dice_at_tolerance(distances, tol_mm))
        hd = float(_surface.compute_robust_hausdorff(distances, 95))
        return nsd, hd
    if not allow_scipy_approx:
        raise ImportError(
            "Exact surfel-weighted NSD/HD95 requires Google DeepMind surface-distance: "
            "pip install git+https://github.com/google-deepmind/surface-distance.git"
        )
    # Approximate unweighted voxel-border EDT fallback (only when explicitly requested)
    t_b = truth ^ binary_erosion(truth, border_value=0)
    p_b = pred ^ binary_erosion(pred, border_value=0)
    if not t_b.any() or not p_b.any():
        return 0.0, 0.0
    dt_t = distance_transform_edt(~t_b, sampling=spacing)
    dt_p = distance_transform_edt(~p_b, sampling=spacing)
    d_tp = dt_p[t_b]
    d_pt = dt_t[p_b]
    nsd = float(((d_tp <= tol_mm).sum() + (d_pt <= tol_mm).sum()) / (len(d_tp) + len(d_pt)))
    hd = float(max(np.percentile(d_tp, 95), np.percentile(d_pt, 95)))
    return nsd, hd


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def score_slice(pred, truth, spacing) -> dict:
    pred, truth = np.asarray(pred), np.asarray(truth)
    spacing = np.asarray(spacing, dtype=float)
    if pred.ndim != 2 or pred.shape != truth.shape or not pred.size:
        raise ValueError("Expected equally sized nonempty 2D arrays")
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Need two positive physical spacings in mm")
    if not np.isin(pred, (0, 1)).all() or not np.isin(truth, (0, 1)).all():
        raise ValueError("Masks must be binary, not probabilities or multiclass maps")
    pred, truth = pred.astype(bool), truth.astype(bool)
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    denominator = 2 * tp + fp + fn
    gt_empty, pred_empty = not truth.any(), not pred.any()
    if gt_empty and pred_empty:
        status, nsd, hd = "both_empty", 1.0, None
    elif gt_empty or pred_empty:
        status = (
            "truth_empty_prediction_nonempty"
            if gt_empty
            else "truth_nonempty_prediction_empty"
        )
        nsd, hd = 0.0, None
    else:
        status = "both_nonempty"
        nsd, hd = _compute_nsd_and_hd95(truth, pred, spacing, 2.0)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pixels": int(pred.size),
        "truth_empty": bool(gt_empty),
        "prediction_empty": bool(pred_empty),
        "empty_status": status,
        "dice": 2 * tp / denominator if denominator else 1.0,
        "nsd_2mm": nsd,
        "hd95_mm": hd,
        "false_positive_area_mm2": float(fp * spacing.prod()) if gt_empty else None,
    }


def aggregate_slices(records) -> dict:
    groups = defaultdict(list)
    seen = set()
    for record in records:
        key = record["subject_id"], record["target"], record["slice_index"]
        if key in seen:
            raise ValueError(f"Duplicate scoring row: {key}")
        seen.add(key)
        groups[key[:2]].append(record)
    patients = []
    for (subject, target), rows in sorted(groups.items()):
        tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
        denom = 2 * tp + fp + fn
        empty = [r for r in rows if r["truth_empty"]]
        positive = [r for r in rows if not r["truth_empty"]]
        finite = [r["hd95_mm"] for r in rows if r["hd95_mm"] is not None]
        patients.append(
            {
                "subject_id": subject,
                "target": target,
                "slices": len(rows),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "dice": 2 * tp / denom if denom else 1.0,
                "nsd_2mm": _mean(r["nsd_2mm"] for r in rows),
                "nsd_2mm_positive_slices": _mean(r["nsd_2mm"] for r in positive),
                "hd95_mm": _mean(finite),
                "hd95_applicable_slices": len(finite),
                "positive_slices": len(positive),
                "empty_slices": len(empty),
                "missed_positive_slices": sum(r["prediction_empty"] for r in positive),
                "empty_fp_slices": sum(not r["prediction_empty"] for r in empty),
                "empty_fp_slice_rate": _mean(
                    int(not r["prediction_empty"]) for r in empty
                ),
                "empty_fp_area_mm2": _mean(r["false_positive_area_mm2"] for r in empty),
                "empty_fp_pixel_rate": (
                    sum(r["fp"] for r in empty) / sum(r["pixels"] for r in empty)
                    if empty
                    else None
                ),
            }
        )

    def summarize(items):
        return {
            **{
                k: _mean(r[k] for r in items)
                for k in (
                    "dice",
                    "nsd_2mm",
                    "nsd_2mm_positive_slices",
                    "hd95_mm",
                    "empty_fp_slice_rate",
                    "empty_fp_area_mm2",
                    "empty_fp_pixel_rate",
                )
            },
            "patient_target_count": len(items),
            "hd95_applicable_patients": sum(r["hd95_mm"] is not None for r in items),
            "empty_applicable_patients": sum(r["empty_slices"] > 0 for r in items),
            "positive_applicable_patients": sum(
                r["positive_slices"] > 0 for r in items
            ),
            "missed_positive_slices": sum(r["missed_positive_slices"] for r in items),
            "empty_fp_slices": sum(r["empty_fp_slices"] for r in items),
        }

    return {
        "metric_version": "2d-slice-nsd2-hd95",
        "patients": patients,
        "per_target": {
            t: summarize([r for r in patients if r["target"] == t])
            for t in sorted({r["target"] for r in patients})
        },
        "macro": summarize(patients),
        "warning": "HD95 is conditional on both masks nonempty; inspect failure denominators",
    }


def score_volume(pred, truth, spacing):
    pred, truth = np.asarray(pred), np.asarray(truth)
    spacing = np.asarray(spacing, dtype=float)
    if pred.ndim != 3 or pred.shape != truth.shape or not pred.size:
        raise ValueError("Expected equally sized nonempty 3D containers")
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("Need three positive physical spacings")
    if not np.isin(pred, (0, 1)).all() or not np.isin(truth, (0, 1)).all():
        raise ValueError("Expected binary masks")
    pred, truth = pred.astype(bool), truth.astype(bool)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    denominator = 2 * tp + fp + fn
    gt_empty, pred_empty = not truth.any(), not pred.any()
    if gt_empty or pred_empty:
        nsd, hd95 = float(gt_empty and pred_empty), None
    else:
        nsd, hd95 = _compute_nsd_and_hd95(truth, pred, spacing, 3.0)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "voxels": int(pred.size),
        "truth_empty": bool(gt_empty),
        "prediction_empty": bool(pred_empty),
        "dice": 2 * tp / denominator if denominator else 1.0,
        "nsd_3mm": nsd,
        "hd95_mm": hd95,
        "publisher_dice": None if gt_empty else 2 * tp / (denominator + 1e-6),
        "publisher_nsd_3mm": None if gt_empty else nsd,
        "empty_fp_volume_mm3": float(fp * spacing.prod()) if gt_empty else None,
        "spacing_xyz_mm": spacing.tolist(),
        "literal_publisher_spacing": bool(np.array_equal(spacing, np.full(3, 1.5))),
    }


def aggregate_volumes(records, subjects, targets):
    subjects, targets = tuple(subjects), tuple(targets)
    if (
        not subjects
        or not targets
        or len(set(subjects)) != len(subjects)
        or len(set(targets)) != len(targets)
    ):
        raise ValueError("Invalid expected patient/target coverage")
    keys = [(r["subject_id"], r["target"]) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate patient/organ volume")
    if set(keys) != {(s, t) for s in subjects for t in targets}:
        raise ValueError("Incomplete or unexpected patient/organ coverage")
    metrics = ("dice", "nsd_3mm", "publisher_dice", "publisher_nsd_3mm", "hd95_mm")

    per_target = {}
    for target in targets:
        rows = [r for r in records if r["target"] == target]
        per_target[target] = {
            **{name: _mean([r[name] for r in rows]) for name in metrics},
            "patient_count": len(rows),
            "positive_reference_groups": sum(not r["truth_empty"] for r in rows),
            "empty_reference_groups": sum(r["truth_empty"] for r in rows),
            "hd95_applicable_groups": sum(r["hd95_mm"] is not None for r in rows),
        }
    return {
        "metric_version": "3d-volume-physical-nsd3-hd95",
        "records": records,
        "patients": list(subjects),
        "targets": list(targets),
        "per_target": per_target,
        "macro": {name: _mean([r[name] for r in per_target.values()]) for name in metrics},
        "macro_applicable_targets": {
            name: sum(r[name] is not None for r in per_target.values())
            for name in metrics
        },
        "empty_reference_groups": sum(r["truth_empty"] for r in records),
        "empty_fp_groups": sum(
            r["truth_empty"] and not r["prediction_empty"] for r in records
        ),
        "missed_positive_groups": sum(
            not r["truth_empty"] and r["prediction_empty"] for r in records
        ),
        "hd95_applicable_groups": sum(r["hd95_mm"] is not None for r in records),
        "empty_fp_volume_mm3_total": sum(
            r["empty_fp_volume_mm3"] or 0 for r in records
        ),
        "warning": (
            "Publisher-formula metrics omit empty references. HD95 is conditional. "
            "Distances use actual physical spacing."
        ),
    }


def build_summary(slice_agg, volume_agg, slice_records) -> dict:
    positive = [r for r in slice_records if not r["truth_empty"]]
    empty = [r for r in slice_records if r["truth_empty"]]
    return {
        "volume_positive_dice_macro": volume_agg["macro"]["publisher_dice"],
        "volume_positive_dice_per_organ": {
            t: v["publisher_dice"] for t, v in volume_agg["per_target"].items()
        },
        "volume_nsd3_positive_macro": volume_agg["macro"]["publisher_nsd_3mm"],
        "volume_hd95_macro": volume_agg["macro"]["hd95_mm"],
        "volume_empty_fp_groups": volume_agg["empty_fp_groups"],
        "volume_missed_positive_groups": volume_agg["missed_positive_groups"],
        "slice_all_group_dice": slice_agg["macro"]["dice"],
        "slice_nsd2_positive_macro": slice_agg["macro"]["nsd_2mm_positive_slices"],
        "slice_hd95_macro": slice_agg["macro"]["hd95_mm"],
        "positive_requests": len(positive),
        "empty_requests": len(empty),
        "missed_positive_requests": sum(r["prediction_empty"] for r in positive),
        "empty_fp_requests": sum(not r["prediction_empty"] for r in empty),
        "mean_positive_request_dice": (
            float(np.mean([r["dice"] for r in positive])) if positive else None
        ),
    }
