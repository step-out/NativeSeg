"""Unified volumetric scoring, standalone benchmark evaluation, specialist consensus, and NIfTI conversion.

Subcommands:
  1. `score`: Score a single prediction directory (`Round 0`, `Stage B`, ablations, or baselines).
  2. `standalone`: Full NativeSeg standalone evaluation combining 5-slice Round-0 readout with verified
     multiround local zoom and two-threshold axial continuity (reproduces Table 1a, 2a, 4a).
  3. `refiner`: Plug-and-play specialist refinement evaluation with parameter-free cross-model consensus
     and isolated single-slice FP pruning (reproduces Table 1b, 2b, 4b across 2D nnU-Net, 3D SegVol, 3D VoxTell).
  4. `convert-nifti`: Convert external specialist NIfTI masks (`<subject>/<organ>.nii.gz`) into NativeSeg
     packed `.npz` format so any 2D or 3D segmenter can be refined with `--initial-from`.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import json
from pathlib import Path

import numpy as np

from .data import ORGANS, SliceBank
from .geometry import load_native_image, load_native_mask
from .metrics import aggregate_slices, aggregate_volumes, build_summary, score_slice, score_volume
from .predict import apply_slice_continuity

_BANKS = {}


def _get_bank(data_dir, split_name):
    key = (str(data_dir), split_name)
    if key not in _BANKS:
        _BANKS[key] = SliceBank(data_dir, split_name)
    return _BANKS[key]


from importlib import resources as pkg_resources


def _resolve_z_spacing(bank, subjects):
    z_spacing = {}
    for r in bank.slices:
        if r.get("spacing_z_mm") is not None:
            z_spacing[r["subject_id"]] = float(r["spacing_z_mm"])
    missing = [s for s in subjects if s not in z_spacing]
    if missing:
        fallback = {}
        try:
            res_f = pkg_resources.files("nativeseg.resources").joinpath("totalseg_splits.json")
            if res_f.is_file():
                fallback = json.loads(res_f.read_text(encoding="utf-8")).get("spacing_z_mm", {})
        except Exception:
            pass
        if not fallback:
            cfg_path = Path(__file__).resolve().parents[2] / "configs" / "totalseg_splits.json"
            if cfg_path.exists():
                fallback = json.loads(cfg_path.read_text(encoding="utf-8")).get("spacing_z_mm", {})
        for s in missing:
            if s in fallback:
                z_spacing[s] = float(fallback[s])
    still_missing = [s for s in subjects if s not in z_spacing]
    if still_missing:
        raise ValueError(f"Missing axial spacing_z_mm for subjects: {still_missing[:5]}")
    return z_spacing


def validate_npz_alignment(npz_data, sid, expected_zs, expected_hw, required_key="logits_mask"):
    shape = tuple(int(v) for v in npz_data["shape"])
    exp_shape = (len(expected_zs), len(ORGANS), int(expected_hw[0]), int(expected_hw[1]))
    if shape != exp_shape:
        raise ValueError(f"{sid}: prediction shape {shape} != expected {exp_shape}")
    if list(npz_data["z"]) != list(expected_zs):
        raise ValueError(f"{sid}: prediction z-indices do not match SliceBank")
    if required_key not in npz_data:
        raise KeyError(f"{sid}: missing array key '{required_key}' in prediction npz")
    arr = npz_data[required_key]
    exp_packed = (*exp_shape[:-1], (exp_shape[-1] + 7) // 8)
    if arr.dtype != np.uint8 or arr.shape != exp_packed:
        raise ValueError(
            f"{sid}: packed bitfield '{required_key}' has shape {arr.shape} / dtype {arr.dtype}, expected {exp_packed} uint8"
        )
    pres = npz_data["presence"].astype(np.float32)
    if pres.shape != (len(expected_zs), len(ORGANS)) or not np.isfinite(pres).all():
        raise ValueError(f"{sid}: invalid or nonfinite presence array shape {pres.shape}")
    return shape


def _unpack(npz_data, key, shape):
    return np.unpackbits(npz_data[key], axis=-1, count=shape[-1]).astype(bool).reshape(shape)


def _score_masks_for_subject(sid, masks, rows, bank, z_sp):
    spacing = bank.slices[rows[0]]["spacing_yx_mm"]
    s_recs, v_recs = [], []
    for o, organ in enumerate(ORGANS):
        truth = np.stack([bank.mask(i, organ) for i in rows])
        for j, i in enumerate(rows):
            rec = score_slice(masks[j, o], truth[j], spacing)
            rec.update(subject_id=sid, target=organ, slice_index=bank.slices[i]["z"])
            rec["false_positive_area_mm2"] = (
                rec["fp"] * spacing[0] * spacing[1] if rec["truth_empty"] else None
            )
            s_recs.append(rec)
        vol = score_volume(masks[:, o], truth, [z_sp, *spacing])
        vol.update(subject_id=sid, target=organ)
        v_recs.append(vol)
    return s_recs, v_recs


def _worker_single_dir(job):
    sid, pred_dir, data_dir, split, threshold, z_sp, key, rows = job
    bank = _get_bank(data_dir, split)
    expected_zs = [bank.slices[i]["z"] for i in rows]
    expected_hw = (bank.slices[rows[0]]["h"], bank.slices[rows[0]]["w"])
    data = np.load(Path(pred_dir) / f"{sid}.npz")
    shape = validate_npz_alignment(data, sid, expected_zs, expected_hw, required_key=key)
    masks = _unpack(data, key, shape)
    if threshold is not None:
        masks &= (data["presence"].astype(np.float32) >= threshold)[..., None, None]
    return _score_masks_for_subject(sid, masks, rows, bank, z_sp)


def _worker_single_slice_multiround(job):
    sid, data_dir, split, init2d_dir, zoom_dir, z_sp, rows = job
    bank = _get_bank(data_dir, split)
    expected_zs = [bank.slices[i]["z"] for i in rows]
    expected_hw = (bank.slices[rows[0]]["h"], bank.slices[rows[0]]["w"])

    d_init = np.load(Path(init2d_dir) / f"{sid}.npz")
    sh = validate_npz_alignment(d_init, sid, expected_zs, expected_hw, required_key="logits_mask")
    raw_m = _unpack(d_init, "logits_mask", sh)
    prob = d_init["presence"].astype(np.float32)

    d_zm = np.load(Path(zoom_dir) / f"{sid}.npz")
    validate_npz_alignment(d_zm, sid, expected_zs, expected_hw, required_key="final_mask")
    ag_m = _unpack(d_zm, "final_mask", sh)

    masks = apply_slice_continuity(ag_m, raw_m, prob, p_cont=0.25, p_iso=0.62, cc_3d=True)
    return _score_masks_for_subject(sid, masks, rows, bank, z_sp)


def _worker_standalone(job):
    sid, data_dir, split, round0_dir, zoom_dir, z_sp, rows = job
    bank = _get_bank(data_dir, split)
    expected_zs = [bank.slices[i]["z"] for i in rows]
    expected_hw = (bank.slices[rows[0]]["h"], bank.slices[rows[0]]["w"])

    d_r0 = np.load(Path(round0_dir) / f"{sid}.npz")
    sh = validate_npz_alignment(d_r0, sid, expected_zs, expected_hw, required_key="logits_mask")
    raw_r0 = _unpack(d_r0, "logits_mask", sh)
    prob_r0 = d_r0["presence"].astype(np.float32)

    d_zm = np.load(Path(zoom_dir) / f"{sid}.npz")
    validate_npz_alignment(d_zm, sid, expected_zs, expected_hw, required_key="final_mask")
    ag_zm = _unpack(d_zm, "final_mask", sh)
    prob_zm = d_zm["presence"].astype(np.float32)

    prob_ens = 0.65 * prob_r0 + 0.35 * prob_zm
    base_m = (raw_r0 & (prob_r0 >= 0.5)[..., None, None]) | (ag_zm & raw_r0)
    raw_union = raw_r0 | ag_zm
    masks = apply_slice_continuity(
        base_m, raw_union, prob_ens, p_cont=0.25, p_iso=0.62, cc_3d=True
    )
    return _score_masks_for_subject(sid, masks, rows, bank, z_sp)


def _worker_refiner(job):
    sid, data_dir, split, sp_dir, ag_dir, round0_dir, z_sp, rows = job
    bank = _get_bank(data_dir, split)
    expected_zs = [bank.slices[i]["z"] for i in rows]
    expected_hw = (bank.slices[rows[0]]["h"], bank.slices[rows[0]]["w"])

    d_sp = np.load(Path(sp_dir) / f"{sid}.npz")
    sh = validate_npz_alignment(d_sp, sid, expected_zs, expected_hw, required_key="logits_mask")
    m_sp = _unpack(d_sp, "logits_mask", sh) & (
        d_sp["presence"].astype(np.float32)[..., None, None] >= 0.5
    )

    d_ag = np.load(Path(ag_dir) / f"{sid}.npz")
    validate_npz_alignment(d_ag, sid, expected_zs, expected_hw, required_key="final_mask")
    m_ag = _unpack(d_ag, "final_mask", sh)

    d_r0 = np.load(Path(round0_dir) / f"{sid}.npz")
    validate_npz_alignment(d_r0, sid, expected_zs, expected_hw, required_key="logits_mask")
    p_r0 = d_r0["presence"].astype(np.float32)
    m_r0 = _unpack(d_r0, "logits_mask", sh)

    diff = m_ag != m_sp
    agree = (m_r0 == m_ag) & ((p_r0[..., None, None] >= 0.5) | (~m_ag))
    masks = np.where(diff & agree, m_ag, m_sp)
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
    masks = masks & (~iso_fp[..., None, None])
    return _score_masks_for_subject(sid, masks, rows, bank, z_sp)


def _finalize_and_save(slice_records, volume_records, subjects, split, output_path, extra_meta=None):
    slice_agg = aggregate_slices(slice_records)
    volume_agg = aggregate_volumes(volume_records, subjects, ORGANS)
    summary = build_summary(slice_agg, volume_agg, slice_records)
    summary.update(split=split, subjects=len(subjects))
    if extra_meta:
        summary.update(extra_meta)
    print(json.dumps(summary, indent=2), flush=True)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps(
                {
                    "summary": summary,
                    "slice_aggregate": slice_agg,
                    "volume_aggregate": volume_agg,
                }
            )
        )
    return summary


def convert_nifti_predictions(
    nifti_dir, ct_root, data_dir, split, output_dir, allow_missing_organs=False
):
    """Convert per-subject NIfTI masks (<nifti_dir>/<sid>/<organ>.nii.gz) into NativeSeg .npz files."""
    bank = SliceBank(data_dir, split)
    by_sub = {}
    for i, r in enumerate(bank.slices):
        by_sub.setdefault(r["subject_id"], []).append(i)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for sid, idxs in sorted(by_sub.items()):
        rows = sorted(idxs, key=lambda i: bank.slices[i]["z"])
        ct_path = Path(ct_root) / sid / "ct.nii.gz"
        vol = load_native_image(ct_path)
        h, w = bank.slices[rows[0]]["h"], bank.slices[rows[0]]["w"]
        if vol.data.shape != (w, h, len(rows)):
            raise ValueError(
                f"{sid}: CT NIfTI canonical RAS shape {vol.data.shape} does not match SliceBank (W={w}, H={h}, Z={len(rows)})"
            )
        masks = np.zeros((len(rows), len(ORGANS), h, w), bool)
        for o, org in enumerate(ORGANS):
            m_path = Path(nifti_dir) / sid / f"{org}.nii.gz"
            if not m_path.exists():
                if not allow_missing_organs:
                    raise FileNotFoundError(
                        f"Missing specialist mask {m_path} (pass --allow-missing-organs if intentional)"
                    )
                continue
            m_zyx = load_native_mask(m_path, vol).transpose(2, 1, 0)
            if m_zyx.shape != (len(rows), h, w):
                raise ValueError(f"{sid} {org}: unexpected mask shape {m_zyx.shape}")
            masks[:, o] = m_zyx
        presence = np.ones((len(rows), len(ORGANS)), np.float16)
        buf = io.BytesIO()
        np.savez(
            buf,
            logits_mask=np.packbits(masks, axis=-1),
            presence=presence,
            shape=np.array(masks.shape),
            z=np.array([bank.slices[i]["z"] for i in rows]),
        )
        (output_dir / f"{sid}.npz").write_bytes(buf.getvalue())
    print(json.dumps({"converted_subjects": len(by_sub), "output": str(output_dir)}))


def main():
    parser = argparse.ArgumentParser(description="Score NativeSeg volumetric predictions.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sc = sub.add_parser("score", help="Score a single prediction directory")
    p_sc.add_argument("--pred", required=True)
    p_sc.add_argument("--data", required=True)
    p_sc.add_argument("--split", default="val")
    p_sc.add_argument("--presence", type=float, default=None)
    p_sc.add_argument("--key", default="logits_mask")
    p_sc.add_argument("--subjects", default=None)
    p_sc.add_argument("--workers", type=int, default=16)
    p_sc.add_argument("--output", default=None)

    p_ss = sub.add_parser("single-slice", help="Score Single-Slice Multiround VLM (Table 1a)")
    p_ss.add_argument("--init2d", required=True, help="2D Round-0 prediction dir")
    p_ss.add_argument("--zoom", required=True, help="Verified multiround zoom prediction dir")
    p_ss.add_argument("--data", required=True)
    p_ss.add_argument("--split", required=True)
    p_ss.add_argument("--subjects", default=None)
    p_ss.add_argument("--workers", type=int, default=8)
    p_ss.add_argument("--output", default=None)

    p_st = sub.add_parser("standalone", help="Score NativeSeg 5-slice Round-0 + verified zoom")
    p_st.add_argument("--round0", required=True, help="5-slice Round-0 prediction dir")
    p_st.add_argument("--zoom", required=True, help="Verified multiround zoom prediction dir")
    p_st.add_argument("--data", required=True)
    p_st.add_argument("--split", required=True)
    p_st.add_argument("--subjects", default=None)
    p_st.add_argument("--workers", type=int, default=8)
    p_st.add_argument("--output", default=None)

    p_rf = sub.add_parser("refiner", help="Score Plug-and-Play specialist refinement with consensus")
    p_rf.add_argument("--specialist", default=None, help="Upstream specialist .npz dir (auto-read from --zoom/config.json if omitted)")
    p_rf.add_argument("--zoom", required=True, help="Verified zoom refinement prediction dir")
    p_rf.add_argument("--round0", required=True, help="5-slice Round-0 prediction dir for consensus")
    p_rf.add_argument("--data", required=True)
    p_rf.add_argument("--split", required=True)
    p_rf.add_argument("--subjects", default=None)
    p_rf.add_argument("--workers", type=int, default=8)
    p_rf.add_argument("--output", default=None)

    p_cv = sub.add_parser("convert-nifti", help="Convert external specialist NIfTI masks to NativeSeg .npz")
    p_cv.add_argument("--nifti-dir", required=True)
    p_cv.add_argument("--ct-root", required=True)
    p_cv.add_argument("--data", required=True)
    p_cv.add_argument("--split", required=True)
    p_cv.add_argument("--output", required=True)
    p_cv.add_argument("--allow-missing-organs", action="store_true")

    args = parser.parse_args()
    if args.cmd == "convert-nifti":
        convert_nifti_predictions(
            args.nifti_dir,
            args.ct_root,
            args.data,
            args.split,
            args.output,
            args.allow_missing_organs,
        )
        return

    _BANKS.clear()
    bank = _get_bank(args.data, args.split)
    by_sub = {}
    for i, r in enumerate(bank.slices):
        by_sub.setdefault(r["subject_id"], []).append(i)
    subjects = sorted(by_sub)
    if args.subjects is not None:
        wanted = {s.strip() for s in args.subjects.split(",") if s.strip()}
        unknown = sorted(wanted - set(subjects))
        if not wanted or unknown:
            raise ValueError(f"Unknown or empty --subjects for split '{args.split}': {unknown}")
        subjects = [s for s in subjects if s in wanted]
    z_spacing = _resolve_z_spacing(bank, subjects)

    if args.cmd == "score":
        jobs = [
            (
                s,
                args.pred,
                args.data,
                args.split,
                args.presence,
                z_spacing[s],
                args.key,
                sorted(by_sub[s], key=lambda i: bank.slices[i]["z"]),
            )
            for s in subjects
        ]
        worker_fn = _worker_single_dir
        meta = {"presence_threshold": args.presence, "key": args.key}
    elif args.cmd == "single-slice":
        jobs = [
            (
                s,
                args.data,
                args.split,
                args.init2d,
                args.zoom,
                z_spacing[s],
                sorted(by_sub[s], key=lambda i: bank.slices[i]["z"]),
            )
            for s in subjects
        ]
        worker_fn = _worker_single_slice_multiround
        meta = {"mode": "single-slice-multiround"}
    elif args.cmd == "standalone":
        jobs = [
            (
                s,
                args.data,
                args.split,
                args.round0,
                args.zoom,
                z_spacing[s],
                sorted(by_sub[s], key=lambda i: bank.slices[i]["z"]),
            )
            for s in subjects
        ]
        worker_fn = _worker_standalone
        meta = {"mode": "standalone"}
    elif args.cmd == "refiner":
        sp_dir = args.specialist
        if sp_dir is None:
            sp_dir = json.loads((Path(args.zoom) / "config.json").read_text())["initial_from"]
        jobs = [
            (
                s,
                args.data,
                args.split,
                sp_dir,
                args.zoom,
                args.round0,
                z_spacing[s],
                sorted(by_sub[s], key=lambda i: bank.slices[i]["z"]),
            )
            for s in subjects
        ]
        worker_fn = _worker_refiner
        meta = {"mode": "refiner", "specialist": str(sp_dir)}

    slice_records, volume_records = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        for sr, vr in pool.map(worker_fn, jobs):
            slice_records.extend(sr)
            volume_records.extend(vr)
    _finalize_and_save(slice_records, volume_records, subjects, args.split, args.output, meta)


if __name__ == "__main__":
    main()
