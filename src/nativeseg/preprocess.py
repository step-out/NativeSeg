"""Dataset preprocessing for TotalSegmentator (NIfTI / zip) and Pediatric-CT-SEG (DICOM+RTSTRUCT / NIfTI).

Produces per-patient canonical RAS slice caches (`slice_cache`) and packs them into flat memmaps (`flat_bank`):
  1. TotalSegmentator (1,081 train / 57 val / 65 test):
     nativeseg-preprocess totalseg --input /path/to/Totalsegmentator_dataset --cache data/cache_totalseg --flat data/flat_bank
  2. Pediatric-CT-SEG (21 pediatric abdominal CT scans from TCIA):
     nativeseg-preprocess pediatric-dicom --dicom-root /path/to/Pediatric-CT-SEG --cache data/cache_pediatric --flat data/flat_bank
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import io
import json
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np

from .data import ORGANS, pack
from .geometry import load_native_image, load_native_mask, sha256_file

PEDIATRIC_ORGAN_ROIS = {
    "spleen": ["Spleen"],
    "kidney_right": ["Kidney Right"],
    "kidney_left": ["Kidney Left"],
    "gallbladder": ["Gall Bladder", "Gallbladder"],
    "liver": ["Liver"],
    "stomach": ["Stomach"],
    "pancreas": ["Pancreas"],
}

PEDIATRIC_NIFTI_FILES = {
    "spleen": "spleen",
    "kidney_right": "right_kidney",
    "kidney_left": "left_kidney",
    "gallbladder": "gallbladder",
    "liver": "liver",
    "stomach": "stomach",
    "pancreas": "pancreas",
}


from importlib import resources as pkg_resources


def _default_splits_config():
    try:
        res_file = pkg_resources.files("nativeseg.resources").joinpath("totalseg_splits.json")
        if res_file.is_file():
            return json.loads(res_file.read_text(encoding="utf-8"))
    except Exception:
        pass
    here = Path(__file__).resolve().parents[2] / "configs" / "totalseg_splits.json"
    if here.exists():
        return json.loads(here.read_text(encoding="utf-8"))
    return None


def _process_totalseg_folder(sid, split, case_dir, out_dir):
    t0 = time.perf_counter()
    volume = load_native_image(Path(case_dir) / "ct.nii.gz")
    if not np.array_equal(volume.data, np.round(volume.data)):
        raise ValueError(f"{sid}: non-integer HU values in source CT")
    ct = np.clip(volume.data, -160, 240).astype(np.int16).transpose(2, 1, 0)
    lab = np.zeros(ct.shape, np.uint8)
    for bit, organ in enumerate(ORGANS):
        mask_path = Path(case_dir) / "segmentations" / f"{organ}.nii.gz"
        if not mask_path.exists():
            raise FileNotFoundError(f"{sid}: missing organ mask {mask_path}")
        mask = load_native_mask(mask_path, volume)
        lab |= mask.transpose(2, 1, 0).astype(np.uint8) << bit
    sx, sy, sz = volume.spacing
    buf = io.BytesIO()
    np.savez(buf, ct=np.ascontiguousarray(ct), lab=np.ascontiguousarray(lab))
    out_path = Path(out_dir) / f"{sid}.npz"
    out_path.write_bytes(buf.getvalue())
    present = [
        [bool(((lab[z] >> b) & 1).any()) for b in range(len(ORGANS))]
        for z in range(len(lab))
    ]
    return {
        "subject_id": sid,
        "split": split,
        "slices": int(ct.shape[0]),
        "hw": [int(ct.shape[1]), int(ct.shape[2])],
        "spacing_yx_mm": [float(sy), float(sx)],
        "spacing_z_mm": float(sz),
        "present": present,
        "file": out_path.name,
        "sha256": sha256_file(out_path),
        "seconds": round(time.perf_counter() - t0, 2),
    }


def _totalseg_worker(job):
    sid, split, src_path, is_zip, out_dir = job
    if not is_zip:
        case_dir = Path(src_path) / sid
        if not (case_dir / "ct.nii.gz").exists():
            raise FileNotFoundError(f"Missing CT volume for expected patient {sid}: {case_dir / 'ct.nii.gz'}")
        return _process_totalseg_folder(sid, split, case_dir, out_dir)
    with zipfile.ZipFile(src_path) as archive, tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / sid
        (folder / "segmentations").mkdir(parents=True)
        prefix = f"Totalsegmentator_dataset/{sid}"
        names = set(archive.namelist())
        if f"{prefix}/ct.nii.gz" not in names:
            prefix = sid
        for rel in ["ct.nii.gz"] + [f"segmentations/{o}.nii.gz" for o in ORGANS]:
            (folder / rel).write_bytes(archive.read(f"{prefix}/{rel}"))
        return _process_totalseg_folder(sid, split, folder, out_dir)


def build_totalseg(
    input_path,
    cache_dir,
    flat_dir=None,
    splits="train,val,test",
    workers=16,
    allow_partial=False,
):
    input_path = Path(input_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    wanted_splits = [s.strip() for s in splits.split(",") if s.strip()]
    is_zip = input_path.is_file() and input_path.suffix == ".zip"

    split_map = {}
    cfg = _default_splits_config()
    if cfg is not None:
        for sp, sids in cfg["splits"].items():
            for sid in sids:
                split_map[sid] = sp
    elif is_zip:
        with zipfile.ZipFile(input_path) as archive:
            meta_candidates = [n for n in archive.namelist() if n.endswith("meta.csv")]
            if not meta_candidates:
                raise FileNotFoundError("Could not find meta.csv inside zip archive")
            text_lines = archive.read(meta_candidates[0]).decode("utf-8").splitlines()
            for row in csv.DictReader(text_lines, delimiter=";"):
                split_map[row["image_id"]] = row["split"]
    elif (input_path / "meta.csv").exists():
        with (input_path / "meta.csv").open() as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                split_map[row["image_id"]] = row["split"]
    else:
        raise FileNotFoundError(
            "Could not find configs/totalseg_splits.json or meta.csv in input_path"
        )

    expected = [(sid, sp) for sid, sp in sorted(split_map.items()) if sp in wanted_splits]
    if allow_partial and not is_zip:
        expected = [
            (sid, sp) for sid, sp in expected if (input_path / sid / "ct.nii.gz").exists()
        ]
    jobs = [(sid, sp, str(input_path), is_zip, str(cache_dir)) for sid, sp in expected]
    records = []
    with ProcessPoolExecutor(workers) as pool:
        for i, rec in enumerate(pool.map(_totalseg_worker, jobs), 1):
            records.append(rec)
            if i % 50 == 0 or i == len(jobs):
                print(json.dumps({"processed": i, "of": len(jobs)}), flush=True)
    index = {
        "version": "nativeseg_slice_cache",
        "organs": list(ORGANS),
        "window_hu": [-160, 240],
        "encoding": "ct int16 HU clipped to [-160, 240], image = (ct + 160) / 400; lab bit i = organs[i]",
        "slice_convention": "canonical RAS volume, slice = data[:, :, z].T (Y, X)",
        "patients": records,
    }
    (cache_dir / "index.json").write_text(json.dumps(index))
    if flat_dir:
        for sp in sorted({r["split"] for r in records}):
            print(json.dumps(pack(cache_dir, flat_dir, sp)), flush=True)


def _referenced_sop_uids(rt_ds):
    uids = set()
    for r_f in getattr(rt_ds, "ReferencedFrameOfReferenceSequence", []):
        for st in getattr(r_f, "RTReferencedStudySequence", []):
            for sr in getattr(st, "RTReferencedSeriesSequence", []):
                for im in getattr(sr, "ContourImageSequence", []):
                    uids.add(str(im.ReferencedSOPInstanceUID))
    for rc in getattr(rt_ds, "ROIContourSequence", []):
        for c in getattr(rc, "ContourSequence", []):
            for im in getattr(c, "ContourImageSequence", []):
                uids.add(str(im.ReferencedSOPInstanceUID))
    return uids


def _rasterize_pediatric_dicom_subject(sid, subject_dir, cache_dir, allow_nonuniform_z=False):
    import pydicom
    from skimage.draw import polygon

    dcm_files = sorted(Path(subject_dir).rglob("*.dcm"))
    ct_by_series = {}
    rt_candidates = []
    for p in dcm_files:
        ds = pydicom.dcmread(p)
        mod = getattr(ds, "Modality", "")
        if mod == "CT":
            ct_by_series.setdefault(str(ds.SeriesInstanceUID), []).append(ds)
        elif mod == "RTSTRUCT":
            rt_candidates.append(ds)
    if not ct_by_series or not rt_candidates:
        raise ValueError(f"{sid}: missing CT or RTSTRUCT DICOM files in {subject_dir}")
    if len(rt_candidates) > 1:
        raise ValueError(f"{sid}: multiple RTSTRUCT files found ({len(rt_candidates)}); disambiguate input folder")
    rt_ds = rt_candidates[0]
    ref_uids = _referenced_sop_uids(rt_ds)

    matched_series = [
        (ser_uid, slices)
        for ser_uid, slices in ct_by_series.items()
        if ref_uids & {str(d.SOPInstanceUID) for d in slices}
    ]
    if len(matched_series) != 1:
        raise ValueError(
            f"{sid}: expected exactly 1 CT series referenced by RTSTRUCT, found {len(matched_series)}"
        )
    ser_uid, ct_ds = matched_series[0]
    if len(ct_ds) < 2:
        raise ValueError(f"{sid}: CT series {ser_uid} has fewer than 2 slices")

    orientation = np.asarray(ct_ds[0].ImageOrientationPatient, float)
    if orientation.shape != (6,) or not np.isfinite(orientation).all():
        raise ValueError(f"{sid}: invalid ImageOrientationPatient")
    col, row = orientation[:3], orientation[3:]
    normal = np.cross(col, row)
    if not np.allclose(normal, [0, 0, 1], atol=1e-3) or not np.allclose(
        col, [1, 0, 0], atol=1e-3
    ):
        raise ValueError(f"{sid}: expected standard axial LPS DICOM orientation, got {orientation.tolist()}")

    rows, cols = int(ct_ds[0].Rows), int(ct_ds[0].Columns)
    spacing = np.asarray(ct_ds[0].PixelSpacing, float)
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError(f"{sid}: invalid PixelSpacing {spacing}")
    pos0 = np.asarray(ct_ds[0].ImagePositionPatient, float)
    inplane0 = np.array([float(pos0 @ col), float(pos0 @ row)])
    for d in ct_ds:
        if (int(d.Rows), int(d.Columns)) != (rows, cols):
            raise ValueError(f"{sid}: inconsistent slice dimensions inside series {ser_uid}")
        if not np.allclose(np.asarray(d.ImageOrientationPatient, float), orientation, atol=1e-4):
            raise ValueError(f"{sid}: inconsistent ImageOrientationPatient inside series {ser_uid}")
        if not np.allclose(np.asarray(d.PixelSpacing, float), spacing, atol=1e-4):
            raise ValueError(f"{sid}: inconsistent PixelSpacing inside series {ser_uid}")
        pos_d = np.asarray(d.ImagePositionPatient, float)
        if pos_d.shape != (3,) or not np.isfinite(pos_d).all():
            raise ValueError(f"{sid}: nonfinite ImagePositionPatient")
        inplane_d = np.array([float(pos_d @ col), float(pos_d @ row)])
        if not np.allclose(inplane_d, inplane0, atol=1e-2):
            raise ValueError(f"{sid}: sheared or drifting in-plane origin across slices")

    ct_ds.sort(key=lambda d: float(np.asarray(d.ImagePositionPatient, float) @ normal))
    positions = [float(np.asarray(d.ImagePositionPatient, float) @ normal) for d in ct_ds]
    gaps = np.diff(positions)
    if (gaps <= 0).any():
        raise ValueError(f"{sid}: duplicate or non-monotonic slice positions in series {ser_uid}")
    if not allow_nonuniform_z and not np.allclose(gaps, gaps[0], atol=0.05):
        raise ValueError(
            f"{sid}: non-uniform axial slice spacing {gaps.min():.3f}..{gaps.max():.3f} mm "
            "(omit --strict-uniform-z for TCIA cases such as Pediatric-CT-SEG-377)"
        )
    by_uid = {str(d.SOPInstanceUID): i for i, d in enumerate(ct_ds)}
    if len(by_uid) != len(ct_ds):
        raise ValueError(f"{sid}: duplicate SOPInstanceUID in series {ser_uid}")
    missing_refs = ref_uids - set(by_uid)
    if missing_refs:
        raise ValueError(
            f"{sid}: RTSTRUCT references {len(missing_refs)} SOPInstanceUIDs absent from CT series {ser_uid}"
        )

    hu = np.stack(
        [
            d.pixel_array.astype(np.float64) * float(d.RescaleSlope)
            + float(d.RescaleIntercept)
            for d in ct_ds
        ]
    )
    if not np.array_equal(hu, np.round(hu)):
        raise ValueError(f"{sid}: non-integer HU values in DICOM series")

    numbers = {str(r.ROIName): int(r.ROINumber) for r in rt_ds.StructureSetROISequence}
    contours = {int(r.ReferencedROINumber): r for r in rt_ds.ROIContourSequence}
    lab = np.zeros((len(ct_ds), rows, cols), np.uint8)
    for bit, organ in enumerate(ORGANS):
        total = np.zeros((len(ct_ds), rows, cols), bool)
        for roi in PEDIATRIC_ORGAN_ROIS[organ]:
            if roi not in numbers or numbers[roi] not in contours:
                continue
            seq = getattr(contours[numbers[roi]], "ContourSequence", [])
            mask = np.zeros_like(total)
            for c in seq:
                uid = str(c.ContourImageSequence[0].ReferencedSOPInstanceUID)
                if uid not in by_uid:
                    raise ValueError(f"{sid}: contour for {roi} references unknown SOPInstanceUID {uid}")
                z = by_uid[uid]
                h = ct_ds[z]
                points = np.asarray(c.ContourData, float).reshape(-1, 3)
                delta = points - np.asarray(h.ImagePositionPatient, float)
                xx, yy = delta @ col / spacing[1], delta @ row / spacing[0]
                rr, cc = polygon(yy, xx, (rows, cols))
                mask[z, rr, cc] ^= True
            total |= mask
        lab |= total.astype(np.uint8) << bit

    # Validated axial LPS (row, col) -> canonical RAS (Y, X) via in-plane axis flip
    ct = np.clip(hu, -160, 240).astype(np.int16)[:, ::-1, ::-1]
    lab = lab[:, ::-1, ::-1]
    buf = io.BytesIO()
    np.savez(buf, ct=np.ascontiguousarray(ct), lab=np.ascontiguousarray(lab))
    out_path = Path(cache_dir) / f"{sid}.npz"
    out_path.write_bytes(buf.getvalue())
    return {
        "subject_id": sid,
        "series_uid": ser_uid,
        "split": "pediatric",
        "slices": int(ct.shape[0]),
        "hw": [int(ct.shape[1]), int(ct.shape[2])],
        "spacing_yx_mm": [float(spacing[0]), float(spacing[1])],
        "spacing_z_mm": float(np.median(gaps)),
        "slice_gap_range_mm": [float(gaps.min()), float(gaps.max())],
        "present": [
            [bool(((lab[z] >> b) & 1).any()) for b in range(len(ORGANS))]
            for z in range(len(lab))
        ],
        "file": out_path.name,
        "sha256": sha256_file(out_path),
    }


def build_pediatric_dicom(dicom_root, cache_dir, flat_dir=None, subjects=None, allow_nonuniform_z=True):
    dicom_root, cache_dir = Path(dicom_root), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg = _default_splits_config()
    available = sorted(d.name for d in dicom_root.iterdir() if d.is_dir())
    if subjects is not None:
        wanted = [s.strip() for s in subjects.split(",") if s.strip()]
        unknown = sorted(set(wanted) - set(available))
        if not wanted or unknown:
            raise ValueError(f"Unknown or empty --subjects in {dicom_root}: {unknown}")
    else:
        wanted = cfg["pediatric_subjects"] if cfg is not None else [d for d in available if d.startswith("Pediatric-CT-SEG-")]
        missing_def = [s for s in wanted if s not in set(available)]
        if missing_def:
            raise ValueError(f"Missing expected Pediatric-CT-SEG subjects in {dicom_root}: {missing_def}")
    records = []
    for sid in wanted:
        sub_dir = dicom_root / sid
        if not sub_dir.exists():
            raise FileNotFoundError(f"Missing Pediatric-CT-SEG subject folder: {sub_dir}")
        rec = _rasterize_pediatric_dicom_subject(
            sid, sub_dir, cache_dir, allow_nonuniform_z=allow_nonuniform_z
        )
        records.append(rec)
        print(
            json.dumps({"subject": sid, "slices": rec["slices"], "hw": rec["hw"]}),
            flush=True,
        )
    index = {
        "version": "nativeseg_slice_cache",
        "organs": list(ORGANS),
        "window_hu": [-160, 240],
        "encoding": "ct int16 HU clipped to [-160, 240], image = (ct + 160) / 400; lab bit i = organs[i]",
        "slice_convention": "canonical RAS volume, slice = data[:, :, z].T (Y, X)",
        "patients": records,
    }
    (cache_dir / "index.json").write_text(json.dumps(index))
    if flat_dir:
        print(json.dumps(pack(cache_dir, flat_dir, "pediatric")), flush=True)


def build_pediatric_nifti(nifti_root, cache_dir, flat_dir=None):
    nifti_root, cache_dir = Path(nifti_root), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for folder in sorted(d for d in nifti_root.iterdir() if d.is_dir()):
        sid = folder.name
        ct_file = (
            folder / "ct.nii.gz"
            if (folder / "ct.nii.gz").exists()
            else folder / "vendor" / "image.nii.gz"
        )
        volume = load_native_image(ct_file)
        if not np.array_equal(volume.data, np.round(volume.data)):
            raise ValueError(f"{sid}: non-integer HU values in NIfTI CT")
        ct = np.clip(volume.data, -160, 240).astype(np.int16).transpose(2, 1, 0)
        lab = np.zeros(ct.shape, np.uint8)
        for bit, organ in enumerate(ORGANS):
            cand1 = folder / "segmentations" / f"{organ}.nii.gz"
            cand2 = folder / f"{PEDIATRIC_NIFTI_FILES[organ]}.nii.gz"
            mask_path = cand1 if cand1.exists() else cand2
            if not mask_path.exists():
                raise FileNotFoundError(f"{sid}: missing organ mask {mask_path}")
            mask = load_native_mask(mask_path, volume)
            lab |= mask.transpose(2, 1, 0).astype(np.uint8) << bit
        sx, sy, sz = volume.spacing
        buf = io.BytesIO()
        np.savez(buf, ct=np.ascontiguousarray(ct), lab=np.ascontiguousarray(lab))
        out_path = cache_dir / f"{sid}.npz"
        out_path.write_bytes(buf.getvalue())
        records.append(
            {
                "subject_id": sid,
                "split": "pediatric",
                "slices": int(ct.shape[0]),
                "hw": [int(ct.shape[1]), int(ct.shape[2])],
                "spacing_yx_mm": [float(sy), float(sx)],
                "spacing_z_mm": float(sz),
                "present": [
                    [bool(((lab[z] >> b) & 1).any()) for b in range(len(ORGANS))]
                    for z in range(len(lab))
                ],
                "file": out_path.name,
                "sha256": sha256_file(out_path),
            }
        )
    index = {
        "version": "nativeseg_slice_cache",
        "organs": list(ORGANS),
        "window_hu": [-160, 240],
        "encoding": "ct int16 HU clipped to [-160, 240], image = (ct + 160) / 400; lab bit i = organs[i]",
        "slice_convention": "canonical RAS volume, slice = data[:, :, z].T (Y, X)",
        "patients": records,
    }
    (cache_dir / "index.json").write_text(json.dumps(index))
    if flat_dir:
        print(json.dumps(pack(cache_dir, flat_dir, "pediatric")), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Preprocess CT benchmarks into NativeSeg slice banks.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ts = sub.add_parser("totalseg", help="Build TotalSegmentator cache and flat memmaps")
    p_ts.add_argument("--input", required=True, help="Totalsegmentator_dataset folder or .zip")
    p_ts.add_argument("--cache", required=True)
    p_ts.add_argument("--flat", default=None)
    p_ts.add_argument("--splits", default="train,val,test")
    p_ts.add_argument("--workers", type=int, default=16)
    p_ts.add_argument("--allow-partial", action="store_true", help="Allow subset folders during development")

    p_pd = sub.add_parser("pediatric-dicom", help="Build Pediatric-CT-SEG from TCIA DICOM + RTSTRUCT")
    p_pd.add_argument("--dicom-root", required=True)
    p_pd.add_argument("--cache", required=True)
    p_pd.add_argument("--flat", default=None)
    p_pd.add_argument("--subjects", default=None)
    p_pd.add_argument("--strict-uniform-z", action="store_true", help="Reject variable-gap DICOM series such as Pediatric-CT-SEG-377")

    p_pn = sub.add_parser("pediatric-nifti", help="Build Pediatric-CT-SEG from converted NIfTI folders")
    p_pn.add_argument("--nifti-root", required=True)
    p_pn.add_argument("--cache", required=True)
    p_pn.add_argument("--flat", default=None)

    p_pk = sub.add_parser("pack", help="Pack an existing slice cache directory into flat memmaps")
    p_pk.add_argument("--cache", required=True)
    p_pk.add_argument("--flat", required=True)
    p_pk.add_argument("--splits", default="train,val")

    args = parser.parse_args()
    if args.cmd == "totalseg":
        build_totalseg(args.input, args.cache, args.flat, args.splits, args.workers, args.allow_partial)
    elif args.cmd == "pediatric-dicom":
        build_pediatric_dicom(
            args.dicom_root, args.cache, args.flat, args.subjects, allow_nonuniform_z=(not args.strict_uniform_z)
        )
    elif args.cmd == "pediatric-nifti":
        build_pediatric_nifti(args.nifti_root, args.cache, args.flat)
    elif args.cmd == "pack":
        for sp in args.splits.split(","):
            print(json.dumps(pack(args.cache, args.flat, sp)), flush=True)


if __name__ == "__main__":
    main()
