"""Box-prompted MedSAM2 specialist-refinement baseline (reproduces `+ MedSAM2` in Tables 1b, 2b, 4b).

Prompts official `MedSAM2` (`sam2.1_hiera_t512.yaml` + `MedSAM2_latest.pt`) on every non-empty
slice of an upstream specialist prediction (`2D nnU-Net` or `3D VoxTell`) using the tight bounding
box of the specialist's slice mask on the `[-160, 240] HU` grayscale CT image.

Requires the official `MedSAM2` (`sam2`) package installed in the Python environment.
"""

import argparse
import io
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import ORGANS, SliceBank
from .evaluate import validate_npz_alignment


def _tight_box(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Run box-prompted MedSAM2 slice refinement baseline.")
    parser.add_argument("--initial-from", required=True, help="Upstream specialist .npz directory")
    parser.add_argument("--data", required=True, help="NativeSeg flat_bank directory")
    parser.add_argument("--split", required=True, help="Benchmark split (val, test, pediatric)")
    parser.add_argument("--output", required=True, help="Output .npz directory")
    parser.add_argument("--medsam2-checkpoint", required=True, help="Path to MedSAM2_latest.pt")
    parser.add_argument("--medsam2-config", default="configs/sam2.1_hiera_t512.yaml")
    parser.add_argument("--presence", type=float, default=0.5)
    parser.add_argument("--subjects", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    torch.set_num_threads(args.threads)
    predictor = SAM2ImagePredictor(
        build_sam2(args.medsam2_config, str(Path(args.medsam2_checkpoint)), device=args.device)
    )
    bank = SliceBank(args.data, args.split)
    rows_of = {}
    for i, r in enumerate(bank.slices):
        rows_of.setdefault(r["subject_id"], []).append(i)
    subjects = sorted(rows_of)
    if args.subjects:
        wanted = {s.strip() for s in args.subjects.split(",") if s.strip()}
        unknown = sorted(wanted - set(subjects))
        if not wanted or unknown:
            raise ValueError(f"Unknown --subjects for split '{args.split}': {unknown}")
        subjects = [s for s in subjects if s in wanted]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for sid in subjects:
        target = output / f"{sid}.npz"
        if target.exists() and not args.overwrite:
            continue
        tick = time.perf_counter()
        rows = sorted(rows_of[sid], key=lambda i: bank.slices[i]["z"])
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

        refined = np.zeros_like(initial)
        prompts = 0
        for j, i in enumerate(rows):
            active_organs = [o for o in range(len(ORGANS)) if initial[j, o].any()]
            if not active_organs:
                continue
            grey = (np.clip(bank.image(i), 0.0, 1.0) * 255.0).astype(np.uint8)
            with torch.inference_mode():
                predictor.set_image(np.repeat(grey[..., None], 3, axis=-1))
                for o in active_organs:
                    masks, _, _ = predictor.predict(
                        box=_tight_box(initial[j, o]), multimask_output=False
                    )
                    refined[j, o] = masks[0] > 0
                    prompts += 1

        buf = io.BytesIO()
        np.savez(
            buf,
            logits_mask=np.packbits(refined, axis=-1),
            presence=np.ones(refined.shape[:2], np.float16),
            shape=np.array(refined.shape),
            z=np.asarray(exp_zs),
        )
        target.with_suffix(".tmp").write_bytes(buf.getvalue())
        target.with_suffix(".tmp").rename(target)
        print(
            json.dumps(
                {
                    "subject": sid,
                    "prompts": prompts,
                    "seconds": round(time.perf_counter() - tick, 1),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
