# NativeSeg: Native Vision Language Segmentation of 3D Computed Tomography via Spatial Visual Tokens and Verified Local Zoom

Official PyTorch implementation and pretrained checkpoints for **NativeSeg**.

**Authors:** Chengqian Ma, Gaole Dai, Zijie Meng, Weixuan Jin, Yijie Du, Yuexiao Ma, Xiawu Zheng, Fei Chao  
**Affiliations:** Peking University, Tsinghua University, Institute of Automation (CAS), Nanyang Technological University, Xiamen University

- **Code Repository:** [https://github.com/step-out/NativeSeg](https://github.com/step-out/NativeSeg)
- **Pretrained Weights:** [https://huggingface.co/ChengqianMa/NativeSeg](https://huggingface.co/ChengqianMa/NativeSeg)

---

## Highlights

- **Native 2D Spatial Token Decoding without External SAM Heads:** Preserves upper-layer `16 x 16` 2D spatial visual tokens and fuses them with `32 x 32` vision patch features inside a five-slice residual FiLM U-Net decoder on LoRA-adapted `Qwen3.8-27B` (`56.00M` trainable parameters per five-slice checkpoint: `29.18M` rank-16 LoRA + `25.37M` five-slice U-Net + `1.44M` spatial query offsets & linear gate heads; `34.65M` per single-slice checkpoint), avoiding external SAM mask decoders and 1D `[SEG]` bottleneck compression (`+19.81` and `+7.04` Round-0 DSC over discrete codebook and 1D `[SEG]` readouts on the identical backbone).
- **Native Presence & Multiscale Error Gates (`<pres>` and `<err>`):** Screens out target-absent CT slices at Round 0 and estimates misclassified canvas area across whole-slice, `2 x 2`, and `4 x 4` regions directly from visual mask overlays without ground truth (`AUROC = 0.957` on own Round-0 masks and `0.942` zero-shot on unseen `2D nnU-Net` masks; `0.88` worst-quadrant hit rate vs. `0.25` chance), halting at 0 executed zooms on `91.8% to 93.0%` of full-volume standalone evaluations (`0.14 to 0.16` mean zoom rounds/slice; `0.17` on the 6-patient validation sweep) and on `93.8% to 96.5%` of specialist refinement evaluations.
- **Standalone & Plug-and-Play Specialist Refiner Modes:**
  1. **Modular Stage-Decoupled Standalone (`benchmark-standalone`):** Sequentially loads `nativeseg_stage_a` (`56.00M`), `nativeseg_single_slice` (`34.65M`), and `nativeseg_local_refiner` (`34.65M`) (`125.30M` combined trainable state on the shared frozen `Qwen3.8-27B` backbone) with 3D axial continuity and component filtering, reaching `86.81%` adult test DSC (`p = 2.3e-7`), `87.04%` zero-shot pediatric DSC (`p = 6.7e-6`), and `86.80%` adult validation DSC.
  2. **Single-Checkpoint Joint Stage B (`stage-b`):** End-to-end single-checkpoint `nativeseg_stage_b` (`56.00M` trainable parameters) uniting five-slice U-Net readout, freeform ROI feathering, continuous logit memory, and 3D axial continuity (`89.09%` DSC on the 6-patient validation ablation subset; `85.83%` adult test, `86.54%` zero-shot pediatric, and `85.44%` adult validation DSC).
  3. **Plug-and-Play Specialist Refiner (`refiner`):** Pairs `nativeseg_local_refiner` with `nativeseg_stage_a` cross-model consensus (`90.65M` combined NativeSeg trainable state; `tau = 0.10, K = 2`) to refine any upstream 2D or 3D segmenter without retraining, safely lifting **2D nnU-Net** (`89.52%` adult; `86.86%` pediatric, `p = 0.024`), **3D SegVol** (`86.81%` pediatric, `+1.90` DSC, `p = 9.5e-6`), and **3D VoxTell** (`89.85%` overall Pediatric-CT-SEG SOTA) where external refiners (`LRN†`, `MedSAM2`) degrade base masks.

---

## Installation

```bash
git clone https://github.com/step-out/NativeSeg.git
cd NativeSeg

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

> `requirements.txt` installs Google DeepMind's `surface-distance` library from GitHub for exact surfel-weighted 3D `NSD@3mm` and conditional bidirectional `HD95` (evaluated over nonempty pairs alongside reported `Miss / FP` slice counts). For bit-exact reproduction, pin your local or Hugging Face `Qwen/Qwen3.8-27B` base model snapshot consistently across stages.

Verify your installation, geometry/continuity routines, and all 7 checkpoint tensor schemas on CPU:

```bash
python -m nativeseg.smoke_test --model-dir Qwen/Qwen3.8-27B --weights-dir checkpoints
```

---

## Pretrained Checkpoints

Download all 7 released checkpoints (`950.75 MiB` / `996.93 MB` total in float32) and verify SHA256 checksums:

```bash
python scripts/download_weights.py --repo-id ChengqianMa/NativeSeg --output checkpoints
```

| Checkpoint | Architecture | Trainable Params | Size (`MiB`) | SHA256 (prefix) | Exact Paper Rows Reproduced |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `nativeseg_stage_a` | `five_slice` | `56.00M` | 213.75 MiB | `a36729a21fbe072a` | `NativeSeg Round 0` (`86.01%` Test / `85.99%` Ped / `85.65%` Val / `88.72%` Val6) & consensus anchor |
| `nativeseg_stage_b` | `five_slice` | `56.00M` | 213.75 MiB | `5a63b7b06baaacd9` | `NativeSeg (+ Joint Stage B)` (`89.09%` Val6 in Table 3b; `85.83%` Test / `86.54%` Ped / `85.44%` Val) |
| `nativeseg_local_refiner` | `single_slice` | `34.65M` | 132.28 MiB | `28813c0a7cb0cb80` | Verified multiscale zoom & `<err>` gate (`AUROC = 0.957` own / `0.942` nnU-Net; standalone & specialist refiner) |
| `nativeseg_single_slice` | `single_slice` | `34.65M` | 132.28 MiB | `8bb9c3c668d6d97c` | `F-LMM†` (`84.83%` Test / `84.97%` Ped / `83.98%` Val / `87.73%` Val6) & stage-decoupled zoom init |
| `baselines/lisa_seg_token` | `single_slice` | `34.65M` | 132.28 MiB | `433e6c9ed26e6e77` | `LISA†` controlled 1D `[SEG]` broadcast baseline (`80.25%` Test / `78.76%` Ped / `78.66%` Val / `81.68%` Val6) |
| `baselines/densemllm_spatial_tokens` | `single_slice` | `34.65M` | 132.28 MiB | `7fd21e8a82db1080` | `DenseMLLM†` controlled 2D spatial-tokens baseline (`82.22%` Test / `81.34%` Ped / `80.59%` Val / `83.73%` Val6) |
| `baselines/lrn_cnn_editor` | `CNNEditor` | `10.05M` | 38.38 MiB | `31667710b2fbc145` | `LRN†` perturbation-trained CNN refiner baseline (`+ LRN†` rows in Tables 1b, 2b, 4b) |

> **Scope Note on External & Two-Stage Baselines:** All NativeSeg rows, controlled single/multi-pass Qwen3.8-27B baselines (`LISA†`, `DenseMLLM†`, `F-LMM†`), `LRN†` (`nativeseg-lrn`), `MedSAM2` (`nativeseg-medsam2`, using official `MedSAM2_latest.pt`), and all 10/10 rows of Table 3(b) + 4/5 rows of Table 3(a) are reproduced directly by the scripts below. Table 3(a) Row 1 (`Discrete Codebook / MedSIGHT†`, `68.91%`) uses a separate two-stage `512`-entry VQ `RegionCodec` + autoregressive token model outside the continuous `NativeSeg` decoder architecture, and upstream specialist masks (`2D nnU-Net`, `3D SegVol`, `3D VoxTell`) are imported from standard NIfTI predictions via `nativeseg-eval convert-nifti`.

---

## Quickstart: Direct 3D NIfTI Inference (`.nii.gz`)

### 1a. Single-Checkpoint Standalone Inference (`nativeseg_stage_b`)

```bash
nativeseg-infer \
  --ct /path/to/ct.nii.gz \
  --mode stage-b \
  --model-dir Qwen/Qwen3.8-27B \
  --checkpoint checkpoints/nativeseg_stage_b \
  --output outputs/standalone_stage_b
```

### 1b. Benchmark-Exact Stage-Decoupled Standalone Inference (Reproducing Tables 1a, 2a, 4a)

```bash
nativeseg-infer \
  --ct /path/to/ct.nii.gz \
  --mode benchmark-standalone \
  --model-dir Qwen/Qwen3.8-27B \
  --round0-checkpoint checkpoints/nativeseg_stage_a \
  --init2d-checkpoint checkpoints/nativeseg_single_slice \
  --zoom-checkpoint checkpoints/nativeseg_local_refiner \
  --output outputs/standalone_benchmark
```

### 2. Plug-and-Play Specialist Refiner Mode (Reproducing Tables 1b, 2b, 4b)

```bash
nativeseg-infer \
  --ct /path/to/ct.nii.gz \
  --organ pancreas \
  --specialist-mask /path/to/upstream_pancreas.nii.gz \
  --model-dir Qwen/Qwen3.8-27B \
  --checkpoint checkpoints/nativeseg_local_refiner \
  --consensus-checkpoint checkpoints/nativeseg_stage_a \
  --output outputs/refined_masks
```

---

## Benchmark Reproduction (TotalSegmentator & Pediatric-CT-SEG)

### 1. Dataset Preparation

1. **TotalSegmentator** ([Zenodo Record 6802614](https://zenodo.org/records/6802614), `1,081` official training scans excluding corrupt `s0864`, `57` validation, `65` test):
   ```bash
   nativeseg-preprocess totalseg \
     --input /path/to/Totalsegmentator_dataset.zip \
     --cache data/cache_totalseg \
     --flat data/flat_bank \
     --splits train,val,test
   ```

2. **Pediatric-CT-SEG** ([TCIA Collection](https://www.cancerimagingarchive.net/collection/pediatric-ct-seg/), `21` external pediatric abdominal CT volumes; supports both raw TCIA DICOM+RTSTRUCT and converted NIfTI folders; variable-gap series `Pediatric-CT-SEG-377` is scored with its `2.0 mm` median slice spacing):
   ```bash
   nativeseg-preprocess pediatric-dicom \
     --dicom-root /path/to/Pediatric-CT-SEG \
     --cache data/cache_pediatric \
     --flat data/flat_bank
   ```

### 2. Authoritative Training Recipes

`scripts/reproduce_training.sh` preserves the exact per-stage distributed world sizes (`8 / 6 / 4 / 1` GPUs) from our original training runs when `>= 8` GPUs are provided (set `TRAIN_BASELINES=1` to also train `LISA†`, `DenseMLLM†`, and `LRN†`):

```bash
MODEL_DIR=Qwen/Qwen3.8-27B DATA_DIR=data/flat_bank OUT_DIR=runs GPUS=0,1,2,3,4,5,6,7 \
  bash scripts/reproduce_training.sh
```

### 3. Full Volumetric Evaluation, Specialist Refinement, and Ablations (Tables 1 to 4 & Figure 4)

```bash
# Reproduce Tables 1a, 2a, 4a + Single-Checkpoint Stage B across Test, Val, and Pediatric:
MODEL_DIR=Qwen/Qwen3.8-27B DATA_DIR=data/flat_bank CKPT_DIR=checkpoints PRED_DIR=predictions GPUS=0,1,2,3,4,5,6,7 \
  bash scripts/reproduce_evaluation.sh

# Reproduce Table 3(a, b) ablations and Figure 4 (<err> calibration + granularity sweep):
MODEL_DIR=Qwen/Qwen3.8-27B DATA_DIR=data/flat_bank CKPT_DIR=checkpoints PRED_DIR=predictions GPUS=0,1,2,3,4,5,6,7 \
  bash scripts/reproduce_ablations.sh

# Convert external specialist NIfTI masks (<subject_id>/<organ>.nii.gz) and run NativeSeg + LRN† (+ optional MedSAM2):
nativeseg-eval convert-nifti \
  --nifti-dir /path/to/voxtell_pediatric_nifti \
  --ct-root /path/to/pediatric_ct_nifti \
  --data data/flat_bank --split pediatric \
  --output predictions/voxtell_pediatric

SKIP_STANDALONE=1 SPECIALIST_NPZ_DIR=predictions/voxtell_pediatric SPECIALIST_SPLIT=pediatric \
  MODEL_DIR=Qwen/Qwen3.8-27B DATA_DIR=data/flat_bank CKPT_DIR=checkpoints PRED_DIR=predictions GPUS=0,1,2,3,4,5,6,7 \
  bash scripts/reproduce_evaluation.sh
```

### 4. Paired Patient-Level Statistical Tests

`nativeseg-stats` reports both the organ-first macro table estimand and paired two-sided Wilcoxon signed-rank `p`-values over target-present patients (`n = 56` on TotalSegmentator adult test; `n = 21` on Pediatric-CT-SEG):

```bash
nativeseg-stats \
  --a predictions/nativeseg_round0_test_score.json \
  --b predictions/nativeseg_standalone_test_score.json \
  --names Round0 NativeSeg
```

---

## Citation

```bibtex
@article{ma2026nativeseg,
  title={NativeSeg: Native Vision Language Segmentation of 3D Computed Tomography via Spatial Visual Tokens and Verified Local Zoom},
  author={Ma, Chengqian and Dai, Gaole and Meng, Zijie and Jin, Weixuan and Du, Yijie and Ma, Yuexiao and Zheng, Xiawu and Chao, Fei},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This project is released under the [Apache-2.0 License](LICENSE).
