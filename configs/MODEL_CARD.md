---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - medical-imaging
  - computed-tomography
  - 3d-segmentation
  - vision-language-model
  - lora
---

# NativeSeg Pretrained Checkpoints

Official pretrained LoRA + spatial decoder + gate head checkpoints and controlled baselines for **NativeSeg: Native Vision Language Segmentation of 3D Computed Tomography via Spatial Visual Tokens and Verified Local Zoom**.

- **Code Repository:** [https://github.com/step-out/NativeSeg](https://github.com/step-out/NativeSeg)
- **Model Weights:** [https://huggingface.co/ChengqianMa/NativeSeg](https://huggingface.co/ChengqianMa/NativeSeg)
- **Base Backbone:** `Qwen/Qwen3.8-27B` (frozen vision tower and frozen lower 48 language layers)

## NativeSeg Checkpoints

| Folder | Decoder | Trainable Params | Size (`MiB`) | SHA256 | Role |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `nativeseg_stage_a/` | `five_slice` | `56.00M` | 213.75 MiB | `a36729a21fbe072a3bf55eb7bcf7d4de8750c20b782085ac6662f60566cb4242` | Stage A five-slice Round-0 readout (`86.01%` Test / `85.99%` Pediatric / `85.65%` Val / `88.72%` Val6 DSC) & consensus anchor |
| `nativeseg_stage_b/` | `five_slice` | `56.00M` | 213.75 MiB | `5a63b7b06baaacd9ac4fb74c6309f2d6ab98b3122b135ed4d16c10c889f4030c` | Stage B joint end-to-end multiround model (`89.09%` Val6 DSC in Table 3b; `85.83%` Test / `86.54%` Pediatric / `85.44%` Val single-checkpoint DSC) |
| `nativeseg_local_refiner/` | `single_slice` | `34.65M` | 132.28 MiB | `28813c0a7cb0cb8056cfb922ab4e2c826665635b3ac983b0605aa42aa6b807b4` | Verified multiscale local zoom & `<err>` gate (`AUROC = 0.957` own / `0.942` 2D nnU-Net, standalone zoom & 2D/3D specialist refinement) |
| `nativeseg_single_slice/` | `single_slice` | `34.65M` | 132.28 MiB | `8bb9c3c668d6d97ca136c4c68fec85fca238178f3f96eb229419228236db3c52` | Single-slice 2D spatial-token + skip Round 0 (`F-LMM†` controlled baseline & stage-decoupled zoom initializer) |

## Controlled Baselines (`baselines/`)

| Folder | Architecture | Trainable Params | Size (`MiB`) | SHA256 | Role |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `baselines/lisa_seg_token/` | `single_slice` (`Qwen3.8-27B`) | `34.65M` | 132.28 MiB | `433e6c9ed26e6e77d5f5a75bd11f794be4987f09e5c0e5a3afe4e2276549d0b4` | Controlled `LISA†` single `[SEG]` broadcast baseline on Qwen3.8-27B (Tables 1a, 2a, 3a, 4a) |
| `baselines/densemllm_spatial_tokens/` | `single_slice` (`Qwen3.8-27B`) | `34.65M` | 132.28 MiB | `7fd21e8a82db10803f4dcfb75c65daefd72d7cb5fa10a9c39781955a38746c87` | Controlled `DenseMLLM†` 2D spatial-tokens-only baseline on Qwen3.8-27B (Tables 1a, 2a, 3a, 4a) |
| `baselines/lrn_cnn_editor/` | `CNNEditor` (2D U-Net) | `10.05M` | 38.38 MiB | `31667710b2fbc14563eb57800fe8243e4b6ddfaa1b215b14cdf044e9728e9129` | Perturbation-trained `LRN†` CNN mask refiner baseline on TotalSegmentator (`+ LRN†` in Tables 1b, 2b, 4b) |
