#!/usr/bin/env bash
# Authoritative Training Recipes for all 7 Released Checkpoints on 1,081 TotalSegmentator Adult CT Scans.
#
# By default, when >= 8 GPUs are supplied in GPUS (e.g. GPUS=0,1,2,3,4,5,6,7), each stage uses the exact
# distributed world size from our original training runs:
#   - nativeseg_single_slice & nativeseg_stage_a : 8 GPUs (per-GPU batch 6 -> global batch 48)
#   - nativeseg_local_refiner & nativeseg_stage_b: 6 GPUs (per-GPU batch 2 + agent_batch 4 / 3)
#   - lisa_seg_token & densemllm_spatial_tokens  : 4 GPUs (per-GPU batch 9 -> global batch 36)
#   - lrn_cnn_editor                             : 1 GPU  (batch 32, 30,000 steps)
# Set FORCE_ALL_GPUS=1 if running on a smaller GPU node.
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-Qwen/Qwen3.8-27B}
DATA_DIR=${DATA_DIR:-data/flat_bank}
OUT_DIR=${OUT_DIR:-runs}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
TOTAL_GPUS=${#GPU_ARR[@]}

slice_gpus() {
  local want=$1
  if [ "${FORCE_ALL_GPUS:-0}" = "1" ] || [ "$TOTAL_GPUS" -lt "$want" ]; then
    echo "$GPUS:$TOTAL_GPUS"
  else
    local sub
    sub=$(IFS=','; echo "${GPU_ARR[*]:0:$want}")
    echo "$sub:$want"
  fi
}

IFS=':' read -r GPUS_8 NPROC_8 <<< "$(slice_gpus 8)"
IFS=':' read -r GPUS_6 NPROC_6 <<< "$(slice_gpus 6)"
IFS=':' read -r GPUS_4 NPROC_4 <<< "$(slice_gpus 4)"
GPU_1="${GPU_ARR[0]}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

# 1. Single-Slice 2D Spatial-Token + Skip Round-0 Checkpoint (nativeseg_single_slice / F-LMM†, 34.65M params, world=8)
CUDA_VISIBLE_DEVICES="$GPUS_8" torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC_8" -m nativeseg.train \
  --model-dir "$MODEL_DIR" \
  --data "$DATA_DIR" \
  --output "$OUT_DIR/nativeseg_single_slice" \
  --stage initial \
  --decoder single_slice \
  --readout fused \
  --steps 12000 \
  --batch 6 \
  --lr-lora 2e-4 \
  --lr-new 3e-4 \
  --scale-aug 0.5 \
  --warmup 200 \
  --save-every 1000 \
  --seed 20261031

# 2. Verified Multiscale Local Zoom & <err> Gate Checkpoint (nativeseg_local_refiner, 34.65M params, world=6)
CUDA_VISIBLE_DEVICES="$GPUS_6" torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC_6" -m nativeseg.train \
  --model-dir "$MODEL_DIR" \
  --data "$DATA_DIR" \
  --output "$OUT_DIR/nativeseg_local_refiner" \
  --init-from "$OUT_DIR/nativeseg_single_slice/step_012000" \
  --stage agent \
  --decoder single_slice \
  --readout fused \
  --steps 3000 \
  --batch 2 \
  --agent-batch 4 \
  --state-residual 0.5 \
  --clean 0.15 \
  --synthetic 0.35 \
  --extra 0.35 \
  --focus-weight 2.0 \
  --lr-lora 1e-4 \
  --lr-new 2e-4 \
  --scale-aug 0.5 \
  --warmup 200 \
  --save-every 500 \
  --seed 20261031

# 3. Five-Slice Spatial-Token U-Net Stage A Checkpoint (nativeseg_stage_a, 56.00M params, world=8; warm-started from local_refiner)
CUDA_VISIBLE_DEVICES="$GPUS_8" torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC_8" -m nativeseg.train \
  --model-dir "$MODEL_DIR" \
  --data "$DATA_DIR" \
  --output "$OUT_DIR/nativeseg_stage_a" \
  --init-from "$OUT_DIR/nativeseg_local_refiner/step_003000" \
  --stage initial \
  --decoder five_slice \
  --readout fused \
  --hard-organs \
  --steps 6000 \
  --batch 6 \
  --lr-lora 1e-4 \
  --lr-new 4e-4 \
  --scale-aug 0.5 \
  --warmup 200 \
  --save-every 500 \
  --seed 20261031

# 4. End-to-End Five-Slice Joint Stage B Checkpoint (nativeseg_stage_b, 56.00M params, world=6; warm-started from local_refiner)
CUDA_VISIBLE_DEVICES="$GPUS_6" torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC_6" -m nativeseg.train \
  --model-dir "$MODEL_DIR" \
  --data "$DATA_DIR" \
  --output "$OUT_DIR/nativeseg_stage_b" \
  --init-from "$OUT_DIR/nativeseg_local_refiner/step_003000" \
  --stage agent \
  --decoder five_slice \
  --readout fused \
  --hard-organs \
  --steps 4000 \
  --batch 2 \
  --agent-batch 3 \
  --state-residual 0.8 \
  --clean 0.15 \
  --synthetic 0.35 \
  --extra 0.40 \
  --focus-weight 2.5 \
  --lr-lora 1e-4 \
  --lr-new 3e-4 \
  --scale-aug 0.5 \
  --warmup 200 \
  --save-every 500 \
  --seed 20261031

# 5. Optional Controlled Baselines (set TRAIN_BASELINES=1: LISA†, DenseMLLM† on world=4, and LRN† CNN editor on 1 GPU)
if [ "${TRAIN_BASELINES:-0}" = "1" ]; then
  for ITEM in "lisa_seg_token:token" "densemllm_spatial_tokens:llm"; do
    NAME="${ITEM%%:*}"
    RO="${ITEM#*:}"
    CUDA_VISIBLE_DEVICES="$GPUS_4" torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC_4" -m nativeseg.train \
      --model-dir "$MODEL_DIR" \
      --data "$DATA_DIR" \
      --output "$OUT_DIR/baselines/${NAME}" \
      --stage initial \
      --decoder single_slice \
      --readout "$RO" \
      --steps 5000 \
      --batch 9 \
      --lr-lora 2e-4 \
      --lr-new 3e-4 \
      --scale-aug 0.5 \
      --warmup 200 \
      --save-every 1000 \
      --seed 20261031
  done

  CUDA_VISIBLE_DEVICES="$GPU_1" python -m nativeseg.lrn_baseline train \
    --data "$DATA_DIR" \
    --output "$OUT_DIR/baselines/lrn_cnn_editor" \
    --steps 30000 \
    --batch 32 \
    --lr 1e-3 \
    --scale-aug 0.5 \
    --seed 20261031
fi
