#!/usr/bin/env bash
# Reproduce NativeSeg, Single-Checkpoint Stage B, Controlled Qwen3.8-27B Readout Baselines,
# and Plug-and-Play Specialist Refinement (LRN†, MedSAM2, NativeSeg) Across TotalSegmentator (Test & Val)
# and Pediatric-CT-SEG (Tables 1, 2, 4, and Appendix B).
# Usage:
#   MODEL_DIR=Qwen/Qwen3.8-27B DATA_DIR=data/flat_bank CKPT_DIR=checkpoints PRED_DIR=predictions GPUS=0,1,2,3,4,5,6,7 bash scripts/reproduce_evaluation.sh
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-Qwen/Qwen3.8-27B}
DATA_DIR=${DATA_DIR:-data/flat_bank}
CKPT_DIR=${CKPT_DIR:-checkpoints}
PRED_DIR=${PRED_DIR:-predictions}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)

export CUDA_VISIBLE_DEVICES="$GPUS"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
mkdir -p "$PRED_DIR"

require_ckpt() {
  local sub=$1
  if [ ! -f "$CKPT_DIR/$sub/trainable.pt" ]; then
    echo "ERROR: Missing required checkpoint $CKPT_DIR/$sub/trainable.pt (run scripts/download_weights.py first)" >&2
    exit 1
  fi
}

for REQ in \
  "nativeseg_stage_a" \
  "nativeseg_stage_b" \
  "nativeseg_single_slice" \
  "nativeseg_local_refiner" \
  "baselines/lisa_seg_token" \
  "baselines/densemllm_spatial_tokens"
do
  require_ckpt "$REQ"
done

# Set SKIP_STANDALONE=1 when running Specialist Refinement only (Section 4) after standalone predictions exist.
if [ "${SKIP_STANDALONE:-0}" != "1" ]; then

# 1. NativeSeg Round 0 (Five-Slice U-Net + <pres>) across Test (65), Val (57), and Pediatric (21)
for SPLIT in test val pediatric; do
  torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" -m nativeseg.predict \
    --model-dir "$MODEL_DIR" --data "$DATA_DIR" \
    --checkpoint "$CKPT_DIR/nativeseg_stage_a" \
    --decoder five_slice --split "$SPLIT" \
    --output "$PRED_DIR/nativeseg_round0_${SPLIT}"

  python -m nativeseg.evaluate score \
    --pred "$PRED_DIR/nativeseg_round0_${SPLIT}" --data "$DATA_DIR" --split "$SPLIT" \
    --presence 0.5 --key logits_mask \
    --output "$PRED_DIR/nativeseg_round0_${SPLIT}_score.json"
done

# 2. F-LMM†, Single-Slice Multiround VLM (Ours), Modular Stage-Decoupled NativeSeg (Ours), and Single-Checkpoint Stage B
for SPLIT in test val pediatric; do
  torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" -m nativeseg.predict \
    --model-dir "$MODEL_DIR" --data "$DATA_DIR" \
    --checkpoint "$CKPT_DIR/nativeseg_single_slice" \
    --decoder single_slice --split "$SPLIT" \
    --output "$PRED_DIR/single_slice_init_${SPLIT}"

  python -m nativeseg.evaluate score \
    --pred "$PRED_DIR/single_slice_init_${SPLIT}" --data "$DATA_DIR" --split "$SPLIT" \
    --presence 0.5 --key logits_mask \
    --output "$PRED_DIR/flmm_single_slice_${SPLIT}_score.json"

  torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" -m nativeseg.predict \
    --model-dir "$MODEL_DIR" --data "$DATA_DIR" \
    --checkpoint "$CKPT_DIR/nativeseg_local_refiner" \
    --decoder single_slice --split "$SPLIT" \
    --initial-from "$PRED_DIR/single_slice_init_${SPLIT}" \
    --policy learned --verify --absent-exit 0.1 --tau 0.05 --max-steps 3 --state-residual 1.0 \
    --output "$PRED_DIR/zoom_standalone_${SPLIT}"

  # Single-Slice Multiround VLM (Ours) (Table 1a: 85.78% Test DSC)
  python -m nativeseg.evaluate single-slice \
    --init2d "$PRED_DIR/single_slice_init_${SPLIT}" \
    --zoom "$PRED_DIR/zoom_standalone_${SPLIT}" \
    --data "$DATA_DIR" --split "$SPLIT" \
    --output "$PRED_DIR/single_slice_multiround_${SPLIT}_score.json"

  # Modular Stage-Decoupled NativeSeg (Ours) (Table 1a: 86.81% Test; Table 2a: 87.04% Pediatric; Table 4a: 86.80% Val)
  python -m nativeseg.evaluate standalone \
    --round0 "$PRED_DIR/nativeseg_round0_${SPLIT}" \
    --zoom "$PRED_DIR/zoom_standalone_${SPLIT}" \
    --data "$DATA_DIR" --split "$SPLIT" \
    --output "$PRED_DIR/nativeseg_standalone_${SPLIT}_score.json"

  # Single-Checkpoint End-to-End Five-Slice Stage B (Appendix B: 85.83% Test / 85.44% Val / 86.54% Pediatric)
  torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" -m nativeseg.predict \
    --model-dir "$MODEL_DIR" --data "$DATA_DIR" \
    --checkpoint "$CKPT_DIR/nativeseg_stage_b" \
    --decoder five_slice --split "$SPLIT" \
    --policy learned --verify --absent-exit 0.1 --tau 0.05 --max-steps 3 --state-residual 1.0 \
    --free-roi --logit-memory --slice-continuity \
    --output "$PRED_DIR/nativeseg_stage_b_${SPLIT}"

  python -m nativeseg.evaluate score \
    --pred "$PRED_DIR/nativeseg_stage_b_${SPLIT}" --data "$DATA_DIR" --split "$SPLIT" \
    --key final_mask \
    --output "$PRED_DIR/nativeseg_stage_b_${SPLIT}_score.json"
done

# 3. Controlled Qwen3.8-27B Readout Baselines: LISA† (token) and DenseMLLM† (llm) (Tables 1a, 2a, 4a)
for ITEM in "baselines/lisa_seg_token:lisa_seg_token:token" "baselines/densemllm_spatial_tokens:densemllm_spatial_tokens:llm"; do
  CK_SUB="${ITEM%%:*}"
  REST="${ITEM#*:}"
  OUT_TAG="${REST%%:*}"
  RO_MODE="${REST#*:}"
  for SPLIT in test val pediatric; do
    torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" -m nativeseg.predict \
      --model-dir "$MODEL_DIR" --data "$DATA_DIR" \
      --checkpoint "$CKPT_DIR/$CK_SUB" \
      --decoder single_slice --readout "$RO_MODE" --split "$SPLIT" \
      --output "$PRED_DIR/${OUT_TAG}_${SPLIT}"

    python -m nativeseg.evaluate score \
      --pred "$PRED_DIR/${OUT_TAG}_${SPLIT}" --data "$DATA_DIR" --split "$SPLIT" \
      --presence 0.5 --key logits_mask \
      --output "$PRED_DIR/${OUT_TAG}_${SPLIT}_score.json"
  done
done

fi # end SKIP_STANDALONE

# 4. Plug-and-Play Specialist Refinement (set SPECIALIST_NPZ_DIR & SPECIALIST_SPLIT to run; Tables 1b, 2b, 4b)
if [ -n "${SPECIALIST_NPZ_DIR:-}" ] && [ -n "${SPECIALIST_SPLIT:-}" ]; then
  if [ ! -d "$PRED_DIR/nativeseg_round0_${SPECIALIST_SPLIT}" ]; then
    torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" -m nativeseg.predict \
      --model-dir "$MODEL_DIR" --data "$DATA_DIR" \
      --checkpoint "$CKPT_DIR/nativeseg_stage_a" \
      --decoder five_slice --split "$SPECIALIST_SPLIT" \
      --output "$PRED_DIR/nativeseg_round0_${SPECIALIST_SPLIT}"
  fi
  TAG=$(basename "$SPECIALIST_NPZ_DIR")

  # 4a. NativeSeg Specialist Refinement (+ cross-model consensus)
  torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" -m nativeseg.predict \
    --model-dir "$MODEL_DIR" --data "$DATA_DIR" \
    --checkpoint "$CKPT_DIR/nativeseg_local_refiner" \
    --decoder single_slice --split "$SPECIALIST_SPLIT" \
    --initial-from "$SPECIALIST_NPZ_DIR" \
    --policy learned --verify --absent-exit 0.1 --tau 0.10 --max-steps 2 --state-residual 1.5 \
    --output "$PRED_DIR/zoom_on_${TAG}"

  python -m nativeseg.evaluate refiner \
    --specialist "$SPECIALIST_NPZ_DIR" \
    --zoom "$PRED_DIR/zoom_on_${TAG}" \
    --round0 "$PRED_DIR/nativeseg_round0_${SPECIALIST_SPLIT}" \
    --data "$DATA_DIR" --split "$SPECIALIST_SPLIT" \
    --output "$PRED_DIR/nativeseg_on_${TAG}_score.json"

  # 4b. LRN† Perturbation-Trained CNN Refiner Baseline
  require_ckpt "baselines/lrn_cnn_editor"
  python -m nativeseg.lrn_baseline predict \
    --data "$DATA_DIR" --split "$SPECIALIST_SPLIT" \
    --checkpoint "$CKPT_DIR/baselines/lrn_cnn_editor" \
    --initial-from "$SPECIALIST_NPZ_DIR" \
    --output "$PRED_DIR/lrn_on_${TAG}"

  python -m nativeseg.evaluate score \
    --pred "$PRED_DIR/lrn_on_${TAG}" --data "$DATA_DIR" --split "$SPECIALIST_SPLIT" \
    --key logits_mask \
    --output "$PRED_DIR/lrn_on_${TAG}_score.json"

  # 4c. Optional MedSAM2 Box-Prompted Refiner Baseline (set MEDSAM2_CKPT=/path/to/MedSAM2_latest.pt)
  if [ -n "${MEDSAM2_CKPT:-}" ]; then
    python -m nativeseg.medsam2_baseline \
      --initial-from "$SPECIALIST_NPZ_DIR" \
      --data "$DATA_DIR" --split "$SPECIALIST_SPLIT" \
      --medsam2-checkpoint "$MEDSAM2_CKPT" \
      --output "$PRED_DIR/medsam2_on_${TAG}"

    python -m nativeseg.evaluate score \
      --pred "$PRED_DIR/medsam2_on_${TAG}" --data "$DATA_DIR" --split "$SPECIALIST_SPLIT" \
      --key logits_mask \
      --output "$PRED_DIR/medsam2_on_${TAG}_score.json"
  fi
fi
