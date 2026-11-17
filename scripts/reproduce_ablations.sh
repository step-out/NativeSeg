#!/usr/bin/env bash
# Reproduce Table 3(a) Continuous/Token Readout Ablations (4/5 rows; note Discrete Codebook MedSIGHT†
# is a separate two-stage RegionCodec VQ + autoregressive model), Table 3(b) Multiround Control &
# Individual Component Ablations (all 10/10 rows), and Figure 4(a, b, c) Native <err> Calibration
# & Granularity Sweeps on the fixed 6-patient validation subset (s0032,s0458,s0650,s0808,s1046,s1233).
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-Qwen/Qwen3.8-27B}
DATA_DIR=${DATA_DIR:-data/flat_bank}
CKPT_DIR=${CKPT_DIR:-checkpoints}
PRED_DIR=${PRED_DIR:-predictions}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
VAL6="s0032,s0458,s0650,s0808,s1046,s1233"

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
  "baselines/lisa_seg_token" \
  "baselines/densemllm_spatial_tokens" \
  "nativeseg_single_slice" \
  "nativeseg_stage_a" \
  "nativeseg_local_refiner" \
  "nativeseg_stage_b"
do
  require_ckpt "$REQ"
done

# ==============================================================================
# Part (a): Round-0 Mask Readout Architecture on Qwen3.8-27B (Table 3a)
# ==============================================================================
for ITEM in \
  "lisa_seg_token_val6:baselines/lisa_seg_token:single_slice:token" \
  "densemllm_spatial_val6:baselines/densemllm_spatial_tokens:single_slice:llm" \
  "single_slice_init_val6:nativeseg_single_slice:single_slice:fused" \
  "nativeseg_round0_val6:nativeseg_stage_a:five_slice:fused"
do
  OUT_TAG="${ITEM%%:*}"
  REST1="${ITEM#*:}"
  CK_NAME="${REST1%%:*}"
  REST2="${REST1#*:}"
  DEC_TYPE="${REST2%%:*}"
  RO_MODE="${REST2#*:}"
  torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" -m nativeseg.predict \
    --model-dir "$MODEL_DIR" --data "$DATA_DIR" \
    --checkpoint "$CKPT_DIR/$CK_NAME" --decoder "$DEC_TYPE" --readout "$RO_MODE" \
    --split val --subjects "$VAL6" \
    --output "$PRED_DIR/${OUT_TAG}"
  python -m nativeseg.evaluate score \
    --pred "$PRED_DIR/${OUT_TAG}" --data "$DATA_DIR" --split val --subjects "$VAL6" \
    --presence 0.5 --key logits_mask \
    --output "$PRED_DIR/${OUT_TAG}_score.json"
done

# ==============================================================================
# Part (b): Multiround Control & Individual Component Ablations (Table 3b)
# Note: The three 'indiv.' rows below test adding each component individually on
# top of Error Gate + Verify on the frozen single-slice checkpoint before
# uniting all components in NativeSeg (+ Joint Stage B).
# ==============================================================================
run_val6_ablation() {
  local NAME=$1
  shift
  torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" -m nativeseg.predict \
    --model-dir "$MODEL_DIR" --data "$DATA_DIR" \
    --checkpoint "$CKPT_DIR/nativeseg_local_refiner" --decoder single_slice \
    --split val --subjects "$VAL6" \
    --initial-from "$PRED_DIR/single_slice_init_val6" \
    "$@" \
    --output "$PRED_DIR/ablate_${NAME}"
  python -m nativeseg.evaluate score \
    --pred "$PRED_DIR/ablate_${NAME}" --data "$DATA_DIR" --split val --subjects "$VAL6" \
    --key final_mask --output "$PRED_DIR/ablate_${NAME}_score.json"
}

run_val6_ablation reread         --policy reread      --max-steps 2 --absent-exit 0.1 --state-residual 1.0
run_val6_ablation sweep_2x2      --policy all2x2      --max-steps 4 --absent-exit 0.1 --state-residual 1.0
run_val6_ablation blank_overlay  --policy learned     --verify --blank-state --tau 0.05 --max-steps 3 --absent-exit 0.1 --state-residual 1.0
run_val6_ablation pixel_entropy  --policy uncertainty --max-steps 2 --absent-exit 0.1 --state-residual 1.0
run_val6_ablation random_region  --policy random      --max-steps 2 --absent-exit 0.1 --state-residual 1.0
run_val6_ablation no_verify      --policy learned     --tau 0.05 --max-steps 3 --absent-exit 0.1 --state-residual 1.0
run_val6_ablation verify_freeroi --policy learned     --verify --free-roi --tau 0.05 --max-steps 3 --absent-exit 0.1 --state-residual 1.0
run_val6_ablation conf_map       --policy learned     --verify --logit-memory --tau 0.05 --max-steps 3 --absent-exit 0.1 --state-residual 1.0
run_val6_ablation axial_cont     --policy learned     --verify --slice-continuity --tau 0.05 --max-steps 3 --absent-exit 0.1 --state-residual 1.0

# Table 3(b) final row: NativeSeg (+ Joint Stage B) single-checkpoint five-slice model (89.09% DSC)
torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" -m nativeseg.predict \
  --model-dir "$MODEL_DIR" --data "$DATA_DIR" \
  --checkpoint "$CKPT_DIR/nativeseg_stage_b" --decoder five_slice \
  --split val --subjects "$VAL6" \
  --policy learned --verify --absent-exit 0.1 --tau 0.05 --max-steps 3 --state-residual 1.0 \
  --free-roi --logit-memory --slice-continuity \
  --output "$PRED_DIR/ablate_joint_stage_b"
python -m nativeseg.evaluate score \
  --pred "$PRED_DIR/ablate_joint_stage_b" --data "$DATA_DIR" --split val --subjects "$VAL6" \
  --key final_mask --output "$PRED_DIR/ablate_joint_stage_b_score.json"

# ==============================================================================
# Figure 4(a, b, c): Native <err> Calibration & Granularity Control Sweep
# ==============================================================================
python -m nativeseg.assess_eval \
  --model-dir "$MODEL_DIR" --checkpoint "$CKPT_DIR/nativeseg_local_refiner" \
  --data "$DATA_DIR" --split val --subjects "$VAL6" \
  --output "$PRED_DIR/assess_own_val6.json"

if [ -n "${SPECIALIST_VAL_NPZ_DIR:-}" ]; then
  python -m nativeseg.assess_eval \
    --model-dir "$MODEL_DIR" --checkpoint "$CKPT_DIR/nativeseg_local_refiner" \
    --initial-from "$SPECIALIST_VAL_NPZ_DIR" \
    --data "$DATA_DIR" --split val --subjects "$VAL6" \
    --output "$PRED_DIR/assess_specialist_val6.json"
fi

run_val6_ablation sweep_rec --policy learned --verify --tau 0.0 --max-steps 4 --absent-exit 0.1 --state-residual 1.0 --record
python -m nativeseg.sweep \
  --pred "$PRED_DIR/ablate_sweep_rec" --data "$DATA_DIR" --split val --subjects "$VAL6" \
  --taus 0,0.02,0.05,0.10,0.20 --budgets 0,1,2,3,4 \
  --output "$PRED_DIR/sweep_val6.json"
