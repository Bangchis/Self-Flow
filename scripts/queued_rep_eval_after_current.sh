#!/usr/bin/env bash
set -uo pipefail

cd /workspace/Self-Flow || exit 1

WAIT_PIDS=("$@")
LOG_DIR=/workspace/Self-Flow/results/queued_runs
RUN_TAG=rep_eval_3models_convnexttiny_defaultcka_seed123_t09
QUEUE_LOG="$LOG_DIR/queued_${RUN_TAG}.log"
mkdir -p "$LOG_DIR"

echo "[$(date -Is)] queued: waiting for current run PIDs: ${WAIT_PIDS[*]:-none}" >> "$QUEUE_LOG"
while true; do
  alive=0
  for pid in "${WAIT_PIDS[@]}"; do
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      alive=1
    fi
  done
  if [[ "$alive" == "0" ]]; then
    break
  fi
  echo "[$(date -Is)] still waiting for current run to finish" >> "$QUEUE_LOG"
  sleep 60
done

echo "[$(date -Is)] starting queued full eval" >> "$QUEUE_LOG"
WANDB_WORKERS=1 \
WANDB_RUN_NAME=${RUN_TAG}_live \
SEED=123 \
EXPERIMENTS=classification,segmentation,cka \
CHECKPOINTS=sit_xl_1m,layersync_800k,lara_400k \
TIMESTEPS=0.9 \
CLASS_PROBES=convnext_tiny_probe \
CLASS_MAIN_PROBE=convnext_tiny_probe \
CLASS_EXTRACT_BATCH_SIZE=${CLASS_EXTRACT_BATCH_SIZE:-2} \
CLASS_CNN_LAYER_BATCH_SIZE=${CLASS_CNN_LAYER_BATCH_SIZE:-1} \
SEG_EXTRACT_BATCH_SIZE=${SEG_EXTRACT_BATCH_SIZE:-${CLASS_EXTRACT_BATCH_SIZE:-2}} \
SEG_TRAIN_BATCH_SIZE=${SEG_TRAIN_BATCH_SIZE:-256} \
SEG_EVAL_BATCH_SIZE=${SEG_EVAL_BATCH_SIZE:-128} \
SEG_CNN_LAYER_BATCH_SIZE=${SEG_CNN_LAYER_BATCH_SIZE:-1} \
SEG_CNN_INPUT_SIZE=${SEG_CNN_INPUT_SIZE:-64} \
CKA_BATCH_SIZE=${CKA_BATCH_SIZE:-8} \
DINO_BATCH_SIZE=${DINO_BATCH_SIZE:-4} \
CKA_MODEL_FEATURES=summary \
CKA_DINO_FEATURES=cls \
CKA_CONDITION_POLICIES=null \
CKA_MAIN_SCENARIO=summary__null__dinov2_cls \
PARALLEL_GPUS=0,1 \
OUT_DIR=results/${RUN_TAG} \
HF_PATH=${RUN_TAG} \
scripts/run_rep_eval_3models.sh >> "$QUEUE_LOG" 2>&1
status=$?
echo "[$(date -Is)] queued full eval finished with exit code $status" >> "$QUEUE_LOG"
exit "$status"
