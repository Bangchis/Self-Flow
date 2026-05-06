#!/usr/bin/env bash
set -euxo pipefail

cd /workspace/Self-Flow

RUN_TAG=rep_eval_3models_convnexttiny_defaultcka_seed123_t09
LOG=/workspace/Self-Flow/results/queued_runs/${RUN_TAG}.log
SHM_OUT=/dev/shm/${RUN_TAG}
FINAL_OUT=/workspace/Self-Flow/results/${RUN_TAG}

mkdir -p /workspace/Self-Flow/results/queued_runs
: > "$LOG"
exec >> "$LOG" 2>&1

echo "[$(date -Is)] starting rerun on /dev/shm"
rm -rf "$SHM_OUT"
mkdir -p "$SHM_OUT"

set +e
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
LIMIT_TINY_TRAIN=${LIMIT_TINY_TRAIN:-} \
LIMIT_TINY_VAL=${LIMIT_TINY_VAL:-} \
LIMIT_VOC_TRAIN=${LIMIT_VOC_TRAIN:-} \
LIMIT_VOC_VAL=${LIMIT_VOC_VAL:-} \
LIMIT_IMAGENET=${LIMIT_IMAGENET:-4000} \
PARALLEL_GPUS=0,1 \
OUT_DIR="$SHM_OUT" \
HF_PATH="$RUN_TAG" \
scripts/run_rep_eval_3models.sh
status=$?
set -e
echo "[$(date -Is)] run finished with exit code $status"
if [[ "$status" == "0" ]]; then
  rm -rf "$FINAL_OUT"
  mkdir -p "$FINAL_OUT"
  cp -a "$SHM_OUT"/. "$FINAL_OUT"/
  echo "[$(date -Is)] copied final outputs to $FINAL_OUT"
fi
exit "$status"
