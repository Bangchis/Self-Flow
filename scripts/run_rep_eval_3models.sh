#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run 3-model representation evaluation.

Environment overrides:
  OUT_DIR=results/rep_eval_3models
  HF_PATH=rep_eval_3models
  SEED=123
  LAYERS=all
  TIMESTEPS=1.0
  EXPERIMENTS=all
  CHECKPOINTS=sit_xl_1m,layersync_800k,lara_400k
  WANDB_RUN_NAME=rep_eval_3models
  WANDB_WORKERS=0
  CLASS_PROBES=convnext_tiny_probe
  CLASS_MAIN_PROBE=convnext_tiny_probe
  CLASS_EPOCHS=50
  SEG_EPOCHS=25
  PARALLEL_GPUS=0,1,2,3
  CLASS_EXTRACT_BATCH_SIZE=8 for linear-only, auto 2 when a CNN probe is enabled
  CLASS_CNN_LAYER_BATCH_SIZE=4 for linear-only, auto 1 when ConvNeXt-Tiny is enabled
  SEG_EXTRACT_BATCH_SIZE=<CLASS_EXTRACT_BATCH_SIZE>
  SEG_TRAIN_BATCH_SIZE=256
  SEG_CNN_LAYER_BATCH_SIZE=1
  SEG_CNN_INPUT_SIZE=64
  CKA_BATCH_SIZE=8
  DINO_BATCH_SIZE=4
  CKA_MODEL_FEATURES=summary
  CKA_DINO_FEATURES=cls
  CKA_CONDITION_POLICIES=null
  CKA_MAIN_SCENARIO=summary__null__dinov2_cls
  KEEP_FEATURE_CACHE=0
  KEEP_WORKER_OUTPUTS=0
  RESUME_WORKERS=0
  LIMIT_TINY_TRAIN=
  LIMIT_TINY_VAL=
  LIMIT_VOC_TRAIN=
  LIMIT_VOC_VAL=
  LIMIT_IMAGENET=4000

Examples:
  scripts/run_rep_eval_3models.sh
  EXPERIMENTS=cka LAYERS=1,7,14,21,28 scripts/run_rep_eval_3models.sh
  CHECKPOINTS=layersync_800k TIMESTEPS=0.8 EXPERIMENTS=classification CLASS_PROBES=convnext_tiny_probe scripts/run_rep_eval_3models.sh
EOF
  exit 0
fi

OUT_DIR="${OUT_DIR:-results/rep_eval_3models}"
HF_PATH="${HF_PATH:-rep_eval_3models}"
SEED="${SEED:-123}"
LAYERS="${LAYERS:-all}"
TIMESTEPS="${TIMESTEPS:-1.0}"
EXPERIMENTS="${EXPERIMENTS:-all}"
CHECKPOINTS="${CHECKPOINTS:-sit_xl_1m,layersync_800k,lara_400k}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-rep_eval_3models}"
WANDB_WORKERS="${WANDB_WORKERS:-0}"
CLASS_PROBES="${CLASS_PROBES:-convnext_tiny_probe}"
CLASS_MAIN_PROBE="${CLASS_MAIN_PROBE:-convnext_tiny_probe}"
CLASS_EPOCHS="${CLASS_EPOCHS:-50}"
SEG_EPOCHS="${SEG_EPOCHS:-25}"
PARALLEL_GPUS="${PARALLEL_GPUS:-0,1,2,3}"
if [[ -z "${CLASS_EXTRACT_BATCH_SIZE+x}" ]]; then
  if [[ ",$CLASS_PROBES," == *",resnet18_probe,"* || ",$CLASS_PROBES," == *",convnext_atto_probe,"* || ",$CLASS_PROBES," == *",convnext_tiny_probe,"* ]]; then
    CLASS_EXTRACT_BATCH_SIZE=2
  else
    CLASS_EXTRACT_BATCH_SIZE=8
  fi
fi
if [[ -z "${CLASS_CNN_LAYER_BATCH_SIZE+x}" ]]; then
  if [[ ",$CLASS_PROBES," == *",convnext_tiny_probe,"* ]]; then
    CLASS_CNN_LAYER_BATCH_SIZE=1
  elif [[ ",$CLASS_PROBES," == *",resnet18_probe,"* || ",$CLASS_PROBES," == *",convnext_atto_probe,"* ]]; then
    CLASS_CNN_LAYER_BATCH_SIZE=2
  else
    CLASS_CNN_LAYER_BATCH_SIZE=4
  fi
fi
SEG_MICROBATCH_SIZE="${SEG_MICROBATCH_SIZE:-1}"
SEG_LAYER_CHUNK_SIZE="${SEG_LAYER_CHUNK_SIZE:-4}"
SEG_EXTRACT_BATCH_SIZE="${SEG_EXTRACT_BATCH_SIZE:-$CLASS_EXTRACT_BATCH_SIZE}"
SEG_TRAIN_BATCH_SIZE="${SEG_TRAIN_BATCH_SIZE:-256}"
SEG_EVAL_BATCH_SIZE="${SEG_EVAL_BATCH_SIZE:-$SEG_TRAIN_BATCH_SIZE}"
SEG_PROBE_PROJ_CHANNELS="${SEG_PROBE_PROJ_CHANNELS:-8}"
SEG_CNN_INPUT_SIZE="${SEG_CNN_INPUT_SIZE:-64}"
SEG_CNN_LAYER_BATCH_SIZE="${SEG_CNN_LAYER_BATCH_SIZE:-1}"
SEG_CNN_OUT_INDICES="${SEG_CNN_OUT_INDICES:-0,1,2,3}"
CKA_MODEL_FEATURES="${CKA_MODEL_FEATURES:-summary}"
CKA_DINO_FEATURES="${CKA_DINO_FEATURES:-cls}"
CKA_BATCH_SIZE="${CKA_BATCH_SIZE:-8}"
DINO_BATCH_SIZE="${DINO_BATCH_SIZE:-4}"
CKA_CONDITION_POLICIES="${CKA_CONDITION_POLICIES:-null}"
CKA_MAIN_SCENARIO="${CKA_MAIN_SCENARIO:-summary__null__dinov2_cls}"
LIMIT_TINY_TRAIN="${LIMIT_TINY_TRAIN:-}"
LIMIT_TINY_VAL="${LIMIT_TINY_VAL:-}"
LIMIT_VOC_TRAIN="${LIMIT_VOC_TRAIN:-}"
LIMIT_VOC_VAL="${LIMIT_VOC_VAL:-}"
LIMIT_IMAGENET="${LIMIT_IMAGENET:-}"
EXTRA_ARGS=()
if [[ "${KEEP_FEATURE_CACHE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--keep-feature-cache)
fi
if [[ "${KEEP_WORKER_OUTPUTS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--keep-worker-outputs)
fi
if [[ "${WANDB_WORKERS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--wandb-workers)
fi
if [[ "${RESUME_WORKERS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--resume-workers)
fi
if [[ -n "$LIMIT_TINY_TRAIN" ]]; then
  EXTRA_ARGS+=(--limit-tiny-train "$LIMIT_TINY_TRAIN")
fi
if [[ -n "$LIMIT_TINY_VAL" ]]; then
  EXTRA_ARGS+=(--limit-tiny-val "$LIMIT_TINY_VAL")
fi
if [[ -n "$LIMIT_VOC_TRAIN" ]]; then
  EXTRA_ARGS+=(--limit-voc-train "$LIMIT_VOC_TRAIN")
fi
if [[ -n "$LIMIT_VOC_VAL" ]]; then
  EXTRA_ARGS+=(--limit-voc-val "$LIMIT_VOC_VAL")
fi
if [[ -n "$LIMIT_IMAGENET" ]]; then
  EXTRA_ARGS+=(--limit-imagenet "$LIMIT_IMAGENET")
fi

CHECKPOINT_ARGS=()
if [[ ",$CHECKPOINTS," == *",sit_xl_1m,"* ]]; then
  CHECKPOINT_ARGS+=(--checkpoint sit_xl_1m=checkpoints/SiT-XL-1M)
fi
if [[ ",$CHECKPOINTS," == *",layersync_800k,"* ]]; then
  CHECKPOINT_ARGS+=(--checkpoint layersync_800k=checkpoints/LayerSync-XL-800k)
fi
if [[ ",$CHECKPOINTS," == *",lara_400k,"* ]]; then
  CHECKPOINT_ARGS+=(--checkpoint lara_400k=checkpoints/LARA-XL/latest)
fi
if [[ "${#CHECKPOINT_ARGS[@]}" -eq 0 ]]; then
  echo "No valid CHECKPOINTS selected: $CHECKPOINTS" >&2
  exit 2
fi

.venv/bin/python representation_eval.py \
  --experiments "$EXPERIMENTS" \
  "${CHECKPOINT_ARGS[@]}" \
  --seed "$SEED" \
  --layers "$LAYERS" \
  --timestep 1.0 \
  --timesteps "$TIMESTEPS" \
  --parallel-gpus "$PARALLEL_GPUS" \
  --global-batch-size 256 \
  --class-epochs "$CLASS_EPOCHS" \
  --seg-epochs "$SEG_EPOCHS" \
  --class-probes "$CLASS_PROBES" \
  --class-main-probe "$CLASS_MAIN_PROBE" \
  --class-extract-batch-size "$CLASS_EXTRACT_BATCH_SIZE" \
  --class-probe-proj-channels 8 \
  --class-cnn-layer-batch-size "$CLASS_CNN_LAYER_BATCH_SIZE" \
  --seg-microbatch-size "$SEG_MICROBATCH_SIZE" \
  --seg-layer-chunk-size "$SEG_LAYER_CHUNK_SIZE" \
  --seg-extract-batch-size "$SEG_EXTRACT_BATCH_SIZE" \
  --seg-train-batch-size "$SEG_TRAIN_BATCH_SIZE" \
  --seg-eval-batch-size "$SEG_EVAL_BATCH_SIZE" \
  --seg-probe-proj-channels "$SEG_PROBE_PROJ_CHANNELS" \
  --seg-cnn-input-size "$SEG_CNN_INPUT_SIZE" \
  --seg-cnn-layer-batch-size "$SEG_CNN_LAYER_BATCH_SIZE" \
  --seg-cnn-out-indices "$SEG_CNN_OUT_INDICES" \
  --cka-batch-size "$CKA_BATCH_SIZE" \
  --dino-batch-size "$DINO_BATCH_SIZE" \
  --cka-model-features "$CKA_MODEL_FEATURES" \
  --cka-dino-features "$CKA_DINO_FEATURES" \
  --cka-condition-policies "$CKA_CONDITION_POLICIES" \
  --cka-main-scenario "$CKA_MAIN_SCENARIO" \
  --wandb \
  --wandb-run-name "$WANDB_RUN_NAME" \
  --hf-upload \
  --hf-repo-id Bangchis/self-flow-representation-eval \
  --hf-path-in-repo "$HF_PATH" \
  --hf-private \
  --output-dir "$OUT_DIR" \
  "${EXTRA_ARGS[@]}"
