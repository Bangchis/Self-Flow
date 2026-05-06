# Representation Evaluation Remote Runbook

This branch contains the LayerSync-style representation evaluation runner for three
checkpoints: SiT-XL 1M, LayerSync 800K, and LARA 400K.

## 1. Rent a Machine

For this workload, VRAM matters more than the GPU name. RTX 5090 has 32GB VRAM, which is
better than RTX 4090 24GB, but batch 16 can still OOM because extraction runs VAE encode
plus full SiT/LayerSync-XL forward and stores all requested layer features.

Recommended order:

1. Fastest/safest: 2-4 GPUs with 48GB or 80GB VRAM each.
2. Good consumer option: 2-4x RTX 5090 32GB.
3. Budget: 2x RTX 4090 24GB, using `CLASS_EXTRACT_BATCH_SIZE=2`.

On RTX 5090, start with:

```bash
CLASS_EXTRACT_BATCH_SIZE=4
SEG_EXTRACT_BATCH_SIZE=4
```

Then try `8`, and only try `16` after `nvidia-smi` shows enough headroom.

## 2. Clone and Install

```bash
cd /workspace
git clone https://github.com/Bangchis/Self-Flow.git
cd Self-Flow
git checkout feat/depth-shortcut-output-distill

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

If the machine uses a different CUDA/JAX stack, reinstall the matching JAX wheel after
`requirements.txt`. For example, use `jax[cuda13]` on CUDA 13 hosts and `jax[cuda12]`
on CUDA 12 hosts.

## 3. Login for Tracking and Downloads

Do not hardcode tokens in shell history or source files. Set them as environment
variables or use each CLI login flow.

```bash
wandb login
hf auth login
```

For Kaggle downloads, place `kaggle.json` at:

```bash
mkdir -p ~/.kaggle
chmod 700 ~/.kaggle
# put kaggle.json in ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## 4. Required Local Paths

The default wrapper expects these paths:

```text
checkpoints/SiT-XL-1M
checkpoints/LayerSync-XL-800k
checkpoints/LARA-XL/latest
checkpoints/vae/sdvae-ema-kaggle-flax
checkpoints/dinov2-g/dinov2_vitg14_pretrain.pth
external/dinov2
data/tiny-imagenet/tiny-imagenet-200
data/pascal-voc/VOCdevkit/VOC2012
data/imagenet-val/imagenet_val_subset_4000_256.txt
```

The repo includes a setup script for most assets:

```bash
scripts/setup_rep_eval_assets.sh
```

By default it downloads/checks:

- SiT-XL 1M, LayerSync 800K, and LARA 400K checkpoints from Hugging Face.
- Stable Diffusion VAE from Kaggle.
- DINOv2-g code and weights.
- Tiny ImageNet.
- PASCAL VOC2012 train/val segmentation.
- 4,000 random ImageNet validation images from Hugging Face `ILSVRC/imagenet-1k`.

ImageNet is special because access is gated. Before running setup, open
`https://huggingface.co/datasets/ILSVRC/imagenet-1k`, accept the terms, then run
`hf auth login`. The setup script streams only 4,000 validation images, so it does
not download full ImageNet.

The old Kaggle full-competition fallback is still available, but usually unnecessary:

```bash
DOWNLOAD_IMAGENET=1 scripts/setup_rep_eval_assets.sh
```

Manual checkpoint download equivalent:

```bash
mkdir -p checkpoints external data
hf download LamTNguyen/sit-dit-xl-imagenet-step1038000 \
  --local-dir checkpoints/SiT-XL-1M
hf download LamTNguyen/Layersync-ckpt-jax \
  --local-dir checkpoints/LayerSync-XL-800k
hf download LamTNguyen/LARA-XL \
  --local-dir checkpoints/LARA-XL
```

DINOv2-g requires the local Meta DINOv2 repo and weights:

```bash
git clone https://github.com/facebookresearch/dinov2.git external/dinov2
mkdir -p checkpoints/dinov2-g
wget -O checkpoints/dinov2-g/dinov2_vitg14_pretrain.pth \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth
```

Dataset preparation is intentionally kept outside git. Put Tiny ImageNet, PASCAL VOC,
and the 4,000-image ImageNet manifest/subset at the paths above before running.

## 5. Full Run

This is the current default experiment: ConvNeXt-Tiny probe for classification and
segmentation, standard centered linear CKA using model summary features vs DINOv2-g CLS,
three checkpoints, one low-noise timestep `t=0.8`.

```bash
cd /workspace/Self-Flow

WANDB_WORKERS=1 \
WANDB_RUN_NAME=rep_eval_3models_convnexttiny_probe_seed123_t08_4x5090_live \
SEED=123 \
EXPERIMENTS=classification,segmentation,cka \
CHECKPOINTS=sit_xl_1m,layersync_800k,lara_400k \
TIMESTEPS=0.8 \
CLASS_PROBES=convnext_tiny_probe \
CLASS_MAIN_PROBE=convnext_tiny_probe \
CLASS_EXTRACT_BATCH_SIZE=4 \
CLASS_CNN_LAYER_BATCH_SIZE=1 \
SEG_EXTRACT_BATCH_SIZE=4 \
SEG_TRAIN_BATCH_SIZE=256 \
SEG_EVAL_BATCH_SIZE=128 \
SEG_CNN_LAYER_BATCH_SIZE=1 \
SEG_CNN_INPUT_SIZE=64 \
CKA_BATCH_SIZE=16 \
DINO_BATCH_SIZE=8 \
CKA_MODEL_FEATURES=summary \
CKA_DINO_FEATURES=cls \
CKA_CONDITION_POLICIES=null \
CKA_MAIN_SCENARIO=summary__null__dinov2_cls \
PARALLEL_GPUS=0,1,2,3 \
OUT_DIR=results/rep_eval_3models_convnexttiny_probe_seed123_t08_4x5090 \
HF_PATH=rep_eval_3models_convnexttiny_probe_seed123_t08_4x5090 \
scripts/run_rep_eval_3models.sh
```

For a detached run:

```bash
setsid bash scripts/rerun_rep_eval_full_t08_shm.sh \
  >/tmp/rep_eval_convnexttiny_defaultcka.nohup 2>&1 < /dev/null &
tail -f results/queued_runs/rep_eval_3models_convnexttiny_probe_seed123_t08_4x5090.log
```

Override batch on a bigger machine:

```bash
CLASS_EXTRACT_BATCH_SIZE=8 SEG_EXTRACT_BATCH_SIZE=8 scripts/rerun_rep_eval_full_t08_shm.sh
```

Use a small random subset for faster debug runs:

```bash
LIMIT_TINY_TRAIN=4000 \
LIMIT_TINY_VAL=2000 \
LIMIT_VOC_TRAIN=500 \
LIMIT_VOC_VAL=500 \
LIMIT_IMAGENET=4000 \
scripts/rerun_rep_eval_full_t08_shm.sh
```

## 6. Outputs

The run writes CSVs, figures, and probe checkpoints under `OUT_DIR`, logs live worker runs
to W&B, and uploads final CSV/figures/probe artifacts to Hugging Face. Feature caches are
ignored by default to avoid very large uploads.
