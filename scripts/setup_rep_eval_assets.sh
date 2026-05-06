#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DOWNLOAD_CHECKPOINTS="${DOWNLOAD_CHECKPOINTS:-1}"
DOWNLOAD_VAE="${DOWNLOAD_VAE:-1}"
DOWNLOAD_DINO="${DOWNLOAD_DINO:-1}"
DOWNLOAD_TINY="${DOWNLOAD_TINY:-1}"
DOWNLOAD_VOC="${DOWNLOAD_VOC:-1}"
DOWNLOAD_IMAGENET="${DOWNLOAD_IMAGENET:-0}"

TINY_ROOT="${TINY_ROOT:-data/tiny-imagenet/tiny-imagenet-200}"
VOC_ROOT="${VOC_ROOT:-data/pascal-voc/VOCdevkit/VOC2012}"
IMAGENET_MANIFEST="${IMAGENET_MANIFEST:-data/imagenet-val/imagenet_val_subset_4000_256.txt}"
IMAGENET_SEARCH_ROOTS="${IMAGENET_SEARCH_ROOTS:-data/imagenet-val/subset_4000_256:data/imagenet-val/subset/ILSVRC2012_img_val_subset:data/imagenet-val/raw}"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing command: $1" >&2
    exit 2
  fi
}

hf_download() {
  local repo="$1"
  local out="$2"
  if [[ -d "$out" && -n "$(find "$out" -mindepth 1 -maxdepth 1 2>/dev/null | head -1)" ]]; then
    echo "[setup] exists: $out"
    return
  fi
  need_cmd huggingface-cli
  mkdir -p "$out"
  echo "[setup] downloading $repo -> $out"
  huggingface-cli download "$repo" --local-dir "$out" --local-dir-use-symlinks False
}

if [[ "$DOWNLOAD_CHECKPOINTS" == "1" ]]; then
  mkdir -p checkpoints
  hf_download LamTNguyen/sit-dit-xl-imagenet-step1038000 checkpoints/SiT-XL-1M
  hf_download LamTNguyen/Layersync-ckpt-jax checkpoints/LayerSync-XL-800k
  hf_download LamTNguyen/LARA-XL checkpoints/LARA-XL

  if [[ ! -e checkpoints/LARA-XL/latest ]]; then
    latest_ckpt="$(find checkpoints/LARA-XL -maxdepth 1 -type d -name 'checkpoint_*' | sort -V | tail -1 || true)"
    if [[ -n "$latest_ckpt" ]]; then
      ln -s "$(basename "$latest_ckpt")" checkpoints/LARA-XL/latest
      echo "[setup] linked checkpoints/LARA-XL/latest -> $(basename "$latest_ckpt")"
    fi
  fi
fi

if [[ "$DOWNLOAD_VAE" == "1" ]]; then
  echo "[setup] downloading/copying Kaggle VAE -> checkpoints/vae/sdvae-ema-kaggle-flax"
  mkdir -p checkpoints/vae/sdvae-ema-kaggle-flax
  python3 - <<'PY'
from pathlib import Path
import shutil

import kagglehub

target = Path("checkpoints/vae/sdvae-ema-kaggle-flax")
handles = [
    "damtrunghieu/sdvae-ema/flax/default/1",
    "damtrunghieu/sdvae-ema/Flax/default/1",
]
last_error = None
for handle in handles:
    try:
        source = Path(kagglehub.model_download(handle))
        break
    except Exception as exc:
        last_error = exc
else:
    raise SystemExit(f"Could not download Kaggle VAE. Last error: {last_error}")

target.mkdir(parents=True, exist_ok=True)
for item in source.iterdir():
    dst = target / item.name
    if item.is_dir():
        shutil.copytree(item, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(item, dst)
print(f"[setup] VAE source: {source}")
PY
fi

if [[ "$DOWNLOAD_DINO" == "1" ]]; then
  need_cmd git
  need_cmd wget
  if [[ ! -d external/dinov2/.git ]]; then
    mkdir -p external
    git clone https://github.com/facebookresearch/dinov2.git external/dinov2
  else
    echo "[setup] exists: external/dinov2"
  fi
  mkdir -p checkpoints/dinov2-g
  if [[ ! -f checkpoints/dinov2-g/dinov2_vitg14_pretrain.pth ]]; then
    wget -O checkpoints/dinov2-g/dinov2_vitg14_pretrain.pth \
      https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth
  else
    echo "[setup] exists: checkpoints/dinov2-g/dinov2_vitg14_pretrain.pth"
  fi
fi

if [[ "$DOWNLOAD_TINY" == "1" ]]; then
  need_cmd wget
  need_cmd unzip
  if [[ ! -f "$TINY_ROOT/wnids.txt" ]]; then
    mkdir -p data/tiny-imagenet
    wget -O data/tiny-imagenet/tiny-imagenet-200.zip \
      http://cs231n.stanford.edu/tiny-imagenet-200.zip
    unzip -q -n data/tiny-imagenet/tiny-imagenet-200.zip -d data/tiny-imagenet
  else
    echo "[setup] exists: $TINY_ROOT"
  fi
fi

if [[ "$DOWNLOAD_VOC" == "1" ]]; then
  need_cmd wget
  need_cmd tar
  if [[ ! -d "$VOC_ROOT/JPEGImages" ]]; then
    mkdir -p data/pascal-voc
    wget -O data/pascal-voc/VOCtrainval_11-May-2012.tar \
      http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    tar -xf data/pascal-voc/VOCtrainval_11-May-2012.tar -C data/pascal-voc
  else
    echo "[setup] exists: $VOC_ROOT"
  fi
fi

if [[ "$DOWNLOAD_IMAGENET" == "1" ]]; then
  need_cmd kaggle
  mkdir -p data/imagenet-val/raw
  echo "[setup] downloading ImageNet competition files. This can be very large and requires accepted Kaggle terms."
  kaggle competitions download -c imagenet-object-localization-challenge -p data/imagenet-val/raw
  find data/imagenet-val/raw -maxdepth 1 -name '*.zip' -print0 | xargs -0 -r -n 1 unzip -q -n -d data/imagenet-val/raw
fi

echo "[setup] creating/checking ImageNet 4k manifest: $IMAGENET_MANIFEST"
IMAGENET_MANIFEST="$IMAGENET_MANIFEST" IMAGENET_SEARCH_ROOTS="$IMAGENET_SEARCH_ROOTS" python3 - <<'PY'
from pathlib import Path
import os

manifest = Path(os.environ["IMAGENET_MANIFEST"])
roots = [Path(item) for item in os.environ["IMAGENET_SEARCH_ROOTS"].split(":") if item]
images = []
for root in roots:
    if not root.exists():
        continue
    for pattern in ("*.JPEG", "*.jpg", "*.jpeg", "*.png"):
        images.extend(root.rglob(pattern))

images = sorted(set(path.resolve() for path in images))
if len(images) < 4000:
    print(f"[setup] WARNING: found only {len(images)} ImageNet-like images. Put a 4k ImageNet val subset under one of:")
    for root in roots:
        print(f"[setup]   - {root}")
else:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w") as handle:
        for path in images[:4000]:
            try:
                rel = path.relative_to(Path.cwd())
            except ValueError:
                rel = path
            handle.write(f"{rel}\n")
    print(f"[setup] wrote {manifest} with 4000 images")
PY

echo "[setup] done. Expected paths:"
printf '  %s\n' \
  checkpoints/SiT-XL-1M \
  checkpoints/LayerSync-XL-800k \
  checkpoints/LARA-XL/latest \
  checkpoints/vae/sdvae-ema-kaggle-flax \
  checkpoints/dinov2-g/dinov2_vitg14_pretrain.pth \
  external/dinov2 \
  "$TINY_ROOT" \
  "$VOC_ROOT" \
  "$IMAGENET_MANIFEST"
