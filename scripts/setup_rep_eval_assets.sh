#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DOWNLOAD_CHECKPOINTS="${DOWNLOAD_CHECKPOINTS:-1}"
DOWNLOAD_VAE="${DOWNLOAD_VAE:-1}"
DOWNLOAD_DINO="${DOWNLOAD_DINO:-1}"
DOWNLOAD_TINY="${DOWNLOAD_TINY:-1}"
DOWNLOAD_VOC="${DOWNLOAD_VOC:-1}"
DOWNLOAD_IMAGENET_HF4K="${DOWNLOAD_IMAGENET_HF4K:-1}"
DOWNLOAD_IMAGENET="${DOWNLOAD_IMAGENET:-0}"
SEED="${SEED:-123}"
IMAGENET_4K_COUNT="${IMAGENET_4K_COUNT:-4000}"

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
  mkdir -p "$out"
  echo "[setup] downloading $repo -> $out"
  if command -v hf >/dev/null 2>&1; then
    hf download "$repo" --local-dir "$out"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$repo" --local-dir "$out" --local-dir-use-symlinks False
  else
    echo "Missing Hugging Face CLI. Install/upgrade huggingface_hub, then run: hf auth login" >&2
    exit 2
  fi
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
config_ok = (target / "config.json").exists()
msgpack_ok = any((target / name).exists() for name in ("vae_params_bf16.msgpack", "diffusion_flax_model.msgpack", "flax_model.msgpack")) or bool(list(target.glob("*.msgpack")))
if config_ok and msgpack_ok:
    print(f"[setup] exists: {target}")
    raise SystemExit(0)

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
config_candidates = sorted(source.rglob("config.json"))
msgpack_candidates = []
for name in ("vae_params_bf16.msgpack", "diffusion_flax_model.msgpack", "flax_model.msgpack"):
    msgpack_candidates.extend(sorted(source.rglob(name)))
if not msgpack_candidates:
    msgpack_candidates = sorted(source.rglob("*.msgpack"))

if not config_candidates:
    raise SystemExit(f"Kaggle VAE download has no config.json under {source}")
if not msgpack_candidates:
    raise SystemExit(f"Kaggle VAE download has no .msgpack params under {source}")

shutil.copy2(config_candidates[0], target / "config.json")
chosen_msgpack = msgpack_candidates[0]
shutil.copy2(chosen_msgpack, target / chosen_msgpack.name)
print(f"[setup] VAE source: {source}")
print(f"[setup] VAE config: {config_candidates[0]}")
print(f"[setup] VAE params: {chosen_msgpack}")
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

if [[ "$DOWNLOAD_IMAGENET_HF4K" == "1" ]]; then
  echo "[setup] downloading 4k random ImageNet validation images from Hugging Face"
  IMAGENET_MANIFEST="$IMAGENET_MANIFEST" IMAGENET_4K_COUNT="$IMAGENET_4K_COUNT" SEED="$SEED" python3 - <<'PY'
from pathlib import Path
import os

from datasets import load_dataset
from huggingface_hub import get_token
from PIL import Image

count = int(os.environ.get("IMAGENET_4K_COUNT", "4000"))
seed = int(os.environ.get("SEED", "123"))
manifest = Path(os.environ["IMAGENET_MANIFEST"])
out_root = Path("data/imagenet-val/subset_4000_256")
repo_root = Path.cwd().resolve()

if manifest.exists():
    existing = []
    for raw in manifest.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists():
            existing.append(path)
    if len(existing) >= count:
        print(f"[setup] exists: {manifest} with {len(existing)} valid images")
        raise SystemExit(0)

token = get_token()
try:
    dataset = load_dataset(
        "ILSVRC/imagenet-1k",
        split="validation",
        streaming=True,
        token=token,
    )
except Exception as exc:
    raise SystemExit(
        "Could not open Hugging Face dataset ILSVRC/imagenet-1k. "
        "Log in with `hf auth login` and accept the dataset terms at "
        "https://huggingface.co/datasets/ILSVRC/imagenet-1k first. "
        f"Original error: {exc}"
    )

out_root.mkdir(parents=True, exist_ok=True)
manifest.parent.mkdir(parents=True, exist_ok=True)
dataset = dataset.shuffle(buffer_size=10000, seed=seed)

written = 0
with open(manifest, "w") as handle:
    for example in dataset:
        image = example["image"]
        label = int(example.get("label", -1))
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
        label_dir = out_root / str(label)
        label_dir.mkdir(parents=True, exist_ok=True)
        path = label_dir / f"imagenet_val_hf_seed{seed}_{written:05d}.JPEG"
        image.save(path, format="JPEG", quality=95)
        handle.write(f"{path.resolve().relative_to(repo_root)}\n")
        written += 1
        if written % 250 == 0:
            print(f"[setup] ImageNet HF subset: {written}/{count}")
        if written >= count:
            break

if written < count:
    raise SystemExit(f"Only wrote {written}/{count} ImageNet images")
print(f"[setup] wrote {manifest} with {written} ImageNet validation images")
PY
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
repo_root = Path.cwd().resolve()
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
                rel = path.resolve().relative_to(repo_root)
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
