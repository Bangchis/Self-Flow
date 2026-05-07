#!/usr/bin/env python3
"""LayerSync-style representation evaluation for Self-Flow checkpoints.

This script evaluates frozen intermediate representations with:
  1. Tiny ImageNet classification probes.
  2. PASCAL VOC semantic segmentation probes.
  3. Linear CKA against DINOv2-g.

The backbone and VAE are always frozen. Only probe heads are trained.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import functools
import hashlib
import json
import math
import os
import queue
import random
import shutil
import subprocess
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
if "--xla_gpu_strict_conv_algorithm_picker=false" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_strict_conv_algorithm_picker=false").strip()
warnings.filterwarnings("ignore", message="Flax classes are deprecated.*")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")

# Import torch before JAX. On this server that makes the CUDA shared libraries
# visible to jax_plugins.xla_cuda13.
import torch

torch.set_grad_enabled(False)

from diffusers.models import FlaxAutoencoderKL
from diffusers.utils import logging as diffusers_logging
import flax.serialization
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
from einops import rearrange
from flax.training import checkpoints
from numpy.lib.format import open_memmap
from PIL import Image
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model import SelfFlowDiT

diffusers_logging.set_verbosity_error()


DIT_VARIANTS = {
    "S": {"hidden_size": 384, "depth": 12, "num_heads": 6},
    "B": {"hidden_size": 768, "depth": 12, "num_heads": 12},
    "L": {"hidden_size": 1024, "depth": 24, "num_heads": 16},
    "XL": {"hidden_size": 1152, "depth": 28, "num_heads": 16},
}

SCALE_FACTOR = 0.18215
NUM_CLASSES_IMAGENET = 1000
NUM_CLASSES_TINY = 200
NUM_CLASSES_VOC = 21
CLASS_LINEAR_PROBES = {"simple_linear", "sra_style"}
CLASS_CNN_PROBES = {"resnet18_probe", "convnext_atto_probe", "convnext_tiny_probe"}
CLASS_PROBE_CHOICES = [
    "simple_linear",
    "sra_style",
    "resnet18_probe",
    "convnext_atto_probe",
    "convnext_tiny_probe",
]
CLASS_PROBE_LABELS = {
    "simple_linear": "MeanPool + Linear",
    "sra_style": "BN + Linear",
    "resnet18_probe": "ResNet18",
    "convnext_atto_probe": "ConvNeXt-Atto",
    "convnext_tiny_probe": "ConvNeXt-Tiny",
}
SEG_PROBE_TYPE = "convnext_tiny_seg"
SEG_PROBE_LABELS = {
    "simple_linear_seg": "1x1 Conv",
    "sra_inspired_dense_seg": "BN2d + 1x1 Conv",
    SEG_PROBE_TYPE: "ConvNeXt-Tiny dense",
}


@dataclass(frozen=True)
class Example:
    image_path: Path
    label: int | None = None
    mask_path: Path | None = None


@dataclass(frozen=True)
class LoadedCheckpoint:
    name: str
    model: SelfFlowDiT
    params: dict
    config: dict


WANDB_RUN = None
WANDB_PROGRESS_LAST: dict[str, float] = {}
TQDM_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"


def log(message: str) -> None:
    print(f"[representation_eval] {message}", flush=True)


def progress(iterable=None, **kwargs):
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("smoothing", 0.05)
    kwargs.setdefault("mininterval", 1.0)
    kwargs.setdefault("bar_format", TQDM_BAR_FORMAT)
    return tqdm(iterable, **kwargs)


def wandb_log(data: dict, step: int | None = None) -> None:
    if WANDB_RUN is None:
        return
    try:
        WANDB_RUN.log(data, step=step)
    except Exception as exc:
        log(f"W&B log skipped: {exc}")


def wandb_log_progress(prefix: str, completed: int, total: int, min_interval_s: float = 30.0, **extra: float | int | str) -> None:
    if WANDB_RUN is None:
        return
    now = time.time()
    last = WANDB_PROGRESS_LAST.get(prefix, 0.0)
    if completed < total and now - last < min_interval_s:
        return
    WANDB_PROGRESS_LAST[prefix] = now
    total_safe = max(int(total), 1)
    payload: dict[str, float | int | str] = {
        f"{prefix}/completed": int(completed),
        f"{prefix}/total": int(total),
        f"{prefix}/progress": float(completed) / float(total_safe),
    }
    for key, value in extra.items():
        payload[f"{prefix}/{key}"] = value
    wandb_log(payload)


def setup_wandb(
    args: argparse.Namespace,
    checkpoint_specs: list[tuple[str, Path]],
    layers: list[int],
    experiments: set[str],
) -> None:
    global WANDB_RUN
    if not args.wandb:
        return

    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    import wandb

    api_key = os.environ.get("WANDB_API_KEY")
    if api_key and args.wandb_mode != "disabled":
        wandb.login(key=api_key, relogin=False)

    WANDB_RUN = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config={
            "experiments": sorted(experiments),
            "checkpoints": [(name, str(path)) for name, path in checkpoint_specs],
            "param_set": args.param_set,
            "model_size": args.model_size,
            "layers": layers,
            "timestep": args.timestep,
            "condition_policy": "null/unconditional" if args.cfg_dropout_rate > 0 else "class_0_fallback",
            "global_batch_size": args.global_batch_size,
            "main_timestep": args.main_timestep,
            "eval_timesteps": args.eval_timesteps,
            "timestep_input_policy": "x_tau=(1-t)*GaussianNoise+t*clean_latent; t=1 is clean/least noisy",
            "classification_epochs": args.class_epochs,
            "segmentation_epochs": args.seg_epochs,
            "segmentation_probe_architecture": SEG_PROBE_TYPE,
            "segmentation_train_batch_size": args.seg_train_batch_size,
            "segmentation_eval_batch_size": args.seg_eval_batch_size,
            "segmentation_probe_projected_channels": args.seg_probe_proj_channels,
            "segmentation_cnn_layer_batch_size": args.seg_cnn_layer_batch_size,
            "segmentation_cnn_input_size": args.seg_cnn_input_size,
            "segmentation_cnn_out_indices": args.seg_cnn_out_indices,
            "classification_probe_architectures": args.class_probes,
            "classification_main_probe": args.class_main_probe,
            "classification_probe_projected_channels": args.class_probe_proj_channels,
            "classification_cnn_layer_batch_size": args.class_cnn_layer_batch_size,
            "classification_cnn_input_size": args.class_cnn_input_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "tiny_imagenet_root": args.tiny_imagenet_root,
            "voc_root": args.voc_root,
            "imagenet_manifest": args.imagenet_manifest,
            "dino_feature": args.dino_feature,
            "cka_model_features": args.cka_model_features,
            "cka_dino_features": args.cka_dino_features,
            "cka_condition_policies": args.cka_condition_policies,
            "cka_main_scenario": args.cka_main_scenario,
            "dino_input_size": args.dino_input_size,
            "parallel_worker": args.parallel_worker,
            "wandb_workers": args.wandb_workers,
        },
    )


def finish_wandb() -> None:
    global WANDB_RUN
    if WANDB_RUN is not None:
        WANDB_RUN.finish()
        WANDB_RUN = None


def wandb_log_table(name: str, rows: list[dict], fieldnames: list[str]) -> None:
    if WANDB_RUN is None or not rows:
        return
    try:
        import wandb

        table = wandb.Table(
            columns=fieldnames,
            data=[[row.get(field) for field in fieldnames] for row in rows],
        )
        wandb_log({f"tables/{name}": table})
    except Exception as exc:
        log(f"W&B table log skipped for {name}: {exc}")


def wandb_log_image(name: str, path: Path) -> None:
    if WANDB_RUN is None or not path.exists():
        return
    try:
        import wandb

        wandb_log({f"figures/{name}": wandb.Image(str(path))})
    except Exception as exc:
        log(f"W&B image log skipped for {name}: {exc}")


def wandb_log_result_artifact(output_dir: Path) -> None:
    if WANDB_RUN is None:
        return
    try:
        import wandb

        artifact_name = sanitize_name(f"{output_dir.name}_outputs")
        artifact = wandb.Artifact(artifact_name, type="representation-eval")
        added = False
        for filename in [
            "classification_probe.csv",
            "segmentation_probe.csv",
            "cka_dinov2g.csv",
            "representation_eval_main.png",
            "probe_arch_ablation.png",
            "classification_probe_heatmap.png",
            "segmentation_probe_heatmap.png",
            "cka_heatmap.png",
            "README.md",
        ]:
            path = output_dir / filename
            if path.exists():
                artifact.add_file(str(path), name=filename)
                added = True
        if added:
            WANDB_RUN.log_artifact(artifact)
    except Exception as exc:
        log(f"W&B artifact log skipped: {exc}")


def wandb_log_metric_rows(prefix: str, rows: list[dict], metric_name: str) -> None:
    if WANDB_RUN is None:
        return
    payload = {}
    for row in rows:
        parts = [prefix, row["model_name"]]
        if "probe_type" in row:
            parts.append(row["probe_type"])
        if "cka_scenario" in row:
            parts.append(row["cka_scenario"])
        if "timestep" in row:
            parts.append(f"t{timestep_tag(float(row['timestep']))}")
        parts.append(f"layer_{row['layer_index']}")
        payload["/".join(parts)] = float(row[metric_name])
    if payload:
        wandb_log(payload)


def tree_to_numpy_dict(tree: dict, prefix: str = "") -> dict[str, np.ndarray]:
    arrays = {}
    for key, value in tree.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            arrays.update(tree_to_numpy_dict(value, name))
        else:
            arrays[name] = np.asarray(jax.device_get(value))
    return arrays


def save_probe_artifact(
    output_dir: Path,
    model_name: str,
    experiment: str,
    arrays: dict,
    metadata: dict,
) -> tuple[Path, Path]:
    probe_dir = output_dir / "probes" / sanitize_name(model_name)
    probe_dir.mkdir(parents=True, exist_ok=True)
    tag = timestep_tag(float(metadata["timestep"])) if "timestep" in metadata else None
    stem = f"{experiment}_t{tag}_probe" if tag else f"{experiment}_probe"
    npz_path = probe_dir / f"{stem}.npz"
    json_path = probe_dir / f"{stem}_metadata.json"
    np.savez_compressed(npz_path, **tree_to_numpy_dict(arrays))
    with open(json_path, "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    log(f"Wrote {npz_path}")
    log(f"Wrote {json_path}")
    return npz_path, json_path


def save_torch_probe_artifact(
    output_dir: Path,
    model_name: str,
    experiment: str,
    state: dict,
    metadata: dict,
) -> tuple[Path, Path]:
    probe_dir = output_dir / "probes" / sanitize_name(model_name)
    probe_dir.mkdir(parents=True, exist_ok=True)
    tag = timestep_tag(float(metadata["timestep"])) if "timestep" in metadata else None
    stem = f"{experiment}_t{tag}_probe" if tag else f"{experiment}_probe"
    pt_path = probe_dir / f"{stem}.pt"
    json_path = probe_dir / f"{stem}_metadata.json"
    cpu_state = {}
    for key, value in state.items():
        if isinstance(value, dict):
            cpu_state[key] = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in value.items()}
        else:
            cpu_state[key] = value.detach().cpu() if torch.is_tensor(value) else value
    torch.save({"state": cpu_state, "metadata": metadata}, pt_path)
    with open(json_path, "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    log(f"Wrote {pt_path}")
    log(f"Wrote {json_path}")
    return pt_path, json_path


def resolve_hf_token(args: argparse.Namespace) -> str | None:
    token = (
        args.hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def write_hf_readme(
    output_dir: Path,
    args: argparse.Namespace,
    checkpoint_specs: list[tuple[str, Path]],
    layers: list[int],
    experiments: set[str],
) -> Path:
    readme_path = output_dir / "README.md"
    lines = [
        "---",
        "license: other",
        "tags:",
            "- representation-evaluation",
            "- cnn-probe",
        "- self-flow",
        "- layersync-style",
        "---",
        "",
        "# Self-Flow Representation Evaluation",
        "",
        "This repository contains LayerSync Figure-4-style representation evaluation outputs.",
        "",
        "## Protocol",
        "",
        f"- Experiments: {', '.join(sorted(experiments))}",
        f"- Layers: {', '.join(str(layer) for layer in layers)}",
        f"- Main timestep: {args.main_timestep}",
        f"- Evaluated timesteps: {', '.join(str(t) for t in args.eval_timesteps)}",
        "- Timestep input policy: `x_tau = (1 - t) * GaussianNoise + t * clean_latent`; repo convention is `t=1` clean/least noisy.",
        f"- Global batch size: {args.global_batch_size}",
        f"- Classification epochs: {args.class_epochs}",
        f"- Segmentation epochs: {args.seg_epochs}",
        f"- Segmentation probe: {SEG_PROBE_LABELS[SEG_PROBE_TYPE]}",
        f"- Segmentation train batch size: {args.seg_train_batch_size}",
        f"- Segmentation layer batch size: {args.seg_cnn_layer_batch_size}",
        f"- CKA model features: {', '.join(args.cka_model_features)}",
        f"- CKA DINO features: {', '.join(args.cka_dino_features)}",
        f"- CKA condition policies: {', '.join(args.cka_condition_policies)}",
        f"- CKA main scenario: {args.cka_main_scenario}",
        "",
        "## Checkpoints",
        "",
    ]
    for name, path in checkpoint_specs:
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `classification_probe.csv`: Tiny ImageNet validation accuracy per layer/probe. Default probe is `convnext_tiny_probe`.",
            "- `segmentation_probe.csv`: PASCAL VOC mIoU per layer/probe.",
            "- `cka_dinov2g.csv`: centered linear CKA vs DINOv2-g. Default scenario is `summary__null__dinov2_cls`.",
            "- `representation_eval_main.png`: main 3-panel figure.",
            "- `probe_arch_ablation.png`: classification/segmentation probe architecture ablations.",
            "- `classification_probe_heatmap.png`: layer × timestep Tiny ImageNet heatmap when multiple timesteps are evaluated.",
            "- `segmentation_probe_heatmap.png`: layer × timestep PASCAL VOC heatmap when multiple timesteps are evaluated.",
            "- `cka_heatmap.png`: layer × timestep CKA heatmap when multiple timesteps are evaluated.",
            "- `probes/`: trained probe parameters plus metadata JSON.",
            "",
            "`feature_cache/`, `_parallel_workers/`, `*.features.npy`, and `*.labels.npy` are excluded by default from uploads because they can be large.",
            "",
        ]
    )
    readme_path.write_text("\n".join(lines))
    return readme_path


def upload_results_to_hf(
    output_dir: Path,
    args: argparse.Namespace,
    checkpoint_specs: list[tuple[str, Path]],
    layers: list[int],
    experiments: set[str],
) -> str | None:
    if not args.hf_upload:
        return None
    token = resolve_hf_token(args)
    if not token:
        raise ValueError("HF upload requested but no token was provided. Set HF_TOKEN or pass --hf-token.")

    from huggingface_hub import HfApi

    api = HfApi()
    repo_id = args.hf_repo_id
    if not repo_id:
        whoami = api.whoami(token=token)
        username = whoami.get("name")
        if not username:
            raise RuntimeError("Could not infer Hugging Face username from token; pass --hf-repo-id.")
        repo_id = f"{username}/self-flow-representation-eval"

    write_hf_readme(output_dir, args, checkpoint_specs, layers, experiments)
    api.create_repo(
        repo_id=repo_id,
        repo_type=args.hf_repo_type,
        private=args.hf_private,
        exist_ok=True,
        token=token,
    )
    ignore_patterns = []
    if not args.hf_include_feature_cache:
        ignore_patterns.extend(
            [
                "feature_cache",
                "feature_cache/**",
                "**/feature_cache",
                "**/feature_cache/**",
                "*.features.npy",
                "**/*.features.npy",
                "*.labels.npy",
                "**/*.labels.npy",
                "shared_cache",
                "shared_cache/**",
                "**/shared_cache",
                "**/shared_cache/**",
            ]
        )
    ignore_patterns.extend(["_parallel_workers", "_parallel_workers/**", "**/_parallel_workers", "**/_parallel_workers/**"])
    if args.hf_ignore_patterns:
        ignore_patterns.extend(pattern.strip() for pattern in args.hf_ignore_patterns.split(",") if pattern.strip())

    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type=args.hf_repo_type,
        folder_path=str(output_dir),
        path_in_repo=args.hf_path_in_repo or None,
        ignore_patterns=ignore_patterns or None,
        delete_patterns=f"{args.hf_path_in_repo.strip('/')}/*" if args.hf_path_in_repo else None,
        commit_message=args.hf_commit_message,
        token=token,
    )
    url = f"https://huggingface.co/{'datasets/' if args.hf_repo_type == 'dataset' else ''}{repo_id}"
    if args.hf_path_in_repo:
        url = f"{url}/tree/main/{args.hf_path_in_repo.strip('/')}"
    log(f"Uploaded results to Hugging Face: {url}")
    wandb_log({"huggingface/repo_url": url, "huggingface/commit": getattr(commit, "oid", "")})
    return url


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def timestep_tag(timestep: float) -> str:
    return f"{float(timestep):.4g}".replace("-", "m").replace(".", "p")


def parse_float_list(value: str | None, default: float) -> list[float]:
    if value is None or str(value).strip() == "":
        return [float(default)]
    items = []
    for raw in str(value).replace(";", ",").split(","):
        raw = raw.strip()
        if raw:
            items.append(float(raw))
    if not items:
        return [float(default)]
    return list(dict.fromkeys(items))


def parse_str_list(value: str | None, default: list[str]) -> list[str]:
    if value is None or str(value).strip() == "":
        return list(default)
    items = []
    for raw in str(value).replace(";", ",").split(","):
        raw = raw.strip()
        if raw:
            items.append(raw)
    return list(dict.fromkeys(items)) if items else list(default)


def parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if value is None or str(value).strip() == "":
        return list(default)
    items = []
    for raw in str(value).replace(";", ",").split(","):
        raw = raw.strip()
        if raw:
            items.append(int(raw))
    return list(dict.fromkeys(items)) if items else list(default)


def parse_class_probes(value: str | None) -> list[str]:
    probes = parse_str_list(value, CLASS_PROBE_CHOICES)
    unknown = set(probes) - set(CLASS_PROBE_CHOICES)
    if unknown:
        raise ValueError(f"Unknown --class-probes: {sorted(unknown)}. Valid: {CLASS_PROBE_CHOICES}")
    return probes


def parse_layers(value: str | None, depth: int) -> list[int]:
    if value is None or value.strip().lower() in {"", "all"}:
        return list(range(1, depth + 1))
    layers = []
    for piece in value.replace(";", ",").split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            start, end = piece.split("-", 1)
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(piece))
    layers = sorted(dict.fromkeys(layers))
    if not layers or layers[0] < 1 or layers[-1] > depth:
        raise ValueError(f"Invalid layer list {layers}; valid range is 1..{depth}")
    return layers


def parse_checkpoint_specs(values: list[str]) -> list[tuple[str, Path]]:
    if not values:
        return [("method", Path("checkpoints/LARA-XL/latest"))]
    specs = []
    for raw in values:
        if "=" in raw:
            name, path = raw.split("=", 1)
            specs.append((sanitize_name(name.strip()), Path(path.strip())))
        else:
            path = Path(raw)
            specs.append((sanitize_name(path.name or "model"), path))
    return specs


def checkpoint_spec_to_cli(spec: tuple[str, Path]) -> str:
    name, path = spec
    return f"{name}={path}"


def build_model_config(model_size: str, class_dropout_prob: float) -> dict:
    model_size = model_size.upper()
    if model_size not in DIT_VARIANTS:
        raise ValueError(f"Unknown model size {model_size!r}; expected one of {sorted(DIT_VARIANTS)}")
    variant = DIT_VARIANTS[model_size]
    return {
        "input_size": 32,
        "patch_size": 2,
        "in_channels": 4,
        "hidden_size": variant["hidden_size"],
        "depth": variant["depth"],
        "num_heads": variant["num_heads"],
        "mlp_ratio": 4.0,
        "num_classes": NUM_CLASSES_IMAGENET,
        "learn_sigma": True,
        "compatibility_mode": True,
        "class_dropout_prob": float(class_dropout_prob),
    }


def init_backbone(model_size: str, class_dropout_prob: float) -> tuple[SelfFlowDiT, dict, dict]:
    config = build_model_config(model_size, class_dropout_prob)
    model = SelfFlowDiT(**config, per_token=False)
    patch_dim = config["in_channels"] * config["patch_size"] ** 2
    n_patches = (config["input_size"] // config["patch_size"]) ** 2
    variables = model.init(
        jax.random.PRNGKey(0),
        jnp.ones((1, n_patches, patch_dim), dtype=jnp.float32),
        timesteps=jnp.ones((1,), dtype=jnp.float32),
        vector=jnp.ones((1,), dtype=jnp.int32),
        deterministic=True,
    )
    return model, variables["params"], config


def resolve_param_checkpoint_dir(root: Path, param_set: str) -> Path:
    root = root.resolve()
    if root.is_file():
        return root.parent
    if root.name.startswith("checkpoint_"):
        return root.parent
    if param_set == "ema" and (root / "ema").is_file():
        return root
    if param_set == "ema" and (root / "ema").is_dir():
        return root / "ema"
    if param_set == "online":
        return root
    if param_set != "ema":
        custom = root / param_set
        if custom.is_dir():
            return custom
    return root


def load_checkpoint(name: str, path: Path, args: argparse.Namespace) -> LoadedCheckpoint:
    model, template, config = init_backbone(args.model_size, args.cfg_dropout_rate)
    ckpt_dir = resolve_param_checkpoint_dir(path, args.param_set)
    log(f"Loading {name} from {ckpt_dir}")
    params = checkpoints.restore_checkpoint(str(ckpt_dir), target=template)
    if isinstance(params, dict) and "backbone" in params:
        params = params["backbone"]
    if isinstance(params, dict) and "feature_head" in params:
        params = {key: value for key, value in params.items() if key != "feature_head"}
    return LoadedCheckpoint(name=name, model=model, params=params, config=config)


def load_flax_vae(path: Path) -> tuple[FlaxAutoencoderKL, dict]:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    msgpack_candidates = [
        path / "flax_model.msgpack",
        path / "diffusion_flax_model.msgpack",
        path / "vae_params_bf16.msgpack",
    ]
    msgpack_path = next((candidate for candidate in msgpack_candidates if candidate.exists()), None)
    if msgpack_path is None:
        msgpacks = sorted(path.glob("*.msgpack"))
        if msgpacks:
            msgpack_path = msgpacks[0]
    if msgpack_path is None:
        msgpacks = sorted(path.rglob("*.msgpack"))
        if msgpacks:
            msgpack_path = msgpacks[0]
    if msgpack_path is None:
        raise FileNotFoundError(f"No Flax msgpack found in {path}")
    config_path = path / "config.json"
    if not config_path.exists():
        configs = sorted(path.rglob("config.json"))
        if configs:
            config_path = configs[0]
        else:
            raise FileNotFoundError(f"Missing VAE config.json in {path}")

    vae = FlaxAutoencoderKL.from_config(str(config_path))
    with open(msgpack_path, "rb") as handle:
        params = flax.serialization.from_bytes(None, handle.read())
    params = jax.tree_util.tree_map(jnp.asarray, params)
    log(f"Loaded VAE params from {msgpack_path}")
    return vae, params


def patchify_latents(latents_nhwc: jax.Array) -> jax.Array:
    latents_nchw = jnp.transpose(latents_nhwc, (0, 3, 1, 2))
    return rearrange(
        latents_nchw,
        "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
        p1=2,
        p2=2,
    )


def encode_images_to_tokens(vae: FlaxAutoencoderKL, vae_params: dict, images_nhwc: jax.Array) -> jax.Array:
    # Diffusers Flax VAE encode in this repo expects NCHW input. The latent
    # distribution mean comes back as NHWC, matching prepare_data_tpu.py.
    images_nchw = jnp.transpose(images_nhwc.astype(jnp.float32), (0, 3, 1, 2))
    latent_dist = vae.apply({"params": vae_params}, images_nchw, method=vae.encode).latent_dist
    return patchify_latents(latent_dist.mean * SCALE_FACTOR)


def apply_eval_timestep_noise(tokens: jax.Array, timestep: float, noise_seed: jax.Array) -> jax.Array:
    # Repo convention: x_tau = (1 - tau) * noise + tau * clean_latent.
    tau = float(timestep)
    if tau >= 1.0 - 1e-8:
        return tokens
    noise = jax.random.normal(jax.random.PRNGKey(noise_seed.astype(jnp.uint32)), tokens.shape, dtype=tokens.dtype)
    return (1.0 - tau) * noise + tau * tokens


def batch_noise_seed(seed: int, timestep: float, indices: np.ndarray, stream: str) -> np.uint32:
    arr = np.asarray(indices, dtype=np.int64).reshape(-1)
    payload = f"{seed}:{float(timestep):.6f}:{stream}:{arr.tolist()}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=4).digest()
    return np.uint32(int.from_bytes(digest, "little"))


def make_summary_extractor(
    ckpt: LoadedCheckpoint,
    vae: FlaxAutoencoderKL,
    vae_params: dict,
    layers: list[int],
    timestep: float,
    null_label: int,
):
    layer_idx = jnp.asarray([layer - 1 for layer in layers], dtype=jnp.int32)
    raw_layers = tuple(int(layer) for layer in layers)
    use_raw_subset = len(raw_layers) < int(ckpt.config["depth"])

    @jax.jit
    def extract_with_params(backbone_params: dict, vae_params_arg: dict, images_nhwc: jax.Array, noise_seed: jax.Array) -> jax.Array:
        tokens = encode_images_to_tokens(vae, vae_params_arg, images_nhwc)
        tokens = apply_eval_timestep_noise(tokens, timestep, noise_seed)
        batch_size = tokens.shape[0]
        timesteps = jnp.full((batch_size,), float(timestep), dtype=jnp.float32)
        labels = jnp.full((batch_size,), int(null_label), dtype=jnp.int32)
        if use_raw_subset:
            _, raw = ckpt.model.apply(
                {"params": backbone_params},
                tokens,
                timesteps=timesteps,
                vector=labels,
                return_raw_features=raw_layers,
                deterministic=True,
            )
            if len(raw_layers) == 1:
                features = jnp.expand_dims(raw, axis=0)
            else:
                features = jnp.stack(raw, axis=0)
            return features.astype(jnp.float32).mean(axis=2)
        _, summaries = ckpt.model.apply(
            {"params": backbone_params},
            tokens,
            timesteps=timesteps,
            vector=labels,
            return_block_summaries=True,
            deterministic=True,
        )
        summaries = summaries.astype(jnp.float32)
        return jnp.take(summaries, layer_idx, axis=0)

    def extract(images_nhwc: jax.Array, noise_seed: jax.Array = jnp.asarray(0, dtype=jnp.uint32)) -> jax.Array:
        return extract_with_params(ckpt.params, vae_params, images_nhwc, noise_seed)

    return extract


def make_dense_extractor(
    ckpt: LoadedCheckpoint,
    vae: FlaxAutoencoderKL,
    vae_params: dict,
    layers: list[int],
    timestep: float,
    null_label: int,
):
    raw_layers = tuple(int(layer) for layer in layers)

    @jax.jit
    def extract_with_params(backbone_params: dict, vae_params_arg: dict, images_nhwc: jax.Array, noise_seed: jax.Array) -> jax.Array:
        tokens = encode_images_to_tokens(vae, vae_params_arg, images_nhwc)
        tokens = apply_eval_timestep_noise(tokens, timestep, noise_seed)
        batch_size = tokens.shape[0]
        timesteps = jnp.full((batch_size,), float(timestep), dtype=jnp.float32)
        labels = jnp.full((batch_size,), int(null_label), dtype=jnp.int32)
        _, raw = ckpt.model.apply(
            {"params": backbone_params},
            tokens,
            timesteps=timesteps,
            vector=labels,
            return_raw_features=raw_layers,
            deterministic=True,
        )
        if len(raw_layers) == 1:
            features = jnp.expand_dims(raw, axis=0)
        else:
            features = jnp.stack(raw, axis=0)
        return features.astype(jnp.float32)

    def extract(images_nhwc: jax.Array, noise_seed: jax.Array = jnp.asarray(0, dtype=jnp.uint32)) -> jax.Array:
        return extract_with_params(ckpt.params, vae_params, images_nhwc, noise_seed)

    return extract


def make_projected_grid_extractor(
    ckpt: LoadedCheckpoint,
    vae: FlaxAutoencoderKL,
    vae_params: dict,
    layers: list[int],
    timestep: float,
    null_label: int,
    proj_channels: int,
    seed: int,
):
    raw_layers = tuple(int(layer) for layer in layers)
    rng = np.random.default_rng(seed)
    projection_np = rng.standard_normal((ckpt.config["hidden_size"], proj_channels), dtype=np.float32)
    projection_np = projection_np / math.sqrt(float(ckpt.config["hidden_size"]))
    projection = jnp.asarray(projection_np, dtype=jnp.float32)

    @jax.jit
    def extract_with_params(backbone_params: dict, vae_params_arg: dict, images_nhwc: jax.Array, noise_seed: jax.Array) -> jax.Array:
        tokens = encode_images_to_tokens(vae, vae_params_arg, images_nhwc)
        tokens = apply_eval_timestep_noise(tokens, timestep, noise_seed)
        batch_size = tokens.shape[0]
        timesteps = jnp.full((batch_size,), float(timestep), dtype=jnp.float32)
        labels = jnp.full((batch_size,), int(null_label), dtype=jnp.int32)
        _, raw = ckpt.model.apply(
            {"params": backbone_params},
            tokens,
            timesteps=timesteps,
            vector=labels,
            return_raw_features=raw_layers,
            deterministic=True,
        )
        if len(raw_layers) == 1:
            features = jnp.expand_dims(raw, axis=0)
        else:
            features = jnp.stack(raw, axis=0)
        n_tokens = features.shape[2]
        grid = int(math.sqrt(int(n_tokens)))
        features = jnp.reshape(features.astype(jnp.float32), (len(raw_layers), batch_size, grid, grid, features.shape[-1]))
        return jnp.einsum("lbhwc,cp->lbphw", features, projection)

    def extract(images_nhwc: jax.Array, noise_seed: jax.Array = jnp.asarray(0, dtype=jnp.uint32)) -> jax.Array:
        return extract_with_params(ckpt.params, vae_params, images_nhwc, noise_seed)

    return extract


def make_projected_grid_and_summary_extractor(
    ckpt: LoadedCheckpoint,
    vae: FlaxAutoencoderKL,
    vae_params: dict,
    layers: list[int],
    timestep: float,
    null_label: int,
    proj_channels: int,
    seed: int,
):
    raw_layers = tuple(int(layer) for layer in layers)
    rng = np.random.default_rng(seed)
    projection_np = rng.standard_normal((ckpt.config["hidden_size"], proj_channels), dtype=np.float32)
    projection_np = projection_np / math.sqrt(float(ckpt.config["hidden_size"]))
    projection = jnp.asarray(projection_np, dtype=jnp.float32)

    @jax.jit
    def extract_with_params(
        backbone_params: dict,
        vae_params_arg: dict,
        images_nhwc: jax.Array,
        noise_seed: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        tokens = encode_images_to_tokens(vae, vae_params_arg, images_nhwc)
        tokens = apply_eval_timestep_noise(tokens, timestep, noise_seed)
        batch_size = tokens.shape[0]
        timesteps = jnp.full((batch_size,), float(timestep), dtype=jnp.float32)
        labels = jnp.full((batch_size,), int(null_label), dtype=jnp.int32)
        _, raw = ckpt.model.apply(
            {"params": backbone_params},
            tokens,
            timesteps=timesteps,
            vector=labels,
            return_raw_features=raw_layers,
            deterministic=True,
        )
        if len(raw_layers) == 1:
            features = jnp.expand_dims(raw, axis=0)
        else:
            features = jnp.stack(raw, axis=0)
        features = features.astype(jnp.float32)
        summaries = jnp.mean(features, axis=2)
        n_tokens = features.shape[2]
        grid = int(math.sqrt(int(n_tokens)))
        grid_features = jnp.reshape(features, (len(raw_layers), batch_size, grid, grid, features.shape[-1]))
        projected = jnp.einsum("lbhwc,cp->lbphw", grid_features, projection)
        return projected, summaries

    def extract(
        images_nhwc: jax.Array,
        noise_seed: jax.Array = jnp.asarray(0, dtype=jnp.uint32),
    ) -> tuple[jax.Array, jax.Array]:
        return extract_with_params(ckpt.params, vae_params, images_nhwc, noise_seed)

    return extract


def make_cka_summary_extractor(
    ckpt: LoadedCheckpoint,
    vae: FlaxAutoencoderKL,
    vae_params: dict,
    layers: list[int],
    timestep: float,
    null_label: int,
    condition_policy: str,
):
    layer_idx = jnp.asarray([layer - 1 for layer in layers], dtype=jnp.int32)
    raw_layers = tuple(int(layer) for layer in layers)
    use_raw_subset = len(raw_layers) < int(ckpt.config["depth"])
    use_labels = condition_policy == "label"

    @jax.jit
    def extract_with_params(
        backbone_params: dict,
        vae_params_arg: dict,
        images_nhwc: jax.Array,
        labels_arg: jax.Array,
        noise_seed: jax.Array,
    ) -> jax.Array:
        tokens = encode_images_to_tokens(vae, vae_params_arg, images_nhwc)
        tokens = apply_eval_timestep_noise(tokens, timestep, noise_seed)
        batch_size = tokens.shape[0]
        timesteps = jnp.full((batch_size,), float(timestep), dtype=jnp.float32)
        labels = labels_arg.astype(jnp.int32) if use_labels else jnp.full((batch_size,), int(null_label), dtype=jnp.int32)
        if use_raw_subset:
            _, raw = ckpt.model.apply(
                {"params": backbone_params},
                tokens,
                timesteps=timesteps,
                vector=labels,
                return_raw_features=raw_layers,
                deterministic=True,
            )
            if len(raw_layers) == 1:
                features = jnp.expand_dims(raw, axis=0)
            else:
                features = jnp.stack(raw, axis=0)
            return features.astype(jnp.float32).mean(axis=2)
        _, summaries = ckpt.model.apply(
            {"params": backbone_params},
            tokens,
            timesteps=timesteps,
            vector=labels,
            return_block_summaries=True,
            deterministic=True,
        )
        return jnp.take(summaries.astype(jnp.float32), layer_idx, axis=0)

    def extract(
        images_nhwc: jax.Array,
        labels: jax.Array,
        noise_seed: jax.Array = jnp.asarray(0, dtype=jnp.uint32),
    ) -> jax.Array:
        return extract_with_params(ckpt.params, vae_params, images_nhwc, labels, noise_seed)

    return extract


def make_cka_raw_mean_extractor(
    ckpt: LoadedCheckpoint,
    vae: FlaxAutoencoderKL,
    vae_params: dict,
    layers: list[int],
    timestep: float,
    null_label: int,
    condition_policy: str,
):
    raw_layers = tuple(int(layer) for layer in layers)
    use_labels = condition_policy == "label"

    @jax.jit
    def extract_with_params(
        backbone_params: dict,
        vae_params_arg: dict,
        images_nhwc: jax.Array,
        labels_arg: jax.Array,
        noise_seed: jax.Array,
    ) -> jax.Array:
        tokens = encode_images_to_tokens(vae, vae_params_arg, images_nhwc)
        tokens = apply_eval_timestep_noise(tokens, timestep, noise_seed)
        batch_size = tokens.shape[0]
        timesteps = jnp.full((batch_size,), float(timestep), dtype=jnp.float32)
        labels = labels_arg.astype(jnp.int32) if use_labels else jnp.full((batch_size,), int(null_label), dtype=jnp.int32)
        _, raw = ckpt.model.apply(
            {"params": backbone_params},
            tokens,
            timesteps=timesteps,
            vector=labels,
            return_raw_features=raw_layers,
            deterministic=True,
        )
        if len(raw_layers) == 1:
            features = jnp.expand_dims(raw, axis=0)
        else:
            features = jnp.stack(raw, axis=0)
        return features.astype(jnp.float32).mean(axis=2)

    def extract(
        images_nhwc: jax.Array,
        labels: jax.Array,
        noise_seed: jax.Array = jnp.asarray(0, dtype=jnp.uint32),
    ) -> jax.Array:
        return extract_with_params(ckpt.params, vae_params, images_nhwc, labels, noise_seed)

    return extract


def make_cka_spatial_extractor(
    ckpt: LoadedCheckpoint,
    vae: FlaxAutoencoderKL,
    vae_params: dict,
    layers: list[int],
    timestep: float,
    null_label: int,
    condition_policy: str,
):
    raw_layers = tuple(int(layer) for layer in layers)
    use_labels = condition_policy == "label"

    @jax.jit
    def extract_with_params(
        backbone_params: dict,
        vae_params_arg: dict,
        images_nhwc: jax.Array,
        labels_arg: jax.Array,
        noise_seed: jax.Array,
    ) -> jax.Array:
        tokens = encode_images_to_tokens(vae, vae_params_arg, images_nhwc)
        tokens = apply_eval_timestep_noise(tokens, timestep, noise_seed)
        batch_size = tokens.shape[0]
        timesteps = jnp.full((batch_size,), float(timestep), dtype=jnp.float32)
        labels = labels_arg.astype(jnp.int32) if use_labels else jnp.full((batch_size,), int(null_label), dtype=jnp.int32)
        _, raw = ckpt.model.apply(
            {"params": backbone_params},
            tokens,
            timesteps=timesteps,
            vector=labels,
            return_raw_features=raw_layers,
            deterministic=True,
        )
        if len(raw_layers) == 1:
            features = jnp.expand_dims(raw, axis=0)
        else:
            features = jnp.stack(raw, axis=0)
        return features.astype(jnp.float32)

    def extract(
        images_nhwc: jax.Array,
        labels: jax.Array,
        noise_seed: jax.Array = jnp.asarray(0, dtype=jnp.uint32),
    ) -> jax.Array:
        return extract_with_params(ckpt.params, vae_params, images_nhwc, labels, noise_seed)

    return extract


def image_to_vae_array(path: Path, size: int = 256, mode: str = "resize") -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        if mode == "fit":
            img = img.resize((size, size), Image.Resampling.BICUBIC)
        else:
            img = img.resize((size, size), Image.Resampling.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return arr


def dino_effective_size(size: int, patch_size: int = 14) -> int:
    if size % patch_size == 0:
        return size
    return (size // patch_size) * patch_size


def image_to_dino_tensor(path: Path, size: int = 256) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        effective = dino_effective_size(size)
        if effective != size:
            left = (size - effective) // 2
            top = (size - effective) // 2
            img = img.crop((left, top, left + effective, top + effective))
        arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (tensor - mean) / std


def dino_patch_tokens_to_grid(tokens: torch.Tensor, target_grid: int) -> torch.Tensor:
    batch, n_tokens, channels = tokens.shape
    source_grid = int(math.sqrt(int(n_tokens)))
    if source_grid * source_grid != int(n_tokens):
        raise ValueError(f"DINO patch token count is not square: {n_tokens}")
    grid = tokens.reshape(batch, source_grid, source_grid, channels).permute(0, 3, 1, 2)
    if source_grid != target_grid:
        grid = torch.nn.functional.interpolate(
            grid.float(),
            size=(target_grid, target_grid),
            mode="bilinear",
            align_corners=False,
        )
    return grid.permute(0, 2, 3, 1).reshape(batch, target_grid * target_grid, channels)


def dino_features_from_output(out: dict, dino_feature: str, target_grid: int | None = None) -> torch.Tensor:
    if dino_feature == "cls":
        return out["x_norm_clstoken"]
    patch_tokens = out["x_norm_patchtokens"]
    if dino_feature == "patch_mean":
        return patch_tokens.mean(dim=1)
    if dino_feature == "patch_tokens":
        if target_grid is None:
            raise ValueError("target_grid is required for DINO patch_tokens features")
        return dino_patch_tokens_to_grid(patch_tokens, target_grid)
    raise ValueError(f"Unknown DINO feature: {dino_feature}")


def load_voc_mask(path: Path, size: int = 256) -> np.ndarray:
    with Image.open(path) as mask:
        mask = mask.resize((size, size), Image.Resampling.NEAREST)
        arr = np.asarray(mask, dtype=np.int32)
    return arr


def limit_examples(examples: list[Example], limit: int | None, seed: int) -> list[Example]:
    if limit is None or limit <= 0 or limit >= len(examples):
        return examples
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    selected = sorted(indices[:limit])
    return [examples[index] for index in selected]


def batched_indices(n_items: int, batch_size: int, shuffle: bool, seed: int) -> Iterable[np.ndarray]:
    indices = np.arange(n_items)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, n_items, batch_size):
        yield indices[start : start + batch_size]


def load_tiny_imagenet(root: Path, seed: int, train_limit: int | None, val_limit: int | None):
    root = root.resolve()
    wnids = [line.strip() for line in (root / "wnids.txt").read_text().splitlines() if line.strip()]
    class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}

    train = []
    for wnid in wnids:
        image_dir = root / "train" / wnid / "images"
        for image_path in sorted(image_dir.glob("*.JPEG")):
            train.append(Example(image_path=image_path, label=class_to_idx[wnid]))

    val_annotations = root / "val" / "val_annotations.txt"
    val = []
    with open(val_annotations) as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            image_name, wnid = parts[:2]
            val.append(Example(image_path=root / "val" / "images" / image_name, label=class_to_idx[wnid]))

    return (
        limit_examples(train, train_limit, seed),
        limit_examples(val, val_limit, seed + 1),
    )


def load_voc(root: Path, seed: int, train_limit: int | None, val_limit: int | None):
    root = root.resolve()

    def read_split(split: str) -> list[Example]:
        ids = [line.strip() for line in (root / "ImageSets" / "Segmentation" / f"{split}.txt").read_text().splitlines()]
        examples = []
        for item_id in ids:
            if not item_id:
                continue
            examples.append(
                Example(
                    image_path=root / "JPEGImages" / f"{item_id}.jpg",
                    mask_path=root / "SegmentationClass" / f"{item_id}.png",
                )
            )
        return examples

    return (
        limit_examples(read_split("train"), train_limit, seed),
        limit_examples(read_split("val"), val_limit, seed + 1),
    )


def load_imagenet_manifest(manifest: Path, repo_root: Path, limit: int | None, seed: int) -> list[Example]:
    examples = []
    with open(manifest) as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            path = Path(raw)
            if not path.is_absolute():
                path = repo_root / path
            label = None
            with contextlib.suppress(ValueError):
                label = int(path.parent.name)
            examples.append(Example(image_path=path, label=label))
    return limit_examples(examples, limit, seed)


def cache_signature(parts: list[str]) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:12]


def feature_cache_paths(cache_dir: Path, prefix: str, model_name: str, split: str, layers: list[int], timestep: float, n_items: int):
    layer_sig = ",".join(str(layer) for layer in layers)
    sig = cache_signature([prefix, model_name, split, layer_sig, f"{timestep:.6f}", str(n_items)])
    base = cache_dir / f"{prefix}_{sanitize_name(model_name)}_{split}_{sig}"
    return base.with_suffix(".features.npy"), base.with_suffix(".labels.npy")


def extract_summary_cache(
    examples: list[Example],
    ckpt: LoadedCheckpoint,
    extract_fn,
    layers: list[int],
    args: argparse.Namespace,
    split: str,
    prefix: str = "tiny",
):
    cache_dir = Path(args.output_dir) / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{prefix}_seed{args.seed}"
    feature_path, label_path = feature_cache_paths(cache_dir, prefix, ckpt.name, split, layers, args.timestep, len(examples))
    if feature_path.exists() and label_path.exists() and not args.recompute_cache:
        return np.load(feature_path, mmap_mode="r"), np.load(label_path, mmap_mode="r")

    if not examples:
        raise ValueError(f"No examples for {split}")
    n_layers = len(layers)
    hidden_size = ckpt.config["hidden_size"]
    features = open_memmap(feature_path, mode="w+", dtype=np.float16, shape=(len(examples), n_layers, hidden_size))
    labels = open_memmap(label_path, mode="w+", dtype=np.int32, shape=(len(examples),))

    log(f"Extracting {prefix}/{split} summary features for {ckpt.name}: {len(examples)} images")
    total_batches = math.ceil(len(examples) / args.class_extract_batch_size)
    progress_prefix = f"extract/{ckpt.name}/t{timestep_tag(args.timestep)}/{prefix}_{split}_summary"
    for batch_id, batch_indices in enumerate(progress(
        batched_indices(len(examples), args.class_extract_batch_size, shuffle=False, seed=args.seed),
        total=total_batches,
        desc=f"extract {prefix}/{split}/{ckpt.name}",
        unit="batch",
    )):
        batch_images = np.stack([image_to_vae_array(examples[int(index)].image_path) for index in batch_indices])
        noise_seed = jnp.asarray(batch_noise_seed(args.seed, args.timestep, batch_indices, f"{split}:summary"), dtype=jnp.uint32)
        summaries = np.asarray(jax.device_get(extract_fn(jnp.asarray(batch_images), noise_seed)), dtype=np.float32)
        features[batch_indices] = np.transpose(summaries, (1, 0, 2)).astype(np.float16)
        labels[batch_indices] = np.asarray([examples[int(index)].label for index in batch_indices], dtype=np.int32)
        wandb_log_progress(
            progress_prefix,
            min((batch_id + 1) * args.class_extract_batch_size, len(examples)),
            len(examples),
            batches=batch_id + 1,
            total_batches=total_batches,
        )
    features.flush()
    labels.flush()
    return np.load(feature_path, mmap_mode="r"), np.load(label_path, mmap_mode="r")


def extract_projected_grid_cache(
    examples: list[Example],
    ckpt: LoadedCheckpoint,
    extract_fn,
    layers: list[int],
    args: argparse.Namespace,
    split: str,
):
    cache_dir = Path(args.output_dir) / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"tinygrid_p{args.class_probe_proj_channels}_seed{args.seed}"
    feature_path, label_path = feature_cache_paths(cache_dir, prefix, ckpt.name, split, layers, args.timestep, len(examples))
    if feature_path.exists() and label_path.exists() and not args.recompute_cache:
        return np.load(feature_path, mmap_mode="r"), np.load(label_path, mmap_mode="r")

    if not examples:
        raise ValueError(f"No examples for {split}")
    n_layers = len(layers)
    proj_channels = args.class_probe_proj_channels
    grid_size = args.class_feature_grid_size
    features = open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(examples), n_layers, proj_channels, grid_size, grid_size),
    )
    labels = open_memmap(label_path, mode="w+", dtype=np.int32, shape=(len(examples),))

    log(
        f"Extracting {prefix}/{split} projected grid features for {ckpt.name}: "
        f"{len(examples)} images, grid={grid_size}x{grid_size}, channels={proj_channels}"
    )
    total_batches = math.ceil(len(examples) / args.class_extract_batch_size)
    progress_prefix = f"extract/{ckpt.name}/t{timestep_tag(args.timestep)}/{prefix}_{split}_grid"
    for batch_id, batch_indices in enumerate(progress(
        batched_indices(len(examples), args.class_extract_batch_size, shuffle=False, seed=args.seed),
        total=total_batches,
        desc=f"extract {prefix}/{split}/{ckpt.name}",
        unit="batch",
    )):
        batch_images = np.stack([image_to_vae_array(examples[int(index)].image_path) for index in batch_indices])
        noise_seed = jnp.asarray(batch_noise_seed(args.seed, args.timestep, batch_indices, f"{split}:grid"), dtype=jnp.uint32)
        projected = np.asarray(jax.device_get(extract_fn(jnp.asarray(batch_images), noise_seed)), dtype=np.float32)
        if projected.shape[-2:] != (grid_size, grid_size):
            raise ValueError(f"Projected grid shape {projected.shape[-2:]} does not match configured {grid_size}x{grid_size}")
        features[batch_indices] = np.transpose(projected, (1, 0, 2, 3, 4)).astype(np.float16)
        labels[batch_indices] = np.asarray([examples[int(index)].label for index in batch_indices], dtype=np.int32)
        wandb_log_progress(
            progress_prefix,
            min((batch_id + 1) * args.class_extract_batch_size, len(examples)),
            len(examples),
            batches=batch_id + 1,
            total_batches=total_batches,
        )
    features.flush()
    labels.flush()
    return np.load(feature_path, mmap_mode="r"), np.load(label_path, mmap_mode="r")


def extract_combined_classification_cache(
    examples: list[Example],
    ckpt: LoadedCheckpoint,
    extract_fn,
    layers: list[int],
    args: argparse.Namespace,
    split: str,
):
    cache_dir = Path(args.output_dir) / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary_prefix = f"tiny_seed{args.seed}"
    grid_prefix = f"tinygrid_p{args.class_probe_proj_channels}_seed{args.seed}"
    summary_feature_path, summary_label_path = feature_cache_paths(
        cache_dir,
        summary_prefix,
        ckpt.name,
        split,
        layers,
        args.timestep,
        len(examples),
    )
    grid_feature_path, grid_label_path = feature_cache_paths(
        cache_dir,
        grid_prefix,
        ckpt.name,
        split,
        layers,
        args.timestep,
        len(examples),
    )
    if (
        summary_feature_path.exists()
        and summary_label_path.exists()
        and grid_feature_path.exists()
        and grid_label_path.exists()
        and not args.recompute_cache
    ):
        return (
            np.load(summary_feature_path, mmap_mode="r"),
            np.load(summary_label_path, mmap_mode="r"),
            np.load(grid_feature_path, mmap_mode="r"),
            np.load(grid_label_path, mmap_mode="r"),
        )

    if not examples:
        raise ValueError(f"No examples for {split}")
    n_layers = len(layers)
    hidden_size = ckpt.config["hidden_size"]
    proj_channels = args.class_probe_proj_channels
    grid_size = args.class_feature_grid_size
    summary_features = open_memmap(
        summary_feature_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(examples), n_layers, hidden_size),
    )
    summary_labels = open_memmap(summary_label_path, mode="w+", dtype=np.int32, shape=(len(examples),))
    grid_features = open_memmap(
        grid_feature_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(examples), n_layers, proj_channels, grid_size, grid_size),
    )
    grid_labels = open_memmap(grid_label_path, mode="w+", dtype=np.int32, shape=(len(examples),))

    log(
        f"Extracting combined summary+grid features for {ckpt.name}/{split}: "
        f"{len(examples)} images, grid={grid_size}x{grid_size}, channels={proj_channels}"
    )
    total_batches = math.ceil(len(examples) / args.class_extract_batch_size)
    progress_prefix = f"extract/{ckpt.name}/t{timestep_tag(args.timestep)}/combined_{split}"
    for batch_id, batch_indices in enumerate(progress(
        batched_indices(len(examples), args.class_extract_batch_size, shuffle=False, seed=args.seed),
        total=total_batches,
        desc=f"extract combined/{split}/{ckpt.name}",
        unit="batch",
    )):
        batch_images = np.stack([image_to_vae_array(examples[int(index)].image_path) for index in batch_indices])
        noise_seed = jnp.asarray(batch_noise_seed(args.seed, args.timestep, batch_indices, f"{split}:combined"), dtype=jnp.uint32)
        projected, summaries = extract_fn(jnp.asarray(batch_images), noise_seed)
        projected = np.asarray(jax.device_get(projected), dtype=np.float32)
        summaries = np.asarray(jax.device_get(summaries), dtype=np.float32)
        if projected.shape[-2:] != (grid_size, grid_size):
            raise ValueError(f"Projected grid shape {projected.shape[-2:]} does not match configured {grid_size}x{grid_size}")
        summary_features[batch_indices] = np.transpose(summaries, (1, 0, 2)).astype(np.float16)
        grid_features[batch_indices] = np.transpose(projected, (1, 0, 2, 3, 4)).astype(np.float16)
        labels_np = np.asarray([examples[int(index)].label for index in batch_indices], dtype=np.int32)
        summary_labels[batch_indices] = labels_np
        grid_labels[batch_indices] = labels_np
        wandb_log_progress(
            progress_prefix,
            min((batch_id + 1) * args.class_extract_batch_size, len(examples)),
            len(examples),
            batches=batch_id + 1,
            total_batches=total_batches,
        )
    summary_features.flush()
    summary_labels.flush()
    grid_features.flush()
    grid_labels.flush()
    return (
        np.load(summary_feature_path, mmap_mode="r"),
        np.load(summary_label_path, mmap_mode="r"),
        np.load(grid_feature_path, mmap_mode="r"),
        np.load(grid_label_path, mmap_mode="r"),
    )


def init_class_probe_params(rng: jax.Array, n_layers: int, hidden_size: int, n_classes: int):
    rng_simple, rng_sra = jax.random.split(rng)
    scale = 0.01
    return {
        "simple": {
            "w": scale * jax.random.normal(rng_simple, (n_layers, hidden_size, n_classes), dtype=jnp.float32),
            "b": jnp.zeros((n_layers, n_classes), dtype=jnp.float32),
        },
        "sra": {
            "bn_scale": jnp.ones((n_layers, hidden_size), dtype=jnp.float32),
            "bn_bias": jnp.zeros((n_layers, hidden_size), dtype=jnp.float32),
            "w": scale * jax.random.normal(rng_sra, (n_layers, hidden_size, n_classes), dtype=jnp.float32),
            "b": jnp.zeros((n_layers, n_classes), dtype=jnp.float32),
        },
    }


def init_bn_state(n_layers: int, hidden_size: int):
    return {
        "mean": jnp.zeros((n_layers, hidden_size), dtype=jnp.float32),
        "var": jnp.ones((n_layers, hidden_size), dtype=jnp.float32),
    }


def bn1d_train(x: jax.Array, params: dict, state: dict, momentum: float = 0.9, eps: float = 1e-5):
    mean = jnp.mean(x, axis=1)
    var = jnp.var(x, axis=1)
    x_norm = (x - mean[:, None, :]) / jnp.sqrt(var[:, None, :] + eps)
    x_norm = x_norm * params["bn_scale"][:, None, :] + params["bn_bias"][:, None, :]
    new_state = {
        "mean": momentum * state["mean"] + (1.0 - momentum) * mean,
        "var": momentum * state["var"] + (1.0 - momentum) * var,
    }
    return x_norm, new_state


def bn1d_eval(x: jax.Array, params: dict, state: dict, eps: float = 1e-5):
    x_norm = (x - state["mean"][:, None, :]) / jnp.sqrt(state["var"][:, None, :] + eps)
    return x_norm * params["bn_scale"][:, None, :] + params["bn_bias"][:, None, :]


def class_logits(params: dict, features_lbc: jax.Array, bn_state: dict | None, train: bool):
    simple = jnp.einsum("lbc,lck->lbk", features_lbc, params["simple"]["w"]) + params["simple"]["b"][:, None, :]
    if train:
        sra_features, new_bn_state = bn1d_train(features_lbc, params["sra"], bn_state)
    else:
        sra_features = bn1d_eval(features_lbc, params["sra"], bn_state)
        new_bn_state = bn_state
    sra = jnp.einsum("lbc,lck->lbk", sra_features, params["sra"]["w"]) + params["sra"]["b"][:, None, :]
    return simple, sra, new_bn_state


def classification_ce_loss(logits_lbk: jax.Array, labels_b: jax.Array) -> jax.Array:
    labels_lbk = jnp.broadcast_to(labels_b[None, :, None], logits_lbk.shape[:2] + (1,))
    log_probs = jax.nn.log_softmax(logits_lbk, axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs, labels_lbk, axis=-1))


def train_linear_classification_probes(
    model_name: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    layers: list[int],
    args: argparse.Namespace,
) -> list[dict]:
    selected = set(args.class_probes) & CLASS_LINEAR_PROBES
    if not selected:
        return []

    n_layers = train_features.shape[1]
    hidden_size = train_features.shape[2]
    params = init_class_probe_params(jax.random.PRNGKey(args.seed + 11), n_layers, hidden_size, NUM_CLASSES_TINY)
    bn_state = init_bn_state(n_layers, hidden_size)
    steps_per_epoch = math.ceil(len(train_labels) / args.class_train_batch_size)
    total_steps = max(1, steps_per_epoch * args.class_epochs)
    lr_schedule = optax.cosine_decay_schedule(args.lr, total_steps)
    tx = optax.adamw(lr_schedule, weight_decay=args.weight_decay)
    opt_state = tx.init(params)

    log(
        f"Training true linear classification probes for {model_name}: "
        f"{', '.join(probe for probe in CLASS_PROBE_CHOICES if probe in selected)}, "
        f"features={train_features.shape}, epochs={args.class_epochs}"
    )

    @jax.jit
    def grad_step(params, opt_state, bn_state, features_blc, labels_b):
        features_lbc = jnp.transpose(features_blc.astype(jnp.float32), (1, 0, 2))
        labels_b = labels_b.astype(jnp.int32)

        def loss_fn(p):
            logits_simple, logits_sra, new_bn_state = class_logits(p, features_lbc, bn_state, train=True)
            loss_simple = classification_ce_loss(logits_simple, labels_b)
            loss_sra = classification_ce_loss(logits_sra, labels_b)
            return 0.5 * (loss_simple + loss_sra), new_bn_state

        (loss, new_bn_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state_next = tx.update(grads, opt_state, params)
        params_next = optax.apply_updates(params, updates)
        return params_next, opt_state_next, new_bn_state, loss

    @jax.jit
    def predict_step(params, bn_state, features_blc):
        features_lbc = jnp.transpose(features_blc.astype(jnp.float32), (1, 0, 2))
        logits_simple, logits_sra, _ = class_logits(params, features_lbc, bn_state, train=False)
        return jnp.argmax(logits_simple, axis=-1), jnp.argmax(logits_sra, axis=-1)

    global_step = 0
    for epoch in range(args.class_epochs):
        losses = []
        train_bar = progress(
            batched_indices(len(train_labels), args.class_train_batch_size, shuffle=True, seed=args.seed + epoch),
            total=steps_per_epoch,
            desc=f"class linear train {model_name} epoch {epoch + 1}/{args.class_epochs}",
            unit="batch",
        )
        for batch_indices in train_bar:
            batch_x = jnp.asarray(np.asarray(train_features[batch_indices], dtype=np.float32))
            batch_y = jnp.asarray(np.asarray(train_labels[batch_indices], dtype=np.int32))
            params, opt_state, bn_state, loss = grad_step(params, opt_state, bn_state, batch_x, batch_y)
            loss_value = float(jax.device_get(loss))
            losses.append(loss_value)
            global_step += 1
            train_bar.set_postfix(loss=f"{loss_value:.4f}", global_step=global_step)

        epoch_loss = float(np.mean(losses))
        log(f"{model_name} true linear classification epoch {epoch + 1}: loss={epoch_loss:.4f}")
        wandb_log(
            {
                f"classification/{model_name}/t{timestep_tag(args.timestep)}/linear_train_loss": epoch_loss,
                "classification/linear_epoch": epoch + 1,
                "classification/linear_global_step": global_step,
                "classification/timestep": args.timestep,
            }
        )

    correct_simple = np.zeros((n_layers,), dtype=np.float64)
    correct_sra = np.zeros((n_layers,), dtype=np.float64)
    total_per_layer = np.zeros((n_layers,), dtype=np.float64)
    for batch_indices in progress(
        batched_indices(len(val_labels), args.class_train_batch_size, shuffle=False, seed=args.seed),
        total=math.ceil(len(val_labels) / args.class_train_batch_size),
        desc=f"class linear eval {model_name}",
        unit="batch",
    ):
        batch_x = jnp.asarray(np.asarray(val_features[batch_indices], dtype=np.float32))
        pred_simple, pred_sra = jax.device_get(predict_step(params, bn_state, batch_x))
        labels_np = np.asarray(val_labels[batch_indices], dtype=np.int64)
        for layer_idx in range(n_layers):
            correct_simple[layer_idx] += np.sum(pred_simple[layer_idx] == labels_np)
            correct_sra[layer_idx] += np.sum(pred_sra[layer_idx] == labels_np)
            total_per_layer[layer_idx] += len(batch_indices)

    rows = []
    for idx, layer in enumerate(layers):
        if "simple_linear" in selected:
            rows.append(
                {
                    "model_name": model_name,
                    "probe_type": "simple_linear",
                    "layer_index": layer,
                    "timestep": args.timestep,
                    "val_accuracy": 100.0 * correct_simple[idx] / max(total_per_layer[idx], 1),
                }
            )
        if "sra_style" in selected:
            rows.append(
                {
                    "model_name": model_name,
                    "probe_type": "sra_style",
                    "layer_index": layer,
                    "timestep": args.timestep,
                    "val_accuracy": 100.0 * correct_sra[idx] / max(total_per_layer[idx], 1),
                }
            )

    save_probe_artifact(
        Path(args.output_dir),
        model_name,
        "classification_linear",
        {"params": params, "bn_state": bn_state},
        {
            "model_name": model_name,
            "experiment": "Tiny ImageNet true linear probing",
            "probe_types": {
                "simple_linear": "MeanPool/block summary + Linear",
                "sra_style": "MeanPool/block summary + BatchNorm + Linear",
            },
            "selected_probe_types": sorted(selected),
            "layers": layers,
            "timestep": args.timestep,
            "global_batch_size": args.global_batch_size,
            "train_batch_size": args.class_train_batch_size,
            "epochs": args.class_epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_classes": NUM_CLASSES_TINY,
            "feature_shape": ["num_examples", "num_layers", "hidden_size"],
        },
    )
    return rows


def train_classification_probes(
    model_name: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    layers: list[int],
    args: argparse.Namespace,
):
    import timm
    import torch.nn as nn
    import torch.nn.functional as F

    selected = set(args.class_probes) & CLASS_CNN_PROBES
    if not selected:
        return []

    class SharedTimmProbe(nn.Module):
        def __init__(self, arch: str, in_channels: int, n_layers: int, n_classes: int):
            super().__init__()
            self.encoder = timm.create_model(
                arch,
                pretrained=False,
                in_chans=in_channels,
                num_classes=0,
                global_pool="avg",
            )
            feature_dim = int(getattr(self.encoder, "num_features"))
            self.head_weight = nn.Parameter(torch.empty(n_layers, feature_dim, n_classes))
            self.head_bias = nn.Parameter(torch.zeros(n_layers, n_classes))
            nn.init.normal_(self.head_weight, std=0.01)

        def forward(self, x: torch.Tensor, layer_indices: torch.Tensor) -> torch.Tensor:
            z = self.encoder(x)
            weight = self.head_weight[layer_indices]
            bias = self.head_bias[layer_indices]
            return torch.bmm(z.unsqueeze(1), weight).squeeze(1) + bias

    def resolve_probe_device() -> torch.device:
        if args.class_probe_device != "auto":
            return torch.device(args.class_probe_device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def layer_ids_for_step(step: int) -> np.ndarray:
        if n_layers <= layer_batch_size:
            return np.arange(n_layers, dtype=np.int64)
        start = (step * layer_batch_size) % n_layers
        return ((start + np.arange(layer_batch_size)) % n_layers).astype(np.int64)

    def make_batch(features: np.ndarray, labels: np.ndarray, batch_indices: np.ndarray, layer_ids: np.ndarray):
        arr = np.asarray(features[batch_indices][:, layer_ids], dtype=np.float32)
        batch_size = arr.shape[0]
        n_selected = arr.shape[1]
        x = torch.from_numpy(arr.reshape(batch_size * n_selected, arr.shape[2], arr.shape[3], arr.shape[4]))
        y_np = np.repeat(np.asarray(labels[batch_indices], dtype=np.int64), n_selected)
        layer_np = np.tile(layer_ids, batch_size).astype(np.int64)
        return x.to(device, non_blocking=True), torch.from_numpy(y_np).to(device), torch.from_numpy(layer_np).to(device)

    def prepare_input(x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        x = (x - mean) / std
        if x.shape[-1] != args.class_cnn_input_size or x.shape[-2] != args.class_cnn_input_size:
            x = F.interpolate(x, size=(args.class_cnn_input_size, args.class_cnn_input_size), mode="bilinear", align_corners=False)
        return x

    n_layers = train_features.shape[1]
    in_channels = train_features.shape[2]
    layer_batch_size = max(1, min(args.class_cnn_layer_batch_size, n_layers))
    eval_layer_batch_size = max(1, min(args.class_cnn_eval_layer_batch_size or layer_batch_size, n_layers))
    device = resolve_probe_device()
    torch.set_grad_enabled(True)
    use_data_parallel = (
        bool(args.class_probe_data_parallel)
        and device.type == "cuda"
        and torch.cuda.device_count() > 1
        and args.class_probe_device in {"auto", "cuda"}
    )
    log(
        f"Training classification CNN probes for {model_name}: "
        f"{', '.join(probe for probe in CLASS_PROBE_CHOICES if probe in selected)} on {device}, "
        f"projected_channels={in_channels}, layer_batch={layer_batch_size}, "
        f"input_size={args.class_cnn_input_size}, data_parallel={use_data_parallel}"
    )
    try:
        jax.clear_caches()
    except Exception:
        pass

    probes: dict[str, nn.Module] = {}
    if "resnet18_probe" in selected:
        probes["resnet18_probe"] = SharedTimmProbe("resnet18", in_channels, n_layers, NUM_CLASSES_TINY).to(device)
    if "convnext_atto_probe" in selected:
        probes["convnext_atto_probe"] = SharedTimmProbe("convnext_atto", in_channels, n_layers, NUM_CLASSES_TINY).to(device)
    if "convnext_tiny_probe" in selected:
        probes["convnext_tiny_probe"] = SharedTimmProbe("convnext_tiny", in_channels, n_layers, NUM_CLASSES_TINY).to(device)
    if use_data_parallel:
        probes = {name: nn.DataParallel(module) for name, module in probes.items()}
    params = [param for module in probes.values() for param in module.parameters()]
    steps_per_epoch = math.ceil(len(train_labels) / args.class_train_batch_size)
    total_steps = max(1, steps_per_epoch * args.class_epochs)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    use_amp = bool(args.class_probe_amp and device.type == "cuda")
    if hasattr(torch, "amp"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def autocast_context():
        if device.type == "cuda":
            if hasattr(torch, "amp"):
                return torch.amp.autocast("cuda", enabled=use_amp)
            return torch.cuda.amp.autocast(enabled=use_amp)
        return contextlib.nullcontext()

    global_step = 0
    for epoch in range(args.class_epochs):
        losses = []
        for probe in probes.values():
            probe.train()
        train_bar = progress(
            batched_indices(len(train_labels), args.class_train_batch_size, shuffle=True, seed=args.seed + epoch),
            total=steps_per_epoch,
            desc=f"class CNN train {model_name} epoch {epoch + 1}/{args.class_epochs}",
            unit="batch",
        )
        for batch_indices in train_bar:
            layer_ids = layer_ids_for_step(global_step)
            batch_x, batch_y, batch_layers = make_batch(train_features, train_labels, batch_indices, layer_ids)
            batch_x = prepare_input(batch_x)
            optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad():
                with autocast_context():
                    step_losses = []
                    for probe in probes.values():
                        step_losses.append(F.cross_entropy(probe(batch_x, batch_layers), batch_y))
                    loss = torch.stack(step_losses).mean()
            if use_amp:
                scale_before = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
            else:
                loss.backward()
                optimizer.step()
                scheduler.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            global_step += 1
            train_bar.set_postfix(loss=f"{loss_value:.4f}", global_step=global_step, layers=",".join(str(layers[i]) for i in layer_ids))
        epoch_loss = float(np.mean(losses))
        log(f"{model_name} classification CNN epoch {epoch + 1}: loss={epoch_loss:.4f}")
        wandb_log(
            {
                f"classification/{model_name}/t{timestep_tag(args.timestep)}/train_loss": epoch_loss,
                "classification/epoch": epoch + 1,
                "classification/global_step": global_step,
                "classification/timestep": args.timestep,
            }
        )

    correct_by_probe = {probe_type: np.zeros((n_layers,), dtype=np.float64) for probe_type in probes}
    total_per_layer = np.zeros((n_layers,), dtype=np.float64)
    for probe in probes.values():
        probe.eval()
    with torch.no_grad():
        for layer_start in progress(
            range(0, n_layers, eval_layer_batch_size),
            total=math.ceil(n_layers / eval_layer_batch_size),
            desc=f"class CNN eval layers {model_name}",
            unit="layer_chunk",
        ):
            layer_ids = np.arange(layer_start, min(layer_start + eval_layer_batch_size, n_layers), dtype=np.int64)
            for batch_indices in batched_indices(len(val_labels), args.class_train_batch_size, shuffle=False, seed=args.seed):
                batch_x, batch_y, batch_layers = make_batch(val_features, val_labels, batch_indices, layer_ids)
                batch_x = prepare_input(batch_x)
                with autocast_context():
                    logits_by_probe = {probe_type: probe(batch_x, batch_layers) for probe_type, probe in probes.items()}
                batch_size = len(batch_indices)
                n_selected = len(layer_ids)
                labels_np = np.asarray(val_labels[batch_indices], dtype=np.int64)
                pred_by_probe = {
                    probe_type: logits.argmax(dim=-1).detach().cpu().numpy().reshape(batch_size, n_selected)
                    for probe_type, logits in logits_by_probe.items()
                }
                for offset, layer_idx in enumerate(layer_ids):
                    for probe_type, pred in pred_by_probe.items():
                        correct_by_probe[probe_type][layer_idx] += np.sum(pred[:, offset] == labels_np)
                    total_per_layer[layer_idx] += batch_size

    rows = []
    for idx, layer in enumerate(layers):
        for probe_type in CLASS_PROBE_CHOICES:
            if probe_type not in correct_by_probe:
                continue
            rows.append(
                {
                    "model_name": model_name,
                    "probe_type": probe_type,
                    "layer_index": layer,
                    "timestep": args.timestep,
                    "val_accuracy": 100.0 * correct_by_probe[probe_type][idx] / max(total_per_layer[idx], 1),
                }
            )

    def unwrap_state_dict(module: nn.Module) -> dict:
        return module.module.state_dict() if isinstance(module, nn.DataParallel) else module.state_dict()

    save_torch_probe_artifact(
        Path(args.output_dir),
        model_name,
        "classification_cnn",
        {probe_type: unwrap_state_dict(probe) for probe_type, probe in probes.items()},
        {
            "model_name": model_name,
            "experiment": "Tiny ImageNet classification CNN probing",
            "probe_types": {
                "resnet18_probe": "ResNet18 trunk + layer-specific classifier",
                "convnext_atto_probe": "ConvNeXt-Atto trunk + layer-specific classifier",
                "convnext_tiny_probe": "ConvNeXt-Tiny trunk + layer-specific classifier",
            },
            "selected_probe_types": sorted(selected),
            "layers": layers,
            "timestep": args.timestep,
            "global_batch_size": args.global_batch_size,
            "train_batch_size": args.class_train_batch_size,
            "layer_batch_size": layer_batch_size,
            "eval_layer_batch_size": eval_layer_batch_size,
            "epochs": args.class_epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_classes": NUM_CLASSES_TINY,
            "projected_channels": in_channels,
            "feature_shape": ["num_examples", "num_layers", "projected_channels", "grid_h", "grid_w"],
            "cnn_input_size": args.class_cnn_input_size,
            "amp": use_amp,
            "data_parallel": use_data_parallel,
            "shared_trunk": True,
            "layer_specific_classifier": True,
        },
    )
    torch.set_grad_enabled(False)
    return rows


def init_seg_probe_params(rng: jax.Array, n_layers: int, hidden_size: int, n_classes: int):
    rng_simple, rng_sra = jax.random.split(rng)
    scale = 0.01
    return {
        "simple": {
            "w": scale * jax.random.normal(rng_simple, (n_layers, hidden_size, n_classes), dtype=jnp.float32),
            "b": jnp.zeros((n_layers, n_classes), dtype=jnp.float32),
        },
        "sra": {
            "bn_scale": jnp.ones((n_layers, hidden_size), dtype=jnp.float32),
            "bn_bias": jnp.zeros((n_layers, hidden_size), dtype=jnp.float32),
            "w": scale * jax.random.normal(rng_sra, (n_layers, hidden_size, n_classes), dtype=jnp.float32),
            "b": jnp.zeros((n_layers, n_classes), dtype=jnp.float32),
        },
    }


def bn2d_train(x: jax.Array, params: dict, state: dict, momentum: float = 0.9, eps: float = 1e-5):
    mean = jnp.mean(x, axis=(1, 2, 3))
    var = jnp.var(x, axis=(1, 2, 3))
    x_norm = (x - mean[:, None, None, None, :]) / jnp.sqrt(var[:, None, None, None, :] + eps)
    x_norm = x_norm * params["bn_scale"][:, None, None, None, :] + params["bn_bias"][:, None, None, None, :]
    new_state = {
        "mean": momentum * state["mean"] + (1.0 - momentum) * mean,
        "var": momentum * state["var"] + (1.0 - momentum) * var,
    }
    return x_norm, new_state


def bn2d_eval(x: jax.Array, params: dict, state: dict, eps: float = 1e-5):
    x_norm = (x - state["mean"][:, None, None, None, :]) / jnp.sqrt(state["var"][:, None, None, None, :] + eps)
    return x_norm * params["bn_scale"][:, None, None, None, :] + params["bn_bias"][:, None, None, None, :]


def dense_to_grid(features_lbnc: jax.Array) -> jax.Array:
    n_tokens = features_lbnc.shape[2]
    grid = int(math.sqrt(int(n_tokens)))
    if grid * grid != int(n_tokens):
        raise ValueError(f"Cannot reshape {n_tokens} tokens to a square grid")
    return rearrange(features_lbnc, "l b (h w) c -> l b h w c", h=grid, w=grid)


def seg_logits(params: dict, features_lbnc: jax.Array, bn_state: dict | None, train: bool):
    grid = dense_to_grid(features_lbnc)
    simple = jnp.einsum("lbhwc,lck->lbhwk", grid, params["simple"]["w"]) + params["simple"]["b"][:, None, None, None, :]
    if train:
        sra_grid, new_bn_state = bn2d_train(grid, params["sra"], bn_state)
    else:
        sra_grid = bn2d_eval(grid, params["sra"], bn_state)
        new_bn_state = bn_state
    sra = jnp.einsum("lbhwc,lck->lbhwk", sra_grid, params["sra"]["w"]) + params["sra"]["b"][:, None, None, None, :]
    return simple, sra, new_bn_state


def upsample_logits(logits_lbhwk: jax.Array, out_h: int, out_w: int) -> jax.Array:
    n_layers, batch_size, height, width, n_classes = logits_lbhwk.shape
    logits = logits_lbhwk.reshape((n_layers * batch_size, height, width, n_classes))
    logits = jax.image.resize(logits, (n_layers * batch_size, out_h, out_w, n_classes), method="linear")
    return logits.reshape((n_layers, batch_size, out_h, out_w, n_classes))


def seg_loss_for_logits(logits_lbhwk: jax.Array, masks_bhw: jax.Array) -> jax.Array:
    labels = jnp.broadcast_to(masks_bhw[None, :, :, :], logits_lbhwk.shape[:4])
    valid = labels != 255
    labels_safe = jnp.where(valid, labels, 0)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits_lbhwk, labels_safe)
    denom = jnp.maximum(jnp.sum(valid), 1)
    return jnp.sum(jnp.where(valid, loss, 0.0)) / denom


def seg_loss_chunked(logits_lbhwk: jax.Array, masks_bhw: jax.Array, chunk_size: int) -> jax.Array:
    n_layers = int(logits_lbhwk.shape[0])
    chunk_size = max(1, int(chunk_size))
    total = jnp.asarray(0.0, dtype=jnp.float32)
    for start in range(0, n_layers, chunk_size):
        end = min(start + chunk_size, n_layers)
        logits_up = upsample_logits(logits_lbhwk[start:end], masks_bhw.shape[1], masks_bhw.shape[2])
        total = total + seg_loss_for_logits(logits_up, masks_bhw) * float(end - start)
    return total / float(max(n_layers, 1))


def load_seg_batch(examples: list[Example], indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    images = []
    masks = []
    for index in indices:
        item = examples[int(index)]
        images.append(image_to_vae_array(item.image_path))
        masks.append(load_voc_mask(item.mask_path))
    return np.stack(images), np.stack(masks).astype(np.int32)


def train_segmentation_probes(
    ckpt: LoadedCheckpoint,
    extract_fn,
    train_examples: list[Example],
    val_examples: list[Example],
    layers: list[int],
    args: argparse.Namespace,
):
    n_layers = len(layers)
    hidden_size = ckpt.config["hidden_size"]
    params = init_seg_probe_params(jax.random.PRNGKey(args.seed + 17), n_layers, hidden_size, NUM_CLASSES_VOC)
    bn_state = init_bn_state(n_layers, hidden_size)
    steps_per_epoch = math.ceil(len(train_examples) / args.global_batch_size)
    total_steps = max(1, steps_per_epoch * args.seg_epochs)
    lr_schedule = optax.cosine_decay_schedule(args.lr, total_steps)
    tx = optax.adamw(lr_schedule, weight_decay=args.weight_decay)
    opt_state = tx.init(params)

    @jax.jit
    def grad_step(params, bn_state, features_lbnc, masks_bhw):
        def loss_fn(p):
            logits_simple, logits_sra, new_bn_state = seg_logits(p, features_lbnc, bn_state, train=True)
            loss_simple = seg_loss_chunked(logits_simple, masks_bhw, args.seg_layer_chunk_size)
            loss_sra = seg_loss_chunked(logits_sra, masks_bhw, args.seg_layer_chunk_size)
            return 0.5 * (loss_simple + loss_sra), new_bn_state

        (loss, new_bn_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        return grads, new_bn_state, loss

    @jax.jit
    def apply_accumulated_grads(params, opt_state, grads):
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    @functools.partial(jax.jit, static_argnames=("out_h", "out_w", "start", "end"))
    def predict_step(params, bn_state, features_lbnc, out_h: int, out_w: int, start: int, end: int):
        logits_simple, logits_sra, _ = seg_logits(params, features_lbnc, bn_state, train=False)
        logits_simple = upsample_logits(logits_simple[start:end], out_h, out_w)
        logits_sra = upsample_logits(logits_sra[start:end], out_h, out_w)
        return jnp.argmax(logits_simple, axis=-1), jnp.argmax(logits_sra, axis=-1)

    global_step = 0
    for epoch in range(args.seg_epochs):
        losses = []
        epoch_bar = progress(
            total=len(train_examples),
            desc=(
                f"seg train {ckpt.name} epoch {epoch + 1}/{args.seg_epochs} "
                f"(global={args.global_batch_size}, micro={args.seg_microbatch_size})"
            ),
            unit="img",
        )
        try:
            for global_indices in batched_indices(
                len(train_examples),
                args.global_batch_size,
                shuffle=True,
                seed=args.seed + 1000 + epoch,
            ):
                grad_accum = jax.tree_util.tree_map(jnp.zeros_like, params)
                loss_accum = 0.0
                for start in range(0, len(global_indices), args.seg_microbatch_size):
                    micro_indices = global_indices[start : start + args.seg_microbatch_size]
                    images, masks = load_seg_batch(train_examples, micro_indices)
                    noise_seed = jnp.asarray(
                        batch_noise_seed(args.seed + epoch, args.timestep, micro_indices, "voc_train"),
                        dtype=jnp.uint32,
                    )
                    features = extract_fn(jnp.asarray(images), noise_seed)
                    grads, bn_state, loss = grad_step(
                        params,
                        bn_state,
                        features,
                        jnp.asarray(masks, dtype=jnp.int32),
                    )
                    weight = float(len(micro_indices)) / float(len(global_indices))
                    grad_accum = jax.tree_util.tree_map(
                        lambda acc, grad: acc + grad * weight,
                        grad_accum,
                        grads,
                    )
                    loss_value = float(jax.device_get(loss))
                    loss_accum += loss_value * weight
                    epoch_bar.update(len(micro_indices))
                    epoch_bar.set_postfix(
                        loss=f"{loss_value:.4f}",
                        opt_step=global_step + 1,
                        micro=args.seg_microbatch_size,
                    )
                params, opt_state = apply_accumulated_grads(params, opt_state, grad_accum)
                losses.append(loss_accum)
                global_step += 1
                epoch_bar.set_postfix(
                    loss=f"{loss_accum:.4f}",
                    opt_step=global_step,
                    global_batch=len(global_indices),
                )
        finally:
            epoch_bar.close()
        epoch_loss = float(np.mean(losses))
        log(f"{ckpt.name} segmentation epoch {epoch + 1}: loss={epoch_loss:.4f}")
        wandb_log(
            {
                f"segmentation/{ckpt.name}/t{timestep_tag(args.timestep)}/train_loss": epoch_loss,
                "segmentation/epoch": epoch + 1,
                "segmentation/optimizer_step": global_step,
                "segmentation/timestep": args.timestep,
            }
        )

    conf_simple = np.zeros((n_layers, NUM_CLASSES_VOC, NUM_CLASSES_VOC), dtype=np.int64)
    conf_sra = np.zeros_like(conf_simple)
    eval_microbatch = args.seg_eval_microbatch_size or args.seg_microbatch_size
    for batch_indices in progress(
        batched_indices(len(val_examples), eval_microbatch, shuffle=False, seed=args.seed),
        total=math.ceil(len(val_examples) / eval_microbatch),
        desc=f"seg eval {ckpt.name}",
        unit="batch",
    ):
        images, masks = load_seg_batch(val_examples, batch_indices)
        noise_seed = jnp.asarray(batch_noise_seed(args.seed, args.timestep, batch_indices, "voc_val"), dtype=jnp.uint32)
        features = extract_fn(jnp.asarray(images), noise_seed)
        for start in range(0, n_layers, args.seg_layer_chunk_size):
            end = min(start + args.seg_layer_chunk_size, n_layers)
            pred_simple, pred_sra = predict_step(params, bn_state, features, masks.shape[1], masks.shape[2], start, end)
            update_confusion(conf_simple[start:end], np.asarray(jax.device_get(pred_simple)), masks)
            update_confusion(conf_sra[start:end], np.asarray(jax.device_get(pred_sra)), masks)

    miou_simple = mean_iou(conf_simple)
    miou_sra = mean_iou(conf_sra)
    rows = []
    for idx, layer in enumerate(layers):
        rows.append(
            {
                "model_name": ckpt.name,
                "probe_type": "simple_linear_seg",
                "layer_index": layer,
                "timestep": args.timestep,
                "mIoU": 100.0 * miou_simple[idx],
            }
        )
        rows.append(
            {
                "model_name": ckpt.name,
                "probe_type": "sra_inspired_dense_seg",
                "layer_index": layer,
                "timestep": args.timestep,
                "mIoU": 100.0 * miou_sra[idx],
            }
        )
    save_probe_artifact(
        Path(args.output_dir),
        ckpt.name,
        "segmentation",
        {"params": params, "bn_state": bn_state},
        {
            "model_name": ckpt.name,
            "experiment": "PASCAL VOC semantic segmentation linear probing",
            "probe_types": {
                "simple_linear_seg": "1x1 Conv",
                "sra_inspired_dense_seg": "BatchNorm2d + 1x1 Conv",
            },
            "layers": layers,
            "timestep": args.timestep,
            "global_batch_size": args.global_batch_size,
            "microbatch_size": args.seg_microbatch_size,
            "epochs": args.seg_epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_classes": NUM_CLASSES_VOC,
            "feature_grid": [16, 16],
        },
    )
    return rows


def voc_cache_paths(cache_dir: Path, prefix: str, model_name: str, split: str, layers: list[int], timestep: float, n_items: int):
    feature_path, _ = feature_cache_paths(cache_dir, prefix, model_name, split, layers, timestep, n_items)
    mask_path = Path(str(feature_path).replace(".features.npy", ".masks.npy"))
    return feature_path, mask_path


def extract_projected_voc_grid_cache(
    examples: list[Example],
    ckpt: LoadedCheckpoint,
    extract_fn,
    layers: list[int],
    args: argparse.Namespace,
    split: str,
):
    cache_dir = Path(args.output_dir) / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"vocgrid_p{args.seg_probe_proj_channels}_seed{args.seed}"
    feature_path, mask_path = voc_cache_paths(cache_dir, prefix, ckpt.name, split, layers, args.timestep, len(examples))
    if feature_path.exists() and mask_path.exists() and not args.recompute_cache:
        return np.load(feature_path, mmap_mode="r"), np.load(mask_path, mmap_mode="r")

    if not examples:
        raise ValueError(f"No VOC examples for {split}")
    n_layers = len(layers)
    proj_channels = args.seg_probe_proj_channels
    grid_size = args.seg_feature_grid_size
    features = open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(examples), n_layers, proj_channels, grid_size, grid_size),
    )
    masks = open_memmap(mask_path, mode="w+", dtype=np.uint8, shape=(len(examples), 256, 256))

    log(
        f"Extracting VOC projected grid features for {ckpt.name}/{split}: "
        f"{len(examples)} images, grid={grid_size}x{grid_size}, channels={proj_channels}"
    )
    total_batches = math.ceil(len(examples) / args.seg_extract_batch_size)
    progress_prefix = f"extract/{ckpt.name}/t{timestep_tag(args.timestep)}/vocgrid_{split}"
    for batch_id, batch_indices in enumerate(progress(
        batched_indices(len(examples), args.seg_extract_batch_size, shuffle=False, seed=args.seed),
        total=total_batches,
        desc=f"extract vocgrid/{split}/{ckpt.name}",
        unit="batch",
    )):
        batch_images = np.stack([image_to_vae_array(examples[int(index)].image_path) for index in batch_indices])
        noise_seed = jnp.asarray(batch_noise_seed(args.seed, args.timestep, batch_indices, f"voc_{split}:grid"), dtype=jnp.uint32)
        projected = np.asarray(jax.device_get(extract_fn(jnp.asarray(batch_images), noise_seed)), dtype=np.float32)
        if projected.shape[-2:] != (grid_size, grid_size):
            raise ValueError(f"Projected VOC grid shape {projected.shape[-2:]} does not match {grid_size}x{grid_size}")
        features[batch_indices] = np.transpose(projected, (1, 0, 2, 3, 4)).astype(np.float16)
        masks[batch_indices] = np.stack([load_voc_mask(examples[int(index)].mask_path) for index in batch_indices]).astype(np.uint8)
        wandb_log_progress(
            progress_prefix,
            min((batch_id + 1) * args.seg_extract_batch_size, len(examples)),
            len(examples),
            batches=batch_id + 1,
            total_batches=total_batches,
        )
    features.flush()
    masks.flush()
    return np.load(feature_path, mmap_mode="r"), np.load(mask_path, mmap_mode="r")


def train_segmentation_convnext_tiny_probe(
    model_name: str,
    train_features: np.ndarray,
    train_masks: np.ndarray,
    val_features: np.ndarray,
    val_masks: np.ndarray,
    layers: list[int],
    args: argparse.Namespace,
) -> list[dict]:
    import timm
    import torch.nn as nn
    import torch.nn.functional as F

    n_layers = train_features.shape[1]
    in_channels = train_features.shape[2]
    train_batch_size = args.seg_train_batch_size or args.global_batch_size
    layer_batch_size = max(1, min(args.seg_cnn_layer_batch_size, n_layers))
    eval_layer_batch_size = max(1, min(args.seg_cnn_eval_layer_batch_size or layer_batch_size, n_layers))
    out_indices = tuple(args.seg_cnn_out_indices)
    device = torch.device(args.seg_probe_device if args.seg_probe_device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_grad_enabled(True)
    use_data_parallel = (
        bool(args.seg_probe_data_parallel)
        and device.type == "cuda"
        and torch.cuda.device_count() > 1
        and args.seg_probe_device in {"auto", "cuda"}
    )

    class ConvNeXtTinyDenseProbe(nn.Module):
        def __init__(self, in_channels: int, n_layers: int, n_classes: int):
            super().__init__()
            self.encoder = timm.create_model(
                "convnext_tiny",
                pretrained=False,
                features_only=True,
                in_chans=in_channels,
                out_indices=out_indices,
            )
            feature_dim = int(sum(self.encoder.feature_info.channels()))
            self.head_weight = nn.Parameter(torch.empty(n_layers, n_classes, feature_dim))
            self.head_bias = nn.Parameter(torch.zeros(n_layers, n_classes))
            nn.init.normal_(self.head_weight, std=0.01)

        def forward(self, x: torch.Tensor, layer_indices: torch.Tensor) -> torch.Tensor:
            features = self.encoder(x)
            target_size = features[0].shape[-2:]
            resized = []
            for feat in features:
                if feat.shape[-2:] != target_size:
                    feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
                resized.append(feat)
            z = torch.cat(resized, dim=1)
            weight = self.head_weight[layer_indices]
            bias = self.head_bias[layer_indices]
            return torch.einsum("bchw,bkc->bkhw", z, weight) + bias[:, :, None, None]

    def prepare_input(x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        x = (x - mean) / std
        if x.shape[-1] != args.seg_cnn_input_size or x.shape[-2] != args.seg_cnn_input_size:
            x = F.interpolate(x, size=(args.seg_cnn_input_size, args.seg_cnn_input_size), mode="bilinear", align_corners=False)
        return x

    def make_batch(features: np.ndarray, masks: np.ndarray, batch_indices: np.ndarray, layer_ids: np.ndarray):
        arr = np.asarray(features[batch_indices][:, layer_ids], dtype=np.float32)
        batch_size = arr.shape[0]
        n_selected = arr.shape[1]
        x = torch.from_numpy(arr.reshape(batch_size * n_selected, arr.shape[2], arr.shape[3], arr.shape[4]))
        masks_np = np.repeat(np.asarray(masks[batch_indices], dtype=np.int64), n_selected, axis=0)
        layer_np = np.tile(layer_ids, batch_size).astype(np.int64)
        return (
            x.to(device, non_blocking=True),
            torch.from_numpy(masks_np).to(device, non_blocking=True),
            torch.from_numpy(layer_np).to(device, non_blocking=True),
            batch_size,
            n_selected,
        )

    log(
        f"Training segmentation ConvNeXt-Tiny probe for {model_name} on {device}: "
        f"batch={train_batch_size}, layer_batch={layer_batch_size}, input_size={args.seg_cnn_input_size}, "
        f"out_indices={out_indices}, projected_channels={in_channels}, data_parallel={use_data_parallel}"
    )
    try:
        jax.clear_caches()
    except Exception:
        pass

    probe: nn.Module = ConvNeXtTinyDenseProbe(in_channels, n_layers, NUM_CLASSES_VOC).to(device)
    if use_data_parallel:
        probe = nn.DataParallel(probe)
    steps_per_epoch = math.ceil(len(train_masks) / train_batch_size) * math.ceil(n_layers / layer_batch_size)
    total_steps = max(1, steps_per_epoch * args.seg_epochs)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    use_amp = bool(args.seg_probe_amp and device.type == "cuda")
    if hasattr(torch, "amp"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def autocast_context():
        if device.type == "cuda":
            if hasattr(torch, "amp"):
                return torch.amp.autocast("cuda", enabled=use_amp)
            return torch.cuda.amp.autocast(enabled=use_amp)
        return contextlib.nullcontext()

    global_step = 0
    for epoch in range(args.seg_epochs):
        probe.train()
        losses = []
        layer_order = np.arange(n_layers, dtype=np.int64)
        epoch_bar = progress(
            total=steps_per_epoch,
            desc=f"seg ConvNeXt-Tiny train {model_name} epoch {epoch + 1}/{args.seg_epochs}",
            unit="step",
        )
        try:
            for batch_indices in batched_indices(len(train_masks), train_batch_size, shuffle=True, seed=args.seed + 1000 + epoch):
                layer_rng = np.random.default_rng(args.seed + 3000 + epoch + global_step)
                layer_rng.shuffle(layer_order)
                for layer_start in range(0, n_layers, layer_batch_size):
                    layer_ids = layer_order[layer_start : layer_start + layer_batch_size]
                    batch_x, batch_masks, batch_layers, _, _ = make_batch(train_features, train_masks, batch_indices, layer_ids)
                    batch_x = prepare_input(batch_x)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.enable_grad():
                        with autocast_context():
                            logits = probe(batch_x, batch_layers)
                            logits = F.interpolate(logits, size=batch_masks.shape[-2:], mode="bilinear", align_corners=False)
                            loss = F.cross_entropy(logits, batch_masks.long(), ignore_index=255)
                    if use_amp:
                        scale_before = scaler.get_scale()
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        if scaler.get_scale() >= scale_before:
                            scheduler.step()
                    else:
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                    loss_value = float(loss.detach().cpu())
                    losses.append(loss_value)
                    global_step += 1
                    epoch_bar.update(1)
                    epoch_bar.set_postfix(loss=f"{loss_value:.4f}", global_step=global_step, layers=",".join(str(layers[i]) for i in layer_ids))
        finally:
            epoch_bar.close()
        epoch_loss = float(np.mean(losses))
        log(f"{model_name} segmentation ConvNeXt-Tiny epoch {epoch + 1}: loss={epoch_loss:.4f}")
        wandb_log(
            {
                f"segmentation/{model_name}/t{timestep_tag(args.timestep)}/convnext_tiny_train_loss": epoch_loss,
                "segmentation/epoch": epoch + 1,
                "segmentation/global_step": global_step,
                "segmentation/timestep": args.timestep,
            }
        )

    conf = np.zeros((n_layers, NUM_CLASSES_VOC, NUM_CLASSES_VOC), dtype=np.int64)
    probe.eval()
    with torch.no_grad():
        for layer_start in progress(
            range(0, n_layers, eval_layer_batch_size),
            total=math.ceil(n_layers / eval_layer_batch_size),
            desc=f"seg ConvNeXt-Tiny eval layers {model_name}",
            unit="layer_chunk",
        ):
            layer_ids = np.arange(layer_start, min(layer_start + eval_layer_batch_size, n_layers), dtype=np.int64)
            for batch_indices in batched_indices(len(val_masks), args.seg_eval_batch_size, shuffle=False, seed=args.seed):
                batch_x, batch_masks, batch_layers, batch_size, n_selected = make_batch(val_features, val_masks, batch_indices, layer_ids)
                batch_x = prepare_input(batch_x)
                with autocast_context():
                    logits = probe(batch_x, batch_layers)
                    logits = F.interpolate(logits, size=batch_masks.shape[-2:], mode="bilinear", align_corners=False)
                pred = logits.argmax(dim=1).detach().cpu().numpy().reshape(batch_size, n_selected, batch_masks.shape[-2], batch_masks.shape[-1])
                pred_lbhw = np.transpose(pred, (1, 0, 2, 3))
                masks_np = np.asarray(val_masks[batch_indices], dtype=np.int32)
                update_confusion(conf[layer_start : layer_start + len(layer_ids)], pred_lbhw, masks_np)

    miou = mean_iou(conf)
    rows = [
        {
            "model_name": model_name,
            "probe_type": SEG_PROBE_TYPE,
            "layer_index": layer,
            "timestep": args.timestep,
            "mIoU": 100.0 * miou[idx],
        }
        for idx, layer in enumerate(layers)
    ]

    def unwrap_state_dict(module: nn.Module) -> dict:
        return module.module.state_dict() if isinstance(module, nn.DataParallel) else module.state_dict()

    save_torch_probe_artifact(
        Path(args.output_dir),
        model_name,
        "segmentation_convnext_tiny",
        {SEG_PROBE_TYPE: unwrap_state_dict(probe)},
        {
            "model_name": model_name,
            "experiment": "PASCAL VOC semantic segmentation ConvNeXt-Tiny probing",
            "probe_types": {SEG_PROBE_TYPE: SEG_PROBE_LABELS[SEG_PROBE_TYPE]},
            "layers": layers,
            "timestep": args.timestep,
            "global_batch_size": args.global_batch_size,
            "train_batch_size": train_batch_size,
            "layer_batch_size": layer_batch_size,
            "eval_layer_batch_size": eval_layer_batch_size,
            "epochs": args.seg_epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_classes": NUM_CLASSES_VOC,
            "projected_channels": in_channels,
            "feature_shape": ["num_examples", "num_layers", "projected_channels", "grid_h", "grid_w"],
            "cnn_input_size": args.seg_cnn_input_size,
            "out_indices": list(out_indices),
            "amp": use_amp,
            "data_parallel": use_data_parallel,
            "shared_trunk": True,
            "layer_specific_classifier": True,
        },
    )
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def update_confusion(conf: np.ndarray, preds_lbhw: np.ndarray, masks_bhw: np.ndarray) -> None:
    valid = masks_bhw != 255
    target = masks_bhw[valid].astype(np.int64)
    for layer_idx in range(preds_lbhw.shape[0]):
        pred = preds_lbhw[layer_idx][valid].astype(np.int64)
        hist = np.bincount(NUM_CLASSES_VOC * target + pred, minlength=NUM_CLASSES_VOC * NUM_CLASSES_VOC)
        conf[layer_idx] += hist.reshape(NUM_CLASSES_VOC, NUM_CLASSES_VOC)


def mean_iou(conf: np.ndarray) -> np.ndarray:
    intersection = np.diagonal(conf, axis1=1, axis2=2).astype(np.float64)
    union = conf.sum(axis=1) + conf.sum(axis=2) - intersection
    valid = union > 0
    iou = np.zeros_like(intersection, dtype=np.float64)
    np.divide(intersection, np.maximum(union, 1.0), out=iou, where=valid)
    return np.sum(iou * valid, axis=1) / np.maximum(valid.sum(axis=1), 1)


def extract_model_cka_features(
    examples: list[Example],
    ckpt: LoadedCheckpoint,
    extract_fn,
    layers: list[int],
    args: argparse.Namespace,
    model_feature: str,
    condition_policy: str,
) -> np.ndarray:
    cache_dir = Path(args.output_dir) / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"cka_model_{model_feature}_{condition_policy}_seed{args.seed}"
    feature_path, _ = feature_cache_paths(cache_dir, prefix, ckpt.name, "imagenet", layers, args.timestep, len(examples))
    if feature_path.exists() and not args.recompute_cache:
        return np.load(feature_path, mmap_mode="r")

    features = open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(examples), len(layers), ckpt.config["hidden_size"]),
    )
    log(
        f"Extracting ImageNet model features for CKA: {ckpt.name}, "
        f"feature={model_feature}, condition={condition_policy}, {len(examples)} images"
    )
    total_batches = math.ceil(len(examples) / args.cka_batch_size)
    progress_prefix = f"extract/{ckpt.name}/t{timestep_tag(args.timestep)}/cka_{model_feature}_{condition_policy}"
    for batch_id, batch_indices in enumerate(progress(
        batched_indices(len(examples), args.cka_batch_size, shuffle=False, seed=args.seed),
        total=total_batches,
        desc=f"cka {model_feature}/{condition_policy} {ckpt.name}",
        unit="batch",
    )):
        batch_images = np.stack([image_to_vae_array(examples[int(index)].image_path) for index in batch_indices])
        batch_labels = np.asarray(
            [examples[int(index)].label if examples[int(index)].label is not None else NUM_CLASSES_IMAGENET for index in batch_indices],
            dtype=np.int32,
        )
        noise_seed = jnp.asarray(
            batch_noise_seed(args.seed, args.timestep, batch_indices, f"cka:{model_feature}:{condition_policy}"),
            dtype=jnp.uint32,
        )
        summaries = np.asarray(
            jax.device_get(extract_fn(jnp.asarray(batch_images), jnp.asarray(batch_labels), noise_seed)),
            dtype=np.float32,
        )
        features[batch_indices] = np.transpose(summaries, (1, 0, 2))
        wandb_log_progress(
            progress_prefix,
            min((batch_id + 1) * args.cka_batch_size, len(examples)),
            len(examples),
            batches=batch_id + 1,
            total_batches=total_batches,
        )
    features.flush()
    return np.load(feature_path, mmap_mode="r")


def load_dinov2_g(args: argparse.Namespace):
    repo = Path(args.dinov2_repo).resolve()
    weights = Path(args.dinov2_weights).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    device = args.dino_device
    if device == "auto":
        device = "cuda:1" if torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"Loading DINOv2-g on {device}")
    model = torch.hub.load(str(repo), "dinov2_vitg14", source="local", pretrained=False)
    state = torch.load(weights, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"DINOv2-g load mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device).eval()
    return model, torch.device(device)


def extract_dino_features(
    examples: list[Example],
    args: argparse.Namespace,
    dino_feature: str | None = None,
    target_grid: int | None = None,
) -> np.ndarray:
    dino_feature = dino_feature or args.dino_feature
    cache_root = Path(args.shared_cache_dir) if args.shared_cache_dir else Path(args.output_dir)
    cache_dir = cache_root / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    effective_size = dino_effective_size(args.dino_input_size)
    image_sig = cache_signature([str(example.image_path) for example in examples])
    sig = cache_signature(
        [
            "dinov2g",
            dino_feature,
            str(target_grid or ""),
            str(len(examples)),
            image_sig,
            str(args.dino_input_size),
            str(effective_size),
        ]
    )
    feature_path = cache_dir / f"dinov2g_{dino_feature}_{sig}.features.npy"
    lock_path = cache_dir / f"{feature_path.name}.lock"

    with open(lock_path, "w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        if feature_path.exists() and not args.recompute_cache:
            return np.load(feature_path, mmap_mode="r")

        model, device = load_dinov2_g(args)
        if effective_size != args.dino_input_size:
            log(
                f"DINOv2-g patch size is 14; center-cropping DINO inputs "
                f"from {args.dino_input_size} to {effective_size}."
            )
        first = True
        features = None
        log(f"Extracting DINOv2-g features for {len(examples)} images")
        total_batches = math.ceil(len(examples) / args.dino_batch_size)
        progress_prefix = f"extract/dinov2g/{dino_feature}"
        for batch_id, batch_indices in enumerate(progress(
            batched_indices(len(examples), args.dino_batch_size, shuffle=False, seed=args.seed),
            total=total_batches,
            desc="cka dino",
            unit="batch",
        )):
            batch = torch.stack([image_to_dino_tensor(examples[int(index)].image_path, args.dino_input_size) for index in batch_indices])
            batch = batch.to(device, non_blocking=True)
            with torch.inference_mode():
                out = model.forward_features(batch)
                feat = dino_features_from_output(out, dino_feature, target_grid)
                feat_np = feat.float().cpu().numpy()
            if first:
                features = open_memmap(feature_path, mode="w+", dtype=np.float32, shape=(len(examples), *feat_np.shape[1:]))
                first = False
            features[batch_indices] = feat_np
            wandb_log_progress(
                progress_prefix,
                min((batch_id + 1) * args.dino_batch_size, len(examples)),
                len(examples),
                batches=batch_id + 1,
                total_batches=total_batches,
            )
        if features is None:
            raise ValueError("No DINO features were extracted")
        features.flush()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return np.load(feature_path, mmap_mode="r")


def centered_linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    xy = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    numerator = np.sum(xy * xy)
    denominator = math.sqrt(np.sum(xx * xx) * np.sum(yy * yy))
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def centered_linear_cka_from_sums(
    xtx: np.ndarray,
    yy: np.ndarray,
    xty: np.ndarray,
    sum_x: np.ndarray,
    sum_y: np.ndarray,
    n_samples: int,
) -> float:
    if n_samples <= 1:
        return 0.0
    inv_n = 1.0 / float(n_samples)
    centered_xtx = xtx - np.outer(sum_x, sum_x) * inv_n
    centered_yy = yy - np.outer(sum_y, sum_y) * inv_n
    centered_xty = xty - np.outer(sum_x, sum_y) * inv_n
    numerator = float(np.sum(centered_xty * centered_xty))
    denominator = math.sqrt(float(np.sum(centered_xtx * centered_xtx)) * float(np.sum(centered_yy * centered_yy)))
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def run_cka(
    ckpt: LoadedCheckpoint,
    extract_fn,
    imagenet_examples: list[Example],
    dino_features: np.ndarray,
    layers: list[int],
    args: argparse.Namespace,
    model_feature: str,
    condition_policy: str,
    dino_feature: str,
):
    model_features = extract_model_cka_features(
        imagenet_examples,
        ckpt,
        extract_fn,
        layers,
        args,
        model_feature,
        condition_policy,
    )
    scenario = f"{model_feature}__{condition_policy}__dinov2_{dino_feature}"
    rows = []
    for idx, layer in enumerate(progress(layers, desc=f"compute cka {ckpt.name}", unit="layer")):
        cka = centered_linear_cka(model_features[:, idx, :], dino_features)
        rows.append(
            {
                "model_name": ckpt.name,
                "cka_scenario": scenario,
                "best_source": "",
                "model_feature": model_feature,
                "condition_policy": condition_policy,
                "dino_feature": dino_feature,
                "cka_sample_axis": "image",
                "layer_index": layer,
                "timestep": args.timestep,
                "cka": cka,
            }
        )
    return rows


def run_spatial_token_cka(
    ckpt: LoadedCheckpoint,
    extract_fn,
    imagenet_examples: list[Example],
    layers: list[int],
    args: argparse.Namespace,
    model_feature: str,
    condition_policy: str,
) -> list[dict]:
    model, device = load_dinov2_g(args)
    n_samples = 0
    sum_x = None
    sum_y = None
    xtx = None
    yy = None
    xty = None
    target_grid = None
    scenario = f"{model_feature}__{condition_policy}__dinov2_patch_tokens"
    log(
        f"Computing streaming spatial/token CKA for {ckpt.name}: "
        f"model tokens vs DINOv2-g patch tokens, condition={condition_policy}"
    )
    try:
        for batch_indices in progress(
            batched_indices(len(imagenet_examples), args.cka_batch_size, shuffle=False, seed=args.seed),
            total=math.ceil(len(imagenet_examples) / args.cka_batch_size),
            desc=f"cka spatial/{condition_policy} {ckpt.name}",
            unit="batch",
        ):
            batch_images = np.stack([image_to_vae_array(imagenet_examples[int(index)].image_path) for index in batch_indices])
            batch_labels = np.asarray(
                [
                    imagenet_examples[int(index)].label
                    if imagenet_examples[int(index)].label is not None
                    else NUM_CLASSES_IMAGENET
                    for index in batch_indices
                ],
                dtype=np.int32,
            )
            noise_seed = jnp.asarray(
                batch_noise_seed(args.seed, args.timestep, batch_indices, f"cka:{model_feature}:{condition_policy}"),
                dtype=jnp.uint32,
            )
            model_tokens = np.asarray(
                jax.device_get(extract_fn(jnp.asarray(batch_images), jnp.asarray(batch_labels), noise_seed)),
                dtype=np.float32,
            )
            n_layers, batch_size, n_tokens, x_dim = model_tokens.shape
            grid = int(math.sqrt(int(n_tokens)))
            if grid * grid != int(n_tokens):
                raise ValueError(f"Model token count is not square: {n_tokens}")
            if target_grid is None:
                target_grid = grid
                scenario = f"{scenario}_{target_grid}x{target_grid}"
            elif target_grid != grid:
                raise ValueError(f"Model token grid changed across batches: {target_grid} vs {grid}")

            batch = torch.stack(
                [image_to_dino_tensor(imagenet_examples[int(index)].image_path, args.dino_input_size) for index in batch_indices]
            )
            batch = batch.to(device, non_blocking=True)
            with torch.inference_mode():
                out = model.forward_features(batch)
                dino_tokens = dino_features_from_output(out, "patch_tokens", target_grid).float().cpu().numpy()

            y = dino_tokens.reshape(batch_size * n_tokens, dino_tokens.shape[-1]).astype(np.float64, copy=False)
            if sum_x is None:
                y_dim = y.shape[1]
                sum_x = np.zeros((n_layers, x_dim), dtype=np.float64)
                sum_y = np.zeros((y_dim,), dtype=np.float64)
                xtx = np.zeros((n_layers, x_dim, x_dim), dtype=np.float64)
                yy = np.zeros((y_dim, y_dim), dtype=np.float64)
                xty = np.zeros((n_layers, x_dim, y_dim), dtype=np.float64)
            assert sum_x is not None and sum_y is not None and xtx is not None and yy is not None and xty is not None
            n_samples += y.shape[0]
            sum_y += y.sum(axis=0)
            yy += y.T @ y
            flat_model_tokens = model_tokens.reshape(n_layers, batch_size * n_tokens, x_dim)
            for layer_idx in range(n_layers):
                x = flat_model_tokens[layer_idx].astype(np.float64, copy=False)
                sum_x[layer_idx] += x.sum(axis=0)
                xtx[layer_idx] += x.T @ x
                xty[layer_idx] += x.T @ y
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if sum_x is None or sum_y is None or xtx is None or yy is None or xty is None:
        raise ValueError("No spatial/token CKA samples were processed")
    rows = []
    for idx, layer in enumerate(progress(layers, desc=f"compute spatial cka {ckpt.name}", unit="layer")):
        cka = centered_linear_cka_from_sums(xtx[idx], yy, xty[idx], sum_x[idx], sum_y, n_samples)
        rows.append(
            {
                "model_name": ckpt.name,
                "cka_scenario": scenario,
                "best_source": "",
                "model_feature": model_feature,
                "condition_policy": condition_policy,
                "dino_feature": f"patch_tokens_{target_grid}x{target_grid}",
                "cka_sample_axis": "image_token",
                "layer_index": layer,
                "timestep": args.timestep,
                "cka": cka,
            }
        )
    return rows


def add_best_cka_rows(rows: list[dict]) -> list[dict]:
    best_by_key: dict[tuple[str, float, int], dict] = {}
    for row in rows:
        key = (row["model_name"], float(row["timestep"]), int(row["layer_index"]))
        if key not in best_by_key or float(row["cka"]) > float(best_by_key[key]["cka"]):
            best_by_key[key] = row
    best_rows = []
    for row in best_by_key.values():
        best = dict(row)
        best["best_source"] = row["cka_scenario"]
        best["cka_scenario"] = "best"
        best_rows.append(best)
    return rows + sorted(best_rows, key=lambda r: (r["model_name"], float(r["timestep"]), int(r["layer_index"])))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    log(f"Wrote {path}")


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as handle:
        return list(csv.DictReader(handle))


MODEL_STYLE = {
    "sit": {"label": "SiT baseline (1M)", "color": "#1f77b4", "order": 0},
    "layersync": {"label": "LayerSync (800k)", "color": "#d62728", "order": 1},
    "lara": {"label": "LARA (400k)", "color": "#7b2cbf", "order": 2},
}


def model_style_key(model: str) -> str | None:
    name = model.lower()
    if "sit" in name:
        return "sit"
    if "layersync" in name or "layer_sync" in name:
        return "layersync"
    if "lara" in name:
        return "lara"
    return None


def display_model_name(model: str) -> str:
    key = model_style_key(model)
    if key is not None:
        return MODEL_STYLE[key]["label"]
    return model


def model_sort_key(model: str) -> tuple[int, str]:
    key = model_style_key(model)
    if key is not None:
        return int(MODEL_STYLE[key]["order"]), model
    return 99, model


def records_to_series(
    rows: list[dict],
    metric: str,
    probe_type: str | None = None,
    timestep: float | None = None,
    cka_scenario: str | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    series: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        if probe_type is not None and row.get("probe_type") != probe_type:
            continue
        if cka_scenario is not None and row.get("cka_scenario") != cka_scenario:
            continue
        if timestep is not None and "timestep" in row:
            if abs(float(row["timestep"]) - float(timestep)) > 1e-8:
                continue
        model = row["model_name"]
        series.setdefault(model, []).append((int(row["layer_index"]), float(row[metric])))
    out = {}
    for model, values in series.items():
        values = sorted(values)
        if not values:
            continue
        out[model] = (np.asarray([item[0] for item in values]), np.asarray([item[1] for item in values]))
    return dict(sorted(out.items(), key=lambda item: model_sort_key(item[0])))


def color_for_model(model: str, index: int) -> str:
    key = model_style_key(model)
    if key is not None:
        return MODEL_STYLE[key]["color"]
    colors = ["#1f77b4", "#d62728", "#7b2cbf", "#2ca02c", "#ff7f0e"]
    return colors[index % len(colors)]


def plot_panel(ax, series: dict[str, tuple[np.ndarray, np.ndarray]], ylabel: str, title: str):
    for idx, (model, (x, y)) in enumerate(series.items()):
        color = color_for_model(model, idx)
        ax.plot(x, y, marker="o", linewidth=2, label=display_model_name(model), color=color)
        avg = float(np.mean(y))
        ax.axhline(avg, linestyle="--", linewidth=1.2, color=color, alpha=0.65)
        max_idx = int(np.argmax(y))
        ax.scatter([x[max_idx]], [y[max_idx]], marker="*", s=130, color="black", zorder=5)
    ax.set_title(title)
    ax.set_xlabel("Layer index")
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.4, alpha=0.3)
    if len(series) >= 2:
        names = list(series)
        base_name, method_name = names[0], names[1]
        base_avg = np.mean(series[base_name][1])
        method_avg = np.mean(series[method_name][1])
        if abs(base_avg) > 1e-12:
            delta = 100.0 * (method_avg - base_avg) / base_avg
            ax.text(0.03, 0.95, f"avg delta: {delta:+.1f}%", transform=ax.transAxes, va="top")


def plot_main_figure(
    output_dir: Path,
    main_timestep: float = 1.0,
    cka_main_scenario: str = "summary__null__dinov2_cls",
    class_main_probe: str = "convnext_tiny_probe",
) -> None:
    cls_rows = read_csv_rows(output_dir / "classification_probe.csv")
    seg_rows = read_csv_rows(output_dir / "segmentation_probe.csv")
    cka_rows = read_csv_rows(output_dir / "cka_dinov2g.csv")
    if not cls_rows and not seg_rows and not cka_rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    plot_panel(
        axes[0],
        records_to_series(cls_rows, "val_accuracy", class_main_probe, timestep=main_timestep),
        "Validation accuracy (%)",
        f"(a) Tiny ImageNet classification accuracy | {CLASS_PROBE_LABELS.get(class_main_probe, class_main_probe)} | t={main_timestep}",
    )
    plot_panel(
        axes[1],
        records_to_series(seg_rows, "mIoU", SEG_PROBE_TYPE, timestep=main_timestep),
        "mIoU (%)",
        f"(b) PASCAL VOC semantic segmentation mIoU | {SEG_PROBE_LABELS[SEG_PROBE_TYPE]} | t={main_timestep}",
    )
    plot_panel(
        axes[2],
        records_to_series(cka_rows, "cka", None, timestep=main_timestep, cka_scenario=cka_main_scenario),
        "Linear CKA",
        f"(c) ImageNet representation alignment vs DINOv2-g | CKA={cka_main_scenario} | t={main_timestep}",
    )
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    by_label = dict(zip(labels, handles))
    if by_label:
        fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=max(1, len(by_label)))
    fig.savefig(output_dir / "representation_eval_main.png", dpi=220)
    plt.close(fig)
    log(f"Wrote {output_dir / 'representation_eval_main.png'}")


def plot_ablation_figure(output_dir: Path, main_timestep: float = 1.0) -> None:
    cls_rows = read_csv_rows(output_dir / "classification_probe.csv")
    seg_rows = read_csv_rows(output_dir / "segmentation_probe.csv")
    if not cls_rows and not seg_rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    plot_probe_ablation(
        axes[0],
        cls_rows,
        "val_accuracy",
        CLASS_PROBE_LABELS,
        f"Tiny ImageNet classification probe ablation | validation accuracy | t={main_timestep}",
        "Validation accuracy (%)",
        timestep=main_timestep,
    )
    plot_probe_ablation(
        axes[1],
        seg_rows,
        "mIoU",
        SEG_PROBE_LABELS,
        f"PASCAL VOC segmentation probe ablation | mIoU | t={main_timestep}",
        "mIoU (%)",
        timestep=main_timestep,
    )
    fig.legend(loc="lower center", ncol=2)
    fig.savefig(output_dir / "probe_arch_ablation.png", dpi=220)
    plt.close(fig)
    log(f"Wrote {output_dir / 'probe_arch_ablation.png'}")


def plot_probe_ablation(
    ax,
    rows: list[dict],
    metric: str,
    labels: dict[str, str],
    title: str,
    ylabel: str,
    timestep: float | None = None,
):
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for row in rows:
        probe = row.get("probe_type")
        if probe not in labels:
            continue
        if timestep is not None and abs(float(row["timestep"]) - float(timestep)) > 1e-8:
            continue
        grouped.setdefault((row["model_name"], probe), []).append((int(row["layer_index"]), float(row[metric])))
    linestyles = {
        "simple_linear": ":",
        "sra_style": "-.",
        "resnet18_probe": "--",
        "convnext_atto_probe": "-",
        "convnext_tiny_probe": "-",
        "simple_linear_seg": "--",
        "sra_inspired_dense_seg": "-",
        SEG_PROBE_TYPE: "-",
    }
    ordered_items = sorted(grouped.items(), key=lambda item: (model_sort_key(item[0][0]), item[0][1]))
    for idx, ((model, probe), values) in enumerate(ordered_items):
        values = sorted(values)
        x = [item[0] for item in values]
        y = [item[1] for item in values]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            linestyle=linestyles.get(probe, "-"),
            label=f"{display_model_name(model)}: {labels[probe]}",
            color=color_for_model(model, idx),
        )
    ax.set_title(title)
    ax.set_xlabel("Layer index")
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.4, alpha=0.3)


def plot_heatmap_file(
    output_dir: Path,
    rows: list[dict],
    *,
    metric: str,
    probe_type: str | None,
    filename: str,
    title: str,
    colorbar_label: str,
) -> None:
    filtered = []
    for row in rows:
        if probe_type is not None and row.get("probe_type") != probe_type:
            continue
        filtered.append(row)
    if not filtered:
        return

    models = sorted({row["model_name"] for row in filtered}, key=model_sort_key)
    layers = sorted({int(row["layer_index"]) for row in filtered})
    timesteps = sorted({float(row["timestep"]) for row in filtered})
    if len(timesteps) < 2:
        return

    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(5.2 * len(models), 4.6),
        squeeze=False,
        constrained_layout=True,
    )
    last_im = None
    for ax, model in zip(axes[0], models):
        matrix = np.full((len(timesteps), len(layers)), np.nan, dtype=np.float32)
        for row in filtered:
            if row["model_name"] != model:
                continue
            t_idx = timesteps.index(float(row["timestep"]))
            l_idx = layers.index(int(row["layer_index"]))
            matrix[t_idx, l_idx] = float(row[metric])
        last_im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(f"{title}: {display_model_name(model)}")
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Timestep")
        ax.set_xticks(np.arange(len(layers)))
        ax.set_xticklabels([str(layer) for layer in layers], rotation=90 if len(layers) > 14 else 0)
        ax.set_yticks(np.arange(len(timesteps)))
        ax.set_yticklabels([f"{t:g}" for t in timesteps])
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), label=colorbar_label)
    path = output_dir / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    log(f"Wrote {path}")


def plot_heatmaps(output_dir: Path, cka_main_scenario: str = "summary__null__dinov2_cls", class_main_probe: str = "convnext_tiny_probe") -> None:
    cls_rows = read_csv_rows(output_dir / "classification_probe.csv")
    seg_rows = read_csv_rows(output_dir / "segmentation_probe.csv")
    cka_rows = read_csv_rows(output_dir / "cka_dinov2g.csv")
    plot_heatmap_file(
        output_dir,
        cls_rows,
        metric="val_accuracy",
        probe_type=class_main_probe,
        filename="classification_probe_heatmap.png",
        title=f"Tiny ImageNet classification accuracy over timestep/layer ({CLASS_PROBE_LABELS.get(class_main_probe, class_main_probe)})",
        colorbar_label="Accuracy (%)",
    )
    plot_heatmap_file(
        output_dir,
        seg_rows,
        metric="mIoU",
        probe_type=SEG_PROBE_TYPE,
        filename="segmentation_probe_heatmap.png",
        title=f"PASCAL VOC segmentation mIoU over timestep/layer ({SEG_PROBE_LABELS[SEG_PROBE_TYPE]})",
        colorbar_label="mIoU (%)",
    )
    plot_heatmap_file(
        output_dir,
        [row for row in cka_rows if row.get("cka_scenario") == cka_main_scenario],
        metric="cka",
        probe_type=None,
        filename="cka_heatmap.png",
        title=f"ImageNet CKA alignment vs DINOv2-g ({cka_main_scenario}) over timestep/layer",
        colorbar_label="CKA",
    )


def run_classification(checkpoints_: list[LoadedCheckpoint], vae, vae_params, layers: list[int], args) -> list[dict]:
    train_examples, val_examples = load_tiny_imagenet(
        Path(args.tiny_imagenet_root),
        args.seed,
        args.limit_tiny_train,
        args.limit_tiny_val,
    )
    rows = []
    null_label = NUM_CLASSES_IMAGENET if args.cfg_dropout_rate > 0 else 0
    for ckpt in checkpoints_:
        needs_linear = bool(set(args.class_probes) & CLASS_LINEAR_PROBES)
        needs_cnn = bool(set(args.class_probes) & CLASS_CNN_PROBES)

        if needs_linear and needs_cnn:
            combined_extract_fn = make_projected_grid_and_summary_extractor(
                ckpt,
                vae,
                vae_params,
                layers,
                args.timestep,
                null_label,
                args.class_probe_proj_channels,
                args.seed,
            )
            train_summary, train_labels, train_grid, train_grid_labels = extract_combined_classification_cache(
                train_examples,
                ckpt,
                combined_extract_fn,
                layers,
                args,
                "train",
            )
            val_summary, val_labels, val_grid, val_grid_labels = extract_combined_classification_cache(
                val_examples,
                ckpt,
                combined_extract_fn,
                layers,
                args,
                "val",
            )
            rows.extend(
                train_linear_classification_probes(
                    ckpt.name,
                    train_summary,
                    train_labels,
                    val_summary,
                    val_labels,
                    layers,
                    args,
                )
            )
            try:
                jax.clear_caches()
            except Exception:
                pass
            rows.extend(train_classification_probes(ckpt.name, train_grid, train_grid_labels, val_grid, val_grid_labels, layers, args))
            continue

        if needs_linear:
            summary_extract_fn = make_summary_extractor(ckpt, vae, vae_params, layers, args.timestep, null_label)
            train_features, train_labels = extract_summary_cache(train_examples, ckpt, summary_extract_fn, layers, args, "train")
            val_features, val_labels = extract_summary_cache(val_examples, ckpt, summary_extract_fn, layers, args, "val")
            rows.extend(
                train_linear_classification_probes(
                    ckpt.name,
                    train_features,
                    train_labels,
                    val_features,
                    val_labels,
                    layers,
                    args,
                )
            )
            try:
                jax.clear_caches()
            except Exception:
                pass

        if needs_cnn:
            grid_extract_fn = make_projected_grid_extractor(
                ckpt,
                vae,
                vae_params,
                layers,
                args.timestep,
                null_label,
                args.class_probe_proj_channels,
                args.seed,
            )
            train_features, train_labels = extract_projected_grid_cache(train_examples, ckpt, grid_extract_fn, layers, args, "train")
            val_features, val_labels = extract_projected_grid_cache(val_examples, ckpt, grid_extract_fn, layers, args, "val")
            rows.extend(train_classification_probes(ckpt.name, train_features, train_labels, val_features, val_labels, layers, args))
    return rows


def run_segmentation(checkpoints_: list[LoadedCheckpoint], vae, vae_params, layers: list[int], args) -> list[dict]:
    train_examples, val_examples = load_voc(
        Path(args.voc_root),
        args.seed,
        args.limit_voc_train,
        args.limit_voc_val,
    )
    rows = []
    null_label = NUM_CLASSES_IMAGENET if args.cfg_dropout_rate > 0 else 0
    for ckpt in checkpoints_:
        extract_fn = make_projected_grid_extractor(
            ckpt,
            vae,
            vae_params,
            layers,
            args.timestep,
            null_label,
            args.seg_probe_proj_channels,
            args.seed + 3000,
        )
        train_features, train_masks = extract_projected_voc_grid_cache(train_examples, ckpt, extract_fn, layers, args, "train")
        val_features, val_masks = extract_projected_voc_grid_cache(val_examples, ckpt, extract_fn, layers, args, "val")
        rows.extend(
            train_segmentation_convnext_tiny_probe(
                ckpt.name,
                train_features,
                train_masks,
                val_features,
                val_masks,
                layers,
                args,
            )
        )
    return rows


def run_all_cka(checkpoints_: list[LoadedCheckpoint], vae, vae_params, layers: list[int], args) -> list[dict]:
    repo_root = Path.cwd()
    imagenet_examples = load_imagenet_manifest(
        Path(args.imagenet_manifest),
        repo_root,
        args.limit_imagenet,
        args.seed,
    )
    global_dino_feature_names = [dino_feature for dino_feature in args.cka_dino_features if dino_feature != "patch_tokens"]
    dino_features_by_name = {
        dino_feature: extract_dino_features(imagenet_examples, args, dino_feature)
        for dino_feature in global_dino_feature_names
    }
    rows = []
    null_label = NUM_CLASSES_IMAGENET if args.cfg_dropout_rate > 0 else 0
    for ckpt in checkpoints_:
        for condition_policy in args.cka_condition_policies:
            if condition_policy == "label" and not all(example.label is not None for example in imagenet_examples):
                log(f"Skipping CKA label condition for {ckpt.name}: ImageNet manifest has missing labels.")
                continue
            for model_feature in args.cka_model_features:
                if model_feature == "summary":
                    extract_fn = make_cka_summary_extractor(
                        ckpt, vae, vae_params, layers, args.timestep, null_label, condition_policy
                    )
                elif model_feature == "raw_mean":
                    extract_fn = make_cka_raw_mean_extractor(
                        ckpt, vae, vae_params, layers, args.timestep, null_label, condition_policy
                    )
                elif model_feature == "spatial_tokens":
                    if "patch_tokens" not in args.cka_dino_features:
                        log("Skipping spatial_tokens CKA because --cka-dino-features does not include patch_tokens.")
                        continue
                    extract_fn = make_cka_spatial_extractor(
                        ckpt, vae, vae_params, layers, args.timestep, null_label, condition_policy
                    )
                    rows.extend(
                        run_spatial_token_cka(
                            ckpt,
                            extract_fn,
                            imagenet_examples,
                            layers,
                            args,
                            model_feature,
                            condition_policy,
                        )
                    )
                    continue
                else:
                    raise ValueError(f"Unknown CKA model feature: {model_feature}")
                for dino_feature, dino_features in dino_features_by_name.items():
                    rows.extend(
                        run_cka(
                            ckpt,
                            extract_fn,
                            imagenet_examples,
                            dino_features,
                            layers,
                            args,
                            model_feature,
                            condition_policy,
                            dino_feature,
                        )
                    )
    scenarios = {row["cka_scenario"] for row in rows}
    if len(scenarios) <= 1:
        return rows
    return add_best_cka_rows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", default=[], help="Checkpoint spec: name=path. Repeat for baseline/method.")
    parser.add_argument("--param-set", default="ema", help="Checkpoint parameter set: ema, online, or a subdir name.")
    parser.add_argument("--model-size", default="XL", choices=sorted(DIT_VARIANTS))
    parser.add_argument("--cfg-dropout-rate", type=float, default=0.1)
    parser.add_argument("--experiments", default="all", help="Comma list: classification,segmentation,cka,all")
    parser.add_argument("--layers", default="all", help="Layer list, e.g. all, 1,14,28, or 1-28")
    parser.add_argument(
        "--timestep",
        type=float,
        default=1.0,
        help="Main timestep used for the Figure-4-style main plots. Repo convention: 0=noise, 1=clean/least noisy.",
    )
    parser.add_argument(
        "--timesteps",
        default="1.0",
        help="Comma list for timestep sweep. Repo convention: 0=noise, 1=clean/least noisy.",
    )
    parser.add_argument("--output-dir", default="results/representation_eval")
    parser.add_argument("--shared-cache-dir", default=None, help="Optional shared cache dir for reusable assets such as DINO features.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--recompute-cache", action="store_true")

    parser.add_argument("--wandb", action="store_true", help="Log CSV tables, figures, and training losses to W&B.")
    parser.add_argument("--wandb-entity", default="Fingerprint_Recognition")
    parser.add_argument("--wandb-project", default="representation_eval")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-api-key", default=None, help="Optional; prefer setting WANDB_API_KEY in the environment.")
    parser.add_argument("--wandb-workers", action="store_true", help="Create live W&B runs for parallel workers.")
    parser.add_argument("--hf-upload", action="store_true", help="Upload CSVs, figures, and probe artifacts to Hugging Face.")
    parser.add_argument("--hf-repo-id", default=None, help="Target repo, e.g. username/self-flow-representation-eval. Defaults to token owner.")
    parser.add_argument("--hf-repo-type", default="dataset", choices=["dataset", "model"])
    parser.add_argument("--hf-token", default=None, help="Optional; prefer setting HF_TOKEN in the environment.")
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--hf-path-in-repo", default=None)
    parser.add_argument("--hf-commit-message", default="Upload Self-Flow representation evaluation results")
    parser.add_argument("--hf-include-feature-cache", action="store_true")
    parser.add_argument("--hf-ignore-patterns", default=None, help="Comma-separated extra ignore patterns.")

    parser.add_argument("--vae-path", default="checkpoints/vae/sdvae-ema-kaggle-flax")
    parser.add_argument("--tiny-imagenet-root", default="data/tiny-imagenet/tiny-imagenet-200")
    parser.add_argument("--voc-root", default="data/pascal-voc/VOCdevkit/VOC2012")
    parser.add_argument("--imagenet-manifest", default="data/imagenet-val/imagenet_val_subset_4000_256.txt")
    parser.add_argument("--dinov2-repo", default="external/dinov2")
    parser.add_argument("--dinov2-weights", default="checkpoints/dinov2-g/dinov2_vitg14_pretrain.pth")
    parser.add_argument("--dino-device", default="auto")
    parser.add_argument("--dino-feature", default="cls", choices=["cls", "patch_mean", "patch_tokens"])
    parser.add_argument("--cka-dino-features", default="cls", help="Comma list: cls,patch_mean,patch_tokens.")
    parser.add_argument(
        "--cka-model-features",
        default="summary",
        help="Comma list: summary,raw_mean,spatial_tokens. spatial_tokens computes streaming token-level CKA.",
    )
    parser.add_argument("--cka-condition-policies", default="null", help="Comma list: null,label.")
    parser.add_argument("--cka-main-scenario", default="summary__null__dinov2_cls", help="CKA scenario used in main plots/heatmaps.")
    parser.add_argument("--dino-input-size", type=int, default=256)

    parser.add_argument("--class-epochs", type=int, default=50)
    parser.add_argument("--seg-epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument(
        "--class-probes",
        default="convnext_tiny_probe",
        help="Comma list: simple_linear,sra_style,resnet18_probe,convnext_atto_probe,convnext_tiny_probe.",
    )
    parser.add_argument(
        "--class-main-probe",
        default="convnext_tiny_probe",
        help="Probe type used in the main classification panel/heatmap. Falls back to the first selected probe if not selected.",
    )
    parser.add_argument("--class-extract-batch-size", type=int, default=8)
    parser.add_argument("--class-train-batch-size", type=int, default=None, help="Defaults to --global-batch-size.")
    parser.add_argument("--class-probe-proj-channels", type=int, default=8, help="Fixed random projection channels for CNN classification probes.")
    parser.add_argument("--class-feature-grid-size", type=int, default=16, help="Spatial grid size of model tokens before CNN upsampling.")
    parser.add_argument("--class-cnn-input-size", type=int, default=32, help="Upsampled spatial size fed to CNN classification probes.")
    parser.add_argument("--class-cnn-layer-batch-size", type=int, default=4, help="Number of layers sampled per classification CNN probe step.")
    parser.add_argument("--class-cnn-eval-layer-batch-size", type=int, default=None, help="Defaults to --class-cnn-layer-batch-size.")
    parser.add_argument("--class-probe-device", default="auto", help="Torch device for CNN probes, e.g. auto, cuda, cuda:0, cpu.")
    parser.add_argument("--no-class-probe-data-parallel", action="store_false", dest="class_probe_data_parallel", help="Disable Torch DataParallel for CNN probes when multiple GPUs are visible in a single worker.")
    parser.add_argument("--no-class-probe-amp", action="store_false", dest="class_probe_amp", help="Disable AMP for Torch CNN probes.")
    parser.add_argument("--seg-microbatch-size", type=int, default=1, help="Memory-safe microbatch for VOC gradient accumulation.")
    parser.add_argument("--seg-eval-microbatch-size", type=int, default=None, help="Defaults to --seg-microbatch-size.")
    parser.add_argument("--seg-batch-size", type=int, default=None, help="Deprecated alias for --seg-microbatch-size.")
    parser.add_argument("--seg-layer-chunk-size", type=int, default=4)
    parser.add_argument("--seg-extract-batch-size", type=int, default=None, help="Defaults to --class-extract-batch-size.")
    parser.add_argument("--seg-train-batch-size", type=int, default=None, help="Torch ConvNeXt-Tiny segmentation batch size. Defaults to --global-batch-size.")
    parser.add_argument("--seg-eval-batch-size", type=int, default=None, help="Defaults to --seg-train-batch-size.")
    parser.add_argument("--seg-probe-proj-channels", type=int, default=8, help="Fixed random projection channels for segmentation ConvNeXt-Tiny probe.")
    parser.add_argument("--seg-feature-grid-size", type=int, default=16, help="Spatial grid size of model tokens before segmentation CNN upsampling.")
    parser.add_argument("--seg-cnn-input-size", type=int, default=64, help="Upsampled spatial size fed to the segmentation ConvNeXt-Tiny probe.")
    parser.add_argument("--seg-cnn-layer-batch-size", type=int, default=1, help="Number of representation layers per segmentation ConvNeXt-Tiny step.")
    parser.add_argument("--seg-cnn-eval-layer-batch-size", type=int, default=None, help="Defaults to --seg-cnn-layer-batch-size.")
    parser.add_argument("--seg-cnn-out-indices", default="0,1,2,3", help="ConvNeXt feature stages used for dense segmentation, e.g. 0,1,2,3.")
    parser.add_argument("--seg-probe-device", default="auto", help="Torch device for segmentation ConvNeXt-Tiny probe.")
    parser.add_argument("--no-seg-probe-data-parallel", action="store_false", dest="seg_probe_data_parallel", help="Disable Torch DataParallel for segmentation probe.")
    parser.add_argument("--no-seg-probe-amp", action="store_false", dest="seg_probe_amp", help="Disable AMP for segmentation ConvNeXt-Tiny probe.")
    parser.add_argument("--cka-batch-size", type=int, default=8)
    parser.add_argument("--dino-batch-size", type=int, default=4)
    parser.add_argument(
        "--parallel-gpus",
        default="auto",
        help="Comma list of GPU ids for timestep-level parallel workers, e.g. 0,1. Use 'none' to force sequential.",
    )

    parser.add_argument("--limit-tiny-train", type=int, default=None)
    parser.add_argument("--limit-tiny-val", type=int, default=None)
    parser.add_argument("--limit-voc-train", type=int, default=None)
    parser.add_argument("--limit-voc-val", type=int, default=None)
    parser.add_argument("--limit-imagenet", type=int, default=None)
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--parallel-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--resume-workers", action="store_true", help="Reuse completed parallel worker directories instead of deleting them.")
    parser.add_argument("--keep-feature-cache", action="store_true", help="Keep generated feature caches after successful runs.")
    parser.add_argument("--keep-worker-outputs", action="store_true", help="Keep per-timestep worker directories after parallel merge.")
    args = parser.parse_args()
    args.main_timestep = float(args.timestep)
    args.eval_timesteps = parse_float_list(args.timesteps, args.main_timestep)
    args.class_probes = parse_class_probes(args.class_probes)
    if args.class_main_probe not in CLASS_PROBE_CHOICES:
        raise ValueError(f"Unknown --class-main-probe: {args.class_main_probe}. Valid: {CLASS_PROBE_CHOICES}")
    if args.class_main_probe not in args.class_probes:
        args.class_main_probe = args.class_probes[0]
    args.cka_dino_features = parse_str_list(args.cka_dino_features, ["cls"])
    args.cka_model_features = parse_str_list(args.cka_model_features, ["summary"])
    args.cka_condition_policies = parse_str_list(args.cka_condition_policies, ["null"])
    args.seg_cnn_out_indices = parse_int_list(args.seg_cnn_out_indices, [0, 1, 2, 3])
    if args.global_batch_size <= 0:
        raise ValueError("--global-batch-size must be positive")
    if args.class_train_batch_size is None:
        args.class_train_batch_size = args.global_batch_size
    if args.seg_batch_size is not None:
        args.seg_microbatch_size = args.seg_batch_size
    if args.seg_microbatch_size <= 0:
        raise ValueError("--seg-microbatch-size must be positive")
    if args.seg_eval_microbatch_size is None:
        args.seg_eval_microbatch_size = args.seg_microbatch_size
    if args.seg_eval_microbatch_size <= 0:
        raise ValueError("--seg-eval-microbatch-size must be positive")
    if args.seg_extract_batch_size is None:
        args.seg_extract_batch_size = args.class_extract_batch_size
    if args.seg_train_batch_size is None:
        args.seg_train_batch_size = args.global_batch_size
    if args.seg_eval_batch_size is None:
        args.seg_eval_batch_size = args.seg_train_batch_size
    if args.seg_extract_batch_size <= 0 or args.seg_train_batch_size <= 0 or args.seg_eval_batch_size <= 0:
        raise ValueError("--seg-extract-batch-size, --seg-train-batch-size, and --seg-eval-batch-size must be positive")
    if args.seg_probe_proj_channels <= 0 or args.seg_feature_grid_size <= 0 or args.seg_cnn_input_size <= 0:
        raise ValueError("--seg-probe-proj-channels, --seg-feature-grid-size, and --seg-cnn-input-size must be positive")
    if args.seg_cnn_layer_batch_size <= 0:
        raise ValueError("--seg-cnn-layer-batch-size must be positive")
    if args.seg_cnn_eval_layer_batch_size is not None and args.seg_cnn_eval_layer_batch_size <= 0:
        raise ValueError("--seg-cnn-eval-layer-batch-size must be positive")
    if any(index < 0 or index > 3 for index in args.seg_cnn_out_indices):
        raise ValueError("--seg-cnn-out-indices must contain ConvNeXt stage indices from 0 to 3")
    if args.class_probe_proj_channels <= 0:
        raise ValueError("--class-probe-proj-channels must be positive")
    if args.class_feature_grid_size <= 0 or args.class_cnn_input_size <= 0:
        raise ValueError("--class-feature-grid-size and --class-cnn-input-size must be positive")
    if args.class_cnn_layer_batch_size <= 0:
        raise ValueError("--class-cnn-layer-batch-size must be positive")
    if args.class_cnn_eval_layer_batch_size is not None and args.class_cnn_eval_layer_batch_size <= 0:
        raise ValueError("--class-cnn-eval-layer-batch-size must be positive")
    valid_dino_features = {"cls", "patch_mean", "patch_tokens"}
    valid_model_features = {"summary", "raw_mean", "spatial_tokens"}
    valid_condition_policies = {"null", "label"}
    if unknown := set(args.cka_dino_features) - valid_dino_features:
        raise ValueError(f"Unknown --cka-dino-features: {sorted(unknown)}")
    if unknown := set(args.cka_model_features) - valid_model_features:
        raise ValueError(f"Unknown --cka-model-features: {sorted(unknown)}")
    if unknown := set(args.cka_condition_policies) - valid_condition_policies:
        raise ValueError(f"Unknown --cka-condition-policies: {sorted(unknown)}")
    return args


def selected_experiments(value: str) -> set[str]:
    parts = {part.strip().lower() for part in value.replace(";", ",").split(",") if part.strip()}
    if "all" in parts or not parts:
        return {"classification", "segmentation", "cka"}
    valid = {"classification", "segmentation", "cka"}
    unknown = parts - valid
    if unknown:
        raise ValueError(f"Unknown experiments: {sorted(unknown)}")
    return parts


CSV_OUTPUTS = {
    "classification": (
        "classification_probe.csv",
        ["model_name", "probe_type", "layer_index", "timestep", "val_accuracy"],
        "val_accuracy",
    ),
    "segmentation": (
        "segmentation_probe.csv",
        ["model_name", "probe_type", "layer_index", "timestep", "mIoU"],
        "mIoU",
    ),
    "cka": (
        "cka_dinov2g.csv",
        [
            "model_name",
            "cka_scenario",
            "best_source",
            "model_feature",
            "condition_policy",
            "dino_feature",
            "cka_sample_axis",
            "layer_index",
            "timestep",
            "cka",
        ],
        "cka",
    ),
}


def resolve_parallel_gpus(value: str | None) -> list[str]:
    if value is None:
        return []
    value = value.strip()
    if value.lower() in {"", "none", "off", "false", "no"}:
        return []
    if value.lower() != "auto":
        return [piece.strip() for piece in value.split(",") if piece.strip()]

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        visible_ids = [piece.strip() for piece in visible.split(",") if piece.strip()]
        if len(visible_ids) > 1:
            return visible_ids
        return []

    count = 0
    try:
        count = int(torch.cuda.device_count())
    except Exception:
        count = 0
    if count <= 0:
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            count = len([line for line in output.splitlines() if line.strip()])
        except Exception:
            count = 0
    return [str(idx) for idx in range(count)] if count > 1 else []


def strip_cli_args(argv: list[str], options_with_values: set[str], flag_options: set[str]) -> list[str]:
    stripped = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token in options_with_values:
            skip_next = True
            continue
        if any(token.startswith(f"{option}=") for option in options_with_values):
            continue
        if token in flag_options:
            continue
        stripped.append(token)
    return stripped


def build_parallel_worker_command(
    args: argparse.Namespace,
    checkpoint_spec: tuple[str, Path],
    timestep: float,
    worker_dir: Path,
    gpu_id: str,
) -> list[str]:
    model_name, _ = checkpoint_spec
    options_with_values = {
        "--checkpoint",
        "--timestep",
        "--timesteps",
        "--output-dir",
        "--parallel-gpus",
        "--wandb-entity",
        "--wandb-project",
        "--wandb-run-name",
        "--wandb-mode",
        "--wandb-api-key",
        "--hf-repo-id",
        "--hf-repo-type",
        "--hf-token",
        "--hf-path-in-repo",
        "--hf-commit-message",
        "--hf-ignore-patterns",
    }
    flag_options = {
        "--parallel-worker",
        "--plots-only",
        "--wandb",
        "--wandb-workers",
        "--hf-upload",
        "--hf-private",
        "--hf-include-feature-cache",
        "--resume-workers",
    }
    worker_args = strip_cli_args(sys.argv[1:], options_with_values, flag_options)
    worker_run_name = sanitize_name(f"{args.wandb_run_name or Path(args.output_dir).name}_{model_name}_t{timestep_tag(timestep)}")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        *worker_args,
        "--checkpoint",
        checkpoint_spec_to_cli(checkpoint_spec),
        "--timestep",
        str(timestep),
        "--timesteps",
        str(timestep),
        "--output-dir",
        str(worker_dir),
        "--parallel-gpus",
        "none",
        "--parallel-worker",
    ]
    if args.wandb and args.wandb_workers:
        cmd.extend(
            [
                "--wandb",
                "--wandb-entity",
                args.wandb_entity,
                "--wandb-project",
                args.wandb_project,
                "--wandb-mode",
                args.wandb_mode,
                "--wandb-run-name",
                worker_run_name,
            ]
        )
    return cmd


def copy_tree_contents(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        target = dst / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def merge_parallel_worker_outputs(output_dir: Path, worker_dirs: list[Path], experiments: set[str]) -> None:
    for worker_dir in worker_dirs:
        copy_tree_contents(worker_dir / "probes", output_dir / "probes")

    for experiment in sorted(experiments):
        filename, fieldnames, _ = CSV_OUTPUTS[experiment]
        rows = []
        for worker_dir in worker_dirs:
            rows.extend(read_csv_rows(worker_dir / filename))
        if not rows:
            continue
        rows.sort(
            key=lambda row: (
                float(row.get("timestep", 0.0)),
                row.get("model_name", ""),
                row.get("probe_type", ""),
                int(row.get("layer_index", 0)),
            )
        )
        write_csv(output_dir / filename, rows, fieldnames)


def log_final_outputs_to_wandb(output_dir: Path, experiments: set[str]) -> None:
    for experiment in sorted(experiments):
        filename, fieldnames, metric = CSV_OUTPUTS[experiment]
        rows = read_csv_rows(output_dir / filename)
        if rows:
            table_name = filename.removesuffix(".csv")
            wandb_log_table(table_name, rows, fieldnames)
            wandb_log_metric_rows(table_name, rows, metric)
    wandb_log_image("representation_eval_main", output_dir / "representation_eval_main.png")
    wandb_log_image("probe_arch_ablation", output_dir / "probe_arch_ablation.png")
    wandb_log_image("classification_probe_heatmap", output_dir / "classification_probe_heatmap.png")
    wandb_log_image("segmentation_probe_heatmap", output_dir / "segmentation_probe_heatmap.png")
    wandb_log_image("cka_heatmap", output_dir / "cka_heatmap.png")
    wandb_log_result_artifact(output_dir)


def cleanup_feature_cache(output_dir: Path, args: argparse.Namespace) -> None:
    if args.keep_feature_cache:
        return
    cache_dir = output_dir / "feature_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        log(f"Removed feature cache {cache_dir}")


def worker_has_completed_outputs(worker_dir: Path, experiments: set[str]) -> bool:
    if not worker_dir.exists():
        return False
    for experiment in experiments:
        filename, _, _ = CSV_OUTPUTS[experiment]
        path = worker_dir / filename
        if not path.exists() or path.stat().st_size <= 0:
            return False
    return True


def run_timestep_worker(
    args: argparse.Namespace,
    checkpoint_spec: tuple[str, Path],
    timestep: float,
    worker_root: Path,
    gpu_queue: "queue.Queue[str]",
    print_lock: threading.Lock,
) -> Path:
    gpu_id = gpu_queue.get()
    model_name, _ = checkpoint_spec
    worker_dir = worker_root / sanitize_name(model_name) / f"t{timestep_tag(timestep)}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_parallel_worker_command(args, checkpoint_spec, timestep, worker_dir, gpu_id)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    if args.wandb:
        env.setdefault("WANDB_RUN_GROUP", args.wandb_run_name or Path(args.output_dir).name)
    prefix = f"[gpu {gpu_id} | {model_name} | t={timestep:g}] "
    try:
        with print_lock:
            log(f"Launching worker on GPU {gpu_id} for {model_name}, timestep {timestep:g}")
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path.cwd()),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            with print_lock:
                print(prefix + line, end="", flush=True)
        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(f"Worker for {model_name}, timestep {timestep:g} on GPU {gpu_id} failed with exit code {return_code}")
        return worker_dir
    finally:
        gpu_queue.put(gpu_id)


def run_parallel_timestep_sweep(
    args: argparse.Namespace,
    output_dir: Path,
    experiments: set[str],
    specs: list[tuple[str, Path]],
    gpu_ids: list[str],
) -> None:
    layers = parse_layers(args.layers, DIT_VARIANTS[args.model_size]["depth"])
    worker_root = output_dir / "_parallel_workers"
    if worker_root.exists() and not args.resume_workers:
        shutil.rmtree(worker_root)
    worker_root.mkdir(parents=True, exist_ok=True)

    tasks = [(spec, float(timestep)) for spec in specs for timestep in args.eval_timesteps]
    log(
        f"Parallel model/timestep sweep on GPUs {gpu_ids}; "
        f"models: {[name for name, _ in specs]}; timesteps: {args.eval_timesteps}"
    )
    log("Each worker loads one checkpoint on one GPU; parent merges aggregate CSV/figures and logs one W&B/HF run.")

    gpu_queue: "queue.Queue[str]" = queue.Queue()
    for gpu_id in gpu_ids:
        gpu_queue.put(gpu_id)
    print_lock = threading.Lock()
    results: dict[tuple[str, float], Path] = {}
    errors: list[BaseException] = []
    pending_tasks = []
    for spec, timestep in tasks:
        model_name, _ = spec
        worker_dir = worker_root / sanitize_name(model_name) / f"t{timestep_tag(timestep)}"
        if args.resume_workers and worker_has_completed_outputs(worker_dir, experiments):
            results[(model_name, timestep)] = worker_dir
            log(f"Reusing completed worker output {worker_dir}")
        else:
            pending_tasks.append((spec, timestep))

    def target(checkpoint_spec: tuple[str, Path], timestep: float) -> None:
        try:
            results[(checkpoint_spec[0], timestep)] = run_timestep_worker(args, checkpoint_spec, timestep, worker_root, gpu_queue, print_lock)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=target, args=(spec, timestep), daemon=False) for spec, timestep in pending_tasks]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise RuntimeError("; ".join(str(error) for error in errors))

    worker_dirs = [results[(spec[0], timestep)] for spec, timestep in tasks]
    merge_parallel_worker_outputs(output_dir, worker_dirs, experiments)
    if not args.keep_worker_outputs:
        shutil.rmtree(worker_root)
        log(f"Removed parallel worker directories {worker_root}")
    plot_main_figure(output_dir, args.main_timestep, args.cka_main_scenario, args.class_main_probe)
    plot_ablation_figure(output_dir, args.main_timestep)
    plot_heatmaps(output_dir, args.cka_main_scenario, args.class_main_probe)

    setup_wandb(args, specs, layers, experiments)
    try:
        log_final_outputs_to_wandb(output_dir, experiments)
        upload_results_to_hf(output_dir, args, specs, layers, experiments)
    finally:
        finish_wandb()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.plots_only:
        experiments = selected_experiments(args.experiments)
        specs = parse_checkpoint_specs(args.checkpoint)
        layers = parse_layers(args.layers, DIT_VARIANTS[args.model_size]["depth"])
        setup_wandb(args, specs, layers, experiments)
        try:
            plot_main_figure(output_dir, args.main_timestep, args.cka_main_scenario, args.class_main_probe)
            plot_ablation_figure(output_dir, args.main_timestep)
            plot_heatmaps(output_dir, args.cka_main_scenario, args.class_main_probe)
            log_final_outputs_to_wandb(output_dir, experiments)
            upload_results_to_hf(output_dir, args, specs, layers, experiments)
        finally:
            finish_wandb()
        return

    experiments = selected_experiments(args.experiments)
    specs = parse_checkpoint_specs(args.checkpoint)
    parallel_gpus = resolve_parallel_gpus(args.parallel_gpus)
    if not args.parallel_worker and parallel_gpus and (len(args.eval_timesteps) * len(specs)) > 1:
        run_parallel_timestep_sweep(args, output_dir, experiments, specs, parallel_gpus)
        return

    checkpoints_ = [load_checkpoint(name, path, args) for name, path in specs]
    layers = parse_layers(args.layers, checkpoints_[0].config["depth"])
    log(f"Using layers: {layers}")
    log(f"Main timestep: {args.main_timestep}; sweep timesteps: {args.eval_timesteps}")
    log(
        f"Using global batch size {args.global_batch_size}; "
        f"classification train batch {args.class_train_batch_size}; "
        f"classification probes {args.class_probes}; "
        f"classification main probe {args.class_main_probe}; "
        f"segmentation probe {SEG_PROBE_TYPE}; "
        f"segmentation train batch {args.seg_train_batch_size}; "
        f"segmentation layer batch {args.seg_cnn_layer_batch_size}."
    )
    if args.cfg_dropout_rate <= 0:
        log("cfg_dropout_rate <= 0: checkpoint has no null label embedding; using class 0 as fallback condition.")

    setup_wandb(args, specs, layers, experiments)

    try:
        vae, vae_params = load_flax_vae(Path(args.vae_path))

        cls_rows: list[dict] = []
        seg_rows: list[dict] = []
        cka_rows: list[dict] = []
        for eval_timestep in args.eval_timesteps:
            args.timestep = float(eval_timestep)
            log(f"=== Evaluating timestep {args.timestep:g} ===")
            if "classification" in experiments:
                cls_rows.extend(run_classification(checkpoints_, vae, vae_params, layers, args))
            if "segmentation" in experiments:
                seg_rows.extend(run_segmentation(checkpoints_, vae, vae_params, layers, args))
            if "cka" in experiments:
                cka_rows.extend(run_all_cka(checkpoints_, vae, vae_params, layers, args))

        args.timestep = args.main_timestep

        if "classification" in experiments:
            cls_fields = CSV_OUTPUTS["classification"][1]
            write_csv(output_dir / "classification_probe.csv", cls_rows, cls_fields)
            wandb_log_table("classification_probe", cls_rows, cls_fields)
            wandb_log_metric_rows("classification_probe", cls_rows, "val_accuracy")

        if "segmentation" in experiments:
            seg_fields = CSV_OUTPUTS["segmentation"][1]
            write_csv(output_dir / "segmentation_probe.csv", seg_rows, seg_fields)
            wandb_log_table("segmentation_probe", seg_rows, seg_fields)
            wandb_log_metric_rows("segmentation_probe", seg_rows, "mIoU")

        if "cka" in experiments:
            cka_fields = CSV_OUTPUTS["cka"][1]
            write_csv(output_dir / "cka_dinov2g.csv", cka_rows, cka_fields)
            wandb_log_table("cka_dinov2g", cka_rows, cka_fields)
            wandb_log_metric_rows("cka_dinov2g", cka_rows, "cka")

        plot_main_figure(output_dir, args.main_timestep, args.cka_main_scenario, args.class_main_probe)
        plot_ablation_figure(output_dir, args.main_timestep)
        plot_heatmaps(output_dir, args.cka_main_scenario, args.class_main_probe)
        wandb_log_image("representation_eval_main", output_dir / "representation_eval_main.png")
        wandb_log_image("probe_arch_ablation", output_dir / "probe_arch_ablation.png")
        wandb_log_image("classification_probe_heatmap", output_dir / "classification_probe_heatmap.png")
        wandb_log_image("segmentation_probe_heatmap", output_dir / "segmentation_probe_heatmap.png")
        wandb_log_image("cka_heatmap", output_dir / "cka_heatmap.png")
        wandb_log_result_artifact(output_dir)
        cleanup_feature_cache(output_dir, args)
        upload_results_to_hf(output_dir, args, specs, layers, experiments)
    finally:
        finish_wandb()


if __name__ == "__main__":
    main()
