#!/usr/bin/env python3
"""
Endpoint-conditioned DeepDTA-style baseline for the full ConfMux-DTA dataset.

The script reuses the data preparation and split functions from the published
ConfMux-DTA training program, then verifies the reconstructed indices against
the feature cache of the completed 100% ConfMux-DTA run. It implements the
categorical DeepDTA architecture described by Ozturk et al. (2018): separate
character embeddings and three Conv1D layers for SMILES and protein sequences,
global max pooling, and a multilayer regression head. A three-level affinity
type indicator is appended to the joint representation so that the comparator
is trained on the same pIC50/pKi/pKd prediction task as ConfMux-DTA.

This is a controlled modern reimplementation of the published architecture,
not the legacy TensorFlow 1.x source distribution.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import random
import resource
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


AFFINITY_ORDER = ("pIC50", "pKi", "pKd")
STOP_REQUESTED = False


class StopRequested(RuntimeError):
    """Raised at a safe batch boundary after a scheduler stop signal."""


def request_stop(signum, frame):
    del frame
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"[signal] received {signum}; stopping after the current batch", flush=True)


def atomic_json_dump(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False, allow_nan=True)
    os.replace(tmp, path)


def index_sha1(indices: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
    return hashlib.sha1(arr.tobytes()).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_confmux_module(path: str):
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"ConfMux-DTA source not found: {source}")
    spec = importlib.util.spec_from_file_location("confmux_reference_source", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import ConfMux-DTA source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_reference_indices(
    reference_cache_dir: str,
    idx_train: np.ndarray,
    idx_valid: np.ndarray,
) -> Dict[str, Any]:
    cache = Path(reference_cache_dir).resolve()
    train_path = cache / "idx_train.npy"
    valid_path = cache / "idx_valid.npy"
    if not train_path.is_file() or not valid_path.is_file():
        raise FileNotFoundError(
            f"Reference cache must contain idx_train.npy and idx_valid.npy: {cache}"
        )
    ref_train = np.asarray(np.load(train_path), dtype=np.int64)
    ref_valid = np.asarray(np.load(valid_path), dtype=np.int64)
    if not np.array_equal(ref_train, np.asarray(idx_train, dtype=np.int64)):
        raise RuntimeError("DeepDTA training indices do not match the completed ConfMux-DTA run")
    if not np.array_equal(ref_valid, np.asarray(idx_valid, dtype=np.int64)):
        raise RuntimeError("DeepDTA validation indices do not match the completed ConfMux-DTA run")
    return {
        "reference_cache_dir": str(cache),
        "train_index_sha1": index_sha1(ref_train),
        "valid_index_sha1": index_sha1(ref_valid),
        "n_train": int(len(ref_train)),
        "n_valid": int(len(ref_valid)),
    }


def prepare_reference_data(args):
    module = import_confmux_module(args.confmux_source)
    module.CONFIG["DATA_FILE"] = str(Path(args.data_file).resolve())
    module.CONFIG["SEED"] = int(args.seed)
    module.CONFIG["TEST_SIZE"] = float(args.test_size)
    module.CONFIG["SPLIT_MODE"] = str(args.split_mode)
    module.CONFIG["RUN_DIR"] = str(Path(args.work_dir).resolve() / "reference_preparation")
    module.CONFIG["BACKUP"]["enable"] = False
    module.CONFIG["DATA_STATS"]["enable"] = False
    module.CONFIG["PLOT_SCATTER"] = False
    module.CONFIG["AFFINITY_UNIFIED"]["enable"] = True
    module.CONFIG["AFFINITY_UNIFIED"]["type_feature"] = "onehot"
    module.apply_unified_names_to_config(module.CONFIG)
    module.init_run_dirs()

    df, mask = module.load_and_prepare_data()
    idx_train, idx_valid, df_all, y_all = module.make_fixed_train_valid_split(df, mask)
    split_info = verify_reference_indices(
        args.reference_cache_dir, idx_train, idx_valid
    )
    return module, df_all, np.asarray(y_all, dtype=np.float32), idx_train, idx_valid, split_info


def character_vocabulary(strings: Sequence[str], used_codes: np.ndarray) -> Dict[str, int]:
    chars = set()
    for code in np.unique(np.asarray(used_codes, dtype=np.int64)):
        chars.update(str(strings[int(code)]))
    return {char: i + 2 for i, char in enumerate(sorted(chars))}


def encode_unique_strings(
    strings: Sequence[str],
    vocab: Dict[str, int],
    max_length: int,
    output_path: Path,
) -> Dict[str, int]:
    arr = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.uint16,
        shape=(len(strings), int(max_length)),
    )
    unk = 1
    n_truncated = 0
    n_unknown = 0
    for row, value in enumerate(strings):
        text = str(value)
        if len(text) > max_length:
            n_truncated += 1
        for col, char in enumerate(text[:max_length]):
            token = vocab.get(char, unk)
            arr[row, col] = token
            if token == unk:
                n_unknown += 1
        if (row + 1) % 100000 == 0 or row + 1 == len(strings):
            print(f"[encode] {output_path.name}: {row + 1}/{len(strings)}", flush=True)
    arr.flush()
    del arr
    return {
        "n_unique": int(len(strings)),
        "n_truncated": int(n_truncated),
        "n_unknown_characters": int(n_unknown),
        "max_length": int(max_length),
    }


def cache_signature(args, split_info: Dict[str, Any]) -> Dict[str, Any]:
    data_path = Path(args.data_file).resolve()
    stat = data_path.stat()
    return {
        "protocol_version": "deepdta-full-controlled-v1",
        "data_file": str(data_path),
        "data_size": int(stat.st_size),
        "data_mtime_ns": int(stat.st_mtime_ns),
        "split_mode": str(args.split_mode),
        "test_size": float(args.test_size),
        "seed": int(args.seed),
        "train_index_sha1": split_info["train_index_sha1"],
        "valid_index_sha1": split_info["valid_index_sha1"],
        "max_smiles_len": int(args.max_smiles_len),
        "max_protein_len": int(args.max_protein_len),
        "affinity_order": list(AFFINITY_ORDER),
    }


def cache_files(cache_dir: Path) -> Dict[str, Path]:
    return {
        "manifest": cache_dir / "manifest.json",
        "smiles_encoded": cache_dir / "smiles_unique_encoded.npy",
        "protein_encoded": cache_dir / "protein_unique_encoded.npy",
        "row_smiles_code": cache_dir / "row_smiles_code.npy",
        "row_protein_code": cache_dir / "row_protein_code.npy",
        "endpoint_code": cache_dir / "endpoint_code.npy",
        "y": cache_dir / "y.npy",
        "weight": cache_dir / "weight.npy",
        "idx_train": cache_dir / "idx_train.npy",
        "idx_valid": cache_dir / "idx_valid.npy",
        "smiles_vocab": cache_dir / "smiles_vocab.json",
        "protein_vocab": cache_dir / "protein_vocab.json",
    }


def build_or_load_cache(
    args,
    module,
    df_all: pd.DataFrame,
    y_all: np.ndarray,
    idx_train: np.ndarray,
    idx_valid: np.ndarray,
    split_info: Dict[str, Any],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = cache_files(cache_dir)
    signature = cache_signature(args, split_info)

    if not args.rebuild_cache and all(path.is_file() and path.stat().st_size > 0 for path in paths.values()):
        with paths["manifest"].open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("signature") != signature:
            raise RuntimeError(
                "Existing DeepDTA cache has a different data/split signature; "
                "use a new --cache_dir or --rebuild_cache"
            )
        print(f"[cache] loaded: {cache_dir}", flush=True)
    else:
        print(f"[cache] building: {cache_dir}", flush=True)
        smiles_col = "smiles_norm" if "smiles_norm" in df_all.columns else module.CONFIG["COL_SMILES"]
        protein_col = "protein_clean" if "protein_clean" in df_all.columns else module.CONFIG["COL_PROTEIN"]
        smiles_values = df_all[smiles_col].astype(str)
        protein_values = df_all[protein_col].astype(str)
        row_smiles_code, unique_smiles = pd.factorize(smiles_values, sort=False)
        row_protein_code, unique_proteins = pd.factorize(protein_values, sort=False)
        row_smiles_code = np.asarray(row_smiles_code, dtype=np.int32)
        row_protein_code = np.asarray(row_protein_code, dtype=np.int32)

        endpoint_map = {name: i for i, name in enumerate(AFFINITY_ORDER)}
        endpoint_values = df_all["affinity_type"].astype(str).map(endpoint_map)
        if endpoint_values.isna().any():
            unknown = sorted(df_all.loc[endpoint_values.isna(), "affinity_type"].astype(str).unique())
            raise ValueError(f"Unexpected affinity types: {unknown}")
        endpoint_code = endpoint_values.to_numpy(dtype=np.uint8)

        smiles_vocab = character_vocabulary(unique_smiles, row_smiles_code[idx_train])
        protein_vocab = character_vocabulary(unique_proteins, row_protein_code[idx_train])
        atomic_json_dump(
            {"PAD": 0, "UNK": 1, "characters": smiles_vocab}, paths["smiles_vocab"]
        )
        atomic_json_dump(
            {"PAD": 0, "UNK": 1, "characters": protein_vocab}, paths["protein_vocab"]
        )
        smiles_stats = encode_unique_strings(
            unique_smiles, smiles_vocab, args.max_smiles_len, paths["smiles_encoded"]
        )
        protein_stats = encode_unique_strings(
            unique_proteins, protein_vocab, args.max_protein_len, paths["protein_encoded"]
        )

        weight_col = module.CONFIG["AFFINITY_UNIFIED"]["keep_weight_col_name"]
        if weight_col in df_all.columns:
            weights = (
                pd.to_numeric(df_all[weight_col], errors="coerce")
                .fillna(1.0)
                .to_numpy(dtype=np.float32, copy=True)
            )
            weights[~np.isfinite(weights) | (weights <= 0)] = 1.0
        else:
            weights = np.ones(len(df_all), dtype=np.float32)
        weights[idx_train] /= float(np.mean(weights[idx_train]) + 1e-12)

        np.save(paths["row_smiles_code"], row_smiles_code)
        np.save(paths["row_protein_code"], row_protein_code)
        np.save(paths["endpoint_code"], endpoint_code)
        np.save(paths["y"], np.asarray(y_all, dtype=np.float32))
        np.save(paths["weight"], weights)
        np.save(paths["idx_train"], np.asarray(idx_train, dtype=np.int64))
        np.save(paths["idx_valid"], np.asarray(idx_valid, dtype=np.int64))

        manifest = {
            "signature": signature,
            "smiles": smiles_stats,
            "protein": protein_stats,
            "smiles_vocab_size_including_pad_unk": int(len(smiles_vocab) + 2),
            "protein_vocab_size_including_pad_unk": int(len(protein_vocab) + 2),
            "n_rows": int(len(df_all)),
            "n_train": int(len(idx_train)),
            "n_valid": int(len(idx_valid)),
        }
        atomic_json_dump(manifest, paths["manifest"])

    with paths["manifest"].open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    arrays = {
        "smiles_encoded": np.load(paths["smiles_encoded"], mmap_mode="r"),
        "protein_encoded": np.load(paths["protein_encoded"], mmap_mode="r"),
        "row_smiles_code": np.load(paths["row_smiles_code"], mmap_mode="r"),
        "row_protein_code": np.load(paths["row_protein_code"], mmap_mode="r"),
        "endpoint_code": np.load(paths["endpoint_code"], mmap_mode="r"),
        "y": np.load(paths["y"], mmap_mode="r"),
        "weight": np.load(paths["weight"], mmap_mode="r"),
        "idx_train": np.load(paths["idx_train"], mmap_mode="r"),
        "idx_valid": np.load(paths["idx_valid"], mmap_mode="r"),
    }
    if not np.array_equal(np.asarray(arrays["idx_train"]), np.asarray(idx_train)):
        raise RuntimeError("Cached DeepDTA training indices changed")
    if not np.array_equal(np.asarray(arrays["idx_valid"]), np.asarray(idx_valid)):
        raise RuntimeError("Cached DeepDTA validation indices changed")
    return arrays, manifest


class RowIndexDataset(Dataset):
    def __init__(self, indices: np.ndarray):
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> int:
        return int(self.indices[item])


class BatchCollator:
    def __init__(self, arrays: Dict[str, np.ndarray], y_mean: float, y_sd: float):
        self.arrays = arrays
        self.y_mean = float(y_mean)
        self.y_sd = float(y_sd)

    def __call__(self, row_ids: Sequence[int]):
        rows = np.asarray(row_ids, dtype=np.int64)
        smi = np.asarray(
            self.arrays["smiles_encoded"][self.arrays["row_smiles_code"][rows]],
            dtype=np.int64,
        )
        pro = np.asarray(
            self.arrays["protein_encoded"][self.arrays["row_protein_code"][rows]],
            dtype=np.int64,
        )
        endpoint = np.array(self.arrays["endpoint_code"][rows], dtype=np.int64, copy=True)
        y_raw = np.array(self.arrays["y"][rows], dtype=np.float32, copy=True)
        y_scaled = (y_raw - self.y_mean) / self.y_sd
        weight = np.array(self.arrays["weight"][rows], dtype=np.float32, copy=True)
        return (
            torch.from_numpy(smi),
            torch.from_numpy(pro),
            torch.from_numpy(endpoint),
            torch.from_numpy(y_scaled),
            torch.from_numpy(weight),
            torch.from_numpy(rows),
        )


class ConvTower(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, filters: int, kernel_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embedding_dim, filters, kernel_size)
        self.conv2 = nn.Conv1d(filters, filters * 2, kernel_size)
        self.conv3 = nn.Conv1d(filters * 2, filters * 3, kernel_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens).transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return torch.amax(x, dim=2)


class EndpointConditionedDeepDTA(nn.Module):
    def __init__(
        self,
        smiles_vocab_size: int,
        protein_vocab_size: int,
        embedding_dim: int = 128,
        filters: int = 32,
        smiles_kernel: int = 4,
        protein_kernel: int = 8,
        endpoint_dim: int = 3,
    ):
        super().__init__()
        self.smiles_tower = ConvTower(
            smiles_vocab_size, embedding_dim, filters, smiles_kernel
        )
        self.protein_tower = ConvTower(
            protein_vocab_size, embedding_dim, filters, protein_kernel
        )
        joint_dim = filters * 6 + endpoint_dim
        self.fc1 = nn.Linear(joint_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self, smiles: torch.Tensor, protein: torch.Tensor, endpoint: torch.Tensor
    ) -> torch.Tensor:
        smi = self.smiles_tower(smiles)
        pro = self.protein_tower(protein)
        ep = F.one_hot(endpoint, num_classes=3).to(dtype=smi.dtype)
        x = torch.cat([smi, pro, ep], dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        return self.out(x).squeeze(1)


def make_loader(
    arrays: Dict[str, np.ndarray],
    indices: np.ndarray,
    y_mean: float,
    y_sd: float,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        RowIndexDataset(indices),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(workers),
        pin_memory=bool(pin_memory),
        # A fresh, deterministically seeded training loader is built each epoch.
        # Persistent workers would add processes without any cross-epoch reuse.
        persistent_workers=False,
        collate_fn=BatchCollator(arrays, y_mean, y_sd),
        generator=generator,
        drop_last=False,
    )


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
    return contextlib.nullcontext()


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def move_batch(batch, device: torch.device):
    smi, pro, endpoint, y, weight, rows = batch
    non_blocking = device.type == "cuda"
    return (
        smi.to(device, non_blocking=non_blocking),
        pro.to(device, non_blocking=non_blocking),
        endpoint.to(device, non_blocking=non_blocking),
        y.to(device, non_blocking=non_blocking),
        weight.to(device, non_blocking=non_blocking),
        rows,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    scaler,
    epoch_completed: int,
    best_epoch: int,
    best_valid_loss: float,
    bad_epochs: int,
    y_mean: float,
    y_sd: float,
    args,
    manifest: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch_completed": int(epoch_completed),
        "best_epoch": int(best_epoch),
        "best_valid_loss": float(best_valid_loss),
        "bad_epochs": int(bad_epochs),
        "y_mean": float(y_mean),
        "y_sd": float(y_sd),
        "architecture": {
            "smiles_vocab_size": int(manifest["smiles_vocab_size_including_pad_unk"]),
            "protein_vocab_size": int(manifest["protein_vocab_size_including_pad_unk"]),
            "embedding_dim": int(args.embedding_dim),
            "filters": int(args.filters),
            "smiles_kernel": int(args.smiles_kernel),
            "protein_kernel": int(args.protein_kernel),
            "endpoint_dim": 3,
        },
        "split_signature": manifest["signature"],
        "args": vars(args),
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_mean: float,
    y_sd: float,
    amp_enabled: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    rows_out = []
    y_out = []
    pred_out = []
    for batch in loader:
        if STOP_REQUESTED:
            raise StopRequested("scheduler stop requested during evaluation")
        smi, pro, endpoint, y_scaled, weight, rows = move_batch(batch, device)
        del weight
        with autocast_context(device, amp_enabled):
            pred_scaled = model(smi, pro, endpoint)
        pred_raw = pred_scaled.float().cpu().numpy() * y_sd + y_mean
        y_raw = y_scaled.float().cpu().numpy() * y_sd + y_mean
        rows_out.append(np.asarray(rows, dtype=np.int64))
        y_out.append(np.asarray(y_raw, dtype=np.float32))
        pred_out.append(np.asarray(pred_raw, dtype=np.float32))
    return (
        np.concatenate(rows_out),
        np.concatenate(y_out),
        np.concatenate(pred_out),
    )


def affine_fit(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    p_centered = p - float(np.mean(p))
    denom = float(np.dot(p_centered, p_centered))
    if denom <= 1e-12:
        return 1.0, 0.0
    slope = float(np.dot(p_centered, y - float(np.mean(y))) / denom)
    intercept = float(np.mean(y) - slope * np.mean(p))
    return slope, intercept


def metric_pack(module, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    residual = y - p
    mse = float(np.mean(residual ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum(residual ** 2) / sst) if sst > 0 else float("nan")
    pearson = float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 else float("nan")
    ci = float(module.concordance_index_optimized(y, p))
    r2m_values = module.r2m_metrics(y, p)
    return {
        "N": int(len(y)),
        "R2": r2,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "PearsonR": pearson,
        "CI": ci,
        "R2m": float(r2m_values["R2m"]),
        "R2m_bar": float(r2m_values["R2m_bar"]),
    }


def metrics_by_endpoint(
    module,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    endpoint_code: np.ndarray,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"overall": metric_pack(module, y_true, y_pred), "by_type": {}}
    for code, endpoint in enumerate(AFFINITY_ORDER):
        mask = np.asarray(endpoint_code) == code
        result["by_type"][endpoint] = metric_pack(module, y_true[mask], y_pred[mask])
    return result


def flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    row = {}
    for key, value in metrics["overall"].items():
        row[f"test_{key}"] = value
        row[f"val_{key}"] = value
    for endpoint, endpoint_metrics in metrics["by_type"].items():
        for key, value in endpoint_metrics.items():
            row[f"{endpoint}_test_{key}"] = value
            row[f"{endpoint}_val_{key}"] = value
    return row


def run_training(
    args,
    module,
    arrays: Dict[str, np.ndarray],
    manifest: Dict[str, Any],
) -> int:
    out_dir = Path(args.work_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "epoch_history.csv"
    last_checkpoint = out_dir / "last_checkpoint.pt"
    best_checkpoint = out_dir / "best_checkpoint.pt"

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
        torch.cuda.reset_peak_memory_stats(device)

    idx_train = np.asarray(arrays["idx_train"], dtype=np.int64)
    idx_valid = np.asarray(arrays["idx_valid"], dtype=np.int64)
    y_train = np.asarray(arrays["y"][idx_train], dtype=np.float64)
    y_mean = float(np.mean(y_train))
    y_sd = float(np.std(y_train))
    if not np.isfinite(y_sd) or y_sd <= 1e-12:
        raise RuntimeError("Training response has zero or non-finite standard deviation")

    architecture = {
        "smiles_vocab_size": int(manifest["smiles_vocab_size_including_pad_unk"]),
        "protein_vocab_size": int(manifest["protein_vocab_size_including_pad_unk"]),
        "embedding_dim": int(args.embedding_dim),
        "filters": int(args.filters),
        "smiles_kernel": int(args.smiles_kernel),
        "protein_kernel": int(args.protein_kernel),
        "endpoint_dim": 3,
    }
    model = EndpointConditionedDeepDTA(**architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    scaler = make_grad_scaler(amp_enabled)

    start_epoch = 1
    best_epoch = 0
    best_valid_loss = float("inf")
    bad_epochs = 0
    if args.resume and last_checkpoint.is_file():
        checkpoint = load_checkpoint(last_checkpoint, device)
        if checkpoint.get("split_signature") != manifest["signature"]:
            raise RuntimeError("Checkpoint data/split signature does not match this run")
        if checkpoint.get("architecture") != architecture:
            raise RuntimeError("Checkpoint architecture does not match this run")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint.get("scaler_state", {}))
        start_epoch = int(checkpoint["epoch_completed"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_valid_loss = float(checkpoint["best_valid_loss"])
        bad_epochs = int(checkpoint["bad_epochs"])
        if not math.isclose(y_mean, float(checkpoint["y_mean"]), rel_tol=0, abs_tol=1e-8):
            raise RuntimeError("Checkpoint target mean changed")
        if not math.isclose(y_sd, float(checkpoint["y_sd"]), rel_tol=0, abs_tol=1e-8):
            raise RuntimeError("Checkpoint target standard deviation changed")
        print(f"[resume] continuing at epoch {start_epoch}", flush=True)

    run_manifest = {
        "status": "training",
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "architecture": architecture,
        "data_signature": manifest["signature"],
        "n_train": int(len(idx_train)),
        "n_valid": int(len(idx_valid)),
        "y_mean": y_mean,
        "y_sd": y_sd,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "official_architecture_reference": "Ozturk et al., Bioinformatics 2018; hkmztrk/DeepDTA",
        "implementation_note": (
            "PyTorch reimplementation with endpoint one-hot conditioning and "
            "the same full-data split as ConfMux-DTA"
        ),
    }
    atomic_json_dump(run_manifest, out_dir / "run_manifest.json")

    valid_loader = make_loader(
        arrays,
        idx_valid,
        y_mean,
        y_sd,
        args.eval_batch_size,
        args.workers,
        False,
        args.seed,
        device.type == "cuda",
    )
    history = []
    if history_path.is_file() and args.resume:
        try:
            history = pd.read_csv(history_path).to_dict("records")
        except Exception:
            history = []

    wall_start = time.time()
    last_completed_epoch = start_epoch - 1
    if bad_epochs >= int(args.patience):
        print("[resume] early-stopping condition was already reached; evaluating best checkpoint", flush=True)

    for epoch in range(start_epoch, int(args.max_epochs) + 1):
        if bad_epochs >= int(args.patience):
            break
        if STOP_REQUESTED:
            break
        train_loader = make_loader(
            arrays,
            idx_train,
            y_mean,
            y_sd,
            args.batch_size,
            args.workers,
            True,
            args.seed + epoch,
            device.type == "cuda",
        )
        model.train()
        train_loss_sum = 0.0
        train_weight_sum = 0.0
        epoch_start = time.time()
        completed_epoch = True
        for step, batch in enumerate(train_loader, start=1):
            if STOP_REQUESTED:
                completed_epoch = False
                break
            smi, pro, endpoint, y_scaled, weight, rows = move_batch(batch, device)
            del rows
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_enabled):
                pred = model(smi, pro, endpoint)
                loss = torch.sum(weight * (pred - y_scaled) ** 2) / torch.sum(weight)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            batch_weight = float(torch.sum(weight).detach().cpu())
            train_loss_sum += float(loss.detach().cpu()) * batch_weight
            train_weight_sum += batch_weight
            if step % int(args.log_every) == 0:
                print(
                    f"[train] epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={train_loss_sum / max(train_weight_sum, 1e-12):.6f}",
                    flush=True,
                )
        del train_loader

        if not completed_epoch:
            print("[interrupt] partial epoch discarded; resume from last completed checkpoint", flush=True)
            run_manifest["status"] = "interrupted"
            run_manifest["last_completed_epoch"] = int(epoch - 1)
            atomic_json_dump(run_manifest, out_dir / "run_manifest.json")
            return 75

        rows_valid, y_valid, pred_valid = predict_loader(
            model, valid_loader, device, y_mean, y_sd, amp_enabled
        )
        del rows_valid
        valid_loss = float(np.mean(((pred_valid - y_valid) / y_sd) ** 2))
        improved = valid_loss < best_valid_loss - float(args.min_delta)
        if improved:
            best_valid_loss = valid_loss
            best_epoch = epoch
            bad_epochs = 0
            best_tmp = best_checkpoint.with_suffix(best_checkpoint.suffix + ".tmp")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "architecture": architecture,
                    "epoch": int(epoch),
                    "valid_scaled_mse": valid_loss,
                    "y_mean": y_mean,
                    "y_sd": y_sd,
                    "split_signature": manifest["signature"],
                },
                best_tmp,
            )
            os.replace(best_tmp, best_checkpoint)
        else:
            bad_epochs += 1

        epoch_row = {
            "epoch": int(epoch),
            "train_scaled_weighted_mse": float(train_loss_sum / max(train_weight_sum, 1e-12)),
            "valid_scaled_mse": valid_loss,
            "valid_raw_rmse": float(np.sqrt(np.mean((pred_valid - y_valid) ** 2))),
            "best_epoch": int(best_epoch),
            "best_valid_scaled_mse": float(best_valid_loss),
            "bad_epochs": int(bad_epochs),
            "epoch_seconds": float(time.time() - epoch_start),
        }
        history.append(epoch_row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        save_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            scaler,
            epoch,
            best_epoch,
            best_valid_loss,
            bad_epochs,
            y_mean,
            y_sd,
            args,
            manifest,
        )
        last_completed_epoch = epoch
        print(
            f"[epoch] {epoch} train={epoch_row['train_scaled_weighted_mse']:.6f} "
            f"valid={valid_loss:.6f} raw_RMSE={epoch_row['valid_raw_rmse']:.6f} "
            f"best={best_epoch} bad={bad_epochs}",
            flush=True,
        )
        if bad_epochs >= int(args.patience):
            print(f"[early_stop] patience={args.patience}", flush=True)
            break

    if STOP_REQUESTED:
        run_manifest["status"] = "interrupted"
        run_manifest["last_completed_epoch"] = int(max(0, last_completed_epoch))
        atomic_json_dump(run_manifest, out_dir / "run_manifest.json")
        return 75
    if not best_checkpoint.is_file():
        raise RuntimeError("Training ended without a valid best checkpoint")

    best = load_checkpoint(best_checkpoint, device)
    if best.get("split_signature") != manifest["signature"]:
        raise RuntimeError("Best checkpoint data/split signature does not match this run")
    if best.get("architecture") != architecture:
        raise RuntimeError("Best checkpoint architecture does not match this run")
    model.load_state_dict(best["model_state"])
    model.to(device)
    model.eval()

    train_eval_loader = make_loader(
        arrays,
        idx_train,
        y_mean,
        y_sd,
        args.eval_batch_size,
        args.workers,
        False,
        args.seed,
        device.type == "cuda",
    )
    train_rows, train_y, train_pred_raw = predict_loader(
        model, train_eval_loader, device, y_mean, y_sd, amp_enabled
    )
    valid_rows, valid_y, valid_pred_raw = predict_loader(
        model, valid_loader, device, y_mean, y_sd, amp_enabled
    )
    slope, intercept = affine_fit(train_y, train_pred_raw)
    train_pred = slope * train_pred_raw + intercept
    valid_pred = slope * valid_pred_raw + intercept

    valid_endpoint = np.asarray(arrays["endpoint_code"][valid_rows], dtype=np.uint8)
    train_endpoint = np.asarray(arrays["endpoint_code"][train_rows], dtype=np.uint8)
    train_metrics = metrics_by_endpoint(module, train_y, train_pred, train_endpoint)
    valid_metrics = metrics_by_endpoint(module, valid_y, valid_pred, valid_endpoint)

    predictions = pd.DataFrame({
        "row_index": valid_rows,
        "affinity_type": [AFFINITY_ORDER[int(x)] for x in valid_endpoint],
        "y_true": valid_y,
        "y_pred_raw": valid_pred_raw,
        "y_pred": valid_pred,
        "abs_err": np.abs(valid_y - valid_pred),
    })
    predictions.to_csv(out_dir / "valid_predictions_for_paper.csv.gz", index=False)

    paper_row = {
        "run_tag": "deepdta_full_controlled",
        "best_epoch": int(best["epoch"]),
        "affine_slope": float(slope),
        "affine_intercept": float(intercept),
    }
    for key, value in train_metrics["overall"].items():
        paper_row[f"train_{key}"] = value
    paper_row.update(flatten_metrics(valid_metrics))
    pd.DataFrame([paper_row]).to_csv(out_dir / "paper_metrics_flat.csv", index=False)

    peak_rss_gb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2))
    peak_gpu_gb = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))
        if device.type == "cuda"
        else 0.0
    )
    result = {
        "status": "complete",
        "model": "endpoint-conditioned DeepDTA-style categorical CNN",
        "best_epoch": int(best["epoch"]),
        "best_valid_scaled_mse": float(best["valid_scaled_mse"]),
        "affine_calibration": {"slope": slope, "intercept": intercept, "fit_set": "training"},
        "train": train_metrics,
        "validation": valid_metrics,
        "test": valid_metrics,
        "split": manifest["signature"],
        "resource": {
            "wall_seconds_this_invocation": float(time.time() - wall_start),
            "peak_rss_gb": peak_rss_gb,
            "peak_gpu_allocated_gb": peak_gpu_gb,
            "host": socket.gethostname(),
            "device": str(device),
        },
    }
    atomic_json_dump(result, out_dir / "metrics.json")
    run_manifest.update({
        "status": "complete",
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_epoch": int(best["epoch"]),
        "metrics_json": "metrics.json",
        "paper_metrics_csv": "paper_metrics_flat.csv",
    })
    atomic_json_dump(run_manifest, out_dir / "run_manifest.json")
    print(json.dumps(result["validation"]["overall"], indent=2), flush=True)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Endpoint-conditioned DeepDTA-style comparator using the exact "
            "ConfMux-DTA full-data training/validation indices"
        ),
    )
    parser.add_argument("--data_file", required=True)
    parser.add_argument(
        "--confmux_source",
        default="confmux_dta_train_predict_scale_controlled.py",
        help="ConfMux-DTA source used only for identical filtering and split reconstruction",
    )
    parser.add_argument("--reference_cache_dir", required=True)
    parser.add_argument(
        "--work_dir", default="confmux_dta_runs/deepdta_full_controlled"
    )
    parser.add_argument(
        "--cache_dir", default="confmux_dta_feature_cache/deepdta_full_controlled"
    )
    parser.add_argument("--seed", type=int, default=5313)
    parser.add_argument(
        "--split_mode", choices=["random", "compound", "protein", "pair"], default="random"
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max_smiles_len", type=int, default=100)
    parser.add_argument("--max_protein_len", type=int, default=1000)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--filters", type=int, default=32)
    parser.add_argument("--smiles_kernel", type=int, default=4)
    parser.add_argument("--protein_kernel", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--eval_batch_size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_delta", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True,
        help="Use mixed precision on CUDA",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument(
        "--prepare_only", action="store_true",
        help="Build and validate the character-encoding cache without training",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if not (0.0 < float(args.test_size) < 1.0):
        raise ValueError("--test_size must be between 0 and 1")
    for value, label in (
        (args.max_smiles_len, "max_smiles_len"),
        (args.max_protein_len, "max_protein_len"),
        (args.batch_size, "batch_size"),
        (args.eval_batch_size, "eval_batch_size"),
        (args.max_epochs, "max_epochs"),
        (args.patience, "patience"),
    ):
        if int(value) <= 0:
            raise ValueError(f"--{label} must be positive")

    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)
    set_seed(int(args.seed))
    Path(args.work_dir).resolve().mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).resolve().mkdir(parents=True, exist_ok=True)

    module, df_all, y_all, idx_train, idx_valid, split_info = prepare_reference_data(args)
    arrays, manifest = build_or_load_cache(
        args, module, df_all, y_all, idx_train, idx_valid, split_info
    )
    del df_all, y_all, idx_train, idx_valid
    atomic_json_dump(
        {
            "status": "cache_ready",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "implementation": "endpoint-conditioned DeepDTA-style categorical CNN",
            "split": manifest["signature"],
            "arguments": vars(args),
        },
        Path(args.work_dir).resolve() / "input_and_split_manifest.json",
    )
    if args.prepare_only:
        print("[done] cache prepared and reference split verified", flush=True)
        return 0
    try:
        return run_training(args, module, arrays, manifest)
    except StopRequested as exc:
        print(f"[interrupt] {exc}; resume from the latest completed epoch", flush=True)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
