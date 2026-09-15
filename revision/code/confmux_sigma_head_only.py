#!/usr/bin/env python3
"""Train only the ConfMux-DTA residual-scale head and export validation sigma.

The expensive point predictor is never retrained. This script reuses:

* the frozen train/validation point predictions,
* the leakage-controlled sparse feature cache and its exact split indices, and
* the SIGMA_UNCERT configuration/function from the original training program.

Alignment is verified against cached labels and split hashes before fitting.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
from scipy import sparse


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(tmp, path)


def sha1_indices(values: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(values, dtype=np.int64).tobytes()).hexdigest()


def import_core(path: Path):
    spec = importlib.util.spec_from_file_location("confmux_core_for_sigma", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import core training program: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_prediction_columns(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    required = {"y_true", "y_pred", "affinity_type"}
    missing = sorted(required.difference(header))
    if missing:
        raise ValueError(f"{path} is missing prediction columns: {missing}")
    frame = pd.read_csv(
        path,
        usecols=["y_true", "y_pred", "affinity_type"],
        dtype={"y_true": "float64", "y_pred": "float64", "affinity_type": "string"},
    )
    frame["affinity_type"] = frame["affinity_type"].fillna("Unknown").astype(str)
    return frame


def verify_alignment(name: str, cached_y: np.ndarray, predictions: pd.DataFrame) -> None:
    observed = predictions["y_true"].to_numpy(float)
    expected = np.asarray(cached_y, float)
    if len(observed) != len(expected):
        raise RuntimeError(
            f"{name} row mismatch: predictions={len(observed)} cached_labels={len(expected)}"
        )
    finite = np.isfinite(observed) & np.isfinite(expected)
    if int(finite.sum()) != len(observed):
        raise RuntimeError(f"{name} contains non-finite cached or exported y_true values")
    max_delta = float(np.max(np.abs(observed - expected))) if len(observed) else 0.0
    if max_delta > 1e-5:
        raise RuntimeError(f"{name} label alignment failed: max_abs_delta={max_delta:.8g}")
    print(f"ALIGNMENT_OK split={name} n={len(observed)} max_abs_delta={max_delta:.3g}", flush=True)


def endpoint_counts(values: Iterable[str]) -> Dict[str, int]:
    counts = pd.Series(values, dtype="string", copy=False).value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-script", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--valid-predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    required_paths = [
        args.core_script,
        args.config,
        args.train_predictions,
        args.valid_predictions,
        args.feature_cache / "Xs.npz",
        args.feature_cache / "y.npy",
        args.feature_cache / "w.npy",
        args.feature_cache / "idx_train.npy",
        args.feature_cache / "idx_valid.npy",
        args.feature_cache / "meta.json",
    ]
    missing = [str(path) for path in required_paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Missing or empty required files:\n" + "\n".join(missing))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "sigma_model.txt"
    output_path = args.out_dir / "valid_predictions_with_sigma.csv.gz"
    manifest_path = args.out_dir / "sigma_head_manifest.json"

    config = load_json(args.config)
    cache_meta = load_json(args.feature_cache / "meta.json")
    if cache_meta.get("fit_scope") != "training_subset_only":
        raise RuntimeError("Refusing a feature cache that was not fitted on training data only")
    if not bool(cache_meta.get("has_affinity_type_feature")):
        raise RuntimeError("Selected cache does not contain the one-hot affinity-type feature")

    idx_train = np.load(args.feature_cache / "idx_train.npy")
    idx_valid = np.load(args.feature_cache / "idx_valid.npy")
    idx_train = np.asarray(idx_train, dtype=np.int64)
    idx_valid = np.asarray(idx_valid, dtype=np.int64)
    if np.intersect1d(idx_train, idx_valid).size:
        raise RuntimeError("Feature-cache train and validation indices overlap")
    if sha1_indices(idx_train) != str(cache_meta.get("train_index_sha1")):
        raise RuntimeError("Training-index hash does not match feature-cache provenance")
    if sha1_indices(idx_valid) != str(cache_meta.get("valid_index_sha1")):
        raise RuntimeError("Validation-index hash does not match feature-cache provenance")

    print("READ_PREDICTIONS_START", flush=True)
    train_pred = read_prediction_columns(args.train_predictions)
    valid_pred = read_prediction_columns(args.valid_predictions)
    print(
        f"READ_PREDICTIONS_DONE n_train={len(train_pred)} n_valid={len(valid_pred)}",
        flush=True,
    )

    y_all = np.load(args.feature_cache / "y.npy")
    w_all = np.load(args.feature_cache / "w.npy")
    if len(y_all) != len(w_all):
        raise RuntimeError("Cached y.npy and w.npy have different lengths")
    verify_alignment("train", y_all[idx_train], train_pred)
    verify_alignment("valid", y_all[idx_valid], valid_pred)

    print(f"LOAD_SPARSE_CACHE_START path={args.feature_cache / 'Xs.npz'}", flush=True)
    x_all = sparse.load_npz(args.feature_cache / "Xs.npz").tocsr()
    if x_all.shape[0] != len(y_all):
        raise RuntimeError(f"Feature/label row mismatch: X={x_all.shape[0]} y={len(y_all)}")
    expected_shape = cache_meta.get("shape")
    if expected_shape and list(x_all.shape) != [int(expected_shape[0]), int(expected_shape[1])]:
        raise RuntimeError(f"Feature shape differs from cache metadata: {x_all.shape} vs {expected_shape}")
    x_train = x_all[idx_train]
    x_valid = x_all[idx_valid]
    del x_all
    gc.collect()
    print(f"LOAD_SPARSE_CACHE_DONE train={x_train.shape} valid={x_valid.shape}", flush=True)

    core = import_core(args.core_script)
    core.CONFIG = config
    threads = int(args.threads or os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    core.CONFIG["NUM_THREADS"] = max(1, threads)
    core.CONFIG["FORCE_DEVICE"] = "cpu"
    core.CONFIG.setdefault("SIGMA_UNCERT", {})["enable"] = True
    core.CONFIG["LOG_PERIOD"] = min(int(core.CONFIG.get("LOG_PERIOD", 100)), 50)

    reuse = model_path.is_file() and model_path.stat().st_size > 0 and not args.force_retrain
    if reuse:
        try:
            core.lgb.Booster(model_file=str(model_path))
            print(f"REUSE_SIGMA_MODEL path={model_path}", flush=True)
        except Exception:
            reuse = False
            print("EXISTING_SIGMA_MODEL_INVALID retraining=1", flush=True)

    if not reuse:
        working_dir = args.out_dir / "training_work"
        working_dir.mkdir(parents=True, exist_ok=True)
        seed = int(config.get("SEED", 2025)) + 17
        print(
            "SIGMA_TRAINING_START "
            f"rounds={core.CONFIG['SIGMA_UNCERT'].get('num_boost_round', 2000)} "
            f"threads={core.CONFIG['NUM_THREADS']} seed={seed}",
            flush=True,
        )
        trained = core.train_sigma_model_lgbm(
            X_tr=x_train,
            y_true_tr=train_pred["y_true"].to_numpy(float),
            y_pred_tr=train_pred["y_pred"].to_numpy(float),
            w_tr=np.asarray(w_all[idx_train], float),
            out_dir=str(working_dir),
            seed=seed,
            device_used="cpu",
        )
        if not trained or not Path(trained).is_file():
            raise RuntimeError("Sigma-head training did not create sigma_model.txt")
        os.replace(trained, model_path)
        print(f"SIGMA_TRAINING_DONE model={model_path}", flush=True)

    print("SIGMA_VALID_PREDICTION_START", flush=True)
    sigma_valid = core.predict_sigma_lgbm(
        str(model_path), x_valid, valid_pred["y_pred"].to_numpy(float)
    )
    if len(sigma_valid) != len(valid_pred):
        raise RuntimeError("Sigma prediction length mismatch")
    if not np.all(np.isfinite(sigma_valid)) or not np.all(sigma_valid > 0):
        raise RuntimeError("Sigma predictions contain invalid or non-positive values")

    compact = pd.DataFrame({
        "source_row": np.arange(len(valid_pred), dtype=np.int64),
        "affinity_type": valid_pred["affinity_type"].astype(str),
        "y_true": valid_pred["y_true"].to_numpy(float),
        "y_pred": valid_pred["y_pred"].to_numpy(float),
        "sigma": np.asarray(sigma_valid, dtype=np.float32),
    })
    tmp_output = output_path.with_name(output_path.name + ".tmp")
    compact.to_csv(tmp_output, index=False, compression={"method": "gzip", "compresslevel": 1})
    os.replace(tmp_output, output_path)

    sigma_float = np.asarray(sigma_valid, float)
    manifest = {
        "complete": True,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "frozen point predictions; sigma head trained on training residuals only",
        "core_script": str(args.core_script.resolve()),
        "config": str(args.config.resolve()),
        "feature_cache": str(args.feature_cache.resolve()),
        "feature_cache_tag": cache_meta.get("tag"),
        "feature_fit_scope": cache_meta.get("fit_scope"),
        "train_index_sha1": sha1_indices(idx_train),
        "valid_index_sha1": sha1_indices(idx_valid),
        "n_train": int(len(train_pred)),
        "n_valid": int(len(valid_pred)),
        "feature_shape": [int(len(y_all)), int(x_train.shape[1])],
        "endpoint_counts_train": endpoint_counts(train_pred["affinity_type"]),
        "endpoint_counts_valid": endpoint_counts(valid_pred["affinity_type"]),
        "sigma_config": core.CONFIG.get("SIGMA_UNCERT", {}),
        "sigma_summary_valid": {
            "min": float(np.min(sigma_float)),
            "p10": float(np.quantile(sigma_float, 0.10)),
            "median": float(np.median(sigma_float)),
            "mean": float(np.mean(sigma_float)),
            "p90": float(np.quantile(sigma_float, 0.90)),
            "max": float(np.max(sigma_float)),
        },
        "sigma_model": str(model_path.resolve()),
        "validation_predictions_with_sigma": str(output_path.resolve()),
    }
    write_json_atomic(manifest_path, manifest)
    print(f"SIGMA_VALID_PREDICTION_DONE output={output_path}", flush=True)
    print("SIGMA_HEAD_COMPLETE=1", flush=True)


if __name__ == "__main__":
    main()
