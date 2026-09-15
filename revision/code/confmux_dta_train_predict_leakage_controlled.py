#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConfMux-DTA leakage-controlled training and prediction pipeline.

This script implements the sequence-only ConfMux-DTA framework described in the
manuscript: unified pIC50/pKi/pKd regression using BPE-derived token documents,
sparse TF-IDF or count features, affinity-type one-hot encoding, LightGBM
regression, optional affine calibration, sigma-based uncertainty estimation,
conformal-style prediction intervals, selective prediction analysis, and a
self-contained prediction bundle for reproducible inference.

This reviewer-facing version enforces a split-before-fit protocol: BPE
tokenizers and TF-IDF/count vectorizers are fitted exclusively on the training
subset. Validation rows are transformed with the frozen training-derived
representations. The code is intended for academic review and reproducibility. It keeps the final
paper-facing workflow in one file while avoiding dependencies on structural
features, pretrained language models, deep-learning frameworks, or GPU-specific
packages for the core pipeline.

Typical usage:
  python confmux_dta_train_predict.py train_test \
      --data_file preprocessed/confmux_dta_bindingdb_agg.csv \
      --split_mode random \
      --total_iters 50000 \
      --stage_iters 2000

  python confmux_dta_train_predict.py predict \
      --bundle_dir path/to/predict_bundle \
      --input_csv examples.csv \
      --output_csv predictions.csv

Core dependencies:
  numpy, pandas, scipy, scikit-learn, lightgbm, tokenizers, rdkit, tqdm,
  matplotlib, joblib
"""

import os
import re
import sys
import csv
import json
import time
import math
import shutil
import joblib
import hashlib
import pathlib
import argparse
import warnings
import textwrap
import subprocess
import threading
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from scipy import sparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, average_precision_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize as sp_normalize

warnings.filterwarnings("ignore")

# ---- Optional dependencies ----
try:
    import psutil
except Exception:
    psutil = None

try:
    import resource
except Exception:
    resource = None

try:
    import pynvml as nvml
except Exception:
    nvml = None

# ---- LightGBM ----
try:
    import lightgbm as lgb
except Exception:
    lgb = None

# ---- RDKit for SMILES standardization ----
try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    try:
        from rdkit.Chem import rdMolStandardize as rdms
    except Exception:
        rdms = None
except Exception:
    Chem = None
    rdms = None

# ---- TensorBoard logging, optional ----
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

# ---- HuggingFace tokenizers for BPE ----
try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.normalizers import NFKC
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
except Exception:
    Tokenizer = None
    BPE = None
    BpeTrainer = None
    ByteLevel = None
    NFKC = None
    ByteLevelDecoder = None


# =============================================================================
# Base configuration
# =============================================================================

PROGRAM_NAME = "confmux_dta"
def generate_program_id():
    return f"{PROGRAM_NAME}_{time.strftime('%Y%m%d_%H%M%S')}"

CONFIG: Dict = {
    # ========== Input data and column definitions ==========
    "DATA_FILE": "preprocessed/confmux_dta_bindingdb_agg.csv",
    "COL_SMILES": "smiles",
    "COL_PROTEIN": "protein",

    # ========== Unified multi-affinity configuration ==========
    "AFFINITY_UNIFIED": {
        "enable": True,
        "p_cols": {"pIC50": "pIC50", "pKi": "pKi", "pKd": "pKd"},
        "w_cols": {"pIC50": "weight_IC50", "pKi": "weight_Ki", "pKd": "weight_Kd"},
        "type_feature": "onehot",
        "type_order": ["pIC50", "pKi", "pKd"],
        "min_types_per_row": 1,
        "keep_type_col_name": "affinity_type",
        "keep_target_col_name": "affinity_value",
        "keep_weight_col_name": "affinity_weight",
    },

    # ========== Single-affinity fallback mode ==========
    "AFFINITY": "pIC50",
    "COL_TARGET": "pIC50",
    "COL_WEIGHT": "weight_IC50",
    "AFFINITY_DISPLAY": "pIC50",
    "PRED_COL": "pred_pIC50",
    "PRED_LO_COL": "pred_pIC50_lo",
    "PRED_HI_COL": "pred_pIC50_hi",

    # ========== SMILES BPE ==========
    "SMILES_BPE": {
        "enable": True,
        "vocab_size": 16000,
        "min_frequency": 2,
        "special_tokens": ["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        "use_byte_level": True,
        "bpe_ngram_range": [1, 2],
    },

    # ========== Protein BPE ==========
    "PROTEIN_BPE": {
        "enable": True,
        "vocab_size": 16000,
        "min_frequency": 2,
        "special_tokens": ["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        "use_byte_level": True,
        "bpe_ngram_range": [1, 1],
    },

    # ========== Run backup settings ==========
    "BACKUP": {
        "enable": True,
        "task_id": generate_program_id(),
        "code_files": [],
        "data_files": [],
        "backup_root": None,
        "keep_versions": 5,
        "compress": False,
    },

    # ========== Data statistics ==========
    "DATA_STATS": {
        "enable": True,
        "output_json": "data_statistics.json",
        "output_csv": "data_statistics_summary.csv",
        "histogram_bins": 20,
        "sequence_length_percentiles": [5, 25, 50, 75, 95],
        "numeric_cols": ["Ki", "Kd", "IC50", "pKi", "pKd", "pIC50", "weight_IC50", "weight_Ki", "weight_Kd"],
    },

    # ========== SMILES normalization ==========
    "SMILES_NORM": {
        "enable": True,
        "output_col": "SMILES_NORM",
        "keep_original": True,
        "drop_invalid": True,
        "version": "rdkit_std_v1",
        "check_normalized": True,
        "check_sample_size": 1000,
    },

    # ========== Data split settings ==========
    "TEST_SIZE": 0.2,
    "SEED": 5313,
    "SPLIT_MODE": "compound",

    # ========== Sparse vectorizer ==========
    "VECTORIZER": {
        "type": "tfidf",
        "min_df": 2,
        "max_features_protein": 50000,
        "max_features_ligand": 50000,
        "norm": "l2",
        "lowercase": False,
        "sublinear_tf": True,
    },

    # ========== LightGBM ==========
    "LGBM_TOTAL_ITERS": 50000,
    "LGBM_STAGE_ITERS": 2000,
    "LGBM_PARAMS": {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 256,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 2,
        "lambda_l2": 1.0,
        "boosting_from_average": True,
        "min_data_in_leaf": 50,
        "verbose": -1,
        "force_col_wise": True,
    },

    # ========== Model-selection objective ==========
    "OPTIMIZE": {
        "by": "rmse",
        "q": 0.99,
    },

    "FORCE_DEVICE": None,
    "NUM_THREADS": 0,

    # ========== Optional sample-weight adjustment ==========
    "TAIL_BOOST_GAMMA": 0.0,
    "TAIL_BOOST_POWER": 1.0,
    "TAIL_CLIP_Z": 3.0,
    "BALANCE": {"enable": False, "n_bins": 10, "power": 1.0, "cap": 3.0},

    # ========== Post-processing ==========
    "STRONG_BINDER_THRESHOLD_PK": 6.0,
    "CALIBRATION": {
        "method": "train_linear",
        "train_sample_cap": 0,
        "clip": None
    },

    # ========== Selective prediction by binned conformal calibration ==========
    "SELECTIVE": {
        "enable": True,
        "alpha": 0.1,
        "n_bins": 12,
        "min_bin_size": 30,
        "default_err_max": 0.8,
        "target_coverages": [0.95, 0.90, 0.85, 0.80, 0.70, 0.60],
        "err_max_grid": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5],
        "plot_max_points": 5000,
    },

    # ========== Sample-level sigma(x) uncertainty model ==========
    # Train a residual-scale model sigma(x), then apply conformal scaling: err_bound = q_scale * sigma(x).
    "SIGMA_UNCERT": {
        "enable": True,
        "target": "log_sq_err",      # log_sq_err | log_abs_err
        "eps": 1e-6,
        "use_pred_feature": True,    # Add base prediction as an extra sigma-model feature.
        "sample_cap": 0,             # Maximum number of samples used for sigma-model training; 0 means no subsampling.
        "valid_fraction": 0.2,       # Internal validation fraction for sigma-model early stopping.
        "num_boost_round": 2000,
        "early_stopping_rounds": 50,
        "min_sigma": 1e-6,           # Lower bound for predicted sigma to avoid division by zero.
        "max_sigma": None,           # Optional upper clipping bound.
        "params": {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 80,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "verbosity": -1,
        },
    },

    # ========== Stage-history training-metric settings ==========
    "TRAIN_METRICS_SAMPLE_CAP": 0,
    "TRAIN_METRICS_RANDOM_SEED": 2027,

    # ========== Output and logging ==========
    "OUT_DIR_BASE": "confmux_dta_runs",
    "EXPERIMENT_NAME": "",
    "RUN_DIR": "",
    "RESUME": False,
    "TB_LOG_DIR_BASE": "confmux_dta_tensorboard",
    "SUMMARY_CSV": "confmux_dta_runs/summary.csv",
    "LOG_FILE": "dti_training.log",
    "VERBOSE": True,
    "GPU_SAMPLE_INTERVAL": 0.2,
    "GPU_PROBE_TIMEOUT_SEC": 12.0,
    "LOG_PERIOD": 100,
    "PLOT_SCATTER": True,

    # ========== Feature cache ==========
    "CACHE": {
        "enable": True,
        "dir": os.path.abspath("confmux_dta_feature_cache"),
        "overwrite": False,
        "resume": False
    },

    "BEST_BY": "rmse",
}

writer = None

# =============================================================================
# Logging and JSON utilities
# =============================================================================

def log(message: str, level: str = "INFO"):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {message}"
    print(line)
    try:
        os.makedirs(os.path.dirname(CONFIG["LOG_FILE"]), exist_ok=True)
        with open(CONFIG["LOG_FILE"], "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.integer, np.int32, np.int64)): return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)

def safe_json_dump(obj, path: str, indent: int = 2):
    def conv(x):
        if isinstance(x, np.ndarray):
            return [conv(v) for v in x.tolist()]
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items()}
        if isinstance(x, (np.integer, np.int32, np.int64)):
            return int(x)
        if isinstance(x, (np.floating, np.float32, np.float64, float)):
            v = float(x)
            return None if (not np.isfinite(v)) else v
        if isinstance(x, np.bool_):
            return bool(x)
        if pd.api.types.is_scalar(x) and pd.isna(x):
            return None
        return x

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(conv(obj), f, ensure_ascii=False, indent=indent, allow_nan=False)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def print_step(step: str, desc: str, is_end: bool = False):
    if is_end:
        log(f"[Done] {step}: {desc}")
    else:
        log(f"[Step] {step}: {desc}")


# =============================================================================
# Affinity targets for single-affinity fallback mode
# =============================================================================

AFFINITY_SPECS = {
    "pIC50": {"target_col": "pIC50", "weight_col": "weight_IC50", "display": "pIC50", "raw_measure": "IC50", "threshold_pk_default": 6.0, "pred_col_prefix": "pred"},
    "pKi":   {"target_col": "pKi",   "weight_col": "weight_Ki",   "display": "pKi",   "raw_measure": "Ki",   "threshold_pk_default": 6.0, "pred_col_prefix": "pred"},
    "pKd":   {"target_col": "pKd",   "weight_col": "weight_Kd",   "display": "pKd",   "raw_measure": "Kd",   "threshold_pk_default": 6.0, "pred_col_prefix": "pred"},
}

def apply_affinity_to_config(cfg: dict, affinity: str):
    if cfg.get("AFFINITY_UNIFIED", {}).get("enable", False):
        return
    a = (affinity or "").strip()
    if a not in AFFINITY_SPECS:
        raise ValueError(f"Unknown affinity={a}; choices: {list(AFFINITY_SPECS.keys())}")
    spec = AFFINITY_SPECS[a]
    cfg["AFFINITY"] = a
    cfg["COL_TARGET"] = spec["target_col"]
    cfg["COL_WEIGHT"] = spec["weight_col"]
    cfg["AFFINITY_DISPLAY"] = spec["display"]
    cfg["AFFINITY_RAW_MEASURE"] = spec.get("raw_measure", "")
    cfg["STRONG_BINDER_THRESHOLD_PK"] = float(spec.get("threshold_pk_default", cfg.get("STRONG_BINDER_THRESHOLD_PK", 6.0)))
    cfg["PRED_COL"] = f"{spec.get('pred_col_prefix','pred')}_{spec['display']}"
    cfg["PRED_LO_COL"] = f"{cfg['PRED_COL']}_lo"
    cfg["PRED_HI_COL"] = f"{cfg['PRED_COL']}_hi"

def apply_unified_names_to_config(cfg: dict):
    """Set field names for unified pX regression."""
    if not cfg.get("AFFINITY_UNIFIED", {}).get("enable", False):
        return
    u = cfg["AFFINITY_UNIFIED"]
    cfg["COL_TARGET"] = u["keep_target_col_name"]
    cfg["COL_WEIGHT"] = u["keep_weight_col_name"]
    cfg["AFFINITY_DISPLAY"] = "pX"
    cfg["PRED_COL"] = "pred_pX"
    cfg["PRED_LO_COL"] = "pred_pX_lo"
    cfg["PRED_HI_COL"] = "pred_pX_hi"
    cfg["AFFINITY"] = "pX"


# =============================================================================
# Convert wide affinity table to unified long-format pX table
# =============================================================================

def expand_affinity_rows(df_wide: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cfg = CONFIG["AFFINITY_UNIFIED"]
    assert cfg.get("enable", False)
    p_cols = cfg["p_cols"]
    w_cols = cfg["w_cols"]
    type_order = list(cfg.get("type_order", list(p_cols.keys())))
    type_col = cfg["keep_type_col_name"]
    y_col = cfg["keep_target_col_name"]
    w_col = cfg["keep_weight_col_name"]

    stats_out = {
        "wide_rows_in": int(len(df_wide)),
        "types": {t: {"p_col": p_cols.get(t), "w_col": w_cols.get(t), "finite": 0} for t in type_order},
        "rows_out": 0
    }

    p_vals = {}
    w_vals = {}
    for t in type_order:
        pc = p_cols.get(t)
        wc = w_cols.get(t)
        p_vals[t] = pd.to_numeric(df_wide[pc], errors="coerce") if (pc in df_wide.columns) else pd.Series([np.nan]*len(df_wide), index=df_wide.index)
        w_vals[t] = pd.to_numeric(df_wide[wc], errors="coerce") if (wc in df_wide.columns) else pd.Series([np.nan]*len(df_wide), index=df_wide.index)

    blocks = []
    for t in type_order:
        pv = p_vals[t]
        m = np.isfinite(pv.to_numpy())
        stats_out["types"][t]["finite"] = int(m.sum())
        if m.sum() == 0:
            continue
        sub = df_wide.loc[m].copy()
        sub[type_col] = t
        sub[y_col] = pv.loc[m].to_numpy(dtype=np.float32)

        # Pandas Copy-on-Write may expose a read-only NumPy view; weights are
        # sanitized in place below, so explicitly request a writable copy.
        ww = w_vals[t].loc[m].to_numpy(dtype=np.float32, copy=True)
        bad = ~np.isfinite(ww) | (ww <= 0)
        if bad.any():
            ww[bad] = 1.0
        sub[w_col] = ww
        blocks.append(sub)

    if not blocks:
        out = df_wide.iloc[:0].copy()
        out[type_col] = []
        out[y_col] = []
        out[w_col] = []
        stats_out["rows_out"] = 0
        return out, stats_out

    df_long = pd.concat(blocks, axis=0, ignore_index=False)
    df_long["_pair_uid"] = df_long["smiles_norm"].astype(str) + "||" + df_long["protein_clean"].astype(str)
    stats_out["rows_out"] = int(len(df_long))
    return df_long, stats_out


# =============================================================================
# Run-directory initialization
# =============================================================================

def init_run_dirs():
    global writer
    fixed_run_dir = str(CONFIG.get("RUN_DIR", "") or "").strip()
    if fixed_run_dir:
        CONFIG["OUT_DIR"] = os.path.abspath(fixed_run_dir)
        run_ts = os.path.basename(os.path.normpath(CONFIG["OUT_DIR"]))
    else:
        run_ts = time.strftime("%Y%m%d-%H%M%S")
        exp_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(CONFIG.get("EXPERIMENT_NAME", "")).strip()).strip("_.-")
        if exp_name:
            run_ts = f"{run_ts}_{exp_name}"
        CONFIG["OUT_DIR"] = os.path.join(CONFIG["OUT_DIR_BASE"], run_ts)
    CONFIG["TB_LOG_DIR"] = os.path.join(CONFIG["TB_LOG_DIR_BASE"], run_ts)
    CONFIG["STATS_DIR"] = os.path.join(CONFIG["OUT_DIR"], "data_statistics")
    CONFIG["LOG_FILE"] = os.path.join(CONFIG["OUT_DIR"], CONFIG["LOG_FILE"])
    os.makedirs(CONFIG["OUT_DIR"], exist_ok=True)
    os.makedirs(CONFIG["STATS_DIR"], exist_ok=True)
    CONFIG["BACKUP"]["backup_root"] = CONFIG["OUT_DIR"]
    try:
        CONFIG["BACKUP"]["code_files"] = [__file__]
    except NameError:
        CONFIG["BACKUP"]["code_files"] = []
    CONFIG["BACKUP"]["data_files"] = [CONFIG["DATA_FILE"]]

    apply_unified_names_to_config(CONFIG)

    if writer is not None:
        try:
            writer.close()
        except Exception:
            pass
        writer = None
    writer = SummaryWriter(CONFIG["TB_LOG_DIR"]) if SummaryWriter is not None else None


# =============================================================================
# Backup helpers
# =============================================================================

def backup_code_and_data(config: dict) -> None:
    if not config["BACKUP"]["enable"]:
        log("[Backup] Backup is disabled")
        return
    backup_dir = os.path.join(config["OUT_DIR"], "backup")
    os.makedirs(backup_dir, exist_ok=True)

    for code_path in config["BACKUP"].get("code_files", []):
        if code_path and os.path.exists(code_path):
            shutil.copy2(code_path, os.path.join(backup_dir, os.path.basename(code_path)))
    for data_path in config["BACKUP"].get("data_files", []):
        if data_path and os.path.exists(data_path):
            shutil.copy2(data_path, os.path.join(backup_dir, os.path.basename(data_path)))

    cfg_snapshot = os.path.join(backup_dir, "config_snapshot.json")
    safe_json_dump(config, cfg_snapshot)
    log(f"[Backup] Completed. Backup directory: {backup_dir}")


# =============================================================================
# Data statistics and diagnostic plots
# =============================================================================

def sha1_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def plot_distribution(data: pd.Series, title: str, save_path: str, bins: int = 20):
    try:
        x = pd.to_numeric(data, errors="coerce").dropna().values
        if len(x) < 5:
            log(f"[Plot] {title}: insufficient valid values ({len(x)}); skipped", "WARNING")
            return
        plt.figure(figsize=(8, 4))
        plt.hist(x, bins=bins, alpha=0.75, edgecolor="black")
        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=160)
        plt.close()
        log(f"[Plot] Saved: {save_path}")
    except Exception as e:
        log(f"[Plot] {title} failed: {e}", "ERROR")

def plot_correlation_matrix(numeric_df: pd.DataFrame, save_path: str):
    try:
        df = numeric_df.apply(lambda x: pd.to_numeric(x, errors="coerce"))
        df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
        if df.shape[1] < 2:
            log("[Plot] Correlation heatmap skipped: fewer than two numeric columns", "WARNING")
            return
        corr = df.corr()
        plt.figure(figsize=(10, 8))
        plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(label="Correlation")
        plt.xticks(range(len(df.columns)), df.columns, rotation=45, ha="right")
        plt.yticks(range(len(df.columns)), df.columns)
        plt.title("Correlation Matrix (Numeric)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=160)
        plt.close()
        log(f"[Plot] Saved correlation heatmap: {save_path}")
    except Exception as e:
        log(f"[Plot] Correlation heatmap failed: {e}", "ERROR")

def analyze_data(df: pd.DataFrame, cfg: dict) -> Dict:
    """Generate data summaries, histograms, a correlation heatmap, JSON, and CSV outputs."""
    log("[DataStats] Start")
    stats_dir = cfg["STATS_DIR"]
    out = {"basic_info": {}, "columns": {}}
    out["basic_info"] = {
        "total_samples": int(len(df)),
        "columns": df.columns.tolist(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_file": os.path.abspath(cfg["DATA_FILE"]),
        "data_sha1": sha1_file(cfg["DATA_FILE"]) if os.path.exists(cfg["DATA_FILE"]) else "",
    }
    bins = int(cfg["DATA_STATS"]["histogram_bins"])
    numeric_cols = set(cfg["DATA_STATS"]["numeric_cols"])

    for col in df.columns:
        try:
            s = df[col]
            if col in numeric_cols:
                v_raw = pd.to_numeric(s, errors="coerce")
                v = v_raw[np.isfinite(v_raw.to_numpy(dtype=float, na_value=np.nan))]
                out["columns"][col] = {
                    "type": "numeric",
                    "stats": {
                        "total": int(len(s)),
                        "non_null": int(v.notna().sum()),
                        "null_ratio": float(1.0 - (len(v) / max(len(s), 1))),
                        "min": None if v.empty else float(v.min()),
                        "max": None if v.empty else float(v.max()),
                        "mean": None if v.empty else float(v.mean()),
                        "std": None if v.empty else float(v.std()),
                        "median": None if v.empty else float(v.median()),
                        "q25": None if v.empty else float(v.quantile(0.25)),
                        "q75": None if v.empty else float(v.quantile(0.75)),
                    }
                }
                plot_distribution(v, f"Distribution of {col}",
                                  os.path.join(stats_dir, f"hist_{col}.png"), bins=bins)
            else:
                out["columns"][col] = {
                    "type": "categorical",
                    "stats": {
                        "total": int(len(s)),
                        "unique_count": int(s.nunique(dropna=True)),
                        "non_null": int(s.notna().sum())
                    }
                }
        except Exception as e:
            log(f"[DataStats] Column {col} failed: {e}", "WARNING")

    numeric_list = [c for c in cfg["DATA_STATS"]["numeric_cols"] if c in df.columns]
    if len(numeric_list) >= 2:
        plot_correlation_matrix(df[numeric_list], os.path.join(stats_dir, "correlation_matrix.png"))

    safe_json_dump(out, os.path.join(stats_dir, cfg["DATA_STATS"]["output_json"]))

    csv_path = os.path.join(stats_dir, cfg["DATA_STATS"]["output_csv"])
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Category", "Metric", "Value"])
            for k, v in out["basic_info"].items():
                w.writerow(["Basic", k, v])
            for col, meta in out["columns"].items():
                if meta["type"] == "numeric":
                    for stat, val in meta["stats"].items():
                        if val is not None:
                            w.writerow([f"Col:{col}", stat, val])
        log(f"[DataStats] Summary CSV: {csv_path}")
    except Exception as e:
        log(f"[DataStats] Failed to write summary CSV: {e}", "WARNING")

    return out


# =============================================================================
# Data cleaning
# =============================================================================

def clean_numeric_column(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().replace(
        ["", "na", "nd", "none", "nan", "N/A", "ND", "None"],
        np.nan, regex=False
    )
    unit_pattern = re.compile(r"^([\d\.\-\+eE]+)\s*([a-zA-Zμ]+)$")
    unit_map = {"nm": 0.001, "μm": 1.0, "um": 1.0, "mm": 1000.0, "mol": 1e6, "m": 1000.0, "u": 1.0, "n": 0.001}

    def parse_value(s):
        if pd.isna(s): return np.nan
        try: return float(s)
        except: pass
        m = unit_pattern.match(str(s))
        if m:
            v_str, unit = m.groups()
            try:
                v = float(v_str)
                unit_lower = unit.lower().replace("μ", "u")
                for k, mult in unit_map.items():
                    if k in unit_lower:
                        return v * mult
                return v
            except:
                return np.nan
        return np.nan

    out = cleaned.apply(parse_value)
    return pd.to_numeric(out, errors="coerce")

def clean_protein(s: str) -> str:
    """Clean protein sequences on the training side by retaining the 20 standard amino acids."""
    AA_RE = re.compile(r"[^A-Za-z]")
    cleaned = AA_RE.sub("", str(s) if s is not None else "").upper()
    keep = set("ACDEFGHIKLMNPQRSTVWY")
    return "".join([c for c in cleaned if c in keep])

def clean_smiles(s: str) -> str:
    SMI_RE = re.compile(r"\s+")
    s = SMI_RE.sub("", str(s) if s is not None else "").strip()
    if "|" in s:
        s = s.split("|", 1)[0]
    return s

def _normalize_smiles_one(s: str) -> Optional[str]:
    s = clean_smiles(s)
    if not s:
        return None
    if Chem is None:
        return s
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        if rdms is not None:
            try: mol = rdms.Normalizer().normalize(mol)
            except: pass
            try: mol = rdms.LargestFragmentChooser().choose(mol)
            except: pass
            try: mol = rdms.Uncharger().uncharge(mol)
            except: pass
            try: mol = rdms.TautomerEnumerator().Canonicalize(mol)
            except: pass
        smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        return smi if smi else None
    except Exception:
        return None

def normalize_smiles_series(series: pd.Series) -> pd.Series:
    log(f"[SMILES normalization] n={len(series)}")
    out = []
    for s in tqdm(series.astype(str).tolist(), desc="Normalize SMILES", unit="records"):
        out.append(_normalize_smiles_one(s))
    return pd.Series(out, index=series.index)


# =============================================================================
# Data filtering and unified long-table expansion
# =============================================================================

def filter_valid_samples(df: pd.DataFrame, out_dir: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    print("\n[Data filtering] Start...")
    stats_out: Dict[str, Any] = {"original": int(len(df))}

    df = df.copy()

    df["protein_clean"] = df[CONFIG["COL_PROTEIN"]].apply(clean_protein)
    before_prot = len(df)
    df = df[df["protein_clean"].astype(str).str.len() > 0].copy()
    stats_out["after_protein_clean"] = int(len(df))
    stats_out["protein_empty_removed"] = before_prot - len(df)
    log(f"[Filter] Removed empty protein records: {stats_out['protein_empty_removed']}")

    if len(df) == 0:
        stats_out["final"] = 0
        return df, stats_out

    if CONFIG["SMILES_NORM"]["enable"]:
        df["smiles_norm"] = normalize_smiles_series(df[CONFIG["COL_SMILES"]])
        ok_s = df["smiles_norm"].notna()
        df = df[ok_s].copy()
    else:
        df["smiles_norm"] = df[CONFIG["COL_SMILES"]].apply(clean_smiles)
    stats_out["after_smiles"] = int(len(df))

    if len(df) == 0:
        stats_out["final"] = 0
        return df, stats_out

    if CONFIG.get("AFFINITY_UNIFIED", {}).get("enable", False):
        df_long, aff_stats = expand_affinity_rows(df)
        stats_out["affinity_expand"] = aff_stats

        if len(df_long) == 0:
            stats_out["final"] = 0
            return df_long, stats_out

        wcol = CONFIG["AFFINITY_UNIFIED"]["keep_weight_col_name"]
        bad_w = df_long[wcol].isna() | (df_long[wcol] <= 0) | ~np.isfinite(df_long[wcol].to_numpy())
        if bad_w.any():
            stats_out["invalid_weight_fixed"] = int(bad_w.sum())
            df_long.loc[bad_w, wcol] = 1.0

        stats_out["final"] = int(len(df_long))
        stats_out["retention_rate"] = 100.0 * stats_out["final"] / max(stats_out["original"], 1)
        stats_out["unique_smiles"] = int(df_long["smiles_norm"].nunique())
        stats_out["unique_proteins"] = int(df_long["protein_clean"].nunique())
        stats_out["unique_pairs"] = int(df_long["_pair_uid"].nunique())
        print(f"[Data filtering] final(long)={stats_out['final']} | pairs={stats_out['unique_pairs']} | retention={stats_out['retention_rate']:.2f}%")
        return df_long, stats_out

    y = pd.to_numeric(df[CONFIG["COL_TARGET"]], errors="coerce")
    finite_y = np.isfinite(y.to_numpy(dtype=float, na_value=np.nan))
    df = df[finite_y].copy()
    df[CONFIG["COL_TARGET"]] = pd.to_numeric(df[CONFIG["COL_TARGET"]], errors="coerce")

    wcol = CONFIG["COL_WEIGHT"]
    if wcol in df.columns:
        df[wcol] = pd.to_numeric(df[wcol], errors="coerce")
        bad_w = ~np.isfinite(df[wcol].to_numpy(dtype=float, na_value=np.nan)) | (df[wcol].to_numpy(dtype=float, na_value=np.nan) <= 0)
        df.loc[bad_w, wcol] = 1.0

    keep_cols = [CONFIG["COL_SMILES"], "smiles_norm", CONFIG["COL_PROTEIN"], "protein_clean",
                 CONFIG["COL_TARGET"], CONFIG["COL_WEIGHT"]]
    df = df[keep_cols].dropna().copy()

    stats_out["final"] = int(len(df))
    stats_out["retention_rate"] = 100.0 * stats_out["final"] / max(stats_out["original"], 1)
    return df, stats_out


# =============================================================================
# BPE utilities
# =============================================================================

def _require_tokenizers():
    if Tokenizer is None:
        raise ImportError("The tokenizers package is required: pip install tokenizers")

def train_bpe_tokenizer_generic(text_list: List[str], bpe_cfg: dict, save_path: str, name: str = "BPE") -> str:
    _require_tokenizers()
    vocab_size = int(bpe_cfg.get("vocab_size", 16000))
    min_freq = int(bpe_cfg.get("min_frequency", 2))
    special_tokens = list(bpe_cfg.get("special_tokens", ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]))
    use_byte_level = bool(bpe_cfg.get("use_byte_level", True))

    text_list = [s for s in (text_list or []) if isinstance(s, str) and len(s) > 0]
    if len(text_list) < 10:
        raise ValueError(f"Too few samples to train {name}: {len(text_list)}")

    tok = Tokenizer(BPE(unk_token="[UNK]"))
    try:
        if NFKC is not None:
            tok.normalizer = NFKC()
    except:
        pass

    if use_byte_level:
        if ByteLevel is not None:
            tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        if ByteLevelDecoder is not None:
            tok.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=special_tokens,
        show_progress=True,
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tok.train_from_iterator(text_list, trainer=trainer)
    tok.save(save_path)
    return save_path

def smiles_to_bpe_doc(tok: "Tokenizer", smi: str) -> str:
    smi = clean_smiles(smi)
    if not smi:
        return ""
    enc = tok.encode(smi)
    return " ".join(enc.tokens)

def protein_to_bpe_doc(tok: "Tokenizer", seq: str) -> str:
    seq = clean_protein(seq)
    if not seq:
        return ""
    enc = tok.encode(seq)
    return " ".join(enc.tokens)


# =============================================================================
# affinity_type one-hot features
# =============================================================================

def affinity_type_onehot(df: pd.DataFrame) -> sparse.csr_matrix:
    cfg = CONFIG["AFFINITY_UNIFIED"]
    types = list(cfg.get("type_order", ["pIC50","pKi","pKd"]))
    col = cfg["keep_type_col_name"]
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")
    t = df[col].astype(str).values
    n = len(t)
    k = len(types)
    idx = {types[i]: i for i in range(k)}
    rows = np.arange(n, dtype=np.int32)
    cols = np.array([idx.get(x, -1) for x in t], dtype=np.int32)
    m = cols >= 0
    rows = rows[m]
    cols = cols[m]
    data = np.ones(rows.size, dtype=np.float32)
    X = sparse.csr_matrix((data, (rows, cols)), shape=(n, k), dtype=np.float32)
    return X


def use_affinity_type_feature() -> bool:
    """Whether unified endpoint identity is appended as a one-hot feature."""
    cfg = CONFIG.get("AFFINITY_UNIFIED", {}) or {}
    return bool(cfg.get("enable", False)) and str(cfg.get("type_feature", "onehot")).lower() == "onehot"


# =============================================================================
# Fast quantile helper
# =============================================================================

def abs_err_quantile(abs_err: np.ndarray, q: float) -> float:
    e = np.asarray(abs_err, float).reshape(-1)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return float("nan")
    q = float(q)
    if q <= 0:
        return float(np.min(e))
    if q >= 1:
        return float(np.max(e))
    k = int(math.ceil(q * (e.size - 1)))
    k = max(0, min(k, e.size - 1))
    return float(np.partition(e, k)[k])


# =============================================================================
# Approximate concordance index by vectorized pair sampling
# =============================================================================

def concordance_index_optimized(y_true, y_pred, max_pairs=500_000, random_state=42) -> float:
    y = np.asarray(y_true, float)
    f = np.asarray(y_pred, float)
    m = np.isfinite(y) & np.isfinite(f)
    y, f = y[m], f[m]
    n = len(y)
    if n < 2:
        return float("nan")

    rng = np.random.default_rng(random_state)
    k = int(min(max_pairs, n * (n - 1) // 2))
    i = rng.integers(0, n, size=k, dtype=np.int64)
    j = rng.integers(0, n, size=k, dtype=np.int64)
    good = i != j
    i, j = i[good], j[good]
    if len(i) == 0:
        return float("nan")

    dy = y[i] - y[j]
    df = f[i] - f[j]
    keep = dy != 0.0
    if not np.any(keep):
        return float("nan")

    prod = dy[keep] * df[keep]
    conc = np.sum(prod > 0.0)
    ties = np.sum(df[keep] == 0.0)
    return float((conc + 0.5 * ties) / prod.size)


# =============================================================================
# Regression and ranking metrics
# =============================================================================

def binarize_affinity(y_true_pk: np.ndarray, threshold_pk: float = 6.0) -> np.ndarray:
    y = np.asarray(y_true_pk, float)
    lab = np.zeros_like(y, dtype=int)
    lab[np.isfinite(y) & (y >= threshold_pk)] = 1
    return lab

def auc_aupr(y_true_pk: np.ndarray, y_pred_pk: np.ndarray, threshold_pk: float = 6.0):
    mask = np.isfinite(y_true_pk) & np.isfinite(y_pred_pk)
    y = y_true_pk[mask]
    s = y_pred_pk[mask]
    if len(y) < 2:
        return {"AUC": float("nan"), "AUPR": float("nan")}
    y_bin = binarize_affinity(y, threshold_pk)
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return {"AUC": float("nan"), "AUPR": float("nan")}
    return {"AUC": float(roc_auc_score(y_bin, s)), "AUPR": float(average_precision_score(y_bin, s))}

def _fit_k_through_origin(y_true, y_pred):
    y = np.asarray(y_true, float).reshape(-1)
    f = np.asarray(y_pred, float).reshape(-1)
    denom = float(np.dot(f, f))
    if denom <= 1e-12:
        return np.nan
    return float(np.dot(y, f) / denom)

def r2m_metrics(y_true, y_pred):
    y = np.asarray(y_true, float).reshape(-1)
    f = np.asarray(y_pred, float).reshape(-1)
    m = np.isfinite(y) & np.isfinite(f)
    y, f = y[m], f[m]
    if y.size < 3:
        return {k: np.nan for k in ["R2m","R2m_prime","R2m_bar","delta_R2m","R2","R0_sq","R0p_sq","k","k_prime"]}

    R2 = float(r2_score(y, f))
    k = _fit_k_through_origin(y, f)
    y_fit0 = k * f
    sse0 = float(np.sum((y - y_fit0) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    R0_sq = float(1.0 - sse0 / sst)

    k_prime = _fit_k_through_origin(f, y)
    f_fit0 = k_prime * y
    sse0p = float(np.sum((f - f_fit0) ** 2))
    sstp = float(np.sum((f - f.mean()) ** 2)) + 1e-12
    R0p_sq = float(1.0 - sse0p / sstp)

    term = abs(R2 - R0_sq)
    termp = abs(R2 - R0p_sq)
    R2m = float(R2 * (1.0 - np.sqrt(term)))
    R2m_prime = float(R2 * (1.0 - np.sqrt(termp)))
    R2m_bar = 0.5 * (R2m + R2m_prime)
    delta = abs(R2m - R2m_prime)

    return {
        "R2m": R2m, "R2m_prime": R2m_prime, "R2m_bar": R2m_bar, "delta_R2m": delta,
        "R2": R2, "R0_sq": R0_sq, "R0p_sq": R0p_sq, "k": k, "k_prime": k_prime
    }

def evaluate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold_pk: float,
    q: Optional[float] = None
) -> Dict[str, float]:
    y = np.asarray(y_true, float)
    f = np.asarray(y_pred, float)
    m = np.isfinite(y) & np.isfinite(f)
    yt = y[m]
    yp = f[m]

    if q is None:
        q = float(CONFIG.get("OPTIMIZE", {}).get("q", 0.99))
    q = float(q)

    if yt.size < 2:
        return {
            "R": float("nan"), "Pearson_p": float("nan"),
            "SpearmanR": float("nan"), "Spearman_p": float("nan"),
            "R2": float("nan"), "RMSE": float("nan"), "MSE": float("nan"),
            "AE_Q99": float("nan"), "AE_Qq": float("nan"),
            "CI": float("nan"), "AUC": float("nan"), "AUPR": float("nan"),
            "N": int(yt.size),
            "R2m": float("nan"), "R2m_prime": float("nan"),
            "R2m_bar": float("nan"), "delta_R2m": float("nan"),
            "R0_sq": float("nan"), "R0p_sq": float("nan"),
            "k0": float("nan"), "k0p": float("nan"),
            "q_used": q,
        }

    # Pearson / Spearman
    try:
        pr, pp = stats.pearsonr(yt, yp)
        R = float(pr); Pearson_p = float(pp)
    except Exception:
        R = float("nan"); Pearson_p = float("nan")

    try:
        sr, spv = stats.spearmanr(yt, yp)
        SpearmanR = float(sr); Spearman_p = float(spv)
    except Exception:
        SpearmanR = float("nan"); Spearman_p = float("nan")

    mse = float(mean_squared_error(yt, yp))
    rmse = float(np.sqrt(mse))
    r2v = float(r2_score(yt, yp))

    abs_err = np.abs(yt - yp)
    ae_qq = abs_err_quantile(abs_err, q)

    ci = concordance_index_optimized(yt, yp, max_pairs=500_000, random_state=42)
    cls = auc_aupr(yt, yp, threshold_pk)
    rm = r2m_metrics(yt, yp)

    return {
        "R": R,
        "Pearson_p": Pearson_p,
        "SpearmanR": SpearmanR,
        "Spearman_p": Spearman_p,
        "R2": r2v,
        "RMSE": rmse,
        "MSE": mse,
        "AE_Q99": ae_qq,      # Backward-compatible field name
        "AE_Qq": ae_qq,       # Preferred field name
        "CI": float(ci) if np.isfinite(ci) else float("nan"),
        "AUC": cls["AUC"],
        "AUPR": cls["AUPR"],
        "N": int(len(yt)),
        "R2m": rm["R2m"], "R2m_prime": rm["R2m_prime"],
        "R2m_bar": rm["R2m_bar"], "delta_R2m": rm["delta_R2m"],
        "R0_sq": rm["R0_sq"], "R0p_sq": rm["R0p_sq"],
        "k0": rm["k"], "k0p": rm["k_prime"],
        "q_used": q,
    }



def strong_binder_probability(y_pred: np.ndarray,
                              sigma: Optional[np.ndarray] = None,
                              threshold_pk: float = 6.0) -> np.ndarray:
    pred = np.asarray(y_pred, float).reshape(-1)
    prob = np.full(pred.shape, np.nan, dtype=np.float64)
    m_pred = np.isfinite(pred)
    if sigma is None:
        prob[m_pred] = (pred[m_pred] >= float(threshold_pk)).astype(np.float64)
        return prob

    sig = np.asarray(sigma, float).reshape(-1)
    m_sig = m_pred & np.isfinite(sig) & (sig > 0)
    if np.any(m_sig):
        z = (float(threshold_pk) - pred[m_sig]) / np.maximum(sig[m_sig], 1e-12)
        prob[m_sig] = stats.norm.sf(z)
    m_det = m_pred & (~m_sig)
    if np.any(m_det):
        prob[m_det] = (pred[m_det] >= float(threshold_pk)).astype(np.float64)
    return np.clip(prob, 0.0, 1.0)


def evaluate_metrics_pack(df_part: pd.DataFrame,
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          threshold_pk: float,
                          q: Optional[float] = None) -> Dict[str, Any]:
    pack = {
        "overall": evaluate_metrics(y_true, y_pred, threshold_pk, q=q),
        "by_type": {}
    }
    if isinstance(df_part, pd.DataFrame) and ("affinity_type" in df_part.columns):
        vals = df_part["affinity_type"].astype(str).fillna("")
        type_order = list((CONFIG.get("AFFINITY_UNIFIED", {}) or {}).get("type_order", ["pIC50", "pKi", "pKd"]))
        ordered = [t for t in type_order if (vals == t).any()]
        extras = [t for t in vals.unique().tolist() if t and (t not in ordered)]
        for t in ordered + extras:
            mask = (vals.values == t)
            pack["by_type"][str(t)] = evaluate_metrics(np.asarray(y_true)[mask], np.asarray(y_pred)[mask], threshold_pk, q=q)
    return pack


def flatten_metrics_for_paper(split_name: str,
                              metrics_pack: Dict[str, Any],
                              alias_splits: Optional[List[str]] = None) -> Dict[str, Any]:
    alias_splits = list(alias_splits or [])
    metric_map = {
        "R2": "R2",
        "CI": "CI",
        "RMSE": "RMSE",
        "MSE": "MSE",
        "R": "PearsonR",
        "N": "N",
    }
    out = {}

    def _write(prefix: str, m: Dict[str, Any]):
        for src, dst in metric_map.items():
            out[f"{prefix}_{dst}"] = m.get(src, np.nan)

    overall = metrics_pack.get("overall", {}) or {}
    _write(split_name, overall)
    for alias in alias_splits:
        _write(alias, overall)

    by_type = metrics_pack.get("by_type", {}) or {}
    for t, m in by_type.items():
        _write(f"{t}_{split_name}", m)
        for alias in alias_splits:
            _write(f"{t}_{alias}", m)
    return out


def build_prediction_export_df(df_part: pd.DataFrame,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               split_name: str,
                               threshold_pk: float,
                               sigma: Optional[np.ndarray] = None,
                               errb: Optional[np.ndarray] = None) -> pd.DataFrame:
    out = df_part.copy().reset_index(drop=True)
    yt = np.asarray(y_true, float).reshape(-1)
    yp = np.asarray(y_pred, float).reshape(-1)
    out["split"] = split_name
    out["y_true"] = yt
    out["y_pred"] = yp
    out["abs_err"] = np.abs(yt - yp)
    out["strong_binder_threshold_pk"] = float(threshold_pk)
    if sigma is not None:
        sig = np.asarray(sigma, float).reshape(-1)
        out["sigma"] = sig
    else:
        sig = None
    if errb is not None:
        eb = np.asarray(errb, float).reshape(-1)
        out["err_bound"] = eb
        out["pred_lo"] = yp - eb
        out["pred_hi"] = yp + eb
    out["prob_strong_binder"] = strong_binder_probability(yp, sig, threshold_pk=float(threshold_pk))
    out["pred_is_strong_binder"] = (out["prob_strong_binder"] >= 0.5).astype(int)
    return out


STAGE_HISTORY_BY_TYPE_METRICS = {
    "R2": "R2",
    "RMSE": "RMSE",
    "R2m": "R2m",
    "CI": "CI",
    "PearsonR": "R",
}


def get_affinity_type_order(df_part: Optional[pd.DataFrame] = None) -> List[str]:
    base = list((CONFIG.get("AFFINITY_UNIFIED", {}) or {}).get("type_order", ["pIC50", "pKi", "pKd"]))
    if not isinstance(df_part, pd.DataFrame) or ("affinity_type" not in df_part.columns):
        return base
    vals = df_part["affinity_type"].astype(str).fillna("")
    ordered = [t for t in base if (vals == t).any()]
    extras = [t for t in vals.unique().tolist() if t and (t not in ordered)]
    return ordered + extras


def stage_history_by_type_columns(prefixes: Optional[List[str]] = None, type_order: Optional[List[str]] = None) -> List[str]:
    prefixes = list(prefixes or ["TR", "VAL"])
    type_order = list(type_order or get_affinity_type_order())
    cols = []
    for pref in prefixes:
        for t in type_order:
            for metric_name in STAGE_HISTORY_BY_TYPE_METRICS.keys():
                cols.append(f"{pref}_{t}_{metric_name}")
    return cols


def stage_history_by_type_values(prefix: str,
                                 metrics_pack: Dict[str, Any],
                                 type_order: Optional[List[str]] = None) -> Dict[str, Any]:
    type_order = list(type_order or get_affinity_type_order())
    out = {}
    by_type = (metrics_pack or {}).get("by_type", {}) or {}
    for t in type_order:
        mt = by_type.get(t, {}) or {}
        for metric_name, src_key in STAGE_HISTORY_BY_TYPE_METRICS.items():
            out[f"{prefix}_{t}_{metric_name}"] = mt.get(src_key, np.nan)
    return out


def get_optimize_spec(opt_by: str) -> Tuple[str, str]:
    """
    Return (metric_key, mode), where mode is either "min" or "max".
    Supported objectives: rmse, mse, ae_q, r2, ci, and r2m.
    For r2m, R2m_bar is used by default as a robust averaged metric.
    """
    by = str(opt_by or "rmse").strip().lower()
    by = by.replace("-", "_")
    if by in ("rmse",):
        return "RMSE", "min"
    if by in ("mse", "l2"):
        return "MSE", "min"
    if by in ("ae_q", "ae_q99", "aeq"):
        return "AE_Qq", "min"
    if by in ("r2", "r^2"):
        return "R2", "max"
    if by in ("ci", "cindex", "concordance"):
        return "CI", "max"
    if by in ("r2m", "r2m_bar", "r2mbar"):
        return "R2m_bar", "max"
    if by in ("r2m_raw",):
        return "R2m", "max"
    log(f"[Warning] Unknown optimization target {opt_by}; using RMSE", "WARNING")
    return "RMSE", "min"

def is_better(current: float, best: float, mode: str, eps: float = 1e-12) -> bool:
    if not np.isfinite(current):
        return False
    if mode == "min":
        return current < best - eps
    return current > best + eps

# =============================================================================
# Training weights
# =============================================================================

def balance_weights_by_quantile(y_tr_raw: np.ndarray, n_bins: int = 10, power: float = 1.0, cap: float = 3.0) -> np.ndarray:
    bins = pd.qcut(pd.Series(y_tr_raw), q=n_bins, labels=False, duplicates="drop")
    counts = pd.Series(bins).value_counts().sort_index().to_numpy().astype(float)
    inv = 1.0 / np.maximum(counts, 1.0)
    w_bin = inv ** float(power)
    w = w_bin[bins]
    if cap > 0:
        w = np.minimum(w, float(cap))
    w = w / (np.mean(w) + 1e-12)
    return w.astype(np.float32)

def make_train_weights(df_tr: pd.DataFrame, y_tr_raw: np.ndarray) -> np.ndarray:
    if CONFIG.get("AFFINITY_UNIFIED", {}).get("enable", False):
        wcol = CONFIG["AFFINITY_UNIFIED"]["keep_weight_col_name"]
        w = df_tr[wcol].fillna(1.0).astype(np.float32).values
    else:
        w = df_tr[CONFIG["COL_WEIGHT"]].fillna(1.0).astype(np.float32).values

    w = np.maximum(w, 1e-6)

    gamma = float(CONFIG["TAIL_BOOST_GAMMA"])
    power = float(CONFIG["TAIL_BOOST_POWER"])
    if gamma > 0:
        mu = float(np.mean(y_tr_raw))
        sd = float(np.std(y_tr_raw) + 1e-8)
        z = (y_tr_raw - mu) / sd
        z = np.clip(z, -CONFIG["TAIL_CLIP_Z"], CONFIG["TAIL_CLIP_Z"])
        w *= (1.0 + gamma * (np.abs(z) ** power)).astype(np.float32)

    bal = CONFIG["BALANCE"]
    if bal.get("enable", False):
        w *= balance_weights_by_quantile(y_tr_raw, n_bins=int(bal["n_bins"]),
                                         power=float(bal["power"]), cap=float(bal["cap"]))
    w = w / (np.mean(w) + 1e-12)
    return w.astype(np.float32)

def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, float)
    f = np.asarray(y_pred, float)
    m = np.isfinite(y) & np.isfinite(f)
    y, f = y[m], f[m]
    if y.size < 2:
        return float("nan")
    return float(np.corrcoef(y, f)[0, 1])


# =============================================================================
# Plotting utilities
# =============================================================================

def plot_scatter(y_true, y_pred, title, out_path):
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        yt = y_true[mask]
        yp = y_pred[mask]
        if len(yt) < 5:
            return
        plt.figure(figsize=(6, 6))
        plt.scatter(yt, yp, s=8, alpha=0.5)
        mn = float(min(np.min(yt), np.min(yp)))
        mx = float(max(np.max(yt), np.max(yp)))
        plt.plot([mn, mx], [mn, mx], "k--", lw=1.1, label="y=x")
        lr = LinearRegression().fit(yt.reshape(-1, 1), yp.reshape(-1, 1))
        a = float(lr.coef_[0][0])
        b = float(lr.intercept_[0])
        xs = np.linspace(mn, mx, 100)
        ys = a * xs + b
        plt.plot(xs, ys, lw=1.1, label=f"y={a:.2f}x+{b:.2f}")
        R = pearson_r(yt, yp)

        abs_err = np.abs(yt - yp)
        q = float(CONFIG.get("OPTIMIZE", {}).get("q", 0.99))
        ae_qq = abs_err_quantile(abs_err, q)

        plt.text(0.02, 0.98, f"R={R:.3f}\nAE_Q{int(q*100)}={ae_qq:.3f}",
                 transform=plt.gca().transAxes, va="top", ha="left",
                 bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

        aff_disp = CONFIG.get("AFFINITY_DISPLAY", "pX")
        plt.xlabel(f"Experimental {aff_disp}")
        plt.ylabel(f"Predicted {aff_disp}")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=170)
        plt.close()
    except Exception as e:
        log(f"[Plot] Scatter plot failed: {e}", "ERROR")

def plot_abs_error_hist(abs_err: np.ndarray, out_path: str, bins: int = 50, title: str = "Absolute Error"):
    try:
        e = np.asarray(abs_err, float)
        e = e[np.isfinite(e)]
        if len(e) < 10:
            return
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.figure(figsize=(7.5, 4.2))
        plt.hist(e, bins=bins, alpha=0.8, edgecolor="black")
        q = float(CONFIG.get("OPTIMIZE", {}).get("q", 0.99))
        ae_qq = abs_err_quantile(e, q)
        plt.axvline(x=ae_qq, color='r', linestyle='--', linewidth=1.5,
                    label=f'AE_Q{int(q*100)}={ae_qq:.3f}')
        plt.title(title)
        plt.xlabel("|error|")
        plt.ylabel("count")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_path, dpi=170)
        plt.close()
    except Exception as ex:
        log(f"[Plot] Absolute-error histogram failed: {ex}", "WARNING")

def plot_curve(x, y, out_path: str, title: str, xlabel: str, ylabel: str):
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.figure(figsize=(7.2, 4.4))
        plt.plot(x, y, marker="o", lw=1.3, ms=4)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_path, dpi=170)
        plt.close()
    except Exception as ex:
        log(f"[Plot] Curve plot failed: {ex}", "WARNING")

def plot_interval_scatter(y_true: np.ndarray, y_pred: np.ndarray, errb: np.ndarray, out_path: str, max_points: int = 5000):
    try:
        yt = np.asarray(y_true, float)
        yp = np.asarray(y_pred, float)
        eb = np.asarray(errb, float)
        m = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(eb)
        yt, yp, eb = yt[m], yp[m], eb[m]
        n = len(yt)
        if n < 30:
            return
        if n > max_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=max_points, replace=False)
            yt, yp, eb = yt[idx], yp[idx], eb[idx]

        order = np.argsort(yp)
        yt, yp, eb = yt[order], yp[order], eb[order]
        x = np.arange(len(yt))

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.figure(figsize=(9.5, 4.8))
        plt.fill_between(x, yp - eb, yp + eb, alpha=0.25, label="pred ± err_bound")
        plt.scatter(x, yt, s=10, alpha=0.6, label="true")
        plt.plot(x, yp, lw=1.0, alpha=0.9, label="pred")
        plt.title("Prediction Interval Visualization (sorted by pred)")
        plt.xlabel("samples (sorted)")
        plt.ylabel(CONFIG.get("AFFINITY_DISPLAY", "pX"))
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(out_path, dpi=170)
        plt.close()
    except Exception as ex:
        log(f"[Plot] Interval scatter plot failed: {ex}", "WARNING")



def plot_scatter_custom(y_true, y_pred, title, out_path, affinity_label: str = "pX"):
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        yt = y_true[mask]
        yp = y_pred[mask]
        if len(yt) < 5:
            return
        plt.figure(figsize=(6, 6))
        plt.scatter(yt, yp, s=8, alpha=0.5)
        mn = float(min(np.min(yt), np.min(yp)))
        mx = float(max(np.max(yt), np.max(yp)))
        plt.plot([mn, mx], [mn, mx], "k--", lw=1.1, label="y=x")
        lr = LinearRegression().fit(yt.reshape(-1, 1), yp.reshape(-1, 1))
        a = float(lr.coef_[0][0])
        b = float(lr.intercept_[0])
        xs = np.linspace(mn, mx, 100)
        ys = a * xs + b
        plt.plot(xs, ys, lw=1.1, label=f"y={a:.2f}x+{b:.2f}")
        R = pearson_r(yt, yp)
        abs_err = np.abs(yt - yp)
        q = float(CONFIG.get("OPTIMIZE", {}).get("q", 0.99))
        ae_qq = abs_err_quantile(abs_err, q)
        plt.text(0.02, 0.98, f"R={R:.3f}\nAE_Q{int(q*100)}={ae_qq:.3f}",
                 transform=plt.gca().transAxes, va="top", ha="left",
                 bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
        plt.xlabel(f"Experimental {affinity_label}")
        plt.ylabel(f"Predicted {affinity_label}")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=170)
        plt.close()
    except Exception as e:
        log(f"[Plot] Custom scatter plot failed: {e}", "ERROR")


def export_per_type_plots(df_part: pd.DataFrame,
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          out_dir: str,
                          split_name: str,
                          errb: Optional[np.ndarray] = None,
                          min_points: int = 5,
                          interval_min_points: int = 30,
                          max_points: int = 5000) -> Dict[str, List[str]]:
    outputs = {"scatter": [], "abs_err_hist": [], "interval": []}
    if not isinstance(df_part, pd.DataFrame) or ("affinity_type" not in df_part.columns):
        return outputs
    os.makedirs(out_dir, exist_ok=True)
    vals = df_part["affinity_type"].astype(str).fillna("").values
    y_true = np.asarray(y_true, float).reshape(-1)
    y_pred = np.asarray(y_pred, float).reshape(-1)
    errb_arr = None if errb is None else np.asarray(errb, float).reshape(-1)
    for t in get_affinity_type_order(df_part):
        mask = (vals == t)
        n = int(mask.sum())
        if n < min_points:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        base = f"{split_name}_{t}"
        scatter_path = os.path.join(out_dir, f"SCATTER_{base}.png")
        plot_scatter_custom(yt, yp,
                            title=f"{split_name} {t} Experimental vs Predicted",
                            out_path=scatter_path,
                            affinity_label=t)
        if os.path.exists(scatter_path):
            outputs["scatter"].append(os.path.basename(scatter_path))
        abs_hist_path = os.path.join(out_dir, f"ABSERR_HIST_{base}.png")
        plot_abs_error_hist(np.abs(yt - yp), abs_hist_path,
                            title=f"{split_name} {t} Absolute Error")
        if os.path.exists(abs_hist_path):
            outputs["abs_err_hist"].append(os.path.basename(abs_hist_path))
        if errb_arr is not None and n >= interval_min_points:
            eb = errb_arr[mask]
            interval_path = os.path.join(out_dir, f"INTERVAL_{base}.png")
            plot_interval_scatter(yt, yp, eb, interval_path, max_points=max_points)
            if os.path.exists(interval_path):
                outputs["interval"].append(os.path.basename(interval_path))
    return outputs


# =============================================================================
# Affine calibration
# =============================================================================

def fit_affine(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    y = np.asarray(y_true, float).reshape(-1, 1)
    p = np.asarray(y_pred, float).reshape(-1, 1)
    m = np.isfinite(y).ravel() & np.isfinite(p).ravel()
    y, p = y[m], p[m]
    if len(y) < 2:
        return 0.0, 1.0
    reg = LinearRegression(fit_intercept=True).fit(p, y)
    a = float(reg.intercept_[0])
    b = float(reg.coef_[0][0])
    return a, b

def apply_affine(p: np.ndarray, a: float, b: float, clip=None) -> np.ndarray:
    q = a + b * np.asarray(p, float)
    if clip is not None and isinstance(clip, (list, tuple)) and len(clip) == 2:
        lo, hi = float(clip[0]), float(clip[1])
        q = np.clip(q, lo, hi)
    return q


# =============================================================================
# Binned conformal selective prediction
# =============================================================================

def build_conformal_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.1,
    n_bins: int = 12,
    min_bin_size: int = 30
) -> Dict:
    y_true = np.asarray(y_true, float).reshape(-1)
    y_pred = np.asarray(y_pred, float).reshape(-1)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]
    y_pred = y_pred[m]
    if len(y_true) < 50:
        raise ValueError(f"Calibration set is too small: {len(y_true)}")

    abs_err = np.abs(y_true - y_pred)
    q_global = float(np.quantile(abs_err, 1.0 - float(alpha)))

    n_bins = int(max(3, n_bins))
    edges = np.quantile(y_pred, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)

    if len(edges) < 4:
        return {
            "alpha": float(alpha),
            "global_q": q_global,
            "bin_edges": None,
            "bin_q": None,
            "n": int(len(y_true)),
            "note": "insufficient unique edges; fallback to global"
        }

    bin_q = []
    bin_n = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mm = (y_pred >= lo) & (y_pred <= hi)
        else:
            mm = (y_pred >= lo) & (y_pred < hi)
        e = abs_err[mm]
        bin_n.append(int(len(e)))
        if len(e) < int(min_bin_size):
            bin_q.append(q_global)
        else:
            bin_q.append(float(np.quantile(e, 1.0 - float(alpha))))

    return {
        "alpha": float(alpha),
        "global_q": q_global,
        "bin_edges": edges.tolist(),
        "bin_q": bin_q,
        "bin_n": bin_n,
        "min_bin_size": int(min_bin_size),
        "n": int(len(y_true))
    }

def err_bound_for_preds(y_pred: np.ndarray, calib: Dict) -> np.ndarray:
    y_pred = np.asarray(y_pred, float).reshape(-1)
    if calib.get("bin_edges") is None or calib.get("bin_q") is None:
        return np.full_like(y_pred, float(calib["global_q"]), dtype=np.float32)
    edges = np.asarray(calib["bin_edges"], float)
    bin_q = np.asarray(calib["bin_q"], float)
    idx = np.searchsorted(edges[1:-1], y_pred, side="right")
    idx = np.clip(idx, 0, len(bin_q) - 1)
    return bin_q[idx].astype(np.float32)

# =============================================================================
# Sample-level sigma(x) uncertainty model
# =============================================================================

def _dense_col_to_csr(x: np.ndarray) -> sparse.csr_matrix:
    x = np.asarray(x, float).reshape(-1, 1)
    return sparse.csr_matrix(x.astype(np.float32))

def make_sigma_features(X: sparse.csr_matrix, y_pred: np.ndarray) -> sparse.csr_matrix:
    """Append the base prediction as an extra feature for the sigma model."""
    if not sparse.isspmatrix_csr(X):
        X = X.tocsr()
    col = _dense_col_to_csr(y_pred)
    return sparse.hstack([X, col], format="csr")

def train_sigma_model_lgbm(
    X_tr: sparse.csr_matrix, y_true_tr: np.ndarray, y_pred_tr: np.ndarray,
    w_tr: Optional[np.ndarray], out_dir: str, seed: int, device_used: str
) -> Optional[str]:
    cfg = CONFIG.get("SIGMA_UNCERT", {}) or {}
    if not cfg.get("enable", True):
        return None

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(int(seed))

    y_true_tr = np.asarray(y_true_tr, float).reshape(-1)
    y_pred_tr = np.asarray(y_pred_tr, float).reshape(-1)
    m = np.isfinite(y_true_tr) & np.isfinite(y_pred_tr)
    idx_all = np.where(m)[0]
    if idx_all.size < 200:
        raise ValueError(f"Too few samples for sigma-model training: {idx_all.size}")

    cap = int(cfg.get("sample_cap", 0) or 0)
    if cap > 0 and idx_all.size > cap:
        idx = rng.choice(idx_all, size=cap, replace=False)
    else:
        idx = idx_all

    eps = float(cfg.get("eps", 1e-6))
    abs_err = np.abs(y_true_tr[idx] - y_pred_tr[idx])
    target = str(cfg.get("target", "log_sq_err")).strip().lower()
    if target == "log_abs_err":
        y_sig = np.log(abs_err + eps)
    else:
        y_sig = np.log(abs_err**2 + eps)

    X_sig = X_tr[idx]
    if cfg.get("use_pred_feature", True):
        X_sig = make_sigma_features(X_sig, y_pred_tr[idx])

    w_sig = None
    if w_tr is not None:
        w_sig = np.asarray(w_tr, float).reshape(-1)[idx]

    n = X_sig.shape[0]
    vf = float(cfg.get("valid_fraction", 0.2))
    n_va = max(1, int(n * vf))
    perm = rng.permutation(n)
    va = perm[:n_va]
    tr = perm[n_va:]
    if tr.size < 50 or va.size < 50:
        # Fallback: train without early stopping
        tr = perm
        va = None

    dtr = lgb.Dataset(X_sig[tr], label=y_sig[tr], weight=(w_sig[tr] if w_sig is not None else None), free_raw_data=False)
    valid_sets = []
    valid_names = []
    if va is not None:
        dva = lgb.Dataset(X_sig[va], label=y_sig[va], weight=(w_sig[va] if w_sig is not None else None), free_raw_data=False)
        valid_sets = [dva]
        valid_names = ["valid_sigma"]

    params = dict(cfg.get("params", {}) or {})
    params.setdefault("objective", "regression")
    params.setdefault("metric", "l2")
    params.setdefault("verbosity", -1)
    if device_used == "gpu":
        params["device"] = "gpu"

    nt = int(CONFIG.get("NUM_THREADS", 0) or 0)
    if nt > 0:
        params["num_threads"] = nt

    num_boost_round = int(cfg.get("num_boost_round", 2000))
    esr = int(cfg.get("early_stopping_rounds", 50))
    callbacks = []
    if va is not None and esr > 0:
        callbacks.append(lgb.early_stopping(stopping_rounds=esr, verbose=False))
    callbacks.append(lgb.log_evaluation(period=int(CONFIG.get("LOG_PERIOD", 100))))

    booster = lgb.train(params, dtr, num_boost_round=num_boost_round, valid_sets=valid_sets, valid_names=valid_names, callbacks=callbacks)

    model_path = os.path.join(out_dir, "sigma_model.txt")
    booster.save_model(model_path)
    return model_path

def predict_sigma_lgbm(model_path: str, X: sparse.csr_matrix, y_pred: np.ndarray) -> np.ndarray:
    cfg = CONFIG.get("SIGMA_UNCERT", {}) or {}
    if not model_path or (not os.path.exists(model_path)):
        raise FileNotFoundError(f"sigma_model not found: {model_path}")
    booster = lgb.Booster(model_file=model_path)
    X_sig = X
    if cfg.get("use_pred_feature", True):
        X_sig = make_sigma_features(X_sig, y_pred)
    s = np.asarray(booster.predict(X_sig), float).reshape(-1)
    s = np.clip(s, -50.0, 50.0)
    target = str(cfg.get("target", "log_sq_err")).strip().lower()
    if target == "log_abs_err":
        sigma = np.exp(s)
    else:
        sigma = np.exp(0.5 * s)
    sigma = np.asarray(sigma, np.float64)
    sigma[~np.isfinite(sigma)] = np.nan
    min_sigma = float(cfg.get("min_sigma", 1e-6))
    mx = cfg.get("max_sigma", None)
    max_sigma = float(mx) if mx is not None else 1e6
    sigma = np.clip(sigma, min_sigma, max_sigma)
    sigma[~np.isfinite(sigma)] = max_sigma
    return sigma.astype(np.float32)

def build_sigma_calibration(y_true: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray, alpha: float, eps: float = 1e-6) -> Dict:
    y_true = np.asarray(y_true, float).reshape(-1)
    y_pred = np.asarray(y_pred, float).reshape(-1)
    sigma = np.asarray(sigma, float).reshape(-1)
    m = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(sigma)
    if m.sum() < 50:
        raise ValueError(f"Sigma calibration set is too small: {int(m.sum())}")
    ratio = np.abs(y_true[m] - y_pred[m]) / np.maximum(sigma[m], float(eps))
    q_scale = float(np.quantile(ratio, 1.0 - float(alpha)))
    return {
        "method": "sigma",
        "alpha": float(alpha),
        "eps": float(eps),
        "q_scale": float(q_scale),
        "n": int(m.sum()),
        "note": "err_bound = q_scale * sigma(x)"
    }

def err_bound_for_sigma(sigma: np.ndarray, calib: Dict) -> np.ndarray:
    sigma = np.asarray(sigma, float).reshape(-1)
    q_scale = float(calib.get("q_scale", 0.0))
    return (q_scale * sigma).astype(np.float32)

def err_bound_for_calib(y_pred: np.ndarray, calib: Dict, sigma: Optional[np.ndarray] = None) -> np.ndarray:
    method = str(calib.get("method", "bin")).strip().lower()
    if method == "sigma":
        if sigma is None:
            raise ValueError("calib.method=sigma but sigma values were not provided")
        return err_bound_for_sigma(sigma, calib)
    return err_bound_for_preds(y_pred, calib)

def selective_metrics_table(y_true: np.ndarray, y_pred: np.ndarray, errb: np.ndarray,
                            err_max_list: List[float]) -> pd.DataFrame:
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    eb = np.asarray(errb, float)
    m = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(eb)
    yt, yp, eb = yt[m], yp[m], eb[m]
    q = float(CONFIG.get("OPTIMIZE", {}).get("q", 0.99))

    rows = []
    for t in err_max_list:
        t = float(t)
        acc = eb <= t
        cov = float(np.mean(acc)) if len(acc) else float("nan")
        if acc.sum() >= 2:
            rmse = float(np.sqrt(mean_squared_error(yt[acc], yp[acc])))
            r2v = float(r2_score(yt[acc], yp[acc]))
            abs_err_acc = np.abs(yt[acc] - yp[acc])
            ae_qq = abs_err_quantile(abs_err_acc, q) if len(abs_err_acc) > 0 else float("nan")
        else:
            rmse = float("nan")
            r2v = float("nan")
            ae_qq = float("nan")
        rows.append({
            "err_max": t,
            "coverage": cov,
            "reject_rate": 1.0 - cov if np.isfinite(cov) else float("nan"),
            "accepted_n": int(acc.sum()),
            "total_n": int(len(acc)),
            "accepted_rmse": rmse,
            "accepted_r2": r2v,
            f"accepted_ae_q{int(q*100)}": ae_qq,
        })
    return pd.DataFrame(rows)

def err_max_for_target_coverage(errb: np.ndarray, target_cov: float) -> float:
    eb = np.asarray(errb, float)
    eb = eb[np.isfinite(eb)]
    if len(eb) == 0:
        return float("nan")
    return float(np.quantile(eb, float(target_cov)))

def run_selective_analysis(y_true: np.ndarray, y_pred: np.ndarray, out_dir: str,
                           tag: str = "valid", extra_prefix: str = "",
                           sigma: Optional[np.ndarray] = None) -> Dict:
    """
    Save selective-prediction analysis and calibration metadata.
    - sigma is None: use the binned conformal fallback.
    - sigma provided: use continuous sigma(x) uncertainty with conformal scaling.
    """
    sel = CONFIG["SELECTIVE"]
    if not sel.get("enable", True):
        return {"enable": False}

    alpha = float(sel.get("alpha", 0.1))
    max_points = int(sel.get("plot_max_points", 5000))
    q = float(CONFIG.get("OPTIMIZE", {}).get("q", 0.99))

    os.makedirs(out_dir, exist_ok=True)

    # calib
    if sigma is not None:
        eps = float((CONFIG.get("SIGMA_UNCERT", {}) or {}).get("eps", 1e-6))
        calib = build_sigma_calibration(y_true=y_true, y_pred=y_pred, sigma=sigma, alpha=alpha, eps=eps)
    else:
        # Backward-compatible binned fallback.
        n_bins = int(sel.get("n_bins", 12))
        min_bin_size = int(sel.get("min_bin_size", 30))
        calib = build_conformal_calibration(y_true=y_true, y_pred=y_pred, alpha=alpha, n_bins=n_bins, min_bin_size=min_bin_size)

    calib_path = os.path.join(out_dir, f"{extra_prefix}selective_calib.json")
    safe_json_dump(calib, calib_path)

    errb = err_bound_for_calib(y_pred, calib, sigma=sigma) if sigma is not None else err_bound_for_preds(y_pred, calib)
    abs_err = np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))
    grid = list(sel.get("err_max_grid", [0.5, 0.8, 1.0]))
    df_tab = selective_metrics_table(y_true, y_pred, errb, grid)

    targets = list(sel.get("target_coverages", [0.9, 0.8, 0.7]))
    cov_map = {str(c): err_max_for_target_coverage(errb, float(c)) for c in targets}

    report = {
        "enable": True,
        "tag": tag,
        "method": str(calib.get("method", "bin")),
        "alpha": float(alpha),
        "calib_path": os.path.basename(calib_path),
        "n_calib": int(calib.get("n", 0)),
        # Bin-only fields; may be None when method != "bin".
        "n_bins": calib.get("n_bins", sel.get("n_bins", None)),
        "min_bin_size": calib.get("min_bin_size", sel.get("min_bin_size", None)),
        "global_q": float(calib.get("global_q", float("nan"))),
        # sigma-only fields
        "q_scale": float(calib.get("q_scale", float("nan"))),
        "targets_err_max": cov_map,
        "grid_table": df_tab.to_dict(orient="records"),
        "abs_err_mean": float(np.nanmean(abs_err)),
        "abs_err_median": float(np.nanmedian(abs_err)),
        "abs_err_q99": abs_err_quantile(abs_err, q),
        "errb_mean": float(np.nanmean(errb)),
        "errb_median": float(np.nanmedian(errb)),
        "affinity": CONFIG.get("AFFINITY", ""),
        "pred_col": CONFIG.get("PRED_COL", ""),
    }

    rep_json = os.path.join(out_dir, f"{extra_prefix}selective_report.json")
    rep_csv = os.path.join(out_dir, f"{extra_prefix}selective_report.csv")
    safe_json_dump(report, rep_json)
    df_tab.to_csv(rep_csv, index=False)

    plot_abs_error_hist(abs_err, os.path.join(out_dir, f"{extra_prefix}abs_err_hist.png"), title=f"Abs Error Histogram ({tag})")

    ebv = np.asarray(errb, float)
    ebv = ebv[np.isfinite(ebv)]
    if len(ebv) >= 20:
        thr = np.quantile(ebv, np.linspace(0.05, 0.95, 19))
        covs = [(ebv <= t).mean() for t in thr]
        plot_curve(thr, covs, os.path.join(out_dir, f"{extra_prefix}coverage_curve.png"),
                   title=f"Coverage vs err_max ({tag})", xlabel="err_max", ylabel="coverage")

    try:
        thr2 = np.quantile(ebv, np.linspace(0.10, 0.95, 18)) if len(ebv) else []
        rmses = []
        q99s = []
        yt = np.asarray(y_true, float)
        yp = np.asarray(y_pred, float)
        m = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(errb)
        yt, yp, eb2 = yt[m], yp[m], np.asarray(errb, float)[m]
        for t in thr2:
            acc = eb2 <= t
            if acc.sum() >= 2:
                rmses.append(float(np.sqrt(mean_squared_error(yt[acc], yp[acc]))))
                abs_err_acc = np.abs(yt[acc] - yp[acc])
                q99s.append(abs_err_quantile(abs_err_acc, q) if len(abs_err_acc) > 0 else float("nan"))
            else:
                rmses.append(float("nan"))
                q99s.append(float("nan"))
        if len(thr2) >= 5:
            plot_curve(thr2, rmses, os.path.join(out_dir, f"{extra_prefix}accepted_rmse_curve.png"),
                       title=f"Accepted RMSE vs err_max ({tag})", xlabel="err_max", ylabel="RMSE(accepted)")
            plot_curve(thr2, q99s, os.path.join(out_dir, f"{extra_prefix}accepted_ae_q{int(q*100)}_curve.png"),
                       title=f"Accepted AE_Q{int(q*100)} vs err_max ({tag})", xlabel="err_max", ylabel=f"AE_Q{int(q*100)}(accepted)")
    except Exception:
        pass

    plot_interval_scatter(y_true, y_pred, errb, os.path.join(out_dir, f"{extra_prefix}interval_vis.png"), max_points=max_points)

    log(f"[Selective] calib={calib_path} report={rep_json} table={rep_csv}")
    return {
        "enable": True,
        "calib_json": calib_path,
        "report_json": rep_json,
        "report_csv": rep_csv,
        "plots": {
            "abs_err_hist": f"{extra_prefix}abs_err_hist.png",
            "coverage_curve": f"{extra_prefix}coverage_curve.png",
            "accepted_rmse_curve": f"{extra_prefix}accepted_rmse_curve.png",
            f"accepted_ae_q{int(q*100)}_curve": f"{extra_prefix}accepted_ae_q{int(q*100)}_curve.png",
            "interval_vis": f"{extra_prefix}interval_vis.png",
        },
        "targets_err_max": cov_map,
    }


# =============================================================================
# Split helpers for random, cold-compound, cold-pair, and cold-target settings
# =============================================================================

def make_groups_optimized(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return group keys for the requested split mode."""
    stats: Dict[str, Any] = {}
    mode = (CONFIG.get("SPLIT_MODE") or "random").lower()
    unified = bool(CONFIG.get("AFFINITY_UNIFIED", {}).get("enable", False))

    # ----- Cold-pair split: group by pair UID -----
    if mode == "pair" and "_pair_uid" in df.columns:
        group_ids, uniques = pd.factorize(df["_pair_uid"].values)
        stats["n_groups"] = len(uniques)
        stats["group_type"] = "pair_uid"
        stats["mode"] = "pair"
        log(f"[Grouping] pair mode uses _pair_uid; n_groups={len(uniques)}")
        return group_ids.astype(np.int32), stats

    # ----- Cold-compound split: group by normalized SMILES -----
    if mode == "compound":
        group_ids, uniques = pd.factorize(df["smiles_norm"].values)
        stats["n_groups"] = len(uniques)
        stats["group_type"] = "compound"
        stats["mode"] = "compound"
        log(f"[Grouping] compound mode; n_groups={len(uniques)}")
        return group_ids.astype(np.int32), stats

    # ----- Cold-target split: group by cleaned protein sequence -----
    if mode == "protein":
        group_ids, uniques = pd.factorize(df["protein_clean"].values)
        stats["n_groups"] = len(uniques)
        stats["group_type"] = "protein"
        stats["mode"] = "protein"
        log(f"[Grouping] protein mode; n_groups={len(uniques)}")
        return group_ids.astype(np.int32), stats

    # ----- Random split: row-level split without grouping -----
    stats["n_groups"] = len(df)
    stats["group_type"] = "random_row"
    stats["mode"] = "random"
    log("[Grouping] random mode uses row-level random split")
    return np.arange(len(df), dtype=np.int32), stats

def group_stratified_split_optimized(
    groups: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int,
    max_groups_for_qcut: int = 50000
) -> Tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(groups, dtype=np.int32)
    y = np.asarray(y, dtype=np.float32)

    start_time = time.time()

    n_groups_total = int(groups.max()) + 1
    group_counts = np.bincount(groups, minlength=n_groups_total)
    valid_groups = np.where(group_counts > 0)[0]
    n_groups = len(valid_groups)

    if n_groups == 0:
        raise ValueError("No valid groups were found")

    if n_groups > 1000:
        sort_idx = np.argsort(groups)
        groups_sorted = groups[sort_idx]
        y_sorted = y[sort_idx]
        group_boundaries = np.concatenate([
            [0],
            np.where(np.diff(groups_sorted))[0] + 1,
            [len(groups_sorted)]
        ])
        group_medians = np.zeros(n_groups, dtype=np.float32)
        for i in range(n_groups):
            start = group_boundaries[i]
            end = group_boundaries[i + 1]
            if end - start > 0:
                if end - start > 10000:
                    sample = y_sorted[start:end][::100]
                    group_medians[i] = np.median(sample)
                else:
                    group_medians[i] = np.median(y_sorted[start:end])
    else:
        group_medians = np.zeros(n_groups, dtype=np.float32)
        for i, g in enumerate(valid_groups):
            mask = groups == g
            group_medians[i] = np.median(y[mask])

    n_bins = 5
    if n_groups > max_groups_for_qcut:
        percentiles = np.percentile(group_medians, np.linspace(0, 100, n_bins + 1))
        group_bins = np.zeros(n_groups, dtype=np.int32)
        for i in range(n_bins):
            if i < n_bins - 1:
                mask = (group_medians >= percentiles[i]) & (group_medians < percentiles[i + 1])
            else:
                mask = (group_medians >= percentiles[i]) & (group_medians <= percentiles[i + 1])
            group_bins[mask] = i
    else:
        try:
            group_bins = pd.qcut(pd.Series(group_medians), q=n_bins, labels=False, duplicates='drop').values
        except Exception:
            group_bins = np.floor(np.linspace(0, n_bins, n_groups, endpoint=False)).astype(int)[:n_groups]

    n_test = max(1, int(round(n_groups * float(test_size))))
    rng = np.random.default_rng(int(seed))

    test_group_indices = set()
    for b in range(n_bins):
        bin_idx = np.where(group_bins == b)[0]
        m = len(bin_idx)
        if m == 0:
            continue
        take = max(1, int(round(m * float(test_size))))
        take = min(take, m)
        sel = rng.choice(bin_idx, size=take, replace=False)
        for idx in sel:
            test_group_indices.add(valid_groups[idx])

    if len(test_group_indices) > n_test:
        test_group_indices = set(rng.choice(list(test_group_indices), size=n_test, replace=False))
    elif len(test_group_indices) < n_test and len(valid_groups) > len(test_group_indices):
        remain = [g for g in valid_groups if g not in test_group_indices]
        if remain:
            add = rng.choice(remain, size=min(n_test - len(test_group_indices), len(remain)), replace=False).tolist()
            test_group_indices.update(add)

    test_group_array = np.array(list(test_group_indices), dtype=np.int32)
    if n_groups_total < 1000000:
        test_mask = np.isin(groups, test_group_array)
    else:
        test_set = set(test_group_array)
        test_mask = np.fromiter((g in test_set for g in groups), dtype=bool, count=len(groups))

    idx_train = np.where(~test_mask)[0]
    idx_test = np.where(test_mask)[0]

    log(f"[Split] train_groups={n_groups - len(test_group_indices)}, test_groups={len(test_group_indices)}")
    log(f"[Split] train_samples={len(idx_train)}, test_samples={len(idx_test)}")
    log(f"[Split] elapsed={time.time()-start_time:.2f}s")

    return idx_train, idx_test


# =============================================================================
# Feature construction and cache handling
# =============================================================================

def build_cache_dir() -> str:
    base = CONFIG["CACHE"]["dir"] or os.path.join(CONFIG["OUT_DIR"], ".cache")
    os.makedirs(base, exist_ok=True)
    return base

def file_sha(meta_dict: dict) -> str:
    raw = json.dumps(meta_dict, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()

def cache_tag_for_features(df_path: str) -> str:
    st = os.stat(df_path)
    tag_meta = {
        "data_path": str(pathlib.Path(df_path).resolve()),
        "size": int(st.st_size),
        "mtime": int(st.st_mtime),
        "data_sha1": sha1_file(df_path),
        "VECTORIZER_type": CONFIG["VECTORIZER"]["type"],
        "VECTORIZER_min_df": CONFIG["VECTORIZER"]["min_df"],
        "VECTORIZER_max_features": (CONFIG["VECTORIZER"]["max_features_ligand"],
                                    CONFIG["VECTORIZER"]["max_features_protein"]),
        "VECTORIZER_norm": CONFIG["VECTORIZER"]["norm"],
        "SMILES_NORM": CONFIG["SMILES_NORM"],
        "SMILES_BPE": CONFIG.get("SMILES_BPE", {}),
        "PROTEIN_BPE": CONFIG.get("PROTEIN_BPE", {}),
        "AFFINITY_UNIFIED": CONFIG.get("AFFINITY_UNIFIED", {}),
        "version": "v7.0-complete-fixed",
    }
    return file_sha(tag_meta)[:16]

def load_or_make_features(df: pd.DataFrame, mask: np.ndarray) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray, str]:
    print_step("Feature processing", "load or build features")
    cache_root = build_cache_dir()
    tag = cache_tag_for_features(CONFIG["DATA_FILE"])
    cdir = os.path.join(cache_root, f"feat_{tag}")

    Xp = os.path.join(cdir, "Xs.npz")
    yp = os.path.join(cdir, "y.npy")
    wp = os.path.join(cdir, "w.npy")
    vec_prot_pkl = os.path.join(cdir, "vectorizer_protein.pkl")
    vec_lig_pkl = os.path.join(cdir, "vectorizer_ligand.pkl")
    meta_json = os.path.join(cdir, "meta.json")
    bpe_tok_sm_json = os.path.join(cdir, "smiles_bpe_tokenizer.json")
    bpe_tok_pr_json = os.path.join(cdir, "protein_bpe_tokenizer.json")

    overwrite = bool(CONFIG["CACHE"]["overwrite"])
    need_files = [Xp, yp, wp, vec_prot_pkl, vec_lig_pkl, meta_json, bpe_tok_sm_json, bpe_tok_pr_json]

    if CONFIG["CACHE"]["enable"] and (not overwrite) and all(os.path.exists(p) for p in need_files):
        log(f"[Cache] Loaded feature cache: {cdir}")
        Xs = sparse.load_npz(Xp)
        y_raw = np.load(yp)
        w = np.load(wp)
        return Xs, y_raw, w, cdir

    log("[Cache] Building new features...")
    vc = CONFIG["VECTORIZER"]
    vec_type = (vc.get("type") or "tfidf").lower().strip()
    if vec_type not in ("tfidf", "count"):
        raise ValueError(f"Unknown vectorizer type: {vec_type}")

    dfv = df.loc[mask].reset_index(drop=True)

    lig_col = "smiles_norm" if "smiles_norm" in dfv.columns else CONFIG["COL_SMILES"]
    pro_col = "protein_clean" if "protein_clean" in dfv.columns else CONFIG["COL_PROTEIN"]
    
    lig_raw = [clean_smiles(s) for s in tqdm(dfv[lig_col].astype(str).tolist(), desc=f"Prepare SMILES({lig_col})")]
    prot_raw = [clean_protein(s) for s in tqdm(dfv[pro_col].astype(str).tolist(), desc=f"Prepare protein({pro_col})")]
    bpe_sm_cfg = CONFIG.get("SMILES_BPE", {}) or {}
    bpe_pr_cfg = CONFIG.get("PROTEIN_BPE", {}) or {}

    _require_tokenizers()
    os.makedirs(cdir, exist_ok=True)

    log(f"[BPE] Training SMILES BPE tokenizer (vocab={bpe_sm_cfg.get('vocab_size',16000)})")
    train_bpe_tokenizer_generic(lig_raw, bpe_sm_cfg, bpe_tok_sm_json, "SMILES-BPE")
    tok_sm = Tokenizer.from_file(bpe_tok_sm_json)

    log(f"[BPE] Training protein BPE tokenizer (vocab={bpe_pr_cfg.get('vocab_size',16000)})")
    train_bpe_tokenizer_generic(prot_raw, bpe_pr_cfg, bpe_tok_pr_json, "Protein-BPE")
    tok_pr = Tokenizer.from_file(bpe_tok_pr_json)

    lig_docs = [smiles_to_bpe_doc(tok_sm, s) for s in tqdm(lig_raw, desc="SMILES->BPE")]
    prot_docs = [protein_to_bpe_doc(tok_pr, s) for s in tqdm(prot_raw, desc="Protein->BPE")]

    lig_ng = tuple(int(x) for x in bpe_sm_cfg.get("bpe_ngram_range", [1, 1]))
    pr_ng = tuple(int(x) for x in bpe_pr_cfg.get("bpe_ngram_range", [1, 1]))

    if vec_type == "tfidf":
        vect_lig = TfidfVectorizer(
            analyzer="word", tokenizer=str.split,
            ngram_range=lig_ng, lowercase=False,
            min_df=int(vc["min_df"]), max_features=int(vc["max_features_ligand"]),
            norm=vc.get("norm", "l2"), use_idf=True, smooth_idf=True,
            sublinear_tf=bool(vc.get("sublinear_tf", False)),
        )
        vect_pr = TfidfVectorizer(
            analyzer="word", tokenizer=str.split,
            ngram_range=pr_ng, lowercase=False,
            min_df=int(vc["min_df"]), max_features=int(vc["max_features_protein"]),
            norm=vc.get("norm", "l2"), use_idf=True, smooth_idf=True,
            sublinear_tf=bool(vc.get("sublinear_tf", False)),
        )
    else:
        vect_lig = CountVectorizer(
            analyzer="word", tokenizer=str.split,
            ngram_range=lig_ng, lowercase=False,
            min_df=int(vc["min_df"]), max_features=int(vc["max_features_ligand"]),
        )
        vect_pr = CountVectorizer(
            analyzer="word", tokenizer=str.split,
            ngram_range=pr_ng, lowercase=False,
            min_df=int(vc["min_df"]), max_features=int(vc["max_features_protein"]),
        )

    L = vect_lig.fit_transform(lig_docs)
    P = vect_pr.fit_transform(prot_docs)

    if vec_type == "count" and vc.get("norm", None):
        L = sp_normalize(L, norm=vc["norm"], copy=False)
        P = sp_normalize(P, norm=vc["norm"], copy=False)

    blocks = [L, P]

    if use_affinity_type_feature():
        log("[Features] Adding affinity_type one-hot features")
        X_aff = affinity_type_onehot(dfv)
        blocks.append(X_aff)
    elif CONFIG.get("AFFINITY_UNIFIED", {}).get("enable", False):
        log("[Features] Unified mixed-endpoint baseline: affinity_type one-hot is DISABLED")

    Xs = sparse.hstack(blocks, format="csr", dtype=np.float32)

    if CONFIG.get("AFFINITY_UNIFIED", {}).get("enable", False):
        ycol = CONFIG["AFFINITY_UNIFIED"]["keep_target_col_name"]
        wcol = CONFIG["AFFINITY_UNIFIED"]["keep_weight_col_name"]
        y_raw = pd.to_numeric(dfv[ycol], errors="coerce").to_numpy(dtype=np.float32)
        w = pd.to_numeric(dfv[wcol], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32, copy=True)
        w[~np.isfinite(w) | (w <= 0)] = 1.0
    else:
        y_raw = pd.to_numeric(dfv[CONFIG["COL_TARGET"]], errors="coerce").to_numpy(dtype=np.float32)
        wcol = CONFIG["COL_WEIGHT"]
        if wcol in dfv.columns:
            w = pd.to_numeric(dfv[wcol], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32, copy=True)
            w[~np.isfinite(w) | (w <= 0)] = 1.0
        else:
            w = np.ones(len(dfv), dtype=np.float32)

    sparse.save_npz(Xp, Xs)
    np.save(yp, y_raw)
    np.save(wp, w)
    joblib.dump(vect_pr, vec_prot_pkl)
    joblib.dump(vect_lig, vec_lig_pkl)

    meta = {
        "tag": tag,
        "data_sha1": sha1_file(CONFIG["DATA_FILE"]),
        "VECTORIZER": {k: v for k, v in vc.items()},
        "SMILES_BPE": bpe_sm_cfg,
        "PROTEIN_BPE": bpe_pr_cfg,
        "AFFINITY_UNIFIED": CONFIG.get("AFFINITY_UNIFIED", {}),
        "shape": [int(Xs.shape[0]), int(Xs.shape[1])],
        "vec_type": vec_type,
        "ligand_bpe_ngram_range": list(lig_ng),
        "protein_bpe_ngram_range": list(pr_ng),
        "has_affinity_type_feature": use_affinity_type_feature(),
    }
    safe_json_dump(meta, meta_json)

    log(f"[Cache] Feature cache completed: {cdir} shape={Xs.shape}")
    return Xs, y_raw, w, cdir


def make_fixed_train_valid_split(df: pd.DataFrame, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """Create the split before any data-dependent representation is fitted."""
    df_all = df.loc[mask].reset_index(drop=True)
    ycol = (CONFIG["AFFINITY_UNIFIED"]["keep_target_col_name"]
            if CONFIG.get("AFFINITY_UNIFIED", {}).get("enable", False)
            else CONFIG["COL_TARGET"])
    y_all = pd.to_numeric(df_all[ycol], errors="coerce").to_numpy(dtype=np.float32)
    all_idx = np.arange(len(y_all), dtype=np.int64)
    mode = (CONFIG.get("SPLIT_MODE") or "random").lower()
    if mode in ("compound", "protein", "pair"):
        groups, _ = make_groups_optimized(df_all)
        idx_tr, idx_va = group_stratified_split_optimized(
            groups, y_all, float(CONFIG["TEST_SIZE"]), int(CONFIG["SEED"])
        )
    else:
        idx_tr, idx_va = train_test_split(
            all_idx, test_size=float(CONFIG["TEST_SIZE"]), shuffle=True,
            random_state=int(CONFIG["SEED"])
        )
    idx_tr, idx_va = np.asarray(idx_tr, dtype=np.int64), np.asarray(idx_va, dtype=np.int64)
    if np.intersect1d(idx_tr, idx_va).size or len(idx_tr) + len(idx_va) != len(df_all):
        raise RuntimeError("Invalid train/validation partition")
    return idx_tr, idx_va, df_all, y_all


def _train_only_cache_tag(df_path: str, idx_tr: np.ndarray, idx_va: np.ndarray) -> str:
    """Fingerprint both representation settings and the exact split."""
    st = os.stat(df_path)
    meta = {
        "fit_scope": "training_subset_only", "protocol_version": "leakage-controlled-v1",
        "data_path": str(pathlib.Path(df_path).resolve()), "data_size": int(st.st_size),
        "data_mtime": int(st.st_mtime), "data_sha1": sha1_file(df_path),
        "split_mode": str(CONFIG["SPLIT_MODE"]), "test_size": float(CONFIG["TEST_SIZE"]),
        "seed": int(CONFIG["SEED"]),
        "train_index_sha1": hashlib.sha1(np.ascontiguousarray(idx_tr).tobytes()).hexdigest(),
        "valid_index_sha1": hashlib.sha1(np.ascontiguousarray(idx_va).tobytes()).hexdigest(),
        "VECTORIZER": CONFIG["VECTORIZER"], "SMILES_BPE": CONFIG.get("SMILES_BPE", {}),
        "PROTEIN_BPE": CONFIG.get("PROTEIN_BPE", {}),
        "AFFINITY_UNIFIED": CONFIG.get("AFFINITY_UNIFIED", {}),
    }
    return file_sha(meta)[:16]


def load_or_make_features_train_only(
    df: pd.DataFrame, mask: np.ndarray, idx_tr: np.ndarray, idx_va: np.ndarray,
    df_all: pd.DataFrame, y_raw: np.ndarray
) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray, str]:
    """Fit BPE/vectorizers on training rows only and transform validation rows."""
    print_step("Leakage-controlled features", "fit=train only; validation=transform only")
    tag = _train_only_cache_tag(CONFIG["DATA_FILE"], idx_tr, idx_va)
    cdir = os.path.join(build_cache_dir(), f"feat_train_only_{tag}")
    paths = {
        "X": os.path.join(cdir, "Xs.npz"), "y": os.path.join(cdir, "y.npy"),
        "w": os.path.join(cdir, "w.npy"), "meta": os.path.join(cdir, "meta.json"),
        "vl": os.path.join(cdir, "vectorizer_ligand.pkl"),
        "vp": os.path.join(cdir, "vectorizer_protein.pkl"),
        "ts": os.path.join(cdir, "smiles_bpe_tokenizer.json"),
        "tp": os.path.join(cdir, "protein_bpe_tokenizer.json"),
        "itr": os.path.join(cdir, "idx_train.npy"), "iva": os.path.join(cdir, "idx_valid.npy"),
    }
    if (CONFIG["CACHE"]["enable"] and not CONFIG["CACHE"]["overwrite"]
            and all(os.path.exists(p) for p in paths.values())):
        meta = load_json(paths["meta"])
        if meta.get("fit_scope") != "training_subset_only":
            raise RuntimeError("Refusing cache without train-only fit provenance")
        if not np.array_equal(np.load(paths["itr"]), idx_tr) or not np.array_equal(np.load(paths["iva"]), idx_va):
            raise RuntimeError("Cached split indices do not match the current split")
        log(f"[Cache] Loaded leakage-controlled cache: {cdir}")
        return sparse.load_npz(paths["X"]), np.load(paths["y"]), np.load(paths["w"]), cdir

    os.makedirs(cdir, exist_ok=True)
    vc = CONFIG["VECTORIZER"]
    vec_type = str(vc.get("type", "tfidf")).lower().strip()
    if vec_type not in ("tfidf", "count"):
        raise ValueError(f"Unknown vectorizer type: {vec_type}")
    lig_col = "smiles_norm" if "smiles_norm" in df_all.columns else CONFIG["COL_SMILES"]
    pro_col = "protein_clean" if "protein_clean" in df_all.columns else CONFIG["COL_PROTEIN"]
    lig_raw = [clean_smiles(s) for s in tqdm(df_all[lig_col].astype(str), desc=f"Prepare SMILES({lig_col})")]
    prot_raw = [clean_protein(s) for s in tqdm(df_all[pro_col].astype(str), desc=f"Prepare protein({pro_col})")]
    lig_train, prot_train = [lig_raw[i] for i in idx_tr], [prot_raw[i] for i in idx_tr]
    bpe_sm_cfg, bpe_pr_cfg = CONFIG.get("SMILES_BPE", {}) or {}, CONFIG.get("PROTEIN_BPE", {}) or {}
    _require_tokenizers()
    log(f"[Leakage control] Training SMILES BPE on {len(idx_tr)} training rows only")
    train_bpe_tokenizer_generic(lig_train, bpe_sm_cfg, paths["ts"], "SMILES-BPE-train-only")
    log(f"[Leakage control] Training protein BPE on {len(idx_tr)} training rows only")
    train_bpe_tokenizer_generic(prot_train, bpe_pr_cfg, paths["tp"], "Protein-BPE-train-only")
    tok_sm, tok_pr = Tokenizer.from_file(paths["ts"]), Tokenizer.from_file(paths["tp"])
    lig_docs = [smiles_to_bpe_doc(tok_sm, s) for s in tqdm(lig_raw, desc="SMILES->fixed BPE")]
    prot_docs = [protein_to_bpe_doc(tok_pr, s) for s in tqdm(prot_raw, desc="Protein->fixed BPE")]
    lig_ng = tuple(int(x) for x in bpe_sm_cfg.get("bpe_ngram_range", [1, 1]))
    pr_ng = tuple(int(x) for x in bpe_pr_cfg.get("bpe_ngram_range", [1, 1]))
    common = dict(analyzer="word", tokenizer=str.split, lowercase=False, min_df=int(vc["min_df"]))
    if vec_type == "tfidf":
        vect_lig = TfidfVectorizer(**common, ngram_range=lig_ng, max_features=int(vc["max_features_ligand"]),
            norm=vc.get("norm", "l2"), use_idf=True, smooth_idf=True,
            sublinear_tf=bool(vc.get("sublinear_tf", False)))
        vect_pr = TfidfVectorizer(**common, ngram_range=pr_ng, max_features=int(vc["max_features_protein"]),
            norm=vc.get("norm", "l2"), use_idf=True, smooth_idf=True,
            sublinear_tf=bool(vc.get("sublinear_tf", False)))
    else:
        vect_lig = CountVectorizer(**common, ngram_range=lig_ng, max_features=int(vc["max_features_ligand"]))
        vect_pr = CountVectorizer(**common, ngram_range=pr_ng, max_features=int(vc["max_features_protein"]))
    # The only fit operations use training documents.
    L_tr = vect_lig.fit_transform([lig_docs[i] for i in idx_tr])
    P_tr = vect_pr.fit_transform([prot_docs[i] for i in idx_tr])
    # Validation uses the frozen training-derived mapping.
    L_va = vect_lig.transform([lig_docs[i] for i in idx_va])
    P_va = vect_pr.transform([prot_docs[i] for i in idx_va])
    if vec_type == "count" and vc.get("norm"):
        L_tr, P_tr = sp_normalize(L_tr, norm=vc["norm"], copy=False), sp_normalize(P_tr, norm=vc["norm"], copy=False)
        L_va, P_va = sp_normalize(L_va, norm=vc["norm"], copy=False), sp_normalize(P_va, norm=vc["norm"], copy=False)
    tr_blocks, va_blocks = [L_tr, P_tr], [L_va, P_va]
    if use_affinity_type_feature():
        tr_blocks.append(affinity_type_onehot(df_all.iloc[idx_tr]))
        va_blocks.append(affinity_type_onehot(df_all.iloc[idx_va]))
    X_tr = sparse.hstack(tr_blocks, format="csr", dtype=np.float32)
    X_va = sparse.hstack(va_blocks, format="csr", dtype=np.float32)
    order = np.concatenate([idx_tr, idx_va])
    stacked = sparse.vstack([X_tr, X_va], format="csr")
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.arange(len(order))
    Xs = stacked[inverse]
    wcol = (CONFIG["AFFINITY_UNIFIED"]["keep_weight_col_name"]
            if CONFIG.get("AFFINITY_UNIFIED", {}).get("enable", False) else CONFIG["COL_WEIGHT"])
    if wcol in df_all.columns:
        w = pd.to_numeric(df_all[wcol], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32, copy=True)
        w[~np.isfinite(w) | (w <= 0)] = 1.0
    else:
        w = np.ones(len(df_all), dtype=np.float32)
    sparse.save_npz(paths["X"], Xs); np.save(paths["y"], y_raw); np.save(paths["w"], w)
    np.save(paths["itr"], idx_tr); np.save(paths["iva"], idx_va)
    joblib.dump(vect_lig, paths["vl"]); joblib.dump(vect_pr, paths["vp"])
    safe_json_dump({
        "tag": tag, "protocol_version": "leakage-controlled-v1",
        "fit_scope": "training_subset_only", "validation_scope": "transform_only",
        "split_mode": str(CONFIG["SPLIT_MODE"]), "test_size": float(CONFIG["TEST_SIZE"]),
        "seed": int(CONFIG["SEED"]), "n_train": int(len(idx_tr)), "n_valid": int(len(idx_va)),
        "train_index_sha1": hashlib.sha1(np.ascontiguousarray(idx_tr).tobytes()).hexdigest(),
        "valid_index_sha1": hashlib.sha1(np.ascontiguousarray(idx_va).tobytes()).hexdigest(),
        "shape": [int(Xs.shape[0]), int(Xs.shape[1])], "vec_type": vec_type,
        "has_affinity_type_feature": use_affinity_type_feature(),
        "unseen_validation_tokens_are_unk_or_ignored": True,
    }, paths["meta"])
    log(f"[Leakage control] Feature cache completed: {cdir} shape={Xs.shape}")
    return Xs, y_raw, w, cdir


# =============================================================================
# Device selection and optional GPU monitoring
# =============================================================================

class GpuProcPeak:
    def __init__(self, pid: int = None, interval: float = 0.2):
        self.pid = pid or os.getpid()
        self.interval = interval
        self._peak_bytes = 0
        self._stop = threading.Event()
        self._th = None
        self._enabled = False
        self._inited = False
        if nvml is not None:
            try:
                nvml.nvmlInit()
                self._inited = True
            except:
                pass

    def _poll(self):
        try:
            while not self._stop.is_set():
                cur = 0
                try:
                    ndev = nvml.nvmlDeviceGetCount()
                    for i in range(ndev):
                        h = nvml.nvmlDeviceGetHandleByIndex(i)
                        procs = nvml.nvmlDeviceGetComputeRunningProcesses(h)
                        for p in procs:
                            if int(p.pid) == self.pid:
                                used = getattr(p, "usedGpuMemory", 0)
                                cur += int(used)
                except:
                    pass
                self._peak_bytes = max(self._peak_bytes, cur)
                time.sleep(self.interval)
        except:
            pass

    def start(self):
        if not self._inited:
            return
        self._enabled = True
        self._stop.clear()
        self._th = threading.Thread(target=self._poll, daemon=True)
        self._th.start()

    def stop(self):
        if not self._enabled:
            return
        self._stop.set()
        if self._th:
            self._th.join(timeout=1.0)
        self._enabled = False

    @property
    def peak_bytes(self) -> Optional[int]:
        return int(self._peak_bytes) if self._peak_bytes > 0 else None

def bytes_to_mb(x: Optional[int]) -> Optional[float]:
    return None if x is None else x / (1024.0 ** 2)

def estimate_csr_bytes(csr: sparse.csr_matrix) -> int:
    b = 0
    if hasattr(csr, "data"): b += csr.data.nbytes
    if hasattr(csr, "indices"): b += csr.indices.nbytes
    if hasattr(csr, "indptr"): b += csr.indptr.nbytes
    return b

def pick_lgbm_device(params: dict) -> Tuple[dict, str]:
    print_step("Device selection", "detect available compute device")
    if lgb is None:
        raise ImportError("lightgbm is required")

    force = (os.environ.get("LGBM_FORCE_DEVICE") or (CONFIG.get("FORCE_DEVICE") or "")).lower()
    p = params.copy()
    p.setdefault("force_col_wise", True)

    if force in ("cpu", "gpu"):
        dev = "GPU" if force == "gpu" else "CPU"
        p["device"] = force
        p["device_type"] = force
        log(f"[Device] Forced device: {dev}")
        return p, dev

    log("[Device] Probing GPU availability...")
    probe_code = textwrap.dedent("""
        import json, numpy as np, lightgbm as lgb
        from scipy import sparse
        X = sparse.csr_matrix(np.zeros((8,2), np.float32))
        y = np.zeros(8, np.float32)
        d = lgb.Dataset(X, label=y, free_raw_data=False)
        params = json.loads(%(PARAMS)s)
        params['device']='gpu'; params['device_type']='gpu'
        lgb.train(params, d, num_boost_round=1, valid_sets=[], valid_names=[])
        print("GPU_OK")
    """).replace("%(PARAMS)s", "r'''"+json.dumps(p)+"'''")

    try:
        cp = subprocess.run([sys.executable, "-c", probe_code],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=float(CONFIG.get("GPU_PROBE_TIMEOUT_SEC", 12.0)), text=True)
        if "GPU_OK" in (cp.stdout or ""):
            log("[Device] GPU is available; using GPU")
            p["device"] = "gpu"; p["device_type"] = "gpu"
            return p, "GPU"
    except Exception:
        pass

    log("[Device] Using CPU")
    p["device"] = "cpu"; p["device_type"] = "cpu"
    return p, "CPU"


# =============================================================================
# Single train/validation run
# =============================================================================

def _train_one_fold(
    fold_dir: str,
    X_tr: sparse.csr_matrix,
    y_tr_raw: np.ndarray,
    df_tr: pd.DataFrame,
    X_va: sparse.csr_matrix,
    y_va_raw: np.ndarray,
    df_va: pd.DataFrame,
    feat_cache_dir: str,
    run_tag: str = "train_test",
) -> Dict:
    os.makedirs(fold_dir, exist_ok=True)

    tr_path = os.path.join(fold_dir, "train.csv")
    va_path = os.path.join(fold_dir, "valid.csv")
    df_tr.to_csv(tr_path, index=False)
    df_va.to_csv(va_path, index=False)

    best_manifest = os.path.join(fold_dir, "best_manifest.json")
    best_model_txt = os.path.join(fold_dir, "best_model.txt")
    scaler_full_pkl = os.path.join(fold_dir, "scaler_full.pkl")
    scaler_pkl = os.path.join(fold_dir, "scaler.pkl")
    stage_hist_csv = os.path.join(fold_dir, "stage_history.csv")
    last_model_txt = os.path.join(fold_dir, "last_checkpoint_model.txt")
    resume_state_json = os.path.join(fold_dir, "resume_state.json")

    print_step(f"{run_tag} data preprocessing", "StandardScaler on y")
    tsc = StandardScaler().fit(y_tr_raw.reshape(-1, 1))
    y_tr_sc = tsc.transform(y_tr_raw.reshape(-1, 1)).ravel()

    w_tr = make_train_weights(df_tr, y_tr_raw)

    cap = int(CONFIG.get("TRAIN_METRICS_SAMPLE_CAP", 0) or 0)
    rng_trm = np.random.default_rng(int(CONFIG.get("TRAIN_METRICS_RANDOM_SEED", 2027)))
    if cap > 0 and len(y_tr_raw) > cap:
        tr_idx_for_metrics = rng_trm.choice(len(y_tr_raw), size=cap, replace=False)
        log(f"[StageHistory] TRAIN subsampling: cap={cap}")
    else:
        tr_idx_for_metrics = np.arange(len(y_tr_raw))

    gpu_mon = GpuProcPeak(interval=CONFIG["GPU_SAMPLE_INTERVAL"])
    gpu_mon.start()

    type_order_for_stage = list((CONFIG.get("AFFINITY_UNIFIED", {}) or {}).get("type_order", ["pIC50", "pKi", "pKd"]))
    cols = [
            "stage","iters_done",
            "TR_R","TR_Pearson_p","TR_SpearmanR","TR_Spearman_p","TR_R2","TR_RMSE","TR_MSE","TR_AE_Qq","TR_CI","TR_AUC","TR_AUPR","TR_N",
            "TR_R2m","TR_R2m_prime","TR_R2m_bar","TR_delta_R2m","TR_R0_sq","TR_R0p_sq","TR_k0","TR_k0p",
            "VAL_R","VAL_Pearson_p","VAL_SpearmanR","VAL_Spearman_p","VAL_R2","VAL_RMSE","VAL_MSE","VAL_AE_Qq","VAL_CI","VAL_AUC","VAL_AUPR","VAL_N",
            "VAL_R2m","VAL_R2m_prime","VAL_R2m_bar","VAL_delta_R2m","VAL_R0_sq","VAL_R0p_sq","VAL_k0","VAL_k0p",
        ] + stage_history_by_type_columns(prefixes=["TR", "VAL"], type_order=type_order_for_stage) + [
            "AFFINE_a","AFFINE_b","CALIB_method",
            "stage_time_sec","iters_per_sec","t_1k_sec","throughput_samples_per_sec",
            "cpu_peak_rss_mb","gpu_peak_mem_mb","csr_train_mb","csr_valid_mb",
            "affinity","target_col","weight_col","pred_col",
    ]
    if (not bool(CONFIG.get("RESUME", False))) or (not os.path.exists(stage_hist_csv)):
        with open(stage_hist_csv, "w", newline="", encoding="utf-8") as fcsv:
            csv.DictWriter(fcsv, fieldnames=cols).writeheader()

    print_step(f"{run_tag} model setup", "lgb.Dataset")
    dtrain = lgb.Dataset(X_tr, label=y_tr_sc, weight=w_tr, free_raw_data=False)

    params_used, device_used = pick_lgbm_device(CONFIG["LGBM_PARAMS"].copy())
    params_used["seed"] = int(CONFIG["SEED"])
    nt = int(CONFIG.get("NUM_THREADS", 0) or 0)
    if nt > 0:
        params_used["num_threads"] = nt

    total_iters = int(CONFIG["LGBM_TOTAL_ITERS"])
    stage_iters = int(CONFIG["LGBM_STAGE_ITERS"])
    stages = (total_iters + stage_iters - 1) // stage_iters

    params_used["bin_construct_sample_cnt"] = max(1, min(len(y_tr_raw), 200000))

    calib_method = (CONFIG["CALIBRATION"]["method"] or "none").lower()
    train_cap = int(CONFIG["CALIBRATION"].get("train_sample_cap", 0) or 0)
    clip_range = CONFIG["CALIBRATION"].get("clip", None)

    if calib_method == "train_linear" and train_cap > 0 and len(y_tr_raw) > train_cap:
        tr_idx_for_calib = rng_trm.choice(len(y_tr_raw), size=train_cap, replace=False)
    else:
        tr_idx_for_calib = np.arange(len(y_tr_raw))

    csr_train_mb = bytes_to_mb(estimate_csr_bytes(X_tr))
    csr_valid_mb = bytes_to_mb(estimate_csr_bytes(X_va))

    print_step(f"{run_tag} model training", f"total={total_iters} stage={stages}")
    booster = None
    it_done = 0
    best_by = str(CONFIG.get("OPTIMIZE", {}).get("by", "rmse")).lower()
    best_q = float(CONFIG.get("OPTIMIZE", {}).get("q", 0.99))
    best_metrics_key, best_mode = get_optimize_spec(best_by)
    best_value = float("inf") if best_mode == "min" else -float("inf")

    best_it = 0
    best_it = 0
    best_metrics = None
    best_affine = {"a": 0.0, "b": 1.0, "method": calib_method}

    resume_enabled = bool(CONFIG.get("RESUME", False))
    if resume_enabled and os.path.exists(last_model_txt) and os.path.exists(resume_state_json):
        state = load_json(resume_state_json)
        expected_signature = {
            "data_file": os.path.abspath(CONFIG["DATA_FILE"]),
            "seed": int(CONFIG["SEED"]),
            "split_mode": str(CONFIG["SPLIT_MODE"]),
            "test_size": float(CONFIG["TEST_SIZE"]),
            "type_feature": str((CONFIG.get("AFFINITY_UNIFIED", {}) or {}).get("type_feature", "onehot")),
            "feat_cache_dir": os.path.abspath(feat_cache_dir),
        }
        if state.get("signature") != expected_signature:
            raise RuntimeError(
                "Checkpoint signature does not match the current data/split/feature configuration; "
                "refusing an unsafe resume. Use a new --run_dir for changed settings."
            )
        booster = lgb.Booster(model_file=last_model_txt)
        it_done = int(state.get("iters_done", booster.current_iteration()))
        best_value = float(state.get("best_value", best_value))
        best_it = int(state.get("best_iteration", 0))
        best_metrics = state.get("best_metrics")
        best_affine = state.get("best_affine", best_affine)
        log(f"[Resume] Loaded checkpoint: iters_done={it_done}/{total_iters}, best_iteration={best_it}")
    elif resume_enabled:
        log("[Resume] No completed model stage found; starting from iteration 0")

    signature = {
        "data_file": os.path.abspath(CONFIG["DATA_FILE"]),
        "seed": int(CONFIG["SEED"]),
        "split_mode": str(CONFIG["SPLIT_MODE"]),
        "test_size": float(CONFIG["TEST_SIZE"]),
        "type_feature": str((CONFIG.get("AFFINITY_UNIFIED", {}) or {}).get("type_feature", "onehot")),
        "feat_cache_dir": os.path.abspath(feat_cache_dir),
    }

    completed_stages = int((it_done + stage_iters - 1) // stage_iters) if it_done > 0 else 0
    with tqdm(total=stages, initial=min(completed_stages, stages), desc=f"{run_tag} stages") as pbar:
        while it_done < total_iters:
            st = int(it_done // stage_iters) + 1
            rounds_this_stage = int(min(stage_iters, total_iters - it_done))
            t1 = time.time()
            booster = lgb.train(
                params_used, dtrain,
                num_boost_round=rounds_this_stage,
                init_model=booster,
                valid_sets=[], valid_names=[],
                keep_training_booster=True,
                callbacks=[lgb.log_evaluation(period=int(CONFIG.get("LOG_PERIOD", 100)))]
            )
            it_done += rounds_this_stage
            dur = time.time() - t1

            if nvml is not None:
                time.sleep(0.02)
            gpu_peak_mb = bytes_to_mb(gpu_mon.peak_bytes)

            pva_sc = booster.predict(X_va, num_iteration=it_done)
            pva = (pva_sc * float(tsc.scale_[0])) + float(tsc.mean_[0])

            a, b, used_method = 0.0, 1.0, "none"
            if calib_method == "train_linear":
                ptr_sc_cal = booster.predict(X_tr[tr_idx_for_calib], num_iteration=it_done)
                ptr_cal = (ptr_sc_cal * float(tsc.scale_[0])) + float(tsc.mean_[0])
                a, b = fit_affine(y_tr_raw[tr_idx_for_calib], ptr_cal)
                pva = apply_affine(pva, a, b, clip=clip_range)
                used_method = "train_linear"
            elif calib_method == "valid_linear":
                a, b = fit_affine(y_va_raw, pva)
                pva = apply_affine(pva, a, b, clip=clip_range)
                used_method = "valid_linear"

            m_val_pack = evaluate_metrics_pack(df_va, y_va_raw, pva, CONFIG["STRONG_BINDER_THRESHOLD_PK"], q=best_q)
            m_val = m_val_pack["overall"]
            m_val_by_type = m_val_pack.get("by_type", {}) or {}

            ptr_sc_m = booster.predict(X_tr[tr_idx_for_metrics], num_iteration=it_done)
            ptr_m = (ptr_sc_m * float(tsc.scale_[0])) + float(tsc.mean_[0])
            ptr_m = apply_affine(ptr_m, a, b, clip=clip_range)
            df_tr_metrics = df_tr.iloc[tr_idx_for_metrics].reset_index(drop=True)
            m_tr_pack = evaluate_metrics_pack(df_tr_metrics, y_tr_raw[tr_idx_for_metrics], ptr_m, CONFIG["STRONG_BINDER_THRESHOLD_PK"], q=best_q)
            m_tr = m_tr_pack["overall"]

            iters_per_sec = float(rounds_this_stage / max(dur, 1e-6))
            t_1k = float(1000.0 / iters_per_sec)
            throughput = float(len(y_tr_raw) / max(dur, 1e-6))
            cpu_peak_mb = None
            try:
                if resource is not None:
                    ru = resource.getrusage(resource.RUSAGE_SELF)
                    val = float(ru.ru_maxrss)
                    cpu_peak_mb = val/1024.0 if val > 10_000 else val/(1024.0**2)
            except Exception:
                pass
            if psutil is not None:
                try:
                    rss = psutil.Process(os.getpid()).memory_info().rss
                    cpu_peak_mb = max(cpu_peak_mb or 0.0, bytes_to_mb(rss) or 0.0)
                except Exception:
                    pass

            with open(stage_hist_csv, "a", newline="", encoding="utf-8") as fcsv:
                w = csv.DictWriter(fcsv, fieldnames=cols)
                row = {
                    "stage": st, "iters_done": it_done,
                    "TR_R": m_tr["R"], "TR_Pearson_p": m_tr.get("Pearson_p", np.nan),
                    "TR_SpearmanR": m_tr.get("SpearmanR", np.nan), "TR_Spearman_p": m_tr.get("Spearman_p", np.nan),
                    "TR_R2": m_tr["R2"], "TR_RMSE": m_tr["RMSE"], "TR_MSE": m_tr["MSE"],
                    "TR_AE_Qq": m_tr["AE_Qq"], "TR_CI": m_tr["CI"], "TR_AUC": m_tr["AUC"], "TR_AUPR": m_tr["AUPR"], "TR_N": m_tr["N"],
                    "TR_R2m": m_tr["R2m"], "TR_R2m_prime": m_tr["R2m_prime"], "TR_R2m_bar": m_tr["R2m_bar"], "TR_delta_R2m": m_tr["delta_R2m"],
                    "TR_R0_sq": m_tr["R0_sq"], "TR_R0p_sq": m_tr["R0p_sq"], "TR_k0": m_tr["k0"], "TR_k0p": m_tr["k0p"],

                    "VAL_R": m_val["R"], "VAL_Pearson_p": m_val.get("Pearson_p", np.nan),
                    "VAL_SpearmanR": m_val.get("SpearmanR", np.nan), "VAL_Spearman_p": m_val.get("Spearman_p", np.nan),
                    "VAL_R2": m_val["R2"], "VAL_RMSE": m_val["RMSE"], "VAL_MSE": m_val["MSE"],
                    "VAL_AE_Qq": m_val["AE_Qq"], "VAL_CI": m_val["CI"], "VAL_AUC": m_val["AUC"], "VAL_AUPR": m_val["AUPR"], "VAL_N": m_val["N"],
                    "VAL_R2m": m_val["R2m"], "VAL_R2m_prime": m_val["R2m_prime"], "VAL_R2m_bar": m_val["R2m_bar"], "VAL_delta_R2m": m_val["delta_R2m"],
                    "VAL_R0_sq": m_val["R0_sq"], "VAL_R0p_sq": m_val["R0p_sq"], "VAL_k0": m_val["k0"], "VAL_k0p": m_val["k0p"],

                    "AFFINE_a": a, "AFFINE_b": b, "CALIB_method": used_method,
                    "stage_time_sec": dur, "iters_per_sec": iters_per_sec, "t_1k_sec": t_1k,
                    "throughput_samples_per_sec": throughput,
                    "cpu_peak_rss_mb": cpu_peak_mb, "gpu_peak_mem_mb": gpu_peak_mb,
                    "csr_train_mb": csr_train_mb, "csr_valid_mb": csr_valid_mb,
                    "affinity": CONFIG.get("AFFINITY",""), "target_col": CONFIG.get("COL_TARGET",""),
                    "weight_col": CONFIG.get("COL_WEIGHT",""), "pred_col": CONFIG.get("PRED_COL",""),
                }
                row.update(stage_history_by_type_values("TR", m_tr_pack, type_order=type_order_for_stage))
                row.update(stage_history_by_type_values("VAL", m_val_pack, type_order=type_order_for_stage))
                w.writerow(row)

            current_value = m_val.get(best_metrics_key, float("inf"))
            if not np.isfinite(current_value):
                current_value = (float("inf") if best_mode == "min" else -float("inf"))

            if is_better(float(current_value), float(best_value), best_mode, eps=1e-12):
                best_value = float(current_value)
                best_it = int(it_done)
                best_metrics = m_val
                best_affine = {"a": float(a), "b": float(b), "method": used_method}

                lgb.Booster(model_str=booster.model_to_string()).save_model(best_model_txt, num_iteration=best_it)

                joblib.dump(tsc, scaler_full_pkl)
                joblib.dump({"target_scaler_mean": float(tsc.mean_[0]),
                           "target_scaler_scale": float(tsc.scale_[0])}, scaler_pkl)

                manifest_data = {
                    "best_by": best_by,
                    "best_mode": best_mode,
                    "best_metrics_key": best_metrics_key,
                    "best_q": best_q if best_by == "ae_q" else None,
                    "best_value": best_value,
                    "best_iteration": int(best_it),
                    "best_metrics": best_metrics,
                    "best_metrics_by_type": m_val_by_type,
                    "device": device_used,
                    "params": params_used,
                    "affine": best_affine,
                    "affinity_unified": CONFIG.get("AFFINITY_UNIFIED", {}),
                }
                safe_json_dump(manifest_data, best_manifest)
                log(f"[Best] stage {st}  {best_metrics_key}={best_value:.4f}")

            checkpoint_tmp = last_model_txt + ".tmp"
            booster.save_model(checkpoint_tmp)
            os.replace(checkpoint_tmp, last_model_txt)
            state_tmp = resume_state_json + ".tmp"
            safe_json_dump({
                "signature": signature,
                "iters_done": int(it_done),
                "total_iters": int(total_iters),
                "stage_iters": int(stage_iters),
                "best_value": float(best_value),
                "best_iteration": int(best_it),
                "best_metrics": best_metrics,
                "best_affine": best_affine,
                "complete": bool(it_done >= total_iters),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, state_tmp)
            os.replace(state_tmp, resume_state_json)
            log(f"[Checkpoint] Saved completed stage {st}: iters_done={it_done}/{total_iters}")

            pbar.update(1)

    gpu_mon.stop()
    if best_metrics is None:
        raise RuntimeError("Training failed")

    booster_best = lgb.Booster(model_file=best_model_txt)
    pva_sc = booster_best.predict(X_va)
    pva = (pva_sc * float(tsc.scale_[0])) + float(tsc.mean_[0])
    aff = best_affine
    pva = apply_affine(pva, aff["a"], aff["b"], clip=CONFIG["CALIBRATION"].get("clip", None))

    if CONFIG["PLOT_SCATTER"]:
        plot_scatter(y_va_raw, pva,
                     title=f"LGBM ({run_tag}) best_by={best_by}={best_value:.4f}",
                     out_path=os.path.join(fold_dir, "SCATTER_VALID.png"))

    selective_out_dir = os.path.join(fold_dir, "selective")
    os.makedirs(selective_out_dir, exist_ok=True)
    sel_res = {"enable": False}
    calib_path = None
    sigma_model_path = None
    if CONFIG["SELECTIVE"]["enable"]:
        try:
            sigma_cfg = CONFIG.get("SIGMA_UNCERT", {}) or {}
            if sigma_cfg.get("enable", True):
                # Train the sigma(x) model using training residuals.
                ptr_sc_full = booster_best.predict(X_tr)
                ptr_full = (ptr_sc_full * float(tsc.scale_[0])) + float(tsc.mean_[0])
                ptr_full = apply_affine(ptr_full, aff["a"], aff["b"], clip=CONFIG["CALIBRATION"].get("clip", None))

                sigma_model_path = train_sigma_model_lgbm(
                    X_tr=X_tr, y_true_tr=y_tr_raw, y_pred_tr=ptr_full,
                    w_tr=w_tr, out_dir=selective_out_dir,
                    seed=int(CONFIG.get("SEED", 2025)) + 17, device_used=device_used
                )

                sigma_va = predict_sigma_lgbm(sigma_model_path, X_va, pva)
                sel_res = run_selective_analysis(y_va_raw, pva, selective_out_dir, tag="valid", sigma=sigma_va)
                calib_path = os.path.join(selective_out_dir, "selective_calib.json")
            else:
                # Fallback to binned conformal calibration.
                sel_res = run_selective_analysis(y_va_raw, pva, selective_out_dir, tag="valid")
                calib_path = os.path.join(selective_out_dir, "selective_calib.json")
        except Exception as e:
            log(f"[Selective/sigma] Calibration failed: {e}", "WARNING")

    pkg_dir = export_predict_bundle(
        fold_dir, feat_cache_dir, CONFIG,
        best_manifest, best_model_txt, scaler_full_pkl, scaler_pkl,
        selective_calib_json=calib_path if sel_res.get("enable") else None,
        sigma_model_txt=sigma_model_path if (sigma_model_path and sel_res.get("enable")) else None,
        example_df=df_va
    )

    # ===== Final best model: full train/validation metrics and paper-facing exports =====
    ptr_sc_full = booster_best.predict(X_tr)
    ptr_full = (ptr_sc_full * float(tsc.scale_[0])) + float(tsc.mean_[0])
    ptr_full = apply_affine(ptr_full, aff["a"], aff["b"], clip=CONFIG["CALIBRATION"].get("clip", None))

    q = float(CONFIG.get("OPTIMIZE", {}).get("q", 0.99))
    threshold_pk = float(CONFIG.get("STRONG_BINDER_THRESHOLD_PK", 6.0))

    final_train_pack = evaluate_metrics_pack(df_tr, y_tr_raw, ptr_full, threshold_pk, q=q)
    final_val_pack = evaluate_metrics_pack(df_va, y_va_raw, pva, threshold_pk, q=q)

    sigma_tr = None
    sigma_va = None
    errb_tr = None
    errb_va = None
    if sigma_model_path and calib_path and os.path.exists(sigma_model_path) and os.path.exists(calib_path):
        try:
            calib_obj = load_json(calib_path)
            sigma_tr = predict_sigma_lgbm(sigma_model_path, X_tr, ptr_full)
            sigma_va = predict_sigma_lgbm(sigma_model_path, X_va, pva)
            errb_tr = err_bound_for_calib(ptr_full, calib_obj, sigma=sigma_tr)
            errb_va = err_bound_for_calib(pva, calib_obj, sigma=sigma_va)
        except Exception as e:
            log(f"[PaperMetrics] Probability/interval export failed: {e}", "WARNING")
            sigma_tr = sigma_va = errb_tr = errb_va = None

    train_pred_df = build_prediction_export_df(
        df_tr, y_tr_raw, ptr_full, split_name="train", threshold_pk=threshold_pk,
        sigma=sigma_tr, errb=errb_tr
    )
    val_pred_df = build_prediction_export_df(
        df_va, y_va_raw, pva, split_name="val", threshold_pk=threshold_pk,
        sigma=sigma_va, errb=errb_va
    )
    train_pred_path = os.path.join(fold_dir, "train_predictions_for_paper.csv")
    val_pred_path = os.path.join(fold_dir, "valid_predictions_for_paper.csv")
    train_pred_df.to_csv(train_pred_path, index=False)
    val_pred_df.to_csv(val_pred_path, index=False)

    plot_outputs = {
        "overall": {},
        "by_type": {"train": {}, "val": {}}
    }
    if CONFIG.get("PLOT_SCATTER", True):
        overall_train_scatter = os.path.join(fold_dir, "SCATTER_TRAIN.png")
        overall_val_scatter = os.path.join(fold_dir, "SCATTER_VALID_BEST.png")
        plot_scatter(y_tr_raw, ptr_full,
                     title=f"LGBM ({run_tag}) TRAIN best_iter={best_it}",
                     out_path=overall_train_scatter)
        plot_scatter(y_va_raw, pva,
                     title=f"LGBM ({run_tag}) VALID best_iter={best_it}",
                     out_path=overall_val_scatter)
        overall_train_abs = os.path.join(fold_dir, "ABSERR_HIST_TRAIN.png")
        overall_val_abs = os.path.join(fold_dir, "ABSERR_HIST_VALID.png")
        plot_abs_error_hist(np.abs(np.asarray(y_tr_raw, float) - np.asarray(ptr_full, float)), overall_train_abs,
                            title="Train Absolute Error")
        plot_abs_error_hist(np.abs(np.asarray(y_va_raw, float) - np.asarray(pva, float)), overall_val_abs,
                            title="Valid Absolute Error")
        plot_outputs["overall"] = {
            "scatter_train": os.path.basename(overall_train_scatter) if os.path.exists(overall_train_scatter) else None,
            "scatter_valid": os.path.basename(overall_val_scatter) if os.path.exists(overall_val_scatter) else None,
            "abs_err_hist_train": os.path.basename(overall_train_abs) if os.path.exists(overall_train_abs) else None,
            "abs_err_hist_valid": os.path.basename(overall_val_abs) if os.path.exists(overall_val_abs) else None,
        }
        if errb_tr is not None:
            overall_train_interval = os.path.join(fold_dir, "INTERVAL_TRAIN.png")
            plot_interval_scatter(y_tr_raw, ptr_full, errb_tr, overall_train_interval,
                                  max_points=int(CONFIG.get("SELECTIVE", {}).get("plot_max_points", 5000)))
            if os.path.exists(overall_train_interval):
                plot_outputs["overall"]["interval_train"] = os.path.basename(overall_train_interval)
        if errb_va is not None:
            overall_val_interval = os.path.join(fold_dir, "INTERVAL_VALID.png")
            plot_interval_scatter(y_va_raw, pva, errb_va, overall_val_interval,
                                  max_points=int(CONFIG.get("SELECTIVE", {}).get("plot_max_points", 5000)))
            if os.path.exists(overall_val_interval):
                plot_outputs["overall"]["interval_valid"] = os.path.basename(overall_val_interval)

        by_type_dir = os.path.join(fold_dir, "plots_by_type")
        plot_outputs["by_type"]["train"] = export_per_type_plots(
            df_tr, y_tr_raw, ptr_full, by_type_dir, split_name="train",
            errb=errb_tr, max_points=int(CONFIG.get("SELECTIVE", {}).get("plot_max_points", 5000))
        )
        plot_outputs["by_type"]["val"] = export_per_type_plots(
            df_va, y_va_raw, pva, by_type_dir, split_name="val",
            errb=errb_va, max_points=int(CONFIG.get("SELECTIVE", {}).get("plot_max_points", 5000))
        )

    paper_metrics_flat = {
        "run_tag": run_tag,
        "best_iteration": int(best_it),
        "best_by": best_by,
        "best_value": best_value,
    }
    paper_metrics_flat.update(flatten_metrics_for_paper("train", final_train_pack))
    paper_metrics_flat.update(flatten_metrics_for_paper("val", final_val_pack, alias_splits=["test"]))
    paper_metrics_json = {
        "run_tag": run_tag,
        "best_iteration": int(best_it),
        "best_by": best_by,
        "best_value": best_value,
        "train": final_train_pack,
        "val": final_val_pack,
        "test": final_val_pack,
    }
    safe_json_dump(paper_metrics_json, os.path.join(fold_dir, "paper_metrics_by_split_and_type.json"))
    pd.DataFrame([paper_metrics_flat]).to_csv(os.path.join(fold_dir, "paper_metrics_flat.csv"), index=False)

    fold_summary = {
        "run_tag": run_tag,
        "fold_dir": fold_dir,
        "device": device_used,
        "best_by": best_by,
        "best_value": best_value,
        "best_iteration": best_it,
        "best_metrics": best_metrics,
        "best_metrics_by_type": m_val_by_type,
        "final_train_metrics": final_train_pack["overall"],
        "final_train_metrics_by_type": final_train_pack.get("by_type", {}),
        "final_val_metrics": final_val_pack["overall"],
        "final_val_metrics_by_type": final_val_pack.get("by_type", {}),
        "paper_metrics_flat": paper_metrics_flat,
        "paper_outputs": {
            "paper_metrics_json": "paper_metrics_by_split_and_type.json",
            "paper_metrics_csv": "paper_metrics_flat.csv",
            "train_predictions_csv": os.path.basename(train_pred_path),
            "valid_predictions_csv": os.path.basename(val_pred_path),
            "plots": plot_outputs,
        },
        "affine": best_affine,
        "selective": sel_res,
        "predict_bundle_dir": pkg_dir,
    }
    safe_json_dump(fold_summary, os.path.join(fold_dir, "fold_summary.json"))

    q = float(CONFIG.get("OPTIMIZE", {}).get("q", 0.99))
    log(f"[{run_tag}] valid: RMSE={best_metrics['RMSE']:.4f} AE_Q{int(q*100)}={best_metrics['AE_Qq']:.4f} R2={best_metrics['R2']:.4f}")

    return fold_summary

# =============================================================================
# train_test run
# =============================================================================

def train_test_run(Xs, y_raw, w, df: pd.DataFrame, mask: np.ndarray, feat_cache_dir: str) -> Dict:
    rs = int(CONFIG["SEED"])
    test_size = float(CONFIG["TEST_SIZE"])
    mode = (CONFIG.get("SPLIT_MODE") or "random").lower()
    unified = bool(CONFIG.get("AFFINITY_UNIFIED", {}).get("enable", False))

    print_step("train_test split", f"split_mode={mode} test_size={test_size}")

    all_idx = np.arange(len(y_raw))
    df_all = df.loc[mask].reset_index(drop=True)

    if mode in ["compound", "protein", "pair"]:
        split_start = time.time()
        groups, group_stats = make_groups_optimized(df_all)
        log(f"[Split] Group construction completed: n_groups={group_stats['n_groups']}, elapsed={time.time()-split_start:.2f}s")

        split_start2 = time.time()
        idx_tr, idx_va = group_stratified_split_optimized(groups, y_raw, test_size, rs)
        log(f"[Split] Group sampling completed, elapsed={time.time()-split_start2:.2f}s")
    else:
        idx_tr, idx_va = train_test_split(all_idx, test_size=test_size, shuffle=True, random_state=rs)

    df_tr = df_all.iloc[idx_tr].reset_index(drop=True)
    df_va = df_all.iloc[idx_va].reset_index(drop=True)

    X_tr, X_va = Xs[idx_tr], Xs[idx_va]
    y_tr_raw2, y_va_raw = y_raw[idx_tr], y_raw[idx_va]

    run_dir = os.path.join(CONFIG["OUT_DIR"], "train_test")
    os.makedirs(run_dir, exist_ok=True)

    fold_summary = _train_one_fold(
        fold_dir=run_dir,
        X_tr=X_tr, y_tr_raw=y_tr_raw2, df_tr=df_tr,
        X_va=X_va, y_va_raw=y_va_raw, df_va=df_va,
        feat_cache_dir=feat_cache_dir,
        run_tag="train_test",
    )

    summary = {
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "mode": "train_test",
        "run_dir": run_dir,
        "affinity_unified": CONFIG.get("AFFINITY_UNIFIED", {}),
        "optimize": CONFIG.get("OPTIMIZE", {}),
        "config": {"SPLIT_MODE": mode, "TEST_SIZE": test_size},
        "fold_summary": fold_summary,
    }
    safe_json_dump(summary, os.path.join(run_dir, "train_test_summary.json"))

    return summary


# =============================================================================
# Build self-contained prediction bundle
# =============================================================================

def export_predict_bundle(
    run_dir: str, feat_cache_dir: str, cfg: dict,
    best_manifest_path: str, best_model_txt: str,
    scaler_full_pkl: str, scaler_pkl: str,
    selective_calib_json: Optional[str] = None,
    sigma_model_txt: Optional[str] = None,
    example_df: Optional[pd.DataFrame] = None
) -> str:
    pkg_dir = os.path.join(run_dir, "predict_bundle")
    os.makedirs(pkg_dir, exist_ok=True)

    for src in [best_model_txt, scaler_full_pkl, scaler_pkl]:
        if src and os.path.exists(src):
            shutil.copy2(src, os.path.join(pkg_dir, os.path.basename(src)))


    if sigma_model_txt and os.path.exists(sigma_model_txt):
        shutil.copy2(sigma_model_txt, os.path.join(pkg_dir, "sigma_model.txt"))
    for fname in ["vectorizer_protein.pkl", "vectorizer_ligand.pkl",
                  "meta.json", "smiles_bpe_tokenizer.json", "protein_bpe_tokenizer.json"]:
        src = os.path.join(feat_cache_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(pkg_dir, fname))

    if selective_calib_json and os.path.exists(selective_calib_json):
        shutil.copy2(selective_calib_json, os.path.join(pkg_dir, "selective_calib.json"))

    aff = {"a": 0.0, "b": 1.0}
    if os.path.exists(best_manifest_path):
        try:
            m = load_json(best_manifest_path)
            aff = m.get("affine", aff)
        except Exception:
            pass

    au = dict(cfg.get("AFFINITY_UNIFIED", {}) or {})
    au["known_types"] = list(au.get("type_order", ["pIC50", "pKi", "pKd"]))
    au["affinity_type_col"] = au.get("keep_type_col_name", "affinity_type")

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "columns": {"smiles": cfg["COL_SMILES"], "protein": cfg["COL_PROTEIN"]},
        "unified": bool(au.get("enable", False)),
        "affinity_unified": au,
        "affine": aff,  # a=intercept, b=slope
        "pred_col": cfg["PRED_COL"],
        "pred_lo_col": cfg["PRED_LO_COL"],
        "pred_hi_col": cfg["PRED_HI_COL"],
        "strong_binder_threshold_pk": float(cfg.get("STRONG_BINDER_THRESHOLD_PK", 6.0)),
        "smiles_norm": cfg["SMILES_NORM"],
        "protein_clean_spec": {"method": "AA20_filter_v1", "output_col": "protein_clean"},
        "vectorizer_type": cfg["VECTORIZER"]["type"],
        "smiles_bpe": cfg["SMILES_BPE"],
        "protein_bpe": cfg["PROTEIN_BPE"],
        "selective": cfg["SELECTIVE"],
        "sigma_uncert": cfg.get("SIGMA_UNCERT", {}),
    }
    safe_json_dump(manifest, os.path.join(pkg_dir, "predict_manifest.json"))

    _write_predict_cli(pkg_dir)
    log(f"[Bundle] Prediction bundle: {pkg_dir}")
    return pkg_dir

def _write_predict_cli(pkg_dir: str):
    """Generate a standalone prediction script for the exported bundle."""
    code = r'''
#!/usr/bin/env python3
import os, json, argparse, numpy as np, pandas as pd, joblib, lightgbm as lgb
from scipy import sparse

try:
    from tokenizers import Tokenizer
except Exception:
    Tokenizer = None

# ---- RDKit normalization consistent with the training pipeline ----
try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    try:
        from rdkit.Chem import rdMolStandardize as rdms
    except Exception:
        rdms = None
except Exception:
    Chem = None
    rdms = None


def clean_protein(s):
    import re
    cleaned = re.sub(r"[^A-Za-z]", "", str(s) if s is not None else "").upper()
    keep = set("ACDEFGHIKLMNPQRSTVWY")
    return "".join([c for c in cleaned if c in keep])


def clean_smiles(s):
    import re
    s = re.sub(r"\s+", "", str(s) if s is not None else "").strip()
    if "|" in s:
        s = s.split("|", 1)[0]
    return s


def _normalize_smiles_one(s: str) -> str:
    """
    Consistent with the training-side _normalize_smiles_one / rdkit_std_v1:
    - clean_smiles
    - MolFromSmiles
    - rdMolStandardize: Normalizer / LargestFragmentChooser / Uncharger / tautomer canonicalization, if available
    - MolToSmiles(isomericSmiles=True, canonical=True)
    Return an empty string for invalid inputs.
    """
    s = clean_smiles(s)
    if not s:
        return ""
    if Chem is None:
        return s

    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return ""

        if rdms is not None:
            try:
                mol = rdms.Normalizer().normalize(mol)
            except Exception:
                pass
            try:
                mol = rdms.LargestFragmentChooser().choose(mol)
            except Exception:
                pass
            try:
                mol = rdms.Uncharger().uncharge(mol)
            except Exception:
                pass
            try:
                mol = rdms.TautomerEnumerator().Canonicalize(mol)
            except Exception:
                pass

        smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        return smi if smi else ""
    except Exception:
        return ""


def normalize_smiles(s: str, norm_cfg: dict) -> str:
    """
    Read the smiles_norm configuration from predict_manifest.json for training/inference consistency:
    - If enable=False: use clean_smiles only.
    - If enable=True: use rdkit_std_v1.
    """
    cfg = norm_cfg or {}
    if not bool(cfg.get("enable", True)):
        return clean_smiles(s)
    return _normalize_smiles_one(s)


def _dense_col_to_csr(x):
    x = np.asarray(x, float).reshape(-1, 1)
    return sparse.csr_matrix(x.astype(np.float32))


def make_sigma_features(X, pred):
    if not sparse.isspmatrix_csr(X):
        X = X.tocsr()
    return sparse.hstack([X, _dense_col_to_csr(pred)], format="csr")


def err_bound_for_preds(y_pred, calib):
    y_pred = np.asarray(y_pred, float).reshape(-1)
    if calib.get("bin_edges") is None or calib.get("bin_q") is None:
        return np.full_like(y_pred, float(calib.get("global_q", 0.0)))
    edges = np.asarray(calib["bin_edges"], float)
    bin_q = np.asarray(calib["bin_q"], float)
    idx = np.searchsorted(edges[1:-1], y_pred, side="right")
    idx = np.clip(idx, 0, len(bin_q) - 1)
    return bin_q[idx]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    man = json.load(open(os.path.join(here, "predict_manifest.json"), "r", encoding="utf-8"))

    col_smi = man["columns"]["smiles"]
    col_pro = man["columns"]["protein"]
    pred_col = man.get("pred_col", "pred")

    unified = bool(man.get("unified", False))
    norm_cfg = man.get("smiles_norm", {}) or {}

    # tokenizers + vectorizers
    if Tokenizer is None:
        raise RuntimeError("tokenizers is required in predict bundle")

    tok_s = Tokenizer.from_file(os.path.join(here, "smiles_bpe_tokenizer.json"))
    tok_p = Tokenizer.from_file(os.path.join(here, "protein_bpe_tokenizer.json"))
    vec_s = joblib.load(os.path.join(here, "vectorizer_ligand.pkl"))
    vec_p = joblib.load(os.path.join(here, "vectorizer_protein.pkl"))

    model = lgb.Booster(model_file=os.path.join(here, "best_model.txt"))

    scaler = joblib.load(os.path.join(here, "scaler_full.pkl")) if os.path.exists(os.path.join(here, "scaler_full.pkl")) else None
    if scaler is None and os.path.exists(os.path.join(here, "scaler.pkl")):
        sp = joblib.load(os.path.join(here, "scaler.pkl"))
        scaler = {"mean": sp["target_scaler_mean"], "scale": sp["target_scaler_scale"]}

    # affine: a + b*pred
    aff = man.get("affine", {"a": 0.0, "b": 1.0})
    a = float(aff.get("a", 0.0))
    b = float(aff.get("b", 1.0))

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_csv", required=True)
    ap.add_argument("--out", dest="output_csv", required=True)
    ap.add_argument("--affinity_type", default="pIC50")
    ap.add_argument("--err_max", type=float, default=float((man.get("selective", {}) or {}).get("default_err_max", 0.8)))
    ap.add_argument("--calib_json", default="")  # optional override
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)

    if col_smi not in df.columns or col_pro not in df.columns:
        raise ValueError(f"Missing required columns: smiles={col_smi}, protein={col_pro}")

    smiles_norm = df[col_smi].astype(str).map(lambda x: normalize_smiles(x, norm_cfg)).tolist()
    protein_clean = df[col_pro].astype(str).map(clean_protein).tolist()

    valid_mask = np.array([(len(s) > 0) and (len(p) > 0) for s, p in zip(smiles_norm, protein_clean)], dtype=bool)

    # Token documents; invalid inputs produce empty documents.
    docs_s = [" ".join(tok_s.encode(s).tokens) if s else "" for s in smiles_norm]
    docs_p = [" ".join(tok_p.encode(p).tokens) if p else "" for p in protein_clean]

    Xs = vec_s.transform(docs_s)
    Xp = vec_p.transform(docs_p)
    X = sparse.hstack([Xs, Xp], format="csr")

    if unified and str((man.get("affinity_unified", {}) or {}).get("type_feature", "onehot")).lower() == "onehot":
        au = man.get("affinity_unified", {}) or {}
        types = au.get("known_types") or au.get("type_order") or ["pIC50", "pKi", "pKd"]
        tcol = au.get("affinity_type_col", au.get("keep_type_col_name", "affinity_type"))

        if tcol in df.columns:
            atype = df[tcol].astype(str).fillna(args.affinity_type).tolist()
        else:
            atype = [args.affinity_type] * len(df)

        idx_map = {t: i for i, t in enumerate(types)}
        rows, cols, data = [], [], []
        for i, t in enumerate(atype):
            if t in idx_map:
                rows.append(i)
                cols.append(idx_map[t])
                data.append(1.0)
        A = sparse.csr_matrix((data, (rows, cols)), shape=(len(df), len(types)), dtype=np.float32)
        X = sparse.hstack([X, A], format="csr")

    pred_s = model.predict(X)

    # inverse scaling
    if isinstance(scaler, dict):
        pred = pred_s * float(scaler["scale"]) + float(scaler["mean"])
    elif scaler is not None:
        pred = scaler.inverse_transform(np.asarray(pred_s).reshape(-1, 1)).ravel()
    else:
        pred = pred_s

    # affine apply (a + b*pred)
    pred = a + b * np.asarray(pred, float)

    # Invalid inputs are set to NaN to avoid false acceptance.
    pred = np.where(valid_mask, pred, np.nan)

    out = df.copy()
    out["smiles_norm"] = smiles_norm
    out["protein_clean"] = protein_clean
    out["valid_input"] = valid_mask.astype(int)

    out[pred_col] = pred

    # interval / selective
    calib_path = args.calib_json.strip() if args.calib_json.strip() else os.path.join(here, "selective_calib.json")
    if os.path.exists(calib_path):
        calib = json.load(open(calib_path, "r", encoding="utf-8"))
        method = str(calib.get("method", "bin")).strip().lower()

        errb = np.full(len(out), np.nan, dtype=float)

        if valid_mask.any():
            if method == "sigma":
                sigma_model_path = os.path.join(here, "sigma_model.txt")
                if not os.path.exists(sigma_model_path):
                    raise FileNotFoundError("calib.method=sigma but sigma_model.txt is missing from the prediction bundle")

                sigma_cfg = man.get("sigma_uncert", {}) or {}
                use_pred_feat = bool(sigma_cfg.get("use_pred_feature", True))

                sigma_model = lgb.Booster(model_file=sigma_model_path)

                # Sigma prediction is computed only for valid rows.
                Xv = X[valid_mask]
                pv = out.loc[valid_mask, pred_col].values.astype(float)

                X_sig = Xv
                if use_pred_feat:
                    X_sig = make_sigma_features(X_sig, pv)

                s = np.asarray(sigma_model.predict(X_sig), float).reshape(-1)
                s = np.clip(s, -50.0, 50.0)

                target = str(sigma_cfg.get("target", "log_sq_err")).strip().lower()
                if target == "log_abs_err":
                    sigma = np.exp(s)
                else:
                    sigma = np.exp(0.5 * s)

                sigma = np.asarray(sigma, np.float64)
                sigma[~np.isfinite(sigma)] = np.nan
                min_sigma = float(sigma_cfg.get("min_sigma", 1e-6))
                mx = sigma_cfg.get("max_sigma", None)
                max_sigma = float(mx) if mx is not None else 1e6
                sigma = np.clip(sigma, min_sigma, max_sigma)
                sigma[~np.isfinite(sigma)] = max_sigma

                # Fill sigma predictions back into the full output array.
                out["sigma"] = np.nan
                out.loc[valid_mask, "sigma"] = sigma

                errb_valid = float(calib.get("q_scale", 0.0)) * sigma
                errb[valid_mask] = errb_valid
            else:
                errb[valid_mask] = err_bound_for_preds(out.loc[valid_mask, pred_col].values, calib)

        out["err_bound"] = errb
        out[man["pred_lo_col"]] = out[pred_col] - errb
        out[man["pred_hi_col"]] = out[pred_col] + errb

        # A prediction is accepted only when it is valid and err_bound <= err_max.
        out["accepted"] = (valid_mask & (errb <= float(args.err_max))).astype(int)

    thr = float(man.get("strong_binder_threshold_pk", 6.0))
    if "sigma" in out.columns:
        sigv = out["sigma"].values.astype(float)
        predv = out[pred_col].values.astype(float)
        prob = np.full(len(out), np.nan, dtype=float)
        mprob = np.isfinite(predv) & np.isfinite(sigv) & (sigv > 0)
        if np.any(mprob):
            z = (thr - predv[mprob]) / np.maximum(sigv[mprob], 1e-12)
            prob[mprob] = stats.norm.sf(z)
        mdet = np.isfinite(predv) & (~mprob)
        if np.any(mdet):
            prob[mdet] = (predv[mdet] >= thr).astype(float)
        out["prob_strong_binder"] = np.clip(prob, 0.0, 1.0)
    else:
        predv = out[pred_col].values.astype(float)
        prob = np.full(len(out), np.nan, dtype=float)
        mdet = np.isfinite(predv)
        prob[mdet] = (predv[mdet] >= thr).astype(float)
        out["prob_strong_binder"] = prob
    out["pred_is_strong_binder"] = (out["prob_strong_binder"] >= 0.5).astype(int)

    out.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
'''
    path = os.path.join(pkg_dir, "predict_cli.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    os.chmod(path, 0o755)


# =============================================================================
# K-fold cross-validation run
# =============================================================================

def kfold_run(Xs, y_raw, w, df: pd.DataFrame, mask: np.ndarray, feat_cache_dir: str, k: int = 5) -> Dict:
    k = int(k)
    rs = int(CONFIG["SEED"])
    print_step("kfold", f"{k}-fold CV")

    df_all = df.loc[mask].reset_index(drop=True)
    groups, _ = make_groups_optimized(df_all)
    unique_groups = np.unique(groups)

    y = y_raw
    group_medians = np.array([np.median(y[groups == g]) for g in unique_groups])
    try:
        group_bins = pd.qcut(group_medians, q=5, labels=False, duplicates="drop").values
    except:
        group_bins = np.zeros(len(unique_groups), dtype=int)

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=rs)

    root_dir = os.path.join(CONFIG["OUT_DIR"], f"kfold_{k}")
    os.makedirs(root_dir, exist_ok=True)

    fold_summaries = []
    for fold, (tr_gi, va_gi) in enumerate(skf.split(np.zeros(len(unique_groups)), group_bins), 1):
        fold_dir = os.path.join(root_dir, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        tr_groups = unique_groups[tr_gi]
        va_groups = unique_groups[va_gi]

        tr_idx = np.where(np.isin(groups, tr_groups))[0]
        va_idx = np.where(np.isin(groups, va_groups))[0]

        df_tr = df_all.iloc[tr_idx].reset_index(drop=True)
        df_va = df_all.iloc[va_idx].reset_index(drop=True)
        X_tr, X_va = Xs[tr_idx], Xs[va_idx]
        y_tr, y_va = y_raw[tr_idx], y_raw[va_idx]

        fs = _train_one_fold(
            fold_dir=fold_dir,
            X_tr=X_tr, y_tr_raw=y_tr, df_tr=df_tr,
            X_va=X_va, y_va_raw=y_va, df_va=df_va,
            feat_cache_dir=feat_cache_dir,
            run_tag=f"kfold{fold}",
        )
        fold_summaries.append(fs)

    r2s = [fs["best_metrics"]["R2"] for fs in fold_summaries]
    rmses = [fs["best_metrics"]["RMSE"] for fs in fold_summaries]
    ae_q99s = [fs["best_metrics"]["AE_Qq"] for fs in fold_summaries]
    r2mbar = [fs["best_metrics"]["R2m_bar"] for fs in fold_summaries]
    pearson = [fs["best_metrics"]["R"] for fs in fold_summaries]

    def _mean_std(arr):
        arr = np.asarray(arr, float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return (float("nan"), float("nan"))
        return (float(np.mean(arr)), float(np.std(arr)))

    agg = {
        "R2_mean": _mean_std(r2s)[0], "R2_std": _mean_std(r2s)[1],
        "RMSE_mean": _mean_std(rmses)[0], "RMSE_std": _mean_std(rmses)[1],
        "AE_Q99_mean": _mean_std(ae_q99s)[0], "AE_Q99_std": _mean_std(ae_q99s)[1],
        "R2m_bar_mean": _mean_std(r2mbar)[0], "R2m_bar_std": _mean_std(r2mbar)[1],
        "PearsonR_mean": _mean_std(pearson)[0], "PearsonR_std": _mean_std(pearson)[1],
    }

    summary = {
        "mode": "kfold", "k": k, "run_dir": root_dir,
        "affinity_unified": CONFIG.get("AFFINITY_UNIFIED", {}),
        "optimize": CONFIG.get("OPTIMIZE", {}),
        "aggregate": agg,
        "fold_summaries": fold_summaries,
    }
    safe_json_dump(summary, os.path.join(root_dir, "kfold_summary.json"))

    q = float(CONFIG.get("OPTIMIZE", {}).get("q", 0.99))
    dfm = []
    for fs in fold_summaries:
        m = fs["best_metrics"]
        dfm.append({
            "fold_dir": fs["fold_dir"],
            "device": fs.get("device", ""),
            "best_by": fs.get("best_by", ""),
            "best_value": fs.get("best_value", float("nan")),
            "best_iteration": fs.get("best_iteration", ""),
            "R": m["R"], "Pearson_p": m.get("Pearson_p", np.nan),
            "SpearmanR": m.get("SpearmanR", np.nan), "Spearman_p": m.get("Spearman_p", np.nan),
            "R2": m["R2"], "RMSE": m["RMSE"], f"AE_Q{int(q*100)}": m["AE_Qq"],
            "CI": m["CI"], "AUC": m["AUC"], "AUPR": m["AUPR"], "N": m["N"],
            "R2m": m["R2m"], "R2m_prime": m["R2m_prime"], "R2m_bar": m["R2m_bar"], "delta_R2m": m["delta_R2m"],
        })
    pd.DataFrame(dfm).to_csv(os.path.join(root_dir, "folds_metrics.csv"), index=False)

    return summary


# =============================================================================
# Data loading
# =============================================================================

def load_and_prepare_data() -> Tuple[pd.DataFrame, np.ndarray]:
    cfg = CONFIG
    print_step("Load data", cfg["DATA_FILE"])
    df = pd.read_csv(cfg["DATA_FILE"])
    log(f"[Data] n={len(df)}")

    for col in cfg["DATA_STATS"]["numeric_cols"]:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])

    if cfg["DATA_STATS"]["enable"]:
        try:
            analyze_data(df, cfg)
        except Exception as e:
            log(f"[DataStats] Failed: {e}", "WARNING")

    df, _ = filter_valid_samples(df, CONFIG["OUT_DIR"])

    if cfg.get("AFFINITY_UNIFIED", {}).get("enable", False):
        y_all = pd.to_numeric(df[cfg["AFFINITY_UNIFIED"]["keep_target_col_name"]], errors="coerce").values.astype(np.float32)
    else:
        y_all = pd.to_numeric(df[cfg["COL_TARGET"]], errors="coerce").values.astype(np.float32)

    mask = np.isfinite(y_all)
    valid = int(mask.sum())
    if valid < 200:
        raise ValueError(f"Too few valid samples: {valid}")

    return df, mask


# =============================================================================
# Prediction with an exported bundle
# =============================================================================

def predict_with_bundle(bundle_dir: str, input_csv: str, output_csv: str,
                        affinity_type: str = "pIC50", err_max: Optional[float] = None,
                        calib_json: str = ""):
    cli_path = os.path.join(bundle_dir, "predict_cli.py")
    if not os.path.exists(cli_path):
        raise FileNotFoundError(f"predict_cli.py not found in {bundle_dir}")

    cmd = [
        sys.executable, cli_path,
        "--in", input_csv,
        "--out", output_csv,
        "--affinity_type", affinity_type,
    ]
    if err_max is not None:
        cmd += ["--err_max", str(err_max)]
    if calib_json:
        cmd += ["--calib_json", calib_json]

    subprocess.run(cmd, check=True)
    log(f"[Predict] Completed: {output_csv}")


# =============================================================================
# CLI
# =============================================================================

def _apply_unified_mode_defaults(enable: bool):
    CONFIG["AFFINITY_UNIFIED"]["enable"] = enable
    if enable:
        CONFIG["AFFINITY"] = "pX"

def build_arg_parser():
    ap = argparse.ArgumentParser(
        prog=os.path.basename(__file__),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="ConfMux-DTA: unified confidence-aware DTA model with BPE, sparse features, LightGBM, and prediction bundles"
    )
    sp = ap.add_subparsers(dest="command", required=True)

    def add_common_train_args(p):
        p.add_argument("--data_file", default=CONFIG["DATA_FILE"],
                       help="Model-ready CSV generated by confmux_dta_bindingdb_preprocess.py")
        p.add_argument("--seed", type=int, default=CONFIG["SEED"])
        p.add_argument("--split_mode", type=str, default=CONFIG["SPLIT_MODE"],
                      choices=["random", "compound", "protein", "pair"])
        p.add_argument("--test_size", type=float, default=CONFIG["TEST_SIZE"])
        p.add_argument("--force_device", type=str, default="", choices=["", "cpu", "gpu"])
        p.add_argument("--num_threads", type=int, default=int(CONFIG.get("NUM_THREADS", 0)))
        p.add_argument("--total_iters", type=int, default=int(CONFIG["LGBM_TOTAL_ITERS"]))
        p.add_argument("--stage_iters", type=int, default=int(CONFIG["LGBM_STAGE_ITERS"]))
        p.add_argument("--no_cache", action="store_true")
        p.add_argument("--cache_overwrite", action="store_true")
        p.add_argument("--no_backup", action="store_true",
                       help="Do not duplicate the input dataset into each run directory")
        p.add_argument("--no_smiles_norm", action="store_true")
        p.add_argument("--experiment_name", default="",
                       help="Label appended to the run directory, e.g. mixed_no_onehot or mixed_onehot")
        p.add_argument("--run_dir", default="",
                       help="Stable output directory used for checkpoint/resume")
        p.add_argument("--resume", action="store_true",
                       help="Resume LightGBM training from the latest completed stage in --run_dir")

        ug = p.add_mutually_exclusive_group()
        ug.add_argument("--unified", action="store_true", help="Enable unified pIC50/pKi/pKd mode")
        ug.add_argument("--no_unified", action="store_true", help="Disable unified mode")

        tg = p.add_mutually_exclusive_group()
        tg.add_argument("--type_onehot", action="store_true",
                        help="In unified mode, append the pIC50/pKi/pKd one-hot feature")
        tg.add_argument("--no_type_onehot", action="store_true",
                        help="Ablation baseline: keep mixed pIC50/pKi/pKd rows but omit endpoint one-hot")

        p.add_argument("--affinity", type=str, default=CONFIG.get("AFFINITY", "pIC50"),
                      choices=["pIC50", "pKi", "pKd"], help="Single-affinity target used when unified mode is disabled")

        p.add_argument("--optimize_by", type=str, default=CONFIG["OPTIMIZE"]["by"],
                      choices=["rmse", "mse", "r2", "ci", "r2m", "ae_q"], help="Model-selection objective")
        p.add_argument("--optimize_q", type=float, default=float(CONFIG["OPTIMIZE"]["q"]),
                      help="Quantile used when optimize_by=ae_q")

        p.add_argument("--calib_method", type=str, default=CONFIG["CALIBRATION"]["method"],
                      choices=["train_linear", "valid_linear", "none"])
        p.add_argument("--no_selective", action="store_true", help="Disable selective prediction calibration")
        p.add_argument("--no_sigma_uncert", action="store_true",
                       help="Disable the auxiliary sigma/uncertainty model for mean-prediction ablations")
        p.add_argument("--selective_alpha", type=float, default=CONFIG["SELECTIVE"]["alpha"])
        p.add_argument("--train_metrics_cap", type=int, default=int(CONFIG.get("TRAIN_METRICS_SAMPLE_CAP", 0)))

    p_tt = sp.add_parser("train_test", help="Single train/validation split")
    add_common_train_args(p_tt)

    p_kf = sp.add_parser("kfold", help="K-fold cross-validation")
    add_common_train_args(p_kf)
    p_kf.add_argument("--k", type=int, default=5)

    p_pr = sp.add_parser("predict", help="Run prediction with an exported bundle")
    p_pr.add_argument("--bundle_dir", required=True)
    p_pr.add_argument("--in", dest="input_csv", required=True)
    p_pr.add_argument("--out", dest="output_csv", required=True)
    p_pr.add_argument("--affinity_type", default="pIC50", choices=["pIC50", "pKi", "pKd"])
    p_pr.add_argument("--err_max", type=float, default=None)
    p_pr.add_argument("--calib_json", default="")

    return ap

def _apply_args_to_config(args):
    # ---- common ----
    if getattr(args, "data_file", None):
        CONFIG["DATA_FILE"] = str(args.data_file)

    if getattr(args, "seed", None) is not None:
        CONFIG["SEED"] = int(args.seed)

    if getattr(args, "split_mode", None):
        CONFIG["SPLIT_MODE"] = str(args.split_mode)

    if getattr(args, "test_size", None) is not None:
        CONFIG["TEST_SIZE"] = float(args.test_size)

    if getattr(args, "force_device", ""):
        CONFIG["FORCE_DEVICE"] = str(args.force_device).strip() or None

    if getattr(args, "num_threads", None) is not None:
        CONFIG["NUM_THREADS"] = int(args.num_threads)

    if getattr(args, "total_iters", None) is not None:
        CONFIG["LGBM_TOTAL_ITERS"] = int(args.total_iters)

    if getattr(args, "stage_iters", None) is not None:
        CONFIG["LGBM_STAGE_ITERS"] = int(args.stage_iters)

    if getattr(args, "no_cache", False):
        CONFIG["CACHE"]["enable"] = False
    if getattr(args, "cache_overwrite", False):
        CONFIG["CACHE"]["overwrite"] = True
    if getattr(args, "no_backup", False):
        CONFIG["BACKUP"]["enable"] = False

    if getattr(args, "no_smiles_norm", False):
        CONFIG["SMILES_NORM"]["enable"] = False

    if getattr(args, "experiment_name", ""):
        CONFIG["EXPERIMENT_NAME"] = str(args.experiment_name)
    if getattr(args, "run_dir", ""):
        CONFIG["RUN_DIR"] = str(args.run_dir)
    if getattr(args, "resume", False):
        CONFIG["RESUME"] = True
        if not str(CONFIG.get("RUN_DIR", "")).strip():
            raise ValueError("--resume requires --run_dir so the checkpoint location is unambiguous")

    # ---- unified toggle ----
    if getattr(args, "unified", False):
        _apply_unified_mode_defaults(True)
    if getattr(args, "no_unified", False):
        _apply_unified_mode_defaults(False)

    if getattr(args, "type_onehot", False):
        CONFIG["AFFINITY_UNIFIED"]["type_feature"] = "onehot"
    if getattr(args, "no_type_onehot", False):
        CONFIG["AFFINITY_UNIFIED"]["type_feature"] = "none"

    # ---- single affinity mode ----
    if not CONFIG.get("AFFINITY_UNIFIED", {}).get("enable", False):
        if getattr(args, "affinity", None):
            apply_affinity_to_config(CONFIG, str(args.affinity))

    # ---- optimize ----
    if getattr(args, "optimize_by", None):
        CONFIG["OPTIMIZE"]["by"] = str(args.optimize_by)
    if getattr(args, "optimize_q", None) is not None:
        CONFIG["OPTIMIZE"]["q"] = float(args.optimize_q)

    # ---- calibration ----
    if getattr(args, "calib_method", None):
        CONFIG["CALIBRATION"]["method"] = str(args.calib_method)

    # ---- selective ----
    if getattr(args, "no_selective", False):
        CONFIG["SELECTIVE"]["enable"] = False
    if getattr(args, "no_sigma_uncert", False):
        CONFIG["SIGMA_UNCERT"]["enable"] = False
    if getattr(args, "selective_alpha", None) is not None:
        CONFIG["SELECTIVE"]["alpha"] = float(args.selective_alpha)

    # ---- stage_history train metrics cap ----
    if getattr(args, "train_metrics_cap", None) is not None:
        CONFIG["TRAIN_METRICS_SAMPLE_CAP"] = int(args.train_metrics_cap)


def main():
    ap = build_arg_parser()
    args = ap.parse_args()

    # ---------- predict: use exported bundle directly ----------
    if args.command == "predict":
        predict_with_bundle(
            bundle_dir=args.bundle_dir,
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            affinity_type=getattr(args, "affinity_type", "pIC50"),
            err_max=getattr(args, "err_max", None),
            calib_json=getattr(args, "calib_json", "") or ""
        )
        return

    # ---------- train / cv ----------
    _apply_args_to_config(args)
    init_run_dirs()
    backup_code_and_data(CONFIG)

    # Save config snapshot
    safe_json_dump(CONFIG, os.path.join(CONFIG["OUT_DIR"], "config_used.json"))

    # load + filter
    df, mask = load_and_prepare_data()

    if args.command == "kfold":
        raise ValueError(
            "This leakage-controlled reviewer experiment supports train_test only. "
            "K-fold evaluation requires separately fitted BPE/vectorizers within every fold."
        )

    # Split first, then fit every data-dependent representation on training rows only.
    idx_tr, idx_va, df_all, y_raw = make_fixed_train_valid_split(df, mask)
    Xs, y_raw, w, feat_cache_dir = load_or_make_features_train_only(
        df, mask, idx_tr, idx_va, df_all, y_raw
    )

    # run
    if args.command == "train_test":
        train_test_run(Xs, y_raw, w, df=df, mask=mask, feat_cache_dir=feat_cache_dir)
        log("[Done] train_test finished.")
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
