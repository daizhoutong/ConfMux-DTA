#!/usr/bin/env python3
"""Leakage-controlled BPE/TF-IDF + sparse Ridge baseline for ConfMux-DTA.

This targeted reviewer ablation reuses the training-only feature cache created
by ``confmux_dta_train_predict_leakage_controlled.py``.  The validation rows are
transformed with preprocessing objects fitted on the training rows only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def load_core(path: str):
    spec = importlib.util.spec_from_file_location("confmux_leakctl", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import core module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure(core, args) -> None:
    cfg = core.CONFIG
    cfg["DATA_FILE"] = os.path.abspath(args.data_file)
    cfg["SEED"] = int(args.seed)
    cfg["SPLIT_MODE"] = "random"
    cfg["TEST_SIZE"] = 0.20
    cfg["FORCE_DEVICE"] = "cpu"
    cfg["NUM_THREADS"] = int(args.num_threads)
    cfg["RUN_DIR"] = os.path.abspath(args.run_dir)
    cfg["EXPERIMENT_NAME"] = "bpe_ridge"
    cfg["RESUME"] = False
    cfg["BACKUP"]["enable"] = False
    cfg["SELECTIVE"]["enable"] = False
    cfg["SIGMA_UNCERT"]["enable"] = False
    cfg["CALIBRATION"]["method"] = "train_linear"
    cfg["AFFINITY_UNIFIED"]["enable"] = True
    cfg["AFFINITY_UNIFIED"]["type_feature"] = "onehot"
    core.init_run_dirs()
    core.safe_json_dump(cfg, os.path.join(cfg["OUT_DIR"], "config_used.json"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", required=True)
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--seed", type=int, default=5313)
    parser.add_argument("--num_threads", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--max_iter", type=int, default=1000)
    args = parser.parse_args()

    core = load_core(os.path.abspath(args.core))
    configure(core, args)

    core.log("[Ridge] Loading data and reconstructing the fixed split")
    df, mask = core.load_and_prepare_data()
    idx_tr, idx_va, df_all, y_raw = core.make_fixed_train_valid_split(df, mask)
    X, _, _, feature_cache_dir = core.load_or_make_features_train_only(
        df, mask, idx_tr, idx_va, df_all, y_raw
    )

    Xtr, Xva = X[idx_tr], X[idx_va]
    ytr, yva = y_raw[idx_tr], y_raw[idx_va]
    dtr = df_all.iloc[idx_tr].reset_index(drop=True)
    dva = df_all.iloc[idx_va].reset_index(drop=True)
    weights = core.make_train_weights(dtr, ytr)

    scaler = StandardScaler().fit(ytr.reshape(-1, 1))
    ytr_scaled = scaler.transform(ytr.reshape(-1, 1)).ravel()
    model = Ridge(
        alpha=float(args.alpha),
        fit_intercept=True,
        solver="lsqr",
        tol=float(args.tol),
        max_iter=int(args.max_iter),
    )

    core.log(
        f"[Ridge] Fit start: alpha={args.alpha:g} tol={args.tol:g} "
        f"max_iter={args.max_iter} shape={Xtr.shape}"
    )
    t0 = time.time()
    model.fit(Xtr, ytr_scaled, sample_weight=weights)
    fit_time_sec = time.time() - t0

    pred_tr_raw = scaler.inverse_transform(
        model.predict(Xtr).reshape(-1, 1)
    ).ravel()
    pred_va_raw = scaler.inverse_transform(
        model.predict(Xva).reshape(-1, 1)
    ).ravel()
    affine_a, affine_b = core.fit_affine(ytr, pred_tr_raw)
    pred_tr = core.apply_affine(pred_tr_raw, affine_a, affine_b, clip=None)
    pred_va = core.apply_affine(pred_va_raw, affine_a, affine_b, clip=None)

    threshold = core.CONFIG["STRONG_BINDER_THRESHOLD_PK"]
    train_pack = core.evaluate_metrics_pack(dtr, ytr, pred_tr, threshold)
    valid_pack = core.evaluate_metrics_pack(dva, yva, pred_va, threshold)

    out_dir = os.path.join(core.CONFIG["OUT_DIR"], "train_test")
    os.makedirs(out_dir, exist_ok=True)
    n_iter = getattr(model, "n_iter_", None)
    if isinstance(n_iter, np.ndarray):
        n_iter = n_iter.tolist()
    elif isinstance(n_iter, np.generic):
        n_iter = n_iter.item()

    flat = {}
    flat.update(core.flatten_metrics_for_paper("Train", train_pack))
    flat.update(core.flatten_metrics_for_paper("Valid", valid_pack))
    flat.update(
        {
            "variant": "bpe_tfidf_sparse_ridge",
            "alpha": float(args.alpha),
            "solver": "lsqr",
            "tol": float(args.tol),
            "max_iter": int(args.max_iter),
            "solver_n_iter": n_iter,
            "fit_time_sec": float(fit_time_sec),
            "affine_a": float(affine_a),
            "affine_b": float(affine_b),
            "feature_cache_dir": feature_cache_dir,
        }
    )

    pd.DataFrame([flat]).to_csv(
        os.path.join(out_dir, "paper_metrics_flat.csv"), index=False
    )
    core.safe_json_dump(
        {"train": train_pack, "valid": valid_pack, "flat": flat},
        os.path.join(out_dir, "paper_metrics.json"),
    )
    core.safe_json_dump(
        {
            "complete": True,
            "model": "Ridge",
            "solver": "lsqr",
            "alpha": float(args.alpha),
            "tol": float(args.tol),
            "max_iter": int(args.max_iter),
            "solver_n_iter": n_iter,
            "fit_time_sec": float(fit_time_sec),
            "valid_R2": float(valid_pack["overall"]["R2"]),
            "valid_RMSE": float(valid_pack["overall"]["RMSE"]),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        os.path.join(out_dir, "resume_state.json"),
    )
    joblib.dump(model, os.path.join(out_dir, "ridge_model.pkl"))
    joblib.dump(scaler, os.path.join(out_dir, "target_scaler.pkl"))
    core.log(
        f"[Ridge] Complete: R2={valid_pack['overall']['R2']:.6f} "
        f"RMSE={valid_pack['overall']['RMSE']:.6f} "
        f"fit_time={fit_time_sec / 60.0:.1f} min"
    )


if __name__ == "__main__":
    main()
