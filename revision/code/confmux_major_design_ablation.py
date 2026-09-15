#!/usr/bin/env python3
"""Focused reviewer ablations for ConfMux-DTA.

Modes
-----
endpoint_lgbm : one endpoint only, using the matching rows from the unified split
char_lgbm : direct character n-gram TF-IDF + LightGBM
bpe_sgd   : training-only BPE/TF-IDF + sparse SGDRegressor

The three endpoint-specific LightGBM runs use the companion
confmux_dta_train_predict_leakage_controlled.py directly from the Slurm script.
"""
import os
import json
import time
import joblib
import argparse
import hashlib
import importlib.util

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


def load_core(path):
    spec = importlib.util.spec_from_file_location("confmux_leakctl", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def configure(core, args):
    c = core.CONFIG
    c["DATA_FILE"] = os.path.abspath(args.data_file)
    c["SEED"] = int(args.seed)
    c["SPLIT_MODE"] = "random"
    c["TEST_SIZE"] = 0.20
    c["FORCE_DEVICE"] = "cpu"
    c["NUM_THREADS"] = int(args.num_threads)
    c["LGBM_TOTAL_ITERS"] = int(args.total_iters)
    c["LGBM_STAGE_ITERS"] = int(args.stage_iters)
    c["RUN_DIR"] = os.path.abspath(args.run_dir)
    c["EXPERIMENT_NAME"] = args.mode
    c["RESUME"] = bool(args.resume)
    c["BACKUP"]["enable"] = False
    c["SELECTIVE"]["enable"] = False
    c["SIGMA_UNCERT"]["enable"] = False
    c["CALIBRATION"]["method"] = "train_linear"
    c["AFFINITY_UNIFIED"]["enable"] = True
    c["AFFINITY_UNIFIED"]["type_feature"] = ("none" if args.mode == "endpoint_lgbm" else "onehot")
    core.init_run_dirs()
    core.safe_json_dump(c, os.path.join(c["OUT_DIR"], "config_used.json"))


def char_features(core, df_all, idx_tr, idx_va, y_raw):
    """Fit direct character n-gram TF-IDF on training rows only."""
    cdir = os.path.join(core.CONFIG["OUT_DIR"], ".cache", "char_ngram_train_only")
    os.makedirs(cdir, exist_ok=True)
    lig_col = "smiles_norm" if "smiles_norm" in df_all.columns else core.CONFIG["COL_SMILES"]
    pro_col = "protein_clean" if "protein_clean" in df_all.columns else core.CONFIG["COL_PROTEIN"]
    lig = [core.clean_smiles(x) for x in df_all[lig_col].astype(str)]
    pro = [core.clean_protein(x) for x in df_all[pro_col].astype(str)]
    vl = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2,
                         max_features=50000, norm="l2", sublinear_tf=True,
                         lowercase=False, dtype=np.float32)
    vp = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2,
                         max_features=50000, norm="l2", sublinear_tf=True,
                         lowercase=False, dtype=np.float32)
    core.log("[Ablation] Fitting character n-gram vectorizers on training rows only")
    Ltr = vl.fit_transform([lig[i] for i in idx_tr])
    Ptr = vp.fit_transform([pro[i] for i in idx_tr])
    Lva = vl.transform([lig[i] for i in idx_va])
    Pva = vp.transform([pro[i] for i in idx_va])
    Xtr = sparse.hstack([Ltr, Ptr, core.affinity_type_onehot(df_all.iloc[idx_tr])],
                        format="csr", dtype=np.float32)
    Xva = sparse.hstack([Lva, Pva, core.affinity_type_onehot(df_all.iloc[idx_va])],
                        format="csr", dtype=np.float32)
    order = np.concatenate([idx_tr, idx_va])
    inv = np.empty(len(order), dtype=np.int64)
    inv[order] = np.arange(len(order))
    X = sparse.vstack([Xtr, Xva], format="csr")[inv]
    sparse.save_npz(os.path.join(cdir, "Xs.npz"), X)
    np.save(os.path.join(cdir, "y.npy"), y_raw)
    np.save(os.path.join(cdir, "idx_train.npy"), idx_tr)
    np.save(os.path.join(cdir, "idx_valid.npy"), idx_va)
    joblib.dump(vl, os.path.join(cdir, "vectorizer_ligand.pkl"))
    joblib.dump(vp, os.path.join(cdir, "vectorizer_protein.pkl"))
    core.safe_json_dump({
        "representation": "direct_character_ngram_tfidf",
        "fit_scope": "training_subset_only", "validation_scope": "transform_only",
        "ligand_ngram_range": [2, 5], "protein_ngram_range": [3, 5],
        "n_train": int(len(idx_tr)), "n_valid": int(len(idx_va)),
        "train_index_sha1": hashlib.sha1(np.ascontiguousarray(idx_tr).tobytes()).hexdigest(),
        "valid_index_sha1": hashlib.sha1(np.ascontiguousarray(idx_va).tobytes()).hexdigest(),
        "shape": [int(X.shape[0]), int(X.shape[1])],
    }, os.path.join(cdir, "meta.json"))
    return X, cdir


def run_char_lgbm(core, df, mask, idx_tr, idx_va, df_all, y_raw):
    X, cdir = char_features(core, df_all, idx_tr, idx_va, y_raw)
    # train_test_run deterministically recreates the same split and uses the
    # unchanged LightGBM settings of the main model.
    w = np.ones(len(y_raw), dtype=np.float32)
    return core.train_test_run(X, y_raw, w, df, mask, cdir)


def run_endpoint_lgbm(core, endpoint, idx_tr, idx_va, df_all):
    """Train one endpoint while preserving the unified model's row assignment."""
    type_col = core.CONFIG["AFFINITY_UNIFIED"]["keep_type_col_name"]
    ycol = core.CONFIG["AFFINITY_UNIFIED"]["keep_target_col_name"]
    ep_global = np.flatnonzero(df_all[type_col].astype(str).to_numpy() == endpoint)
    assignment = np.full(len(df_all), -1, dtype=np.int8)
    assignment[np.asarray(idx_tr, dtype=np.int64)] = 0
    assignment[np.asarray(idx_va, dtype=np.int64)] = 1
    ep_assignment = assignment[ep_global]
    ep_train_mask = ep_assignment == 0
    ep_valid_mask = ep_assignment == 1
    idx_ep_tr = np.flatnonzero(ep_train_mask).astype(np.int64)
    idx_ep_va = np.flatnonzero(ep_valid_mask).astype(np.int64)
    if len(idx_ep_tr) == 0 or len(idx_ep_va) == 0 or np.any(ep_train_mask & ep_valid_mask):
        raise RuntimeError(f"Invalid endpoint-preserving split for {endpoint}")
    df_ep = df_all.iloc[ep_global].reset_index(drop=True)
    y_ep = pd.to_numeric(df_ep[ycol], errors="coerce").to_numpy(dtype=np.float32)
    core.CONFIG["AFFINITY_UNIFIED"]["type_feature"] = "none"
    X, _, _, cdir = core.load_or_make_features_train_only(
        df_ep, np.ones(len(df_ep), dtype=bool), idx_ep_tr, idx_ep_va, df_ep, y_ep
    )
    out = os.path.join(core.CONFIG["OUT_DIR"], "train_test")
    summary = core._train_one_fold(
        fold_dir=out,
        X_tr=X[idx_ep_tr], y_tr_raw=y_ep[idx_ep_tr], df_tr=df_ep.iloc[idx_ep_tr].reset_index(drop=True),
        X_va=X[idx_ep_va], y_va_raw=y_ep[idx_ep_va], df_va=df_ep.iloc[idx_ep_va].reset_index(drop=True),
        feat_cache_dir=cdir, run_tag=f"endpoint_{endpoint}",
    )
    core.safe_json_dump({
        "mode": "endpoint_specific", "endpoint": endpoint,
        "split_source": "unified_global_split", "fold_summary": summary,
    }, os.path.join(out, "train_test_summary.json"))
    return summary


def run_bpe_sgd(core, df, mask, idx_tr, idx_va, df_all, y_raw, epochs, batch_size):
    X, _, _, cdir = core.load_or_make_features_train_only(
        df, mask, idx_tr, idx_va, df_all, y_raw
    )
    Xtr, Xva = X[idx_tr], X[idx_va]
    ytr, yva = y_raw[idx_tr], y_raw[idx_va]
    dtr, dva = df_all.iloc[idx_tr].reset_index(drop=True), df_all.iloc[idx_va].reset_index(drop=True)
    out = os.path.join(core.CONFIG["OUT_DIR"], "train_test")
    os.makedirs(out, exist_ok=True)
    model_p = os.path.join(out, "sgd_checkpoint.pkl")
    state_p = os.path.join(out, "resume_state.json")
    scaler_p = os.path.join(out, "scaler.pkl")
    scaler = StandardScaler().fit(ytr.reshape(-1, 1))
    ytr_s = scaler.transform(ytr.reshape(-1, 1)).ravel()
    weights = core.make_train_weights(dtr, ytr)
    if os.path.exists(model_p) and os.path.exists(state_p):
        model = joblib.load(model_p)
        state = core.load_json(state_p)
        start_epoch = int(state.get("epochs_done", 0))
        core.log(f"[Resume] SGD epochs={start_epoch}/{epochs}")
    else:
        model = SGDRegressor(
            loss="squared_error", penalty="elasticnet", alpha=1e-5,
            l1_ratio=0.15, learning_rate="optimal", average=True,
            max_iter=1, tol=None, shuffle=False, random_state=int(core.CONFIG["SEED"])
        )
        start_epoch = 0
    for epoch in range(start_epoch, int(epochs)):
        rng = np.random.default_rng(int(core.CONFIG["SEED"]) + epoch)
        order = rng.permutation(len(ytr_s))
        for lo in range(0, len(order), int(batch_size)):
            ii = order[lo:lo + int(batch_size)]
            model.partial_fit(Xtr[ii], ytr_s[ii], sample_weight=weights[ii])
        pva = scaler.inverse_transform(model.predict(Xva).reshape(-1, 1)).ravel()
        met = core.evaluate_metrics(yva, pva, core.CONFIG["STRONG_BINDER_THRESHOLD_PK"])
        joblib.dump(model, model_p)
        core.safe_json_dump({
            "epochs_done": epoch + 1, "total_epochs": int(epochs),
            "valid_R2": met["R2"], "valid_RMSE": met["RMSE"],
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "complete": epoch + 1 >= int(epochs),
        }, state_p)
        core.log(f"[SGD] epoch={epoch+1}/{epochs} R2={met['R2']:.4f} RMSE={met['RMSE']:.4f}")
    ptr0 = scaler.inverse_transform(model.predict(Xtr).reshape(-1, 1)).ravel()
    pva0 = scaler.inverse_transform(model.predict(Xva).reshape(-1, 1)).ravel()
    a, b = core.fit_affine(ytr, ptr0)
    ptr = core.apply_affine(ptr0, a, b, clip=None)
    pva = core.apply_affine(pva0, a, b, clip=None)
    tr_pack = core.evaluate_metrics_pack(dtr, ytr, ptr, core.CONFIG["STRONG_BINDER_THRESHOLD_PK"])
    va_pack = core.evaluate_metrics_pack(dva, yva, pva, core.CONFIG["STRONG_BINDER_THRESHOLD_PK"])
    flat = {}
    flat.update(core.flatten_metrics_for_paper("Train", tr_pack))
    flat.update(core.flatten_metrics_for_paper("Valid", va_pack))
    flat.update({"variant": "bpe_tfidf_sgd_elasticnet", "affine_a": a, "affine_b": b})
    pd.DataFrame([flat]).to_csv(os.path.join(out, "paper_metrics_flat.csv"), index=False)
    core.safe_json_dump({"train": tr_pack, "valid": va_pack, "flat": flat},
                        os.path.join(out, "paper_metrics.json"))
    core.build_prediction_export_df(dva, yva, pva, "Valid",
        core.CONFIG["STRONG_BINDER_THRESHOLD_PK"]).to_csv(
            os.path.join(out, "valid_predictions.csv"), index=False)
    joblib.dump(scaler, scaler_p)
    joblib.dump(model, os.path.join(out, "sgd_final.pkl"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["endpoint_lgbm", "char_lgbm", "bpe_sgd"])
    ap.add_argument("--endpoint", choices=["pIC50", "pKi", "pKd"])
    ap.add_argument("--core", required=True)
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--seed", type=int, default=5313)
    ap.add_argument("--num_threads", type=int, default=96)
    ap.add_argument("--total_iters", type=int, default=50000)
    ap.add_argument("--stage_iters", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=100000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    core = load_core(os.path.abspath(args.core))
    configure(core, args)
    df, mask = core.load_and_prepare_data()
    idx_tr, idx_va, df_all, y_raw = core.make_fixed_train_valid_split(df, mask)
    if args.mode == "endpoint_lgbm":
        if not args.endpoint:
            raise ValueError("--endpoint is required for endpoint_lgbm")
        run_endpoint_lgbm(core, args.endpoint, idx_tr, idx_va, df_all)
    elif args.mode == "char_lgbm":
        run_char_lgbm(core, df, mask, idx_tr, idx_va, df_all, y_raw)
    else:
        run_bpe_sgd(core, df, mask, idx_tr, idx_va, df_all, y_raw,
                    args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
