#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版：屏幕+日志双输出 | 校正前后结果分存 | 校正前后全量误差图表 + 概率内容 + 
区间/置信区间内容 + 累计绝对误差阈值条形图（raw vs calibrated）+
观察值-预测值透明度散点图/hexbin图
"""

import os
import json
import math
import argparse
import subprocess
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt

# --------------------------
# RDKit 标准化
# --------------------------
def _require_rdkit():
    try:
        from rdkit import Chem
        return True
    except Exception:
        return False


def canonicalize_smiles(smiles: str) -> str:
    from rdkit import Chem
    if smiles is None:
        return ""
    s = str(smiles).strip()
    if not s or s.lower() == "nan":
        return ""
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def clean_protein(seq: str) -> str:
    if seq is None:
        return ""
    s = str(seq).replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
    return s.upper()


# --------------------------
# 通用数组清洗
# --------------------------
def _paired_valid_arrays(y, yp):
    y = np.asarray(y, dtype=float)
    yp = np.asarray(yp, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yp)
    return y[mask], yp[mask], mask


def _paired_valid_triplet(y, yp, z):
    y = np.asarray(y, dtype=float)
    yp = np.asarray(yp, dtype=float)
    z = np.asarray(z, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yp) & np.isfinite(z)
    return y[mask], yp[mask], z[mask], mask


AE_THRESHOLDS_DEFAULT = (0.5, 1.0, 1.5)


def summarize_absolute_error_thresholds(
    y_true, y_pred, thresholds=AE_THRESHOLDS_DEFAULT,
    dataset=None, aff_type=None, mode=None
):
    y, yp, _ = _paired_valid_arrays(y_true, y_pred)
    ae = np.abs(y - yp)
    n = len(ae)
    rows = []
    for thr in thresholds:
        count = int(np.sum(ae <= float(thr))) if n else 0
        ratio = float(count / n) if n else float("nan")
        rows.append({
            "dataset": dataset,
            "aff_type": aff_type,
            "mode": mode,
            "sample_count": int(n),
            "ae_threshold": float(thr),
            "count_le_threshold": count,
            "ratio_le_threshold": ratio,
            "percentage_le_threshold": float(ratio * 100.0) if n else float("nan"),
        })
    return pd.DataFrame(rows)


def ae_threshold_metrics_dict(summary_df):
    out = {}
    if summary_df is None or len(summary_df) == 0:
        return out
    for _, row in summary_df.iterrows():
        thr = float(row["ae_threshold"])
        key = str(thr).replace('.', '_')
        out[f"AE_le_{key}_count"] = int(row["count_le_threshold"])
        out[f"AE_le_{key}_ratio"] = float(row["ratio_le_threshold"])
        out[f"AE_le_{key}_pct"] = float(row["percentage_le_threshold"])
    return out


def format_ae_threshold_log(summary_df):
    if summary_df is None or len(summary_df) == 0:
        return "无有效样本"
    parts = []
    for _, row in summary_df.iterrows():
        thr = float(row["ae_threshold"])
        pct = float(row["percentage_le_threshold"])
        cnt = int(row["count_le_threshold"])
        n = int(row["sample_count"])
        parts.append(f"|AE|≤{thr:.1f}: {pct:.2f}% ({cnt}/{n})")
    return " | ".join(parts)


def to_jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (np.ndarray,)):
        return v.tolist()
    if isinstance(v, Path):
        return str(v)
    if pd.isna(v):
        return None
    return v


def save_json(obj, path: Path):
    def convert(x):
        if isinstance(x, dict):
            return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, list):
            return [convert(v) for v in x]
        return to_jsonable(x)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert(obj), f, ensure_ascii=False, indent=2)


# --------------------------
# 评估指标
# --------------------------
def rmse(y, yp):
    y, yp, _ = _paired_valid_arrays(y, yp)
    return float(np.sqrt(np.mean((y - yp) ** 2))) if len(y) else float("nan")


def mse(y, yp):
    y, yp, _ = _paired_valid_arrays(y, yp)
    return float(np.mean((y - yp) ** 2)) if len(y) else float("nan")


def mae(y, yp):
    y, yp, _ = _paired_valid_arrays(y, yp)
    return float(np.mean(np.abs(y - yp))) if len(y) else float("nan")


def r2(y, yp):
    y, yp, _ = _paired_valid_arrays(y, yp)
    if len(y) < 2:
        return float("nan")
    ss_res = np.sum((y - yp) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def pearsonr(y, yp):
    y, yp, _ = _paired_valid_arrays(y, yp)
    if len(y) < 2:
        return float("nan")
    if np.allclose(y, y[0]) or np.allclose(yp, yp[0]):
        return float("nan")
    return float(np.corrcoef(y, yp)[0, 1])


def spearmanr(y, yp):
    y, yp, _ = _paired_valid_arrays(y, yp)
    if len(y) < 2:
        return float("nan")
    ry = pd.Series(y).rank(method="average")
    ryp = pd.Series(yp).rank(method="average")
    return float(np.corrcoef(ry, ryp)[0, 1])


def concordance_index(y, yp):
    y, yp, _ = _paired_valid_arrays(y, yp)
    n = len(y)
    if n < 2:
        return float("nan")
    concordant = 0.0
    permissible = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if y[i] == y[j]:
                continue
            permissible += 1
            if yp[i] == yp[j]:
                concordant += 0.5
            elif (yp[i] > yp[j] and y[i] > y[j]) or (yp[i] < yp[j] and y[i] < y[j]):
                concordant += 1
    return float(concordant / permissible) if permissible > 0 else float("nan")


def r2m_metrics(y, yp):
    y, yp, _ = _paired_valid_arrays(y, yp)
    r2_val = r2(y, yp)
    if len(y) < 2 or np.isnan(r2_val):
        return {"R2m": np.nan, "R2m_bar": np.nan}
    denom = np.sum(yp ** 2)
    if denom <= 0:
        return {"R2m": np.nan, "R2m_bar": np.nan}
    k = np.sum(y * yp) / denom
    sst = np.sum((y - np.mean(y)) ** 2)
    if sst <= 0:
        return {"R2m": np.nan, "R2m_bar": np.nan}
    r02 = 1 - np.sum((y - k * yp) ** 2) / sst
    r2m = r2_val * (1 - math.sqrt(abs(r2_val - r02)))
    return {"R2m": float(r2m), "R2m_bar": float(r2m)}


# --------------------------
# 概率指标
# --------------------------
def auc_aupr(y_true_bin, score):
    y = np.asarray(y_true_bin, dtype=int)
    s = np.asarray(score, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    y = y[mask]
    s = s[mask]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    P = np.sum(y == 1)
    N = np.sum(y == 0)
    if P == 0 or N == 0:
        return float("nan"), float("nan")
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    tpr = tps / P
    fpr = fps / N
    roc_x = np.concatenate([[0.0], fpr, [1.0]])
    roc_y = np.concatenate([[0.0], tpr, [1.0]])
    roc_auc = float(np.trapz(roc_y, roc_x))
    prec = tps / np.maximum(tps + fps, 1)
    rec = tps / P
    pr_x = np.concatenate([[0.0], rec])
    pr_y = np.concatenate([[1.0], prec])
    pr_auc = float(np.trapz(pr_y, pr_x))
    return roc_auc, pr_auc


def ece_20bins(y_true_bin, prob):
    y = np.asarray(y_true_bin, dtype=int)
    p = np.asarray(prob, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    n = len(y)
    if n == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, 21)
    ece = 0.0
    for i in range(20):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & (p < hi) if i < 19 else (p >= lo) & (p <= hi)
        if not np.any(m):
            continue
        acc = np.mean(y[m])
        conf = np.mean(p[m])
        ece += (np.sum(m) / n) * abs(acc - conf)
    return float(ece)


def brier_score(y_true_bin, prob):
    y = np.asarray(y_true_bin, dtype=int)
    p = np.asarray(prob, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def log_loss_binary(y_true_bin, prob, eps=1e-15):
    y = np.asarray(y_true_bin, dtype=int)
    p = np.asarray(prob, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return float("nan")
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# --------------------------
# 线性校正（纯后处理） y_true = a * y_pred + b
# --------------------------
def linear_calibration(y_true, y_pred):
    y_true, y_pred, _ = _paired_valid_arrays(y_true, y_pred)
    a, b, _, _, _ = linregress(y_pred, y_true)
    y_pred_corrected = a * y_pred + b
    return float(a), float(b), y_pred_corrected


# --------------------------
# 概率 / 区间工具
# --------------------------
def infer_pred_col(df):
    for c in ["pred_pX", "pred_final", "y_pred", "pred"]:
        if c in df.columns:
            return c
    raise ValueError(f"未找到预测列，现有列：{list(df.columns)}")


def infer_sigma_col(df):
    for c in ["sigma", "pred_sigma", "sigma_pred", "uncertainty", "pred_std"]:
        if c in df.columns:
            return c
    return None


def infer_interval_cols(df):
    lo_col = None
    hi_col = None
    err_col = None
    acc_col = None
    for c in ["pred_pX_lo", "pred_lo", "lo"]:
        if c in df.columns:
            lo_col = c
            break
    for c in ["pred_pX_hi", "pred_hi", "hi"]:
        if c in df.columns:
            hi_col = c
            break
    for c in ["err_bound", "error_bound", "pred_err_bound"]:
        if c in df.columns:
            err_col = c
            break
    for c in ["accepted", "is_accepted"]:
        if c in df.columns:
            acc_col = c
            break
    return lo_col, hi_col, err_col, acc_col


def _norm_cdf(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def gaussian_prob_ge_threshold(pred, sigma, threshold_pk, sigma_scale=1.0):
    pred = np.asarray(pred, dtype=float)
    sigma = np.asarray(sigma, dtype=float) * float(sigma_scale)
    sigma = np.maximum(sigma, 1e-12)
    z = (threshold_pk - pred) / sigma
    prob = 1.0 - _norm_cdf(z)
    return np.clip(prob, 0.0, 1.0)


def gaussian_interval(pred, sigma, alpha=0.95, sigma_scale=1.0):
    alpha_to_z = {
        0.50: 0.67448975,
        0.80: 1.28155157,
        0.90: 1.64485363,
        0.95: 1.95996398,
        0.98: 2.32634787,
        0.99: 2.57582930,
    }
    if alpha not in alpha_to_z:
        raise ValueError(f"暂只支持 alpha in {sorted(alpha_to_z.keys())}")
    z = alpha_to_z[alpha]
    pred = np.asarray(pred, dtype=float)
    sigma = np.asarray(sigma, dtype=float) * float(sigma_scale)
    sigma = np.maximum(sigma, 1e-12)
    lo = pred - z * sigma
    hi = pred + z * sigma
    return lo, hi


def interval_metrics(y_true, lo, hi):
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    y = y[mask]
    lo = lo[mask]
    hi = hi[mask]
    if len(y) == 0:
        return {
            "coverage": float("nan"),
            "width_mean": float("nan"),
            "width_median": float("nan"),
            "below_rate": float("nan"),
            "above_rate": float("nan"),
        }
    width = hi - lo
    return {
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "width_mean": float(np.mean(width)),
        "width_median": float(np.median(width)),
        "below_rate": float(np.mean(y < lo)),
        "above_rate": float(np.mean(y > hi)),
    }


def transform_interval_linear(lo, hi, a, b):
    lo2 = a * np.asarray(lo, dtype=float) + b
    hi2 = a * np.asarray(hi, dtype=float) + b
    lo_new = np.minimum(lo2, hi2)
    hi_new = np.maximum(lo2, hi2)
    return lo_new, hi_new


# --------------------------
# 绘图函数：观察值-预测值透明度散点图 / hexbin图
# --------------------------
def plot_obs_vs_pred_scatter(
    y_true, y_pred, out_path, title,
    alpha=0.35, s=20, show_grid=False, grid_alpha=0.0,
    show_diag=True, show_r2=True, show_rmse=True,
    show_ci=True, show_r2m=True
):
    """
    绘制观察值 vs 预测值散点图（带透明度）
    左上角默认显示 R2、RMSE、CI、R2m
    """
    y_true, y_pred, _ = _paired_valid_arrays(y_true, y_pred)
    if len(y_true) == 0:
        return

    plt.figure(figsize=(6, 6), dpi=150)

    # 散点图
    plt.scatter(y_true, y_pred, alpha=alpha, s=s, c="#4472c4", edgecolors="none")

    # 对角线 y=x
    if show_diag:
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.5, alpha=0.7, label="y=x")

    # 计算并显示指标
    r2_val = r2(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    ci_val = concordance_index(y_true, y_pred)
    r2m_val = r2m_metrics(y_true, y_pred).get("R2m", float("nan"))

    stats_text = []
    if show_r2 and not np.isnan(r2_val):
        stats_text.append(f"$R^2$ = {r2_val:.3f}")
    if show_rmse and not np.isnan(rmse_val):
        stats_text.append(f"RMSE = {rmse_val:.3f}")
    if show_ci and not np.isnan(ci_val):
        stats_text.append(f"CI = {ci_val:.3f}")
    if show_r2m and not np.isnan(r2m_val):
        stats_text.append(f"$R^2_m$ = {r2m_val:.3f}")

    if stats_text:
        plt.text(
            0.05, 0.95, "\n".join(stats_text),
            transform=plt.gca().transAxes,
            fontsize=10, ha="left", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray")
        )

    plt.xlabel("Observed Value")
    plt.ylabel("Predicted Value")
    plt.title(title)
    if show_grid:
        plt.grid(alpha=grid_alpha)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_obs_vs_pred_hexbin(
    y_true, y_pred, out_path, title,
    gridsize=50, cmap="Blues", show_diag=True,
    show_r2=True, show_rmse=True, show_grid=False, grid_alpha=0.0
):
    """
    绘制观察值 vs 预测值 hexbin 图（用于大数据量）
    """
    y_true, y_pred, _ = _paired_valid_arrays(y_true, y_pred)
    if len(y_true) == 0:
        return
    
    plt.figure(figsize=(6, 6), dpi=150)
    
    # hexbin 图
    hb = plt.hexbin(y_true, y_pred, gridsize=gridsize, cmap=cmap, mincnt=1, alpha=0.8)
    plt.colorbar(hb, label="Count")
    
    # 对角线 y=x
    if show_diag:
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, alpha=0.7, label="y=x")
    
    # 计算并显示指标
    r2_val = r2(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    mae_val = mae(y_true, y_pred)
    
    stats_text = []
    if show_r2 and not np.isnan(r2_val):
        stats_text.append(f"$R^2$ = {r2_val:.3f}")
    if show_rmse and not np.isnan(rmse_val):
        stats_text.append(f"RMSE = {rmse_val:.3f}")
    if not np.isnan(mae_val):
        stats_text.append(f"MAE = {mae_val:.3f}")
    
    if stats_text:
        plt.text(0.05, 0.95, "\n".join(stats_text),
                 transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    plt.xlabel("Observed Value")
    plt.ylabel("Predicted Value")
    plt.title(title)
    if show_grid:
        plt.grid(alpha=grid_alpha)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# --------------------------
# 通用误差绘图函数（支持校正前/后）
# --------------------------
def plot_error_hist(y_true, y_pred, out_path, title):
    ae = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    bins = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    counts, edges = np.histogram(ae, bins=bins)
    x_labels = [f"[{bins[i]:.2f},{bins[i+1]:.2f})" for i in range(len(bins)-1)]
    
    plt.figure(figsize=(8, 4), dpi=150)
    plt.bar(x_labels, counts, color="#4472c4", alpha=0.8)
    plt.xlabel("Absolute Error (|pred-obs|)")
    plt.ylabel("Number of Samples")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_cumulative_error(
    y_true, y_pred, out_path, title,
    thresholds=AE_THRESHOLDS_DEFAULT,
    show_grid=False, grid_alpha=0.0,
    csv_path=None, dataset=None, aff_type=None, mode=None
):
    ae = np.sort(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)))
    ae = ae[np.isfinite(ae)]
    n = len(ae)
    if n == 0:
        return None, None
    percent = np.arange(1, n + 1) / n * 100.0
    summary_df = summarize_absolute_error_thresholds(
        y_true, y_pred, thresholds=thresholds, dataset=dataset, aff_type=aff_type, mode=mode
    )

    plot_df = pd.DataFrame({
        "dataset": dataset,
        "aff_type": aff_type,
        "mode": mode,
        "rank": np.arange(1, n + 1, dtype=int),
        "absolute_error_threshold": ae,
        "cumulative_percentage": percent,
    })

    if summary_df is not None and len(summary_df) > 0:
        thr_map = {
            float(row["ae_threshold"]): float(row["percentage_le_threshold"])
            for _, row in summary_df.iterrows()
        }
        plot_df["is_key_threshold"] = plot_df["absolute_error_threshold"].isin(list(thr_map.keys()))
    else:
        plot_df["is_key_threshold"] = False

    if csv_path is not None:
        plot_df.to_csv(csv_path, index=False, encoding="utf-8")

    plt.figure(figsize=(8, 4), dpi=150)
    plt.plot(ae, percent, color="#e74c3c", linewidth=2)

    marker_cycle = ["o", "s", "^", "D", "P", "X"]
    legend_lines = []
    for i, (_, row) in enumerate(summary_df.iterrows()):
        thr = float(row["ae_threshold"])
        pct = float(row["percentage_le_threshold"])
        mk = marker_cycle[i % len(marker_cycle)]
        plt.axvline(thr, linestyle="--", linewidth=1.0, color="gray", alpha=0.8)
        plt.axhline(pct, linestyle=":", linewidth=1.0, color="gray", alpha=0.8)
        plt.scatter([thr], [pct], s=34, zorder=3, marker=mk, color="#e74c3c", edgecolors="black", linewidths=0.4)
        legend_lines.append(f"{mk}  ≤{thr:.1f}: {pct:.2f}%")

    x_max = max(float(np.max(ae)), max(float(t) for t in thresholds))
    x_pad = 0.30 * x_max if x_max > 0 else 0.5
    plt.xlim(left=0.0, right=x_max + x_pad)
    plt.ylim(0, 100)
    plt.xlabel("Absolute Error Threshold")
    plt.ylabel("Percentage of Samples ≤ Threshold (%)")
    plt.title(title)

    if legend_lines:
        legend_text = "\n".join(legend_lines)
        plt.text(
            0.985, 0.50, legend_text,
            transform=plt.gca().transAxes,
            fontsize=9, ha="right", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.90, edgecolor="gray")
        )

    if show_grid:
        plt.grid(alpha=grid_alpha)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return summary_df, plot_df


def plot_ae_threshold_bar(summary_df, out_path, title, value_col="percentage_le_threshold"):
    """
    按 AE 阈值绘制条形图：
    - 若 summary_df 含 raw / calibrated，则画分组条形图
    - 若只含单一 mode，则画单组条形图
    """
    if summary_df is None or len(summary_df) == 0:
        return
    
    df = summary_df.copy()
    df["ae_threshold"] = pd.to_numeric(df["ae_threshold"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df[df["ae_threshold"].notna() & df[value_col].notna()].copy()
    if len(df) == 0:
        return
    
    thresholds = sorted(df["ae_threshold"].unique().tolist())
    
    if "mode" in df.columns and df["mode"].notna().any():
        modes = [m for m in ["raw", "calibrated"] if m in set(df["mode"].astype(str))]
        if len(modes) == 0:
            modes = sorted(df["mode"].dropna().astype(str).unique().tolist())
    else:
        modes = ["summary"]
        df["mode"] = "summary"
    
    x = np.arange(len(thresholds), dtype=float)
    width = 0.36 if len(modes) <= 2 else 0.8 / max(len(modes), 1)
    
    plt.figure(figsize=(7.2, 4.6), dpi=150)
    color_map = {
        "raw": "#4472c4",
        "calibrated": "#e74c3c",
        "summary": "#4472c4",
    }
    
    for i, mode in enumerate(modes):
        sub = df[df["mode"].astype(str) == str(mode)].copy()
        vals = []
        counts = []
        totals = []
        
        for thr in thresholds:
            row = sub[np.isclose(sub["ae_threshold"].to_numpy(dtype=float), float(thr))]
            if len(row) == 0:
                vals.append(np.nan)
                counts.append(0)
                totals.append(0)
            else:
                row0 = row.iloc[0]
                vals.append(float(row0[value_col]))
                counts.append(int(row0["count_le_threshold"]) if "count_le_threshold" in row0 else 0)
                totals.append(int(row0["sample_count"]) if "sample_count" in row0 else 0)
        
        offset = (i - (len(modes) - 1) / 2.0) * width
        xpos = x + offset
        bars = plt.bar(
            xpos, vals,
            width=width * 0.92,
            label=mode.capitalize(),
            color=color_map.get(mode, None),
            alpha=0.88,
            edgecolor="black",
            linewidth=0.6,
        )
        
        for j, bar in enumerate(bars):
            h = bar.get_height()
            if not np.isfinite(h):
                continue
            txt = f"{h:.2f}%"
            if counts[j] > 0 and totals[j] > 0:
                txt += f"\n({counts[j]}/{totals[j]})"
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 1.2,
                txt,
                ha="center", va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.85, edgecolor="gray"),
            )
    
    ymax = float(np.nanmax(df[value_col].to_numpy(dtype=float))) if len(df) else 100.0
    ymax = max(100.0 if ymax > 92 else ymax + 12.0, 15.0)
    
    plt.xticks(x, [f"≤{thr:.1f}" for thr in thresholds])
    plt.ylim(0, ymax)
    plt.xlabel("Absolute Error Threshold")
    plt.ylabel("Percentage of Samples ≤ Threshold (%)")
    plt.title(title)
    if len(modes) > 1:
        plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# --------------------------
# 模型预测
# --------------------------
def run_bundle_predict(bundle_dir: Path, in_csv: Path, out_csv: Path, calib_json: Path = None):
    pred_cli = bundle_dir / "predict_cli.py"
    cmd = ["python", str(pred_cli), "--in", str(in_csv), "--out", str(out_csv)]
    if calib_json is not None:
        cmd.extend(["--calib_json", str(calib_json)])
    
    logging.info(f"Running: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"模型预测失败！\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
    if r.stdout.strip():
        logging.info(r.stdout.strip())
    if r.stderr.strip():
        logging.info(r.stderr.strip())


# --------------------------
# 核心：单数据集处理
# --------------------------
def process_dataset(
    name,
    src_csv,
    smi_col,
    prot_col,
    y_col,
    aff_type,
    bundle_dir,
    out_dir,
    threshold_pk,
    gauss_alpha,
    prob_sigma_scale,
    q_scale,
    skip_predict=False,
    pred_csv_override=None,
):
    logging.info(f"\n===== {name} ({aff_type}) =====")
    
    raw = pd.read_csv(src_csv)
    
    # 预处理 + UID
    raw["smiles_norm"] = raw[smi_col].apply(canonicalize_smiles)
    raw["protein_clean"] = raw[prot_col].apply(clean_protein)
    raw["affinity_value"] = pd.to_numeric(raw[y_col], errors="coerce")
    raw["affinity_type"] = aff_type
    raw["_pair_uid"] = f"{name}_" + raw.index.astype(str)
    
    # 过滤无效
    df = raw[
        (raw["smiles_norm"] != "") &
        (raw["protein_clean"] != "") &
        (raw["affinity_value"].notna())
    ].copy()
    
    logging.info(f"原始样本：{len(raw)}")
    logging.info(f"有效样本：{len(df)}")
    
    # 模型输入
    model_in = df[["smiles_norm", "protein_clean", "affinity_type", "_pair_uid"]].rename(
        columns={"smiles_norm": "smiles", "protein_clean": "protein"}
    )
    in_csv = out_dir / f"{name}_input.csv"
    model_in.to_csv(in_csv, index=False)
    
    # 预测
    pred_csv = out_dir / f"{name}_pred.csv" if pred_csv_override is None else Path(pred_csv_override)
    if not skip_predict:
        run_bundle_predict(bundle_dir, in_csv, pred_csv)
    
    # UID 精准合并
    pred_df = pd.read_csv(pred_csv)
    pred_col = infer_pred_col(pred_df)
    sigma_col = infer_sigma_col(pred_df)
    lo_col, hi_col, err_col, acc_col = infer_interval_cols(pred_df)
    
    merged = pd.merge(pred_df, df[["_pair_uid", "affinity_value"]], on="_pair_uid", how="inner")
    
    y_true = merged["affinity_value"].values.astype(float)
    y_pred = pd.to_numeric(merged[pred_col], errors="coerce").values.astype(float)
    
    # 只保留 paired valid
    y_true_v, y_pred_v, mask = _paired_valid_arrays(y_true, y_pred)
    merged = merged.loc[mask].copy()
    y_true = y_true_v
    y_pred = y_pred_v
    
    logging.info(f"合并后 paired valid 样本：{len(y_true)}")
    logging.info(f"预测列：{pred_col}")
    logging.info(f"sigma列：{sigma_col}")
    logging.info(f"区间列：lo={lo_col}, hi={hi_col}, err={err_col}, accepted={acc_col}")
    
    # =====================
    # 原始预测指标
    # =====================
    raw_rmse = rmse(y_true, y_pred)
    raw_mse = mse(y_true, y_pred)
    raw_mae = mae(y_true, y_pred)
    raw_r2 = r2(y_true, y_pred)
    raw_pearson = pearsonr(y_true, y_pred)
    raw_spearman = spearmanr(y_true, y_pred)
    raw_ci = concordance_index(y_true, y_pred)
    raw_r2m = r2m_metrics(y_true, y_pred)
    
    # =====================
    # 线性校正
    # =====================
    a, b, y_corr = linear_calibration(y_true, y_pred)
    
    # =====================
    # 校正后指标
    # =====================
    corr_rmse = rmse(y_true, y_corr)
    corr_mse = mse(y_true, y_corr)
    corr_mae = mae(y_true, y_corr)
    corr_r2 = r2(y_true, y_corr)
    corr_pearson = pearsonr(y_true, y_corr)
    corr_spearman = spearmanr(y_true, y_corr)
    corr_ci = concordance_index(y_true, y_corr)
    corr_r2m = r2m_metrics(y_true, y_corr)
    
    # =====================
    # raw / corr merged 明细
    # =====================
    raw_merged = merged.copy()
    raw_merged["y_true"] = y_true
    raw_merged["y_pred_raw"] = y_pred
    raw_merged["abs_error_raw"] = np.abs(y_true - y_pred)
    
    corr_merged = merged.copy()
    corr_merged["y_true"] = y_true
    corr_merged["y_pred_calibrated"] = y_corr
    corr_merged["abs_error_calibrated"] = np.abs(y_true - y_corr)
    
    # =====================
    # accepted rate
    # =====================
    accepted_rate = float("nan")
    n_accepted = float("nan")
    if acc_col is not None and acc_col in merged.columns:
        acc = pd.to_numeric(merged[acc_col], errors="coerce").fillna(0).astype(int).to_numpy()
        accepted_rate = float(np.mean(acc == 1))
        n_accepted = int(np.sum(acc == 1))
    
    # =====================
    # 概率内容：仅 sigma 存在时
    # =====================
    raw_prob_metrics = {}
    corr_prob_metrics = {}
    
    if sigma_col is not None and sigma_col in merged.columns:
        sigma = pd.to_numeric(merged[sigma_col], errors="coerce").to_numpy(dtype=float)
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)
        
        y_g, p_raw_g, s_g, mask_g = _paired_valid_triplet(y_true, y_pred, sigma)
        _, p_corr_g, s_corr_g, _ = _paired_valid_triplet(y_true, y_corr, np.abs(a) * sigma)
        
        if len(y_g) > 0:
            prob_raw = gaussian_prob_ge_threshold(
                p_raw_g, s_g, threshold_pk,
                sigma_scale=prob_sigma_scale
            )
            prob_corr = gaussian_prob_ge_threshold(
                p_corr_g, s_corr_g, threshold_pk,
                sigma_scale=prob_sigma_scale
            )
            
            y_bin = (y_g >= threshold_pk).astype(int)
            
            auc_pred_raw, aupr_pred_raw = auc_aupr(y_bin, p_raw_g)
            auc_prob_raw, aupr_prob_raw = auc_aupr(y_bin, prob_raw)
            auc_pred_corr, aupr_pred_corr = auc_aupr(y_bin, p_corr_g)
            auc_prob_corr, aupr_prob_corr = auc_aupr(y_bin, prob_corr)
            
            raw_prob_metrics = {
                "strong_prevalence": float(np.mean(y_bin)),
                "AUC_score_pred": auc_pred_raw,
                "AUPR_score_pred": aupr_pred_raw,
                "AUC_score_prob": auc_prob_raw,
                "AUPR_score_prob": aupr_prob_raw,
                "ECE_20bins": ece_20bins(y_bin, prob_raw),
                "Brier": brier_score(y_bin, prob_raw),
                "LogLoss": log_loss_binary(y_bin, prob_raw),
                "prob_sigma_scale": float(prob_sigma_scale),
            }
            
            corr_prob_metrics = {
                "strong_prevalence": float(np.mean(y_bin)),
                "AUC_score_pred": auc_pred_corr,
                "AUPR_score_pred": aupr_pred_corr,
                "AUC_score_prob": auc_prob_corr,
                "AUPR_score_prob": aupr_prob_corr,
                "ECE_20bins": ece_20bins(y_bin, prob_corr),
                "Brier": brier_score(y_bin, prob_corr),
                "LogLoss": log_loss_binary(y_bin, prob_corr),
                "prob_sigma_scale": float(prob_sigma_scale),
            }
            
            # 写回明细
            raw_merged.loc[mask_g, "sigma_raw"] = s_g
            raw_merged.loc[mask_g, "prob_ge_threshold_raw"] = prob_raw
            corr_merged.loc[mask_g, "sigma_calibrated"] = s_corr_g
            corr_merged.loc[mask_g, "prob_ge_threshold_calibrated"] = prob_corr
    
    # =====================
    # 区间内容
    # =====================
    raw_interval_metrics = {}
    corr_interval_metrics = {}
    raw_lo = raw_hi = None
    corr_lo = corr_hi = None
    
    # 优先已有 lo/hi
    if lo_col is not None and hi_col is not None and lo_col in merged.columns and hi_col in merged.columns:
        raw_lo = pd.to_numeric(merged[lo_col], errors="coerce").to_numpy(dtype=float)
        raw_hi = pd.to_numeric(merged[hi_col], errors="coerce").to_numpy(dtype=float)
        corr_lo, corr_hi = transform_interval_linear(raw_lo, raw_hi, a, b)
    
    # 其次 err_bound
    elif err_col is not None and err_col in merged.columns:
        errb = pd.to_numeric(merged[err_col], errors="coerce").to_numpy(dtype=float)
        raw_lo = y_pred - errb
        raw_hi = y_pred + errb
        corr_lo = y_corr - np.abs(a) * errb
        corr_hi = y_corr + np.abs(a) * errb
    
    # 最后 q_scale * sigma
    elif sigma_col is not None and sigma_col in merged.columns and q_scale is not None:
        sigma = pd.to_numeric(merged[sigma_col], errors="coerce").to_numpy(dtype=float)
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)
        raw_lo = y_pred - float(q_scale) * sigma
        raw_hi = y_pred + float(q_scale) * sigma
        corr_lo = y_corr - float(q_scale) * np.abs(a) * sigma
        corr_hi = y_corr + float(q_scale) * np.abs(a) * sigma
    
    if raw_lo is not None and raw_hi is not None:
        raw_interval_metrics = interval_metrics(y_true, raw_lo, raw_hi)
        raw_merged["interval_lo_raw"] = raw_lo
        raw_merged["interval_hi_raw"] = raw_hi
    
    if corr_lo is not None and corr_hi is not None:
        corr_interval_metrics = interval_metrics(y_true, corr_lo, corr_hi)
        corr_merged["interval_lo_calibrated"] = corr_lo
        corr_merged["interval_hi_calibrated"] = corr_hi
    
    # =====================
    # Gaussian PI 单独统计
    # =====================
    raw_gauss_metrics = {}
    corr_gauss_metrics = {}
    
    if sigma_col is not None and sigma_col in merged.columns:
        sigma = pd.to_numeric(merged[sigma_col], errors="coerce").to_numpy(dtype=float)
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)
        
        y_g, p_raw_g, s_g, mask_g = _paired_valid_triplet(y_true, y_pred, sigma)
        _, p_corr_g, s_corr_g, _ = _paired_valid_triplet(y_true, y_corr, np.abs(a) * sigma)
        
        if len(y_g) > 0:
            lo_g_raw, hi_g_raw = gaussian_interval(
                p_raw_g, s_g,
                alpha=gauss_alpha,
                sigma_scale=prob_sigma_scale
            )
            lo_g_corr, hi_g_corr = gaussian_interval(
                p_corr_g, s_corr_g,
                alpha=gauss_alpha,
                sigma_scale=prob_sigma_scale
            )
            
            alpha_tag = int(gauss_alpha * 100)
            g_raw = interval_metrics(y_g, lo_g_raw, hi_g_raw)
            g_corr = interval_metrics(y_g, lo_g_corr, hi_g_corr)
            
            raw_gauss_metrics = {
                f"gauss_PI_{alpha_tag}_coverage": g_raw["coverage"],
                f"gauss_PI_{alpha_tag}_width_mean": g_raw["width_mean"],
                f"gauss_PI_{alpha_tag}_width_median": g_raw["width_median"],
                f"gauss_PI_{alpha_tag}_below_rate": g_raw["below_rate"],
                f"gauss_PI_{alpha_tag}_above_rate": g_raw["above_rate"],
            }
            
            corr_gauss_metrics = {
                f"gauss_PI_{alpha_tag}_coverage": g_corr["coverage"],
                f"gauss_PI_{alpha_tag}_width_mean": g_corr["width_mean"],
                f"gauss_PI_{alpha_tag}_width_median": g_corr["width_median"],
                f"gauss_PI_{alpha_tag}_below_rate": g_corr["below_rate"],
                f"gauss_PI_{alpha_tag}_above_rate": g_corr["above_rate"],
            }
            
            raw_merged.loc[mask_g, f"gauss_lo_raw_{alpha_tag}"] = lo_g_raw
            raw_merged.loc[mask_g, f"gauss_hi_raw_{alpha_tag}"] = hi_g_raw
            corr_merged.loc[mask_g, f"gauss_lo_calibrated_{alpha_tag}"] = lo_g_corr
            corr_merged.loc[mask_g, f"gauss_hi_calibrated_{alpha_tag}"] = hi_g_corr
    
    # =====================
    # 绝对误差累计阈值汇总 + 条形图
    # =====================
    raw_ae_summary = summarize_absolute_error_thresholds(
        y_true, y_pred,
        thresholds=AE_THRESHOLDS_DEFAULT,
        dataset=name, aff_type=aff_type, mode="raw"
    )
    corr_ae_summary = summarize_absolute_error_thresholds(
        y_true, y_corr,
        thresholds=AE_THRESHOLDS_DEFAULT,
        dataset=name, aff_type=aff_type, mode="calibrated"
    )
    
    ae_summary_df = pd.concat([raw_ae_summary, corr_ae_summary], ignore_index=True)
    ae_summary_csv = out_dir / f"{name}_absolute_error_threshold_summary.csv"
    ae_summary_df.to_csv(ae_summary_csv, index=False, encoding="utf-8")
    
    ae_bar_png = out_dir / f"{name}_absolute_error_threshold_bar.png"
    raw_ae_bar_png = out_dir / f"{name}_raw_absolute_error_threshold_bar.png"
    corr_ae_bar_png = out_dir / f"{name}_corr_absolute_error_threshold_bar.png"
    
    plot_ae_threshold_bar(
        ae_summary_df, ae_bar_png,
        f"{name} Absolute Error Threshold Comparison"
    )
    plot_ae_threshold_bar(
        raw_ae_summary, raw_ae_bar_png,
        f"{name} Raw Absolute Error Threshold Bar"
    )
    plot_ae_threshold_bar(
        corr_ae_summary, corr_ae_bar_png,
        f"{name} Corrected Absolute Error Threshold Bar"
    )
    
    # =====================
    # 绘制校正前后图表
    # =====================
    # 误差直方图
    plot_error_hist(
        y_true, y_pred,
        out_dir / f"{name}_raw_error_hist.png",
        f"{name} Raw |Error| Distribution"
    )
    plot_error_hist(
        y_true, y_corr,
        out_dir / f"{name}_corr_error_hist.png",
        f"{name} Corrected |Error| Distribution"
    )
    
    # 累计误差图 + 作图 CSV
    raw_cumulative_csv = out_dir / f"{name}_raw_cumulative_plot.csv"
    corr_cumulative_csv = out_dir / f"{name}_corr_cumulative_plot.csv"

    plot_cumulative_error(
        y_true, y_pred,
        out_dir / f"{name}_raw_cumulative.png",
        f"{name} Raw Cumulative Error Percentage",
        thresholds=AE_THRESHOLDS_DEFAULT,
        csv_path=raw_cumulative_csv,
        dataset=name, aff_type=aff_type, mode="raw"
    )
    plot_cumulative_error(
        y_true, y_corr,
        out_dir / f"{name}_corr_cumulative.png",
        f"{name} Corrected Cumulative Error Percentage",
        thresholds=AE_THRESHOLDS_DEFAULT,
        csv_path=corr_cumulative_csv,
        dataset=name, aff_type=aff_type, mode="calibrated"
    )
    
    # =====================
    # 观察值-预测值散点图/hexbin图
    # =====================
    # 原始数据散点图
    plot_obs_vs_pred_scatter(
        y_true, y_pred,
        out_dir / f"{name}_raw_scatter.png",
        f"{name} Raw: Observed vs Predicted",
        alpha=0.35, s=20
    )
    
    # 校正后数据散点图
    plot_obs_vs_pred_scatter(
        y_true, y_corr,
        out_dir / f"{name}_corr_scatter.png",
        f"{name} Corrected: Observed vs Predicted",
        alpha=0.35, s=20
    )
    
    # 如果样本量较大，同时生成 hexbin 图
    if len(y_true) > 1000:
        plot_obs_vs_pred_hexbin(
            y_true, y_pred,
            out_dir / f"{name}_raw_hexbin.png",
            f"{name} Raw: Observed vs Predicted (Hexbin)",
            gridsize=50, cmap="Blues"
        )
        plot_obs_vs_pred_hexbin(
            y_true, y_corr,
            out_dir / f"{name}_corr_hexbin.png",
            f"{name} Corrected: Observed vs Predicted (Hexbin)",
            gridsize=50, cmap="Blues"
        )
    
    logging.info(f"已生成 {name} 校正前后全量误差图表（累计曲线 + 条形图 + 散点图）")
    
    # =====================
    # 日志打印
    # =====================
    logging.info("\n模型原始结果：")
    logging.info(
        f"RMSE = {raw_rmse:.4f} | MSE = {raw_mse:.4f} | MAE = {raw_mae:.4f} | "
        f"R2 = {raw_r2:.4f} | R2m = {raw_r2m['R2m']:.4f}"
    )
    logging.info(
        f"Pearson = {raw_pearson:.4f} | Spearman = {raw_spearman:.4f} | CI = {raw_ci:.4f}"
    )
    
    logging.info(f"\n线性拟合公式：y_true = {a:.4f} * y_pred + {b:.4f}")
    
    logging.info("\n线性校正后结果：")
    logging.info(
        f"RMSE = {corr_rmse:.4f} | MSE = {corr_mse:.4f} | MAE = {corr_mae:.4f} | "
        f"R2 = {corr_r2:.4f} | R2m = {corr_r2m['R2m']:.4f}"
    )
    logging.info(
        f"Pearson = {corr_pearson:.4f} | Spearman = {corr_spearman:.4f} | CI = {corr_ci:.4f}"
    )
    
    logging.info("原始累计绝对误差关键阈值：" + format_ae_threshold_log(raw_ae_summary))
    logging.info("校正后累计绝对误差关键阈值：" + format_ae_threshold_log(corr_ae_summary))
    
    if not np.isnan(accepted_rate):
        logging.info(f"\naccepted_rate = {accepted_rate:.4f} | N_accepted = {int(n_accepted)}")
    
    if raw_prob_metrics:
        logging.info("\n原始概率结果：")
        logging.info(
            f"Prevalence = {raw_prob_metrics['strong_prevalence']:.4f} | "
            f"AUC(pred) = {raw_prob_metrics['AUC_score_pred']:.4f} | "
            f"AUPR(pred) = {raw_prob_metrics['AUPR_score_pred']:.4f}"
        )
        logging.info(
            f"AUC(prob) = {raw_prob_metrics['AUC_score_prob']:.4f} | "
            f"AUPR(prob) = {raw_prob_metrics['AUPR_score_prob']:.4f} | "
            f"ECE = {raw_prob_metrics['ECE_20bins']:.4f} | "
            f"Brier = {raw_prob_metrics['Brier']:.4f} | "
            f"LogLoss = {raw_prob_metrics['LogLoss']:.4f}"
        )
    
    if corr_prob_metrics:
        logging.info("\n校正后概率结果：")
        logging.info(
            f"Prevalence = {corr_prob_metrics['strong_prevalence']:.4f} | "
            f"AUC(pred) = {corr_prob_metrics['AUC_score_pred']:.4f} | "
            f"AUPR(pred) = {corr_prob_metrics['AUPR_score_pred']:.4f}"
        )
        logging.info(
            f"AUC(prob) = {corr_prob_metrics['AUC_score_prob']:.4f} | "
            f"AUPR(prob) = {corr_prob_metrics['AUPR_score_prob']:.4f} | "
            f"ECE = {corr_prob_metrics['ECE_20bins']:.4f} | "
            f"Brier = {corr_prob_metrics['Brier']:.4f} | "
            f"LogLoss = {corr_prob_metrics['LogLoss']:.4f}"
        )
    
    if raw_interval_metrics:
        logging.info("\n原始区间结果：")
        logging.info(
            f"Coverage = {raw_interval_metrics['coverage']:.4f} | "
            f"MedianWidth = {raw_interval_metrics['width_median']:.4f} | "
            f"MeanWidth = {raw_interval_metrics['width_mean']:.4f} | "
            f"Below = {raw_interval_metrics['below_rate']:.4f} | "
            f"Above = {raw_interval_metrics['above_rate']:.4f}"
        )
    
    if corr_interval_metrics:
        logging.info("\n校正后区间结果：")
        logging.info(
            f"Coverage = {corr_interval_metrics['coverage']:.4f} | "
            f"MedianWidth = {corr_interval_metrics['width_median']:.4f} | "
            f"MeanWidth = {corr_interval_metrics['width_mean']:.4f} | "
            f"Below = {corr_interval_metrics['below_rate']:.4f} | "
            f"Above = {corr_interval_metrics['above_rate']:.4f}"
        )
    
    if raw_gauss_metrics:
        logging.info("\n原始 Gaussian PI 结果：")
        logging.info(str(raw_gauss_metrics))
    
    if corr_gauss_metrics:
        logging.info("\n校正后 Gaussian PI 结果：")
        logging.info(str(corr_gauss_metrics))
    
    # =====================
    # 保存 metrics
    # =====================
    raw_metrics = {
        "dataset": name,
        "aff_type": aff_type,
        "sample_count": len(y_true),
        "threshold_pk": threshold_pk,
        "gauss_alpha": gauss_alpha,
        "accepted_rate": accepted_rate,
        "N_accepted": n_accepted,
        "rmse": raw_rmse,
        "mse": raw_mse,
        "mae": raw_mae,
        "r2": raw_r2,
        "r2m": raw_r2m["R2m"],
        "pearson": raw_pearson,
        "spearman": raw_spearman,
        "ci": raw_ci,
    }
    raw_metrics.update(ae_threshold_metrics_dict(raw_ae_summary))
    raw_metrics.update(raw_prob_metrics)
    raw_metrics.update({f"interval_{k}": v for k, v in raw_interval_metrics.items()})
    raw_metrics.update(raw_gauss_metrics)
    
    calib_metrics = {
        "dataset": name,
        "aff_type": aff_type,
        "sample_count": len(y_true),
        "threshold_pk": threshold_pk,
        "gauss_alpha": gauss_alpha,
        "calib_slope_a": a,
        "calib_intercept_b": b,
        "accepted_rate": accepted_rate,
        "N_accepted": n_accepted,
        "rmse": corr_rmse,
        "mse": corr_mse,
        "mae": corr_mae,
        "r2": corr_r2,
        "r2m": corr_r2m["R2m"],
        "pearson": corr_pearson,
        "spearman": corr_spearman,
        "ci": corr_ci,
    }
    calib_metrics.update(ae_threshold_metrics_dict(corr_ae_summary))
    calib_metrics.update(corr_prob_metrics)
    calib_metrics.update({f"interval_{k}": v for k, v in corr_interval_metrics.items()})
    calib_metrics.update(corr_gauss_metrics)
    
    pd.DataFrame([raw_metrics]).to_csv(
        out_dir / f"{name}_raw_metrics.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame([calib_metrics]).to_csv(
        out_dir / f"{name}_calibrated_metrics.csv", index=False, encoding="utf-8"
    )
    
    # =====================
    # 保存 merged 明细
    # =====================
    raw_merged.to_csv(out_dir / f"{name}_raw_merged.csv", index=False, encoding="utf-8")
    corr_merged.to_csv(out_dir / f"{name}_calibrated_merged.csv", index=False, encoding="utf-8")
    
    logging.info(f"已保存 {name} 原始/校正后指标文件、merged 明细与累计误差阈值汇总 CSV")
    
    return {
        "dataset": name,
        "aff_type": aff_type,
        "raw_metrics_csv": str(out_dir / f"{name}_raw_metrics.csv"),
        "calibrated_metrics_csv": str(out_dir / f"{name}_calibrated_metrics.csv"),
        "raw_merged_csv": str(out_dir / f"{name}_raw_merged.csv"),
        "calibrated_merged_csv": str(out_dir / f"{name}_calibrated_merged.csv"),
        "ae_threshold_summary_csv": str(ae_summary_csv),
        "raw_cumulative_png": str(out_dir / f"{name}_raw_cumulative.png"),
        "corr_cumulative_png": str(out_dir / f"{name}_corr_cumulative.png"),
        "raw_cumulative_plot_csv": str(out_dir / f"{name}_raw_cumulative_plot.csv"),
        "corr_cumulative_plot_csv": str(out_dir / f"{name}_corr_cumulative_plot.csv"),
        "ae_threshold_bar_png": str(ae_bar_png),
        "raw_ae_threshold_bar_png": str(raw_ae_bar_png),
        "corr_ae_threshold_bar_png": str(corr_ae_bar_png),
        "raw_scatter_png": str(out_dir / f"{name}_raw_scatter.png"),
        "corr_scatter_png": str(out_dir / f"{name}_corr_scatter.png"),
        "sample_count": int(len(y_true)),
        "raw_ae_thresholds": raw_ae_summary.to_dict(orient="records"),
        "calibrated_ae_thresholds": corr_ae_summary.to_dict(orient="records"),
    }


# --------------------------
# 主程序
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_dir", required=True, type=str)
    ap.add_argument("--davis_csv", type=str, default=None)
    ap.add_argument("--metz_csv", type=str, default=None)
    ap.add_argument("--selective_calib_json", type=str, default=None)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--threshold_pk", type=float, default=6.0)
    ap.add_argument("--gauss_alpha", type=float, default=0.95)
    ap.add_argument("--prob_sigma_scale", type=float, default=None)
    ap.add_argument("--skip_predict", action="store_true")
    ap.add_argument("--davis_pred", type=str, default=None)
    ap.add_argument("--metz_pred", type=str, default=None)
    args = ap.parse_args()

    if not _require_rdkit():
        raise SystemExit("请安装 RDKit")

    if args.davis_csv is None and args.metz_csv is None:
        raise SystemExit("请至少提供 --davis_csv 或 --metz_csv 之一")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 双输出日志
    log_file = out_dir / "evaluation.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

    logging.info(f"结果输出目录：{out_dir.resolve()}")
    logging.info(f"日志文件已保存至：{log_file.resolve()}")

    q_scale = None
    prob_sigma_scale = args.prob_sigma_scale if args.prob_sigma_scale is not None else 1.0

    selective_calib_loaded = None
    if args.selective_calib_json:
        with open(args.selective_calib_json, encoding="utf-8") as f:
            selective_calib_loaded = json.load(f)
        q_scale = selective_calib_loaded.get("q_scale")
        for k in ["prob_sigma_scale", "sigma_scale", "prob_scale"]:
            if k in selective_calib_loaded and selective_calib_loaded[k] is not None and args.prob_sigma_scale is None:
                prob_sigma_scale = float(selective_calib_loaded[k])
                break

    logging.info(f"threshold_pk = {args.threshold_pk}")
    logging.info(f"gauss_alpha = {args.gauss_alpha}")
    logging.info(f"prob_sigma_scale = {prob_sigma_scale}")
    logging.info(f"q_scale = {q_scale}")

    bundle_dir = Path(args.bundle_dir)
    results = []

    if args.davis_csv:
        results.append(
            process_dataset(
                name="DAVIS",
                src_csv=args.davis_csv,
                smi_col="smiles",
                prot_col="protein",
                y_col="pKd",
                aff_type="pKd",
                bundle_dir=bundle_dir,
                out_dir=out_dir,
                threshold_pk=args.threshold_pk,
                gauss_alpha=args.gauss_alpha,
                prob_sigma_scale=prob_sigma_scale,
                q_scale=q_scale,
                skip_predict=args.skip_predict,
                pred_csv_override=args.davis_pred,
            )
        )

    if args.metz_csv:
        results.append(
            process_dataset(
                name="Metz",
                src_csv=args.metz_csv,
                smi_col="smiles",
                prot_col="protein",
                y_col="pKi",
                aff_type="pKi",
                bundle_dir=bundle_dir,
                out_dir=out_dir,
                threshold_pk=args.threshold_pk,
                gauss_alpha=args.gauss_alpha,
                prob_sigma_scale=prob_sigma_scale,
                q_scale=q_scale,
                skip_predict=args.skip_predict,
                pred_csv_override=args.metz_pred,
            )
        )

    run_config = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "bundle_dir": str(bundle_dir),
        "davis_csv": args.davis_csv,
        "metz_csv": args.metz_csv,
        "davis_pred": args.davis_pred,
        "metz_pred": args.metz_pred,
        "skip_predict": bool(args.skip_predict),
        "out_dir": str(out_dir),
        "threshold_pk": float(args.threshold_pk),
        "gauss_alpha": float(args.gauss_alpha),
        "prob_sigma_scale": float(prob_sigma_scale),
        "q_scale": None if q_scale is None else float(q_scale),
        "selective_calib_json": args.selective_calib_json,
        "selective_calib_loaded": selective_calib_loaded,
        "results": results,
        "log_file": str(log_file),
    }

    save_json(run_config, out_dir / "run_config.json")

    logging.info(f"\n全部计算完成！所有结果/日志/图表已保存至: {out_dir.resolve()}")


if __name__ == "__main__":
    main()