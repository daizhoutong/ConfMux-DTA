#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
外部单数据集验证（统一多亲和力模型专用，最终增强版）
================================================================================
适用于 unified multi-affinity 模型的外部验证。
外部数据集与训练 CSV 同格式宽表，例如：
    smiles, protein, pIC50, pKi, pKd

主要功能：
- 读取一个外部 CSV（如 bindingdb_diff_only_in_new.csv）
- 自动标准化 smiles / protein
- 将宽表展开成长表：affinity_value + affinity_type
- 调用 bundle_dir/predict_cli.py 进行预测
- 计算原始 / 线性校正后指标
- 计算概率 / 区间 / Gaussian PI 指标（若预测文件含 sigma / 区间）
- 保存 merged 明细、总汇总指标、分 affinity_type 指标、图表、日志
- 输出 run_config.json 和 best_manifest.json

增强内容：
- 使用 O(n log n) 的 concordance_index，避免 O(n^2) 极慢
- merge 后统一 affinity_type 列名，避免 KeyError: 'affinity_type'
- smiles/protein 标准化加缓存，减少重复 RDKit 开销
- 每个 affinity_type 单独输出 raw/calibrated 概率 / 区间 / Gaussian PI 结果
- 大样本 observed vs predicted 改为 hexbin（适合高密度点云）
- 额外输出总体 + 分类型 absolute error cumulative plots
- 在 AE cumulative 图中标出 |AE|<=0.5 / 1.0 / 1.5 的比例，并写入 CSV 与日志
================================================================================
"""

import json
import math
import argparse
import subprocess
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt


AE_THRESHOLDS = [0.5, 1.0, 1.5]
PANEL_TYPES = ["ALL", "pIC50", "pKi", "pKd"]


# --------------------------
# RDKit 标准化
# --------------------------
def _require_rdkit():
    try:
        from rdkit import Chem  # noqa: F401
        return True
    except Exception:
        return False


def canonicalize_smiles(smiles: str) -> str:
    try:
        from rdkit import Chem
    except Exception:
        return "" if smiles is None else str(smiles).strip()

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


def _norm_cdf(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _format_metric(v, nd=4):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "nan"
    return f"{v:.{nd}f}"


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
    """O(n log n) 计算 CI"""
    y, yp, _ = _paired_valid_arrays(y, yp)
    n = len(y)
    if n < 2:
        return float("nan")

    order = np.argsort(y, kind="mergesort")
    y = y[order]
    yp = yp[order]

    _, rank = np.unique(yp, return_inverse=True)
    rank = rank + 1
    m = int(rank.max())

    class FenwickTree:
        def __init__(self, n):
            self.n = n
            self.bit = np.zeros(n + 1, dtype=np.int64)

        def add(self, idx, val=1):
            i = idx
            while i <= self.n:
                self.bit[i] += val
                i += i & -i

        def sum(self, idx):
            s = 0
            i = idx
            while i > 0:
                s += self.bit[i]
                i -= i & -i
            return s

    ft = FenwickTree(m)
    permissible = 0.0
    concordant = 0.0
    n_prev = 0

    start = 0
    while start < n:
        end = start + 1
        while end < n and y[end] == y[start]:
            end += 1

        group_ranks = rank[start:end]
        for r in group_ranks:
            less = ft.sum(r - 1)
            leq = ft.sum(r)
            equal_prev = leq - less
            concordant += less + 0.5 * equal_prev
            permissible += n_prev

        for r in group_ranks:
            ft.add(r, 1)

        n_prev += (end - start)
        start = end

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
# 线性校正
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
# AE 统计工具
# --------------------------
def absolute_error_threshold_stats(y_true, y_pred, thresholds=None):
    thresholds = AE_THRESHOLDS if thresholds is None else thresholds
    y_true, y_pred, _ = _paired_valid_arrays(y_true, y_pred)
    ae = np.abs(y_true - y_pred)
    n = len(ae)
    rows = []
    for thr in thresholds:
        cnt = int(np.sum(ae <= thr)) if n else 0
        prop = float(cnt / n) if n else float("nan")
        rows.append({
            "ae_threshold": float(thr),
            "count_le_threshold": cnt,
            "proportion_le_threshold": prop,
        })
    return rows


def log_ae_threshold_summary(prefix, y_true, y_pred, thresholds=None):
    y_true_v, y_pred_v, _ = _paired_valid_arrays(y_true, y_pred)
    rows = absolute_error_threshold_stats(y_true_v, y_pred_v, thresholds=thresholds)
    n = len(y_true_v)
    parts = []
    for r in rows:
        parts.append(
            f"|AE|<={r['ae_threshold']:.1f}: {r['proportion_le_threshold']:.4f} ({r['count_le_threshold']}/{n})"
        )
    logging.info(prefix + " | " + " | ".join(parts))
    return rows


# --------------------------
# 绘图函数
# --------------------------
def plot_error_hist(y_true, y_pred, out_path, title):
    ae = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    bins = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    counts, _ = np.histogram(ae, bins=bins)
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


def plot_cumulative_error(y_true, y_pred, out_path, title, thresholds=None):
    thresholds = AE_THRESHOLDS if thresholds is None else thresholds
    y_true, y_pred, _ = _paired_valid_arrays(y_true, y_pred)
    ae = np.sort(np.abs(y_true - y_pred))
    n = len(ae)
    if n == 0:
        return []

    percent = np.arange(1, n + 1) / n * 100.0
    summary_rows = absolute_error_threshold_stats(y_true, y_pred, thresholds=thresholds)

    plt.figure(figsize=(8.2, 5.2), dpi=180)
    plt.plot(ae, percent, linewidth=2)

    xmax = float(np.max(ae)) if n else 1.0
    xmax = max(xmax, max(thresholds) * 1.05)

    txt_lines = []
    for row in summary_rows:
        thr = row["ae_threshold"]
        prop = row["proportion_le_threshold"]
        pct = 100.0 * prop if np.isfinite(prop) else float("nan")
        plt.axvline(thr, linestyle="--", linewidth=1.2, alpha=0.8)
        plt.axhline(pct, linestyle=":", linewidth=1.1, alpha=0.8)
        plt.scatter([thr], [pct], s=28, zorder=3)
        txt_lines.append(f"|AE|≤{thr:.1f}: {pct:.2f}%")

    plt.xlim(0, xmax * 1.02)
    plt.ylim(0, 100.5)
    plt.xlabel("Absolute Error Threshold")
    plt.ylabel("Percentage of Samples ≤ Threshold (%)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.text(
        0.98, 0.05,
        "\n".join(txt_lines),
        transform=plt.gca().transAxes,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.88, edgecolor="gray")
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return summary_rows


def _hexbin_single_ax(ax, y_true, y_pred, title, xlim=None, ylim=None, add_colorbar=False):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        return None

    if xlim is None or ylim is None:
        vmin = float(min(np.min(y_true), np.min(y_pred)))
        vmax = float(max(np.max(y_true), np.max(y_pred)))
        pad = 0.03 * (vmax - vmin) if vmax > vmin else 0.5
        lo = vmin - pad
        hi = vmax + pad
        xlim = (lo, hi)
        ylim = (lo, hi)

    hb = ax.hexbin(y_true, y_pred, gridsize=70, bins='log', mincnt=1)
    ax.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]], '--', linewidth=1.3)

    if len(y_true) >= 2 and (not np.allclose(y_true, y_true[0])):
        k, b = np.polyfit(y_true, y_pred, 1)
        xx = np.array([xlim[0], xlim[1]], dtype=float)
        yy = k * xx + b
        ax.plot(xx, yy, '-', linewidth=1.2)
        fit_txt = f"fit={k:.3f}x+{b:.3f}"
    else:
        fit_txt = "fit=NA"

    r2_val = r2(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    mae_val = mae(y_true, y_pred)
    p_val = pearsonr(y_true, y_pred)

    txt = (
        f"N={len(y_true)}\n"
        f"R²={_format_metric(r2_val)}\n"
        f"Pearson={_format_metric(p_val)}\n"
        f"RMSE={_format_metric(rmse_val)}\n"
        f"MAE={_format_metric(mae_val)}\n"
        f"{fit_txt}"
    )
    ax.text(
        0.03, 0.97, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.88, edgecolor="gray")
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Observed pX")
    ax.set_ylabel("Predicted pX")
    ax.set_title(title)
    ax.grid(alpha=0.20)

    if add_colorbar:
        cb = plt.colorbar(hb, ax=ax)
        cb.set_label("log10(count)")
    return hb


def plot_true_vs_pred_hexbin(y_true, y_pred, out_path, title):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return

    vmin = float(min(np.min(y_true), np.min(y_pred)))
    vmax = float(max(np.max(y_true), np.max(y_pred)))
    pad = 0.03 * (vmax - vmin) if vmax > vmin else 0.5
    lo = vmin - pad
    hi = vmax + pad

    fig, ax = plt.subplots(figsize=(5.9, 5.5), dpi=180)
    _hexbin_single_ax(ax, y_true, y_pred, title, xlim=(lo, hi), ylim=(lo, hi), add_colorbar=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_hexbin_panels(df, pred_col, out_path, title_prefix):
    if pred_col not in df.columns or "y_true" not in df.columns or "affinity_type" not in df.columns:
        return

    all_y = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
    all_p = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=float)
    yv, pv, _ = _paired_valid_arrays(all_y, all_p)
    if len(yv) == 0:
        return

    vmin = float(min(np.min(yv), np.min(pv)))
    vmax = float(max(np.max(yv), np.max(pv)))
    pad = 0.03 * (vmax - vmin) if vmax > vmin else 0.5
    lo = vmin - pad
    hi = vmax + pad

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=180)
    axes = axes.ravel()
    hb_ref = None

    for ax, t in zip(axes, PANEL_TYPES):
        if t == "ALL":
            sub = df.copy()
            title = f"{title_prefix} Overall"
        else:
            sub = df[df["affinity_type"] == t].copy()
            title = f"{title_prefix} {t}"

        y_true = pd.to_numeric(sub["y_true"], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(sub[pred_col], errors="coerce").to_numpy(dtype=float)
        hb = _hexbin_single_ax(ax, y_true, y_pred, title, xlim=(lo, hi), ylim=(lo, hi), add_colorbar=False)
        if hb is not None:
            hb_ref = hb

    if hb_ref is not None:
        cbar = fig.colorbar(hb_ref, ax=axes.tolist(), shrink=0.92)
        cbar.set_label("log10(count)")

    fig.suptitle(title_prefix, y=1.01, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_error_panels(df, pred_col, out_path, title_prefix, thresholds=None):
    thresholds = AE_THRESHOLDS if thresholds is None else thresholds
    if pred_col not in df.columns or "y_true" not in df.columns or "affinity_type" not in df.columns:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=180)
    axes = axes.ravel()

    for ax, t in zip(axes, PANEL_TYPES):
        if t == "ALL":
            sub = df.copy()
            title = f"{title_prefix} Overall"
        else:
            sub = df[df["affinity_type"] == t].copy()
            title = f"{title_prefix} {t}"

        y_true = pd.to_numeric(sub["y_true"], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(sub[pred_col], errors="coerce").to_numpy(dtype=float)
        y_true, y_pred, _ = _paired_valid_arrays(y_true, y_pred)
        ae = np.sort(np.abs(y_true - y_pred))
        n = len(ae)
        if n == 0:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
            continue

        percent = np.arange(1, n + 1) / n * 100.0
        ax.plot(ae, percent, linewidth=2)
        xmax = max(float(np.max(ae)), max(thresholds) * 1.05)

        txt_lines = []
        for row in absolute_error_threshold_stats(y_true, y_pred, thresholds=thresholds):
            thr = row["ae_threshold"]
            prop = row["proportion_le_threshold"]
            pct = 100.0 * prop if np.isfinite(prop) else float("nan")
            ax.axvline(thr, linestyle="--", linewidth=1.1, alpha=0.8)
            ax.axhline(pct, linestyle=":", linewidth=1.0, alpha=0.8)
            ax.scatter([thr], [pct], s=22, zorder=3)
            txt_lines.append(f"≤{thr:.1f}: {pct:.1f}%")

        ax.set_xlim(0, xmax * 1.02)
        ax.set_ylim(0, 100.5)
        ax.set_xlabel("Absolute Error Threshold")
        ax.set_ylabel("Samples ≤ Threshold (%)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.text(
            0.98, 0.05,
            "\n".join(txt_lines),
            transform=ax.transAxes,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.88, edgecolor="gray")
        )

    fig.suptitle(title_prefix, y=1.01, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


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
# JSON 工具
# --------------------------
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
# 汇总函数：overall / per-type 都复用
# raw_df: y_true + y_pred_raw + 可能的 sigma/prob/interval/gauss 列
# corr_df: y_true + y_pred_calibrated + 对应校正后列
# --------------------------
def summarize_partition(raw_df, corr_df, threshold_pk, gauss_alpha):
    alpha_tag = int(gauss_alpha * 100)

    y_true = pd.to_numeric(raw_df["y_true"], errors="coerce").to_numpy(dtype=float)
    y_pred_raw = pd.to_numeric(raw_df["y_pred_raw"], errors="coerce").to_numpy(dtype=float)
    y_pred_corr = pd.to_numeric(corr_df["y_pred_calibrated"], errors="coerce").to_numpy(dtype=float)

    accepted_rate = float("nan")
    n_accepted = float("nan")
    if "accepted" in raw_df.columns:
        acc = pd.to_numeric(raw_df["accepted"], errors="coerce").fillna(0).astype(int).to_numpy()
        accepted_rate = float(np.mean(acc == 1))
        n_accepted = int(np.sum(acc == 1))

    raw_metrics = {
        "sample_count": len(raw_df),
        "accepted_rate": accepted_rate,
        "N_accepted": n_accepted,
        "rmse": rmse(y_true, y_pred_raw),
        "mse": mse(y_true, y_pred_raw),
        "mae": mae(y_true, y_pred_raw),
        "r2": r2(y_true, y_pred_raw),
        "r2m": r2m_metrics(y_true, y_pred_raw)["R2m"],
        "r2m_bar": r2m_metrics(y_true, y_pred_raw)["R2m_bar"],
        "pearson": pearsonr(y_true, y_pred_raw),
        "spearman": spearmanr(y_true, y_pred_raw),
        "ci": concordance_index(y_true, y_pred_raw),
    }

    corr_metrics = {
        "sample_count": len(corr_df),
        "accepted_rate": accepted_rate,
        "N_accepted": n_accepted,
        "rmse": rmse(y_true, y_pred_corr),
        "mse": mse(y_true, y_pred_corr),
        "mae": mae(y_true, y_pred_corr),
        "r2": r2(y_true, y_pred_corr),
        "r2m": r2m_metrics(y_true, y_pred_corr)["R2m"],
        "r2m_bar": r2m_metrics(y_true, y_pred_corr)["R2m_bar"],
        "pearson": pearsonr(y_true, y_pred_corr),
        "spearman": spearmanr(y_true, y_pred_corr),
        "ci": concordance_index(y_true, y_pred_corr),
    }

    # 概率内容：raw
    y_bin = (y_true >= float(threshold_pk)).astype(int)
    raw_metrics["strong_prevalence"] = float(np.mean(y_bin)) if len(y_bin) else float("nan")
    raw_metrics["AUC_score_pred"], raw_metrics["AUPR_score_pred"] = auc_aupr(y_bin, y_pred_raw)

    if "prob_ge_threshold_raw" in raw_df.columns:
        prob_raw = pd.to_numeric(raw_df["prob_ge_threshold_raw"], errors="coerce").to_numpy(dtype=float)
        raw_metrics["AUC_score_prob"], raw_metrics["AUPR_score_prob"] = auc_aupr(y_bin, prob_raw)
        raw_metrics["ECE_20bins"] = ece_20bins(y_bin, prob_raw)
        raw_metrics["Brier"] = brier_score(y_bin, prob_raw)
        raw_metrics["LogLoss"] = log_loss_binary(y_bin, prob_raw)

    # 概率内容：corr
    corr_metrics["strong_prevalence"] = float(np.mean(y_bin)) if len(y_bin) else float("nan")
    corr_metrics["AUC_score_pred"], corr_metrics["AUPR_score_pred"] = auc_aupr(y_bin, y_pred_corr)

    if "prob_ge_threshold_calibrated" in corr_df.columns:
        prob_corr = pd.to_numeric(corr_df["prob_ge_threshold_calibrated"], errors="coerce").to_numpy(dtype=float)
        corr_metrics["AUC_score_prob"], corr_metrics["AUPR_score_prob"] = auc_aupr(y_bin, prob_corr)
        corr_metrics["ECE_20bins"] = ece_20bins(y_bin, prob_corr)
        corr_metrics["Brier"] = brier_score(y_bin, prob_corr)
        corr_metrics["LogLoss"] = log_loss_binary(y_bin, prob_corr)

    # 区间内容：raw
    if "interval_lo_raw" in raw_df.columns and "interval_hi_raw" in raw_df.columns:
        im = interval_metrics(
            y_true,
            pd.to_numeric(raw_df["interval_lo_raw"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(raw_df["interval_hi_raw"], errors="coerce").to_numpy(dtype=float),
        )
        for k, v in im.items():
            raw_metrics[f"interval_{k}"] = v

    # 区间内容：corr
    if "interval_lo_calibrated" in corr_df.columns and "interval_hi_calibrated" in corr_df.columns:
        im = interval_metrics(
            y_true,
            pd.to_numeric(corr_df["interval_lo_calibrated"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(corr_df["interval_hi_calibrated"], errors="coerce").to_numpy(dtype=float),
        )
        for k, v in im.items():
            corr_metrics[f"interval_{k}"] = v

    # Gaussian PI：raw
    lo_raw_col = f"gauss_lo_raw_{alpha_tag}"
    hi_raw_col = f"gauss_hi_raw_{alpha_tag}"
    if lo_raw_col in raw_df.columns and hi_raw_col in raw_df.columns:
        im = interval_metrics(
            y_true,
            pd.to_numeric(raw_df[lo_raw_col], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(raw_df[hi_raw_col], errors="coerce").to_numpy(dtype=float),
        )
        raw_metrics[f"gauss_PI_{alpha_tag}_coverage"] = im["coverage"]
        raw_metrics[f"gauss_PI_{alpha_tag}_width_mean"] = im["width_mean"]
        raw_metrics[f"gauss_PI_{alpha_tag}_width_median"] = im["width_median"]
        raw_metrics[f"gauss_PI_{alpha_tag}_below_rate"] = im["below_rate"]
        raw_metrics[f"gauss_PI_{alpha_tag}_above_rate"] = im["above_rate"]

    # Gaussian PI：corr
    lo_corr_col = f"gauss_lo_calibrated_{alpha_tag}"
    hi_corr_col = f"gauss_hi_calibrated_{alpha_tag}"
    if lo_corr_col in corr_df.columns and hi_corr_col in corr_df.columns:
        im = interval_metrics(
            y_true,
            pd.to_numeric(corr_df[lo_corr_col], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(corr_df[hi_corr_col], errors="coerce").to_numpy(dtype=float),
        )
        corr_metrics[f"gauss_PI_{alpha_tag}_coverage"] = im["coverage"]
        corr_metrics[f"gauss_PI_{alpha_tag}_width_mean"] = im["width_mean"]
        corr_metrics[f"gauss_PI_{alpha_tag}_width_median"] = im["width_median"]
        corr_metrics[f"gauss_PI_{alpha_tag}_below_rate"] = im["below_rate"]
        corr_metrics[f"gauss_PI_{alpha_tag}_above_rate"] = im["above_rate"]

    return raw_metrics, corr_metrics


# --------------------------
# 核心：单外部数据集处理（宽表 -> 长表）
# --------------------------
def process_dataset(
    name,
    src_csv,
    smi_col,
    prot_col,
    y_cols,
    bundle_dir,
    out_dir,
    threshold_pk,
    gauss_alpha,
    prob_sigma_scale,
    q_scale,
    skip_predict=False,
    pred_csv_override=None,
):
    logging.info(f"\n===== {name} =====")
    raw = pd.read_csv(src_csv, low_memory=False)

    # 缓存标准化
    smiles_series = raw[smi_col].fillna("").astype(str)
    prot_series = raw[prot_col].fillna("").astype(str)

    uniq_smiles = smiles_series.unique().tolist()
    uniq_prot = prot_series.unique().tolist()

    smiles_map = {s: canonicalize_smiles(s) for s in uniq_smiles}
    prot_map = {s: clean_protein(s) for s in uniq_prot}

    raw["smiles_norm"] = smiles_series.map(smiles_map)
    raw["protein_clean"] = prot_series.map(prot_map)

    logging.info(f"原始样本（宽表行数）：{len(raw)}")

    # 宽表 -> 长表
    blocks = []
    type_counts = {}
    for aff_type, y_col in y_cols.items():
        if y_col not in raw.columns:
            logging.info(f"跳过 {aff_type}：列 {y_col} 不存在")
            continue

        sub = raw[[smi_col, prot_col, "smiles_norm", "protein_clean"]].copy()
        sub["affinity_value"] = pd.to_numeric(raw[y_col], errors="coerce")
        sub["affinity_type"] = aff_type
        sub["_source_y_col"] = y_col
        sub["_pair_uid"] = [f"{name}_{aff_type}_{i}" for i in raw.index]

        sub = sub[
            (sub["smiles_norm"] != "") &
            (sub["protein_clean"] != "") &
            (sub["affinity_value"].notna())
        ].copy()

        type_counts[aff_type] = int(len(sub))
        logging.info(f"{aff_type} 有效样本：{len(sub)}")
        blocks.append(sub)

    if not blocks:
        raise RuntimeError("展开后没有任何有效样本，请检查 smiles/protein/pIC50/pKi/pKd 列")

    df = pd.concat(blocks, ignore_index=True)
    logging.info(f"展开后总有效样本（长表行数）：{len(df)}")
    logging.info("affinity_type 分布：")
    logging.info("\n" + df["affinity_type"].value_counts().to_string())

    df.to_csv(out_dir / f"{name}_expanded_long.csv", index=False, encoding="utf-8")

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

    pred_df = pd.read_csv(pred_csv, low_memory=False)
    pred_col = infer_pred_col(pred_df)
    sigma_col = infer_sigma_col(pred_df)
    lo_col, hi_col, err_col, acc_col = infer_interval_cols(pred_df)

    merged = pd.merge(
        pred_df,
        df[["_pair_uid", "affinity_value", "affinity_type", "_source_y_col"]],
        on="_pair_uid",
        how="inner",
        suffixes=("", "_true")
    )

    # 统一列名
    if "affinity_type" not in merged.columns:
        if "affinity_type_true" in merged.columns:
            merged["affinity_type"] = merged["affinity_type_true"]
        elif "affinity_type_x" in merged.columns:
            merged["affinity_type"] = merged["affinity_type_x"]
        elif "affinity_type_y" in merged.columns:
            merged["affinity_type"] = merged["affinity_type_y"]

    if "_source_y_col" not in merged.columns:
        if "_source_y_col_true" in merged.columns:
            merged["_source_y_col"] = merged["_source_y_col_true"]

    y_true = pd.to_numeric(merged["affinity_value"], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(merged[pred_col], errors="coerce").to_numpy(dtype=float)

    y_true_v, y_pred_v, mask = _paired_valid_arrays(y_true, y_pred)
    merged = merged.loc[mask].copy().reset_index(drop=True)
    y_true = y_true_v
    y_pred = y_pred_v

    logging.info(f"合并后 paired valid 样本：{len(y_true)}")
    logging.info(f"预测列：{pred_col}")
    logging.info(f"sigma列：{sigma_col}")
    logging.info(f"区间列：lo={lo_col}, hi={hi_col}, err={err_col}, accepted={acc_col}")

    # 原始 / 校正
    a, b, y_corr = linear_calibration(y_true, y_pred)

    raw_merged = merged.copy()
    raw_merged["y_true"] = y_true
    raw_merged["y_pred_raw"] = y_pred
    raw_merged["abs_error_raw"] = np.abs(y_true - y_pred)

    corr_merged = merged.copy()
    corr_merged["y_true"] = y_true
    corr_merged["y_pred_calibrated"] = y_corr
    corr_merged["abs_error_calibrated"] = np.abs(y_true - y_corr)

    # sigma / prob
    sigma = None
    if sigma_col is not None and sigma_col in merged.columns:
        sigma = pd.to_numeric(merged[sigma_col], errors="coerce").to_numpy(dtype=float)
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)
        raw_merged["sigma_raw"] = sigma
        corr_merged["sigma_calibrated"] = np.abs(a) * sigma

        mask_prob = np.isfinite(sigma)
        raw_merged["prob_ge_threshold_raw"] = np.nan
        corr_merged["prob_ge_threshold_calibrated"] = np.nan

        if np.any(mask_prob):
            raw_merged.loc[mask_prob, "prob_ge_threshold_raw"] = gaussian_prob_ge_threshold(
                raw_merged.loc[mask_prob, "y_pred_raw"].to_numpy(dtype=float),
                sigma[mask_prob],
                threshold_pk,
                sigma_scale=prob_sigma_scale,
            )
            corr_merged.loc[mask_prob, "prob_ge_threshold_calibrated"] = gaussian_prob_ge_threshold(
                corr_merged.loc[mask_prob, "y_pred_calibrated"].to_numpy(dtype=float),
                np.abs(a) * sigma[mask_prob],
                threshold_pk,
                sigma_scale=prob_sigma_scale,
            )

    # 区间内容
    raw_lo = raw_hi = None
    corr_lo = corr_hi = None

    if lo_col is not None and hi_col is not None and lo_col in merged.columns and hi_col in merged.columns:
        raw_lo = pd.to_numeric(merged[lo_col], errors="coerce").to_numpy(dtype=float)
        raw_hi = pd.to_numeric(merged[hi_col], errors="coerce").to_numpy(dtype=float)
        corr_lo, corr_hi = transform_interval_linear(raw_lo, raw_hi, a, b)

    elif err_col is not None and err_col in merged.columns:
        errb = pd.to_numeric(merged[err_col], errors="coerce").to_numpy(dtype=float)
        raw_lo = y_pred - errb
        raw_hi = y_pred + errb
        corr_lo = y_corr - np.abs(a) * errb
        corr_hi = y_corr + np.abs(a) * errb

    elif sigma is not None and q_scale is not None:
        raw_lo = y_pred - float(q_scale) * sigma
        raw_hi = y_pred + float(q_scale) * sigma
        corr_lo = y_corr - float(q_scale) * np.abs(a) * sigma
        corr_hi = y_corr + float(q_scale) * np.abs(a) * sigma

    if raw_lo is not None and raw_hi is not None:
        raw_merged["interval_lo_raw"] = raw_lo
        raw_merged["interval_hi_raw"] = raw_hi
    if corr_lo is not None and corr_hi is not None:
        corr_merged["interval_lo_calibrated"] = corr_lo
        corr_merged["interval_hi_calibrated"] = corr_hi

    # Gaussian PI
    alpha_tag = int(gauss_alpha * 100)
    if sigma is not None:
        mask_g = np.isfinite(sigma)
        raw_merged[f"gauss_lo_raw_{alpha_tag}"] = np.nan
        raw_merged[f"gauss_hi_raw_{alpha_tag}"] = np.nan
        corr_merged[f"gauss_lo_calibrated_{alpha_tag}"] = np.nan
        corr_merged[f"gauss_hi_calibrated_{alpha_tag}"] = np.nan

        if np.any(mask_g):
            lo_g_raw, hi_g_raw = gaussian_interval(
                raw_merged.loc[mask_g, "y_pred_raw"].to_numpy(dtype=float),
                sigma[mask_g],
                alpha=gauss_alpha,
                sigma_scale=prob_sigma_scale,
            )
            lo_g_corr, hi_g_corr = gaussian_interval(
                corr_merged.loc[mask_g, "y_pred_calibrated"].to_numpy(dtype=float),
                np.abs(a) * sigma[mask_g],
                alpha=gauss_alpha,
                sigma_scale=prob_sigma_scale,
            )

            raw_merged.loc[mask_g, f"gauss_lo_raw_{alpha_tag}"] = lo_g_raw
            raw_merged.loc[mask_g, f"gauss_hi_raw_{alpha_tag}"] = hi_g_raw
            corr_merged.loc[mask_g, f"gauss_lo_calibrated_{alpha_tag}"] = lo_g_corr
            corr_merged.loc[mask_g, f"gauss_hi_calibrated_{alpha_tag}"] = hi_g_corr

    # 总体汇总
    overall_raw, overall_corr = summarize_partition(
        raw_merged, corr_merged, threshold_pk=threshold_pk, gauss_alpha=gauss_alpha
    )
    overall_raw["dataset"] = name
    overall_corr["dataset"] = name
    overall_raw["threshold_pk"] = threshold_pk
    overall_corr["threshold_pk"] = threshold_pk
    overall_raw["gauss_alpha"] = gauss_alpha
    overall_corr["gauss_alpha"] = gauss_alpha
    overall_raw["calib_slope_a"] = a
    overall_raw["calib_intercept_b"] = b
    overall_corr["calib_slope_a"] = a
    overall_corr["calib_intercept_b"] = b
    overall_raw["prob_sigma_scale"] = prob_sigma_scale
    overall_corr["prob_sigma_scale"] = prob_sigma_scale
    overall_raw["q_scale"] = q_scale
    overall_corr["q_scale"] = q_scale

    pd.DataFrame([overall_raw]).to_csv(out_dir / "summary_overall_raw.csv", index=False, encoding="utf-8")
    pd.DataFrame([overall_corr]).to_csv(out_dir / "summary_overall_calibrated.csv", index=False, encoding="utf-8")

    raw_merged.to_csv(out_dir / "merged_raw.csv", index=False, encoding="utf-8")
    corr_merged.to_csv(out_dir / "merged_calibrated.csv", index=False, encoding="utf-8")

    # 分 affinity_type 汇总：把概率 / 区间 / Gaussian PI 也带上
    per_type_rows = []
    for t in ["pIC50", "pKi", "pKd"]:
        g_raw = raw_merged[raw_merged["affinity_type"] == t].copy()
        if len(g_raw) == 0:
            continue
        g_corr = corr_merged[corr_merged["affinity_type"] == t].copy()

        raw_m, corr_m = summarize_partition(g_raw, g_corr, threshold_pk=threshold_pk, gauss_alpha=gauss_alpha)

        row = {"affinity_type": t, "n": len(g_raw)}
        for k, v in raw_m.items():
            if k != "sample_count":
                row[f"raw_{k}"] = v
        for k, v in corr_m.items():
            if k != "sample_count":
                row[f"calibrated_{k}"] = v
        per_type_rows.append(row)

    per_type_df = pd.DataFrame(per_type_rows)
    per_type_df.to_csv(out_dir / "summary_per_type.csv", index=False, encoding="utf-8")

    # AE 阈值覆盖率（overall + per_type, raw + calibrated）
    ae_rows = []
    for subset_name, g_raw, g_corr in [("ALL", raw_merged, corr_merged)] + [
        (t, raw_merged[raw_merged["affinity_type"] == t].copy(), corr_merged[corr_merged["affinity_type"] == t].copy())
        for t in ["pIC50", "pKi", "pKd"] if (raw_merged[raw_merged["affinity_type"] == t].shape[0] > 0)
    ]:
        y_sub = pd.to_numeric(g_raw["y_true"], errors="coerce").to_numpy(dtype=float)
        pred_raw_sub = pd.to_numeric(g_raw["y_pred_raw"], errors="coerce").to_numpy(dtype=float)
        pred_corr_sub = pd.to_numeric(g_corr["y_pred_calibrated"], errors="coerce").to_numpy(dtype=float)
        y_sub_v, pred_raw_v, _ = _paired_valid_arrays(y_sub, pred_raw_sub)
        y_sub_c, pred_corr_v, _ = _paired_valid_arrays(y_sub, pred_corr_sub)

        for row in absolute_error_threshold_stats(y_sub_v, pred_raw_v, thresholds=AE_THRESHOLDS):
            ae_rows.append({
                "dataset": name,
                "subset": subset_name,
                "mode": "raw",
                "n": len(y_sub_v),
                **row,
            })
        for row in absolute_error_threshold_stats(y_sub_c, pred_corr_v, thresholds=AE_THRESHOLDS):
            ae_rows.append({
                "dataset": name,
                "subset": subset_name,
                "mode": "calibrated",
                "n": len(y_sub_c),
                **row,
            })

    ae_df = pd.DataFrame(ae_rows)
    ae_df.to_csv(out_dir / "absolute_error_threshold_summary.csv", index=False, encoding="utf-8")

    # 图：overall raw/corrected 单图 + 4-panel 图
    plot_true_vs_pred_hexbin(y_true, y_pred, out_dir / f"{name}_raw_true_vs_pred_hexbin.png", f"{name} Raw Observed vs Predicted")
    plot_true_vs_pred_hexbin(y_true, y_corr, out_dir / f"{name}_corr_true_vs_pred_hexbin.png", f"{name} Corrected Observed vs Predicted")

    plot_hexbin_panels(raw_merged, "y_pred_raw", out_dir / f"{name}_raw_true_vs_pred_hexbin_panels.png", f"{name} Raw Observed vs Predicted")
    plot_hexbin_panels(corr_merged, "y_pred_calibrated", out_dir / f"{name}_corr_true_vs_pred_hexbin_panels.png", f"{name} Corrected Observed vs Predicted")

    plot_error_hist(y_true, y_pred, out_dir / f"{name}_raw_error_hist.png", f"{name} Raw |Error| Distribution")
    plot_error_hist(y_true, y_corr, out_dir / f"{name}_corr_error_hist.png", f"{name} Corrected |Error| Distribution")

    # AE cumulative：overall raw/corrected 单图 + 4-panel 图
    plot_cumulative_error(y_true, y_pred, out_dir / f"{name}_raw_cumulative.png", f"{name} Raw Cumulative Absolute Error")
    plot_cumulative_error(y_true, y_corr, out_dir / f"{name}_corr_cumulative.png", f"{name} Corrected Cumulative Absolute Error")
    plot_cumulative_error_panels(raw_merged, "y_pred_raw", out_dir / f"{name}_raw_cumulative_panels.png", f"{name} Raw Cumulative Absolute Error")
    plot_cumulative_error_panels(corr_merged, "y_pred_calibrated", out_dir / f"{name}_corr_cumulative_panels.png", f"{name} Corrected Cumulative Absolute Error")

    logging.info(f"已生成 {name} 原始/校正后的 hexbin scatter（overall + panels）、误差分布图、累计绝对误差图")

    # 屏幕和日志中明确输出 overall AE 阈值覆盖率
    log_ae_threshold_summary(f"{name} RAW AE coverage", y_true, y_pred, thresholds=AE_THRESHOLDS)
    log_ae_threshold_summary(f"{name} CALIBRATED AE coverage", y_true, y_corr, thresholds=AE_THRESHOLDS)
    for t in ["pIC50", "pKi", "pKd"]:
        g_raw = raw_merged[raw_merged["affinity_type"] == t].copy()
        g_corr = corr_merged[corr_merged["affinity_type"] == t].copy()
        if len(g_raw) == 0:
            continue
        log_ae_threshold_summary(
            f"{name} {t} RAW AE coverage",
            g_raw["y_true"].to_numpy(dtype=float),
            g_raw["y_pred_raw"].to_numpy(dtype=float),
            thresholds=AE_THRESHOLDS,
        )
        log_ae_threshold_summary(
            f"{name} {t} CALIBRATED AE coverage",
            g_corr["y_true"].to_numpy(dtype=float),
            g_corr["y_pred_calibrated"].to_numpy(dtype=float),
            thresholds=AE_THRESHOLDS,
        )

    # best manifest：按 calibrated_rmse 最小选 best affinity_type
    best_type = None
    if len(per_type_df) > 0 and "calibrated_rmse" in per_type_df.columns:
        tmp = per_type_df[pd.to_numeric(per_type_df["calibrated_rmse"], errors="coerce").notna()].copy()
        if len(tmp) > 0:
            tmp["calibrated_rmse"] = pd.to_numeric(tmp["calibrated_rmse"], errors="coerce")
            best_row = tmp.sort_values("calibrated_rmse", ascending=True).iloc[0]
            best_type = str(best_row["affinity_type"])

    manifest = {
        "dataset": name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "external_csv": str(src_csv),
        "bundle_dir": str(bundle_dir),
        "pred_csv": str(pred_csv),
        "expanded_long_csv": str(out_dir / f"{name}_expanded_long.csv"),
        "merged_raw_csv": str(out_dir / "merged_raw.csv"),
        "merged_calibrated_csv": str(out_dir / "merged_calibrated.csv"),
        "summary_overall_raw_csv": str(out_dir / "summary_overall_raw.csv"),
        "summary_overall_calibrated_csv": str(out_dir / "summary_overall_calibrated.csv"),
        "summary_per_type_csv": str(out_dir / "summary_per_type.csv"),
        "absolute_error_threshold_summary_csv": str(out_dir / "absolute_error_threshold_summary.csv"),
        "log_file": str(out_dir.parent / "evaluation.log"),
        "sample_count_wide": int(len(raw)),
        "sample_count_long_valid": int(len(raw_merged)),
        "type_counts": type_counts,
        "pred_col": pred_col,
        "sigma_col": sigma_col,
        "interval_cols": {"lo": lo_col, "hi": hi_col, "err": err_col, "accepted": acc_col},
        "calibration": {"slope_a": a, "intercept_b": b},
        "ae_thresholds": AE_THRESHOLDS,
        "best_by": "calibrated_rmse",
        "best_affinity_type": best_type,
    }
    save_json(manifest, out_dir / "best_manifest.json")

    logging.info("\n模型原始结果：")
    logging.info(
        f"RMSE = {overall_raw['rmse']:.4f} | MSE = {overall_raw['mse']:.4f} | MAE = {overall_raw['mae']:.4f} | "
        f"R2 = {overall_raw['r2']:.4f} | R2m = {overall_raw['r2m']:.4f}"
    )
    logging.info(
        f"Pearson = {overall_raw['pearson']:.4f} | Spearman = {overall_raw['spearman']:.4f} | CI = {overall_raw['ci']:.4f}"
    )
    logging.info(f"\n线性拟合公式：y_true = {a:.4f} * y_pred + {b:.4f}")
    logging.info("\n线性校正后结果：")
    logging.info(
        f"RMSE = {overall_corr['rmse']:.4f} | MSE = {overall_corr['mse']:.4f} | MAE = {overall_corr['mae']:.4f} | "
        f"R2 = {overall_corr['r2']:.4f} | R2m = {overall_corr['r2m']:.4f}"
    )
    logging.info(
        f"Pearson = {overall_corr['pearson']:.4f} | Spearman = {overall_corr['spearman']:.4f} | CI = {overall_corr['ci']:.4f}"
    )
    logging.info(f"best_affinity_type(by calibrated_rmse) = {best_type}")
    logging.info(f"absolute_error_threshold_summary.csv 已保存：{out_dir / 'absolute_error_threshold_summary.csv'}")

    return {
        "wide_count": int(len(raw)),
        "long_count": int(len(raw_merged)),
        "type_counts": type_counts,
        "overall_raw": overall_raw,
        "overall_corr": overall_corr,
        "best_type": best_type,
    }


# --------------------------
# 主程序
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_dir", required=True, type=str)
    ap.add_argument("--external_csv", required=True, type=str)
    ap.add_argument("--selective_calib_json", type=str, default=None)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--threshold_pk", type=float, default=6.0)
    ap.add_argument("--gauss_alpha", type=float, default=0.95)
    ap.add_argument("--prob_sigma_scale", type=float, default=None)
    ap.add_argument("--skip_predict", action="store_true")
    ap.add_argument("--external_pred", type=str, default=None)

    ap.add_argument("--smiles_col", type=str, default="smiles")
    ap.add_argument("--protein_col", type=str, default="protein")
    ap.add_argument("--pic50_col", type=str, default="pIC50")
    ap.add_argument("--pki_col", type=str, default="pKi")
    ap.add_argument("--pkd_col", type=str, default="pKd")
    args = ap.parse_args()

    if not _require_rdkit():
        raise SystemExit("请安装 RDKit")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / "evaluation.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()]
    )

    q_scale = None
    prob_sigma_scale = args.prob_sigma_scale if args.prob_sigma_scale is not None else 1.0
    selective_calib = None
    if args.selective_calib_json:
        with open(args.selective_calib_json, encoding="utf-8") as f:
            selective_calib = json.load(f)
        q_scale = selective_calib.get("q_scale")
        for k in ["prob_sigma_scale", "sigma_scale", "prob_scale"]:
            if k in selective_calib and selective_calib[k] is not None and args.prob_sigma_scale is None:
                prob_sigma_scale = float(selective_calib[k])
                break

    logging.info(f"结果输出目录：{out_dir.resolve()}")
    logging.info(f"日志文件已保存至：{log_file.resolve()}")
    logging.info(f"external_csv = {args.external_csv}")
    logging.info(f"threshold_pk = {args.threshold_pk}")
    logging.info(f"gauss_alpha = {args.gauss_alpha}")
    logging.info(f"prob_sigma_scale = {prob_sigma_scale}")
    logging.info(f"q_scale = {q_scale}")
    logging.info(f"AE coverage thresholds = {AE_THRESHOLDS}")

    bundle_dir = Path(args.bundle_dir)
    y_cols = {
        "pIC50": args.pic50_col,
        "pKi": args.pki_col,
        "pKd": args.pkd_col,
    }

    ext_out_dir = out_dir / "BindingDB_diff_only_in_new"
    ext_out_dir.mkdir(parents=True, exist_ok=True)

    result_info = process_dataset(
        name="BindingDB_diff_only_in_new",
        src_csv=args.external_csv,
        smi_col=args.smiles_col,
        prot_col=args.protein_col,
        y_cols=y_cols,
        bundle_dir=bundle_dir,
        out_dir=ext_out_dir,
        threshold_pk=args.threshold_pk,
        gauss_alpha=args.gauss_alpha,
        prob_sigma_scale=prob_sigma_scale,
        q_scale=q_scale,
        skip_predict=args.skip_predict,
        pred_csv_override=args.external_pred,
    )

    # run config
    run_config = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": "external_cross_dataset_validation_newbdb_enhanced.py",
        "bundle_dir": str(bundle_dir),
        "external_csv": str(args.external_csv),
        "external_pred": args.external_pred,
        "skip_predict": bool(args.skip_predict),
        "out_dir": str(out_dir),
        "sub_out_dir": str(ext_out_dir),
        "threshold_pk": float(args.threshold_pk),
        "gauss_alpha": float(args.gauss_alpha),
        "prob_sigma_scale": float(prob_sigma_scale),
        "q_scale": None if q_scale is None else float(q_scale),
        "ae_thresholds": AE_THRESHOLDS,
        "columns": {
            "smiles_col": args.smiles_col,
            "protein_col": args.protein_col,
            "pic50_col": args.pic50_col,
            "pki_col": args.pki_col,
            "pkd_col": args.pkd_col,
        },
        "y_cols": y_cols,
        "selective_calib_json": args.selective_calib_json,
        "selective_calib_loaded": selective_calib,
        "result_info": result_info,
        "artifacts": {
            "log_file": str(log_file),
            "summary_overall_raw_csv": str(ext_out_dir / "summary_overall_raw.csv"),
            "summary_overall_calibrated_csv": str(ext_out_dir / "summary_overall_calibrated.csv"),
            "summary_per_type_csv": str(ext_out_dir / "summary_per_type.csv"),
            "absolute_error_threshold_summary_csv": str(ext_out_dir / "absolute_error_threshold_summary.csv"),
            "best_manifest_json": str(ext_out_dir / "best_manifest.json"),
            "expanded_long_csv": str(ext_out_dir / "BindingDB_diff_only_in_new_expanded_long.csv"),
            "merged_raw_csv": str(ext_out_dir / "merged_raw.csv"),
            "merged_calibrated_csv": str(ext_out_dir / "merged_calibrated.csv"),
            "raw_hexbin_panels_png": str(ext_out_dir / "BindingDB_diff_only_in_new_raw_true_vs_pred_hexbin_panels.png"),
            "corr_hexbin_panels_png": str(ext_out_dir / "BindingDB_diff_only_in_new_corr_true_vs_pred_hexbin_panels.png"),
            "raw_cumulative_panels_png": str(ext_out_dir / "BindingDB_diff_only_in_new_raw_cumulative_panels.png"),
            "corr_cumulative_panels_png": str(ext_out_dir / "BindingDB_diff_only_in_new_corr_cumulative_panels.png"),
        },
    }
    save_json(run_config, out_dir / "run_config.json")

    logging.info(f"\n全部计算完成！所有结果/日志/图表已保存至: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
