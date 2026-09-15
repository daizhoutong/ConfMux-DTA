#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
posthoc_best_model_full_plots.py
============================================================
用途：
1) 复用训练结束后、基于最佳模型导出的
   - train_predictions_for_paper.csv
   - valid_predictions_for_paper.csv
   生成“整体数据图”与论文友好的组合图；
2) 补齐原训练代码里没有直接给出的：
   - overall_all（train+valid 合并）的整体散点图 / 误差图 / 区间图
   - overall_all 按 affinity_type 的分图
   - Fig4 风格 4-panel interval figure（Overall + pIC50 + pKi + pKd）
   - Fig5 风格 coverage-vs-RMSE / coverage-vs-R2 figure
   - selective curve table / JSON summary

设计原则：
- 不重训，不改模型；
- 直接使用“最佳模型”已导出的 prediction csv，避免后处理逻辑漂移；
- 若 train / valid 只存在一个，也可单独出图；
- 兼容 unified 模式（affinity_type 列）与单 affinity 模式。

运行示例：
python posthoc_best_model_full_plots.py \
  --run_dir /home/lhy/cpi/models_lgbm_bpe_unified/20260309-085141/train_test \
  --out_subdir posthoc_full_plots

可选：
python posthoc_best_model_full_plots.py \
  --run_dir /path/to/run_dir \
  --use_splits val \
  --err_grid 0.6,0.8,1.0,1.2,1.5
"""

import os
import json
import math
import argparse
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


TYPE_COLORS = {
    "pIC50": "#1f77b4",
    "pKi": "#2ca02c",
    "pKd": "#9467bd",
    "pX": "#1f77b4",
    "overall": "#444444",
}

SPLIT_COLORS = {
    "train": "#2ca02c",
    "valid": "#1f77b4",
    "val": "#1f77b4",
    "overall_all": "#444444",
    "overall": "#444444",
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default)


def _json_default(x):
    if isinstance(x, (np.integer, )):
        return int(x)
    if isinstance(x, (np.floating, )):
        v = float(x)
        return None if not np.isfinite(v) else v
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"Unsupported type for JSON: {type(x)}")


# -----------------------------------------------------------------------------
# 基础统计
# -----------------------------------------------------------------------------

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


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, float)
    p = np.asarray(y_pred, float)
    m = np.isfinite(y) & np.isfinite(p)
    y = y[m]
    p = p[m]
    if len(y) < 2:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def ci_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Harrell-style concordance index for continuous regression targets.
    Ignores pairs tied on y_true and gives 0.5 credit to ties on y_pred.
    O(n log n), suitable for large posthoc prediction tables.
    """
    y = np.asarray(y_true, float)
    p = np.asarray(y_pred, float)
    m = np.isfinite(y) & np.isfinite(p)
    y = y[m]
    p = p[m]
    n = len(y)
    if n < 2:
        return float("nan")

    order = np.argsort(y, kind="mergesort")
    y = y[order]
    p = p[order]

    uniq_p, inv = np.unique(p, return_inverse=True)
    ranks = inv + 1  # 1-based Fenwick rank
    bit = np.zeros(len(uniq_p) + 2, dtype=np.int64)
    eq = np.zeros(len(uniq_p) + 2, dtype=np.int64)

    def bit_add(i: int, delta: int):
        while i < len(bit):
            bit[i] += delta
            i += i & -i

    def bit_sum(i: int) -> int:
        s = 0
        while i > 0:
            s += int(bit[i])
            i -= i & -i
        return s

    concordant = 0.0
    comparable = 0
    inserted = 0
    start = 0
    while start < n:
        end = start + 1
        while end < n and y[end] == y[start]:
            end += 1

        for k in range(start, end):
            r = int(ranks[k])
            num_less = bit_sum(r - 1)
            num_equal = int(eq[r])
            concordant += float(num_less) + 0.5 * float(num_equal)
            comparable += inserted

        for k in range(start, end):
            r = int(ranks[k])
            bit_add(r, 1)
            eq[r] += 1
            inserted += 1

        start = end

    if comparable == 0:
        return float("nan")
    return float(concordant / float(comparable))


def _r2_origin(y_ref: np.ndarray, y_fit: np.ndarray) -> float:
    """Squared determination for origin-constrained regression y_ref ~ k * y_fit."""
    yr = np.asarray(y_ref, float)
    yf = np.asarray(y_fit, float)
    m = np.isfinite(yr) & np.isfinite(yf)
    yr = yr[m]
    yf = yf[m]
    if len(yr) < 2:
        return float("nan")
    denom = float(np.dot(yf, yf))
    if denom <= 0:
        return float("nan")
    k = float(np.dot(yr, yf) / denom)
    sse0 = float(np.sum((yr - k * yf) ** 2))
    sst = float(np.sum((yr - np.mean(yr)) ** 2))
    if sst <= 0:
        return float("nan")
    return float(1.0 - sse0 / sst)


def r2m_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric R2m used in QSAR/DTI literature.
    Returns mean of forward and reverse Rm^2 variants for stability.
    """
    y = np.asarray(y_true, float)
    p = np.asarray(y_pred, float)
    m = np.isfinite(y) & np.isfinite(p)
    y = y[m]
    p = p[m]
    if len(y) < 2:
        return float("nan")

    r = pearson_r(y, p)
    if not np.isfinite(r):
        return float("nan")
    r2 = float(r * r)

    r0_sq = _r2_origin(y, p)
    r0p_sq = _r2_origin(p, y)
    if not np.isfinite(r0_sq) or not np.isfinite(r0p_sq):
        return float("nan")

    rm2 = r2 * (1.0 - math.sqrt(max(0.0, abs(r2 - r0_sq))))
    rm2p = r2 * (1.0 - math.sqrt(max(0.0, abs(r2 - r0p_sq))))
    return float(0.5 * (rm2 + rm2p))


def summarize_prediction_df(df: pd.DataFrame, q: float = 0.99) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "n": int(len(df)),
        "n_valid_pred": 0,
        "n_interval": 0,
        "n_sigma": 0,
    }
    if len(df) == 0:
        return out

    yt = pd.to_numeric(df.get("y_true"), errors="coerce").to_numpy(dtype=float)
    yp = pd.to_numeric(df.get("y_pred"), errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(yt) & np.isfinite(yp)
    out["n_valid_pred"] = int(m.sum())
    if m.sum() >= 2:
        abs_err = np.abs(yt[m] - yp[m])
        out.update({
            "rmse": float(np.sqrt(mean_squared_error(yt[m], yp[m]))),
            "r2": float(r2_score(yt[m], yp[m])),
            "pearson_r": pearson_r(yt[m], yp[m]),
            "ci": ci_score(yt[m], yp[m]),
            "r2m": r2m_score(yt[m], yp[m]),
            "abs_err_mean": float(np.mean(abs_err)),
            "abs_err_median": float(np.median(abs_err)),
            f"abs_err_q{int(q*100)}": abs_err_quantile(abs_err, q),
        })

    if "err_bound" in df.columns:
        eb = pd.to_numeric(df["err_bound"], errors="coerce").to_numpy(dtype=float)
        me = np.isfinite(eb)
        out["n_interval"] = int(me.sum())
        if me.sum() > 0:
            out.update({
                "err_bound_mean": float(np.mean(eb[me])),
                "err_bound_median": float(np.median(eb[me])),
                f"err_bound_q{int(q*100)}": abs_err_quantile(eb[me], q),
            })

    if "sigma" in df.columns:
        sg = pd.to_numeric(df["sigma"], errors="coerce").to_numpy(dtype=float)
        ms = np.isfinite(sg)
        out["n_sigma"] = int(ms.sum())
        if ms.sum() > 0:
            out.update({
                "sigma_mean": float(np.mean(sg[ms])),
                "sigma_median": float(np.median(sg[ms])),
            })

    if "affinity_type" in df.columns:
        out["affinity_type_counts"] = df["affinity_type"].astype(str).value_counts(dropna=False).to_dict()
    if "split" in df.columns:
        out["split_counts"] = df["split"].astype(str).value_counts(dropna=False).to_dict()
    return out


# -----------------------------------------------------------------------------
# 出图
# -----------------------------------------------------------------------------

def _pick_color(split_name: str = "", affinity_label: str = "overall") -> str:
    if affinity_label in TYPE_COLORS:
        return TYPE_COLORS[affinity_label]
    if split_name in SPLIT_COLORS:
        return SPLIT_COLORS[split_name]
    return "#1f77b4"


def export_scatter_points_csv(df: pd.DataFrame, out_csv: str):
    keep_cols = [
        c for c in [
            "split", "affinity_type", "y_true", "y_pred", "abs_err",
            "sigma", "err_bound", "pred_lo", "pred_hi",
            "prob_strong_binder", "pred_is_strong_binder",
            "smiles", "SMILES_NORM", "smiles_norm",
            "protein", "protein_clean",
            "pIC50", "pKi", "pKd", "affinity_value", "affinity_weight"
        ] if c in df.columns
    ]
    ensure_dir(os.path.dirname(os.path.abspath(out_csv)))
    df[keep_cols].to_csv(out_csv, index=False)



def plot_scatter(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 title: str,
                 out_path: str,
                 affinity_label: str = "overall",
                 split_name: str = "overall"):
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    m = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[m]
    yp = yp[m]
    if len(yt) < 5:
        return

    color = _pick_color(split_name=split_name, affinity_label=affinity_label)
    mn = float(min(np.min(yt), np.min(yp)))
    mx = float(max(np.max(yt), np.max(yp)))
    pad = 0.02 * (mx - mn + 1e-9)
    lo = mn - pad
    hi = mx + pad

    lr = LinearRegression().fit(yt.reshape(-1, 1), yp.reshape(-1, 1))
    beta = float(lr.coef_[0][0])
    alpha = float(lr.intercept_[0])
    r = pearson_r(yt, yp)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))

    plt.figure(figsize=(6.2, 6.0))
    plt.scatter(yt, yp, s=10, alpha=0.60, color=color, edgecolors="none", rasterized=True)
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="black")
    xs = np.array([lo, hi], dtype=float)
    ys = alpha + beta * xs
    plt.plot(xs, ys, linestyle="-", linewidth=1.0, color="black")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel(f"Observed {affinity_label}")
    plt.ylabel(f"Predicted {affinity_label}")
    plt.title(title)
    plt.text(
        0.03, 0.97,
        f"Fit: Pred = α + β·Obs\nα={alpha:.4f}, β={beta:.4f}\nR={r:.4f}, RMSE={rmse:.4f}\nN={len(yt)}",
        transform=plt.gca().transAxes,
        va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.15, edgecolor="none")
    )
    plt.tight_layout()
    ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    plt.savefig(out_path, dpi=180)
    plt.close()



def plot_abs_error_hist(abs_err: np.ndarray, out_path: str, title: str, q: float = 0.99, bins: int = 50, show_grid: bool = False):
    e = np.asarray(abs_err, float)
    e = e[np.isfinite(e)]
    if len(e) < 10:
        return
    ae_qq = abs_err_quantile(e, q)
    plt.figure(figsize=(7.2, 4.4))
    plt.hist(e, bins=bins, alpha=0.8, edgecolor="black")
    plt.axvline(x=ae_qq, color="r", linestyle="--", linewidth=1.5, label=f"AE_Q{int(q*100)}={ae_qq:.3f}")
    plt.title(title)
    plt.xlabel("|error|")
    plt.ylabel("count")
    plt.legend()
    if show_grid:
        plt.grid(alpha=0.25)
    plt.tight_layout()
    ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    plt.savefig(out_path, dpi=170)
    plt.close()



def plot_interval_scatter(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          errb: np.ndarray,
                          out_path: str,
                          title: str,
                          ylabel: str = "pX",
                          max_points: int = 5000,
                          show_grid: bool = False):
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    eb = np.asarray(errb, float)
    m = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(eb)
    yt = yt[m]
    yp = yp[m]
    eb = eb[m]
    if len(yt) < 30:
        return
    if len(yt) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(yt), size=max_points, replace=False)
        yt = yt[idx]
        yp = yp[idx]
        eb = eb[idx]
    order = np.argsort(yp)
    yt = yt[order]
    yp = yp[order]
    eb = eb[order]
    x = np.arange(len(yt))

    plt.figure(figsize=(9.5, 4.8))
    # 更醒目的误差/区间带颜色，避免原默认浅色在高密度点图中不明显
    plt.fill_between(x, yp - eb, yp + eb, color="#FF8C00", alpha=0.40, label="pred ± err_bound")
    plt.scatter(x, yt, s=10, alpha=0.65, color="#1f77b4", label="true")
    plt.plot(x, yp, lw=1.2, alpha=0.95, color="#8B0000", label="pred")
    plt.title(title)
    plt.xlabel("samples (sorted by pred)")
    plt.ylabel(ylabel)
    plt.legend()
    if show_grid:
        plt.grid(alpha=0.2)
    plt.tight_layout()
    ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    plt.savefig(out_path, dpi=170)
    plt.close()



def plot_fig4_interval_4panel(df: pd.DataFrame, out_path: str, split_label: str = "overall_all", max_points: int = 3000, show_grid: bool = False):
    if not {"y_true", "y_pred", "err_bound"}.issubset(df.columns):
        return

    panels: List[Tuple[str, pd.DataFrame]] = [("Overall", df)]
    if "affinity_type" in df.columns:
        for t in ["pIC50", "pKi", "pKd"]:
            sub = df[df["affinity_type"].astype(str) == t].copy()
            if len(sub) > 0:
                panels.append((t, sub))
    panels = panels[:4]
    if len(panels) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes = axes.ravel()

    for ax in axes[len(panels):]:
        ax.axis("off")

    for i, (name, sub) in enumerate(panels):
        yt = pd.to_numeric(sub["y_true"], errors="coerce").to_numpy(dtype=float)
        yp = pd.to_numeric(sub["y_pred"], errors="coerce").to_numpy(dtype=float)
        eb = pd.to_numeric(sub["err_bound"], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(eb)
        yt = yt[m]
        yp = yp[m]
        eb = eb[m]
        if len(yt) < 30:
            axes[i].axis("off")
            continue
        if len(yt) > max_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(yt), size=max_points, replace=False)
            yt = yt[idx]
            yp = yp[idx]
            eb = eb[idx]
        order = np.argsort(yp)
        yt = yt[order]
        yp = yp[order]
        eb = eb[order]
        x = np.arange(len(yt))
        ax = axes[i]
        # 统一使用更醒目的橙色区间带，便于 Fig4 多面板中快速辨认误差区域
        ax.fill_between(x, yp - eb, yp + eb, color="#FF8C00", alpha=0.40)
        ax.scatter(x, yt, s=8, alpha=0.60, color="#1f77b4")
        ax.plot(x, yp, lw=1.2, color="#8B0000")
        ax.set_title(f"{name}")
        ax.set_xlabel("samples (sorted by pred)")
        ax.set_ylabel("affinity")
        if show_grid:
            ax.grid(alpha=0.2)

    fig.suptitle(f"Fig4-style calibrated interval visualization ({split_label})", fontsize=14)
    ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# selective / coverage 分析
# -----------------------------------------------------------------------------

def selective_metrics_table(df: pd.DataFrame, err_max_list: List[float], q: float = 0.99) -> pd.DataFrame:
    if not {"y_true", "y_pred", "err_bound"}.issubset(df.columns):
        return pd.DataFrame()

    yt = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
    yp = pd.to_numeric(df["y_pred"], errors="coerce").to_numpy(dtype=float)
    eb = pd.to_numeric(df["err_bound"], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(eb)
    yt = yt[m]
    yp = yp[m]
    eb = eb[m]
    rows = []
    if len(yt) == 0:
        return pd.DataFrame()

    for t in err_max_list:
        t = float(t)
        acc = eb <= t
        cov = float(np.mean(acc))
        if acc.sum() >= 2:
            rmse = float(np.sqrt(mean_squared_error(yt[acc], yp[acc])))
            r2v = float(r2_score(yt[acc], yp[acc]))
            civ = ci_score(yt[acc], yp[acc])
            r2mv = r2m_score(yt[acc], yp[acc])
            abs_err_acc = np.abs(yt[acc] - yp[acc])
            ae_qq = abs_err_quantile(abs_err_acc, q) if len(abs_err_acc) > 0 else float("nan")
        else:
            rmse = float("nan")
            r2v = float("nan")
            civ = float("nan")
            r2mv = float("nan")
            ae_qq = float("nan")
        rows.append({
            "err_max": t,
            "coverage": cov,
            "reject_rate": 1.0 - cov,
            "accepted_n": int(acc.sum()),
            "total_n": int(len(acc)),
            "accepted_rmse": rmse,
            "accepted_r2": r2v,
            "accepted_ci": civ,
            "accepted_r2m": r2mv,
            f"accepted_ae_q{int(q*100)}": ae_qq,
        })
    return pd.DataFrame(rows)



def plot_curve(x, y, out_path: str, title: str, xlabel: str, ylabel: str, show_grid: bool = False):
    if len(x) == 0 or len(y) == 0:
        return
    plt.figure(figsize=(7.2, 4.4))
    plt.plot(x, y, marker="o", lw=1.3, ms=4)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show_grid:
        plt.grid(alpha=0.25)
    plt.tight_layout()
    ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    plt.savefig(out_path, dpi=170)
    plt.close()



def plot_fig5_coverage_vs_metrics(df_tab: pd.DataFrame, out_path: str, show_grid: bool = False):
    need = {"coverage", "accepted_rmse", "accepted_r2", "accepted_ci", "accepted_r2m"}
    if df_tab.empty or not need.issubset(df_tab.columns):
        return

    sub = df_tab.copy()
    sub = sub.sort_values("coverage")

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.8), constrained_layout=True)
    axes = axes.ravel()

    panels = [
        ("Coverage vs accepted RMSE", "accepted_rmse", "accepted-subset RMSE"),
        ("Coverage vs accepted R²", "accepted_r2", "accepted-subset R²"),
        ("Coverage vs accepted CI", "accepted_ci", "accepted-subset CI"),
        ("Coverage vs accepted R2m", "accepted_r2m", "accepted-subset R2m"),
    ]

    for ax, (title, col, ylabel) in zip(axes, panels):
        ax.plot(sub["coverage"], sub[col], marker="o", lw=1.3)
        ax.set_title(title)
        ax.set_xlabel("retained coverage")
        ax.set_ylabel(ylabel)
        if show_grid:
            ax.grid(alpha=0.25)

    fig.suptitle("Fig5-style selective prediction trade-off", fontsize=14)
    ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)



# -----------------------------------------------------------------------------
# 数据加载
# -----------------------------------------------------------------------------

def read_prediction_csv(path: str, split_name: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "split" not in df.columns and split_name:
        df["split"] = split_name
    if "abs_err" not in df.columns and {"y_true", "y_pred"}.issubset(df.columns):
        yt = pd.to_numeric(df["y_true"], errors="coerce")
        yp = pd.to_numeric(df["y_pred"], errors="coerce")
        df["abs_err"] = (yt - yp).abs()
    return df



def load_available_prediction_frames(run_dir: str, use_splits: List[str]) -> Dict[str, pd.DataFrame]:
    mapping = {
        "train": os.path.join(run_dir, "train_predictions_for_paper.csv"),
        "val": os.path.join(run_dir, "valid_predictions_for_paper.csv"),
        "valid": os.path.join(run_dir, "valid_predictions_for_paper.csv"),
    }
    out: Dict[str, pd.DataFrame] = {}
    seen_paths = set()
    for s in use_splits:
        p = mapping.get(s)
        if not p or (not os.path.exists(p)):
            continue
        key = "val" if s in {"val", "valid"} else s
        if p in seen_paths:
            continue
        out[key] = read_prediction_csv(p, split_name=key)
        seen_paths.add(p)
    return out



def parse_err_grid(s: str, df: Optional[pd.DataFrame]) -> List[float]:
    s = (s or "").strip()
    if s:
        vals = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            vals.append(float(part))
        vals = sorted(set(vals))
        return vals

    # 自动从 err_bound 分位数生成阈值
    if df is None or "err_bound" not in df.columns:
        return [0.6, 0.8, 1.0, 1.2, 1.5]
    eb = pd.to_numeric(df["err_bound"], errors="coerce").to_numpy(dtype=float)
    eb = eb[np.isfinite(eb)]
    if len(eb) < 20:
        return [0.6, 0.8, 1.0, 1.2, 1.5]
    qs = np.quantile(eb, [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
    vals = sorted(set([round(float(x), 3) for x in qs.tolist()]))
    return vals


# -----------------------------------------------------------------------------
# 单个 split / 合并 split 的完整导出
# -----------------------------------------------------------------------------

def export_one_group(df: pd.DataFrame, out_dir: str, label: str, q: float, err_grid: List[float], show_grid: bool = False):
    ensure_dir(out_dir)
    summary = summarize_prediction_df(df, q=q)
    save_json(summary, os.path.join(out_dir, f"summary_{label}.json"))
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, f"summary_{label}.csv"), index=False)

    # 散点点坐标
    export_scatter_points_csv(df, os.path.join(out_dir, f"SCATTERPTS_{label}.csv"))

    # overall scatter
    if {"y_true", "y_pred"}.issubset(df.columns):
        yt = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
        yp = pd.to_numeric(df["y_pred"], errors="coerce").to_numpy(dtype=float)
        plot_scatter(
            yt, yp,
            title=f"{label.upper()} Observed vs Predicted",
            out_path=os.path.join(out_dir, f"SCATTER_{label}.png"),
            affinity_label="pX" if "affinity_type" in df.columns else "affinity",
            split_name=label,
        )
        plot_abs_error_hist(
            np.abs(yt - yp),
            os.path.join(out_dir, f"ABSERR_HIST_{label}.png"),
            title=f"{label.upper()} Absolute Error",
            q=q,
            show_grid=show_grid,
        )

    # overall interval
    if {"y_true", "y_pred", "err_bound"}.issubset(df.columns):
        plot_interval_scatter(
            pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(df["y_pred"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(df["err_bound"], errors="coerce").to_numpy(dtype=float),
            os.path.join(out_dir, f"INTERVAL_{label}.png"),
            title=f"Prediction Interval Visualization ({label})",
            ylabel="affinity",
            show_grid=show_grid,
        )
        plot_fig4_interval_4panel(df, os.path.join(out_dir, f"FIG4_INTERVAL_4PANEL_{label}.png"), split_label=label, show_grid=show_grid)

        df_tab = selective_metrics_table(df, err_grid, q=q)
        if not df_tab.empty:
            df_tab.to_csv(os.path.join(out_dir, f"selective_curve_table_{label}.csv"), index=False)
            plot_curve(
                df_tab["err_max"].to_numpy(dtype=float),
                df_tab["coverage"].to_numpy(dtype=float),
                os.path.join(out_dir, f"COVERAGE_vs_ERRMAX_{label}.png"),
                title=f"Coverage vs err_max ({label})",
                xlabel="err_max",
                ylabel="coverage",
                show_grid=show_grid,
            )
            plot_curve(
                df_tab["err_max"].to_numpy(dtype=float),
                df_tab["accepted_rmse"].to_numpy(dtype=float),
                os.path.join(out_dir, f"ACCEPTED_RMSE_vs_ERRMAX_{label}.png"),
                title=f"Accepted RMSE vs err_max ({label})",
                xlabel="err_max",
                ylabel="accepted RMSE",
                show_grid=show_grid,
            )
            plot_curve(
                df_tab["err_max"].to_numpy(dtype=float),
                df_tab["accepted_r2"].to_numpy(dtype=float),
                os.path.join(out_dir, f"ACCEPTED_R2_vs_ERRMAX_{label}.png"),
                title=f"Accepted R² vs err_max ({label})",
                xlabel="err_max",
                ylabel="accepted R²",
                show_grid=show_grid,
            )
            plot_curve(
                df_tab["err_max"].to_numpy(dtype=float),
                df_tab["accepted_ci"].to_numpy(dtype=float),
                os.path.join(out_dir, f"ACCEPTED_CI_vs_ERRMAX_{label}.png"),
                title=f"Accepted CI vs err_max ({label})",
                xlabel="err_max",
                ylabel="accepted CI",
                show_grid=show_grid,
            )
            plot_curve(
                df_tab["err_max"].to_numpy(dtype=float),
                df_tab["accepted_r2m"].to_numpy(dtype=float),
                os.path.join(out_dir, f"ACCEPTED_R2M_vs_ERRMAX_{label}.png"),
                title=f"Accepted R2m vs err_max ({label})",
                xlabel="err_max",
                ylabel="accepted R2m",
                show_grid=show_grid,
            )
            plot_fig5_coverage_vs_metrics(df_tab, os.path.join(out_dir, f"FIG5_COVERAGE_TRADEOFF_{label}.png"), show_grid=show_grid)

            selective_summary = {
                "label": label,
                "err_grid": err_grid,
                "best_rows": df_tab.to_dict(orient="records"),
            }
            save_json(selective_summary, os.path.join(out_dir, f"selective_summary_{label}.json"))

    # per-type
    if "affinity_type" in df.columns:
        by_type_dir = os.path.join(out_dir, "plots_by_type")
        ensure_dir(by_type_dir)
        for t in ["pIC50", "pKi", "pKd"]:
            sub = df[df["affinity_type"].astype(str) == t].copy()
            if len(sub) < 5:
                continue
            yt = pd.to_numeric(sub["y_true"], errors="coerce").to_numpy(dtype=float)
            yp = pd.to_numeric(sub["y_pred"], errors="coerce").to_numpy(dtype=float)
            export_scatter_points_csv(sub, os.path.join(by_type_dir, f"SCATTERPTS_{label}_{t}.csv"))
            plot_scatter(
                yt, yp,
                title=f"{label.upper()} {t} Observed vs Predicted",
                out_path=os.path.join(by_type_dir, f"SCATTER_{label}_{t}.png"),
                affinity_label=t,
                split_name=label,
            )
            plot_abs_error_hist(
                np.abs(yt - yp),
                os.path.join(by_type_dir, f"ABSERR_HIST_{label}_{t}.png"),
                title=f"{label.upper()} {t} Absolute Error",
                q=q,
                show_grid=show_grid,
            )
            if {"err_bound"}.issubset(sub.columns) and len(sub) >= 30:
                eb = pd.to_numeric(sub["err_bound"], errors="coerce").to_numpy(dtype=float)
                plot_interval_scatter(
                    yt, yp, eb,
                    os.path.join(by_type_dir, f"INTERVAL_{label}_{t}.png"),
                    title=f"Prediction Interval Visualization ({label} {t})",
                    ylabel=t,
                    show_grid=show_grid,
                )


# -----------------------------------------------------------------------------
# 主程序
# -----------------------------------------------------------------------------

def build_arg_parser():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Posthoc plotting for best-model prediction csv exported by the original training code"
    )
    ap.add_argument("--run_dir", required=True, help="训练输出目录，内含 train_predictions_for_paper.csv / valid_predictions_for_paper.csv")
    ap.add_argument("--out_subdir", default="posthoc_full_plots", help="输出子目录名（位于 run_dir 下）")
    ap.add_argument("--use_splits", default="train,val", help="使用哪些 split，逗号分隔：train,val")
    ap.add_argument("--q", type=float, default=0.99, help="AE_Qq 分位数")
    ap.add_argument("--err_grid", default="", help="选择性筛选阈值列表，如 0.6,0.8,1.0,1.2,1.5；留空则自动按 err_bound 分位数生成")
    ap.add_argument("--grid", action="store_true", help="开启网格线；默认关闭")
    return ap



def main():
    args = build_arg_parser().parse_args()
    run_dir = os.path.abspath(args.run_dir)
    out_dir = os.path.join(run_dir, args.out_subdir)
    ensure_dir(out_dir)

    use_splits = [x.strip() for x in str(args.use_splits).split(",") if x.strip()]
    frames = load_available_prediction_frames(run_dir, use_splits)
    if not frames:
        raise FileNotFoundError(
            "未找到可用 prediction csv。请确认 run_dir 下存在 train_predictions_for_paper.csv 和/或 valid_predictions_for_paper.csv"
        )

    manifest = {
        "run_dir": run_dir,
        "available_splits": list(frames.keys()),
        "out_dir": out_dir,
        "grid": bool(args.grid),
    }
    save_json(manifest, os.path.join(out_dir, "posthoc_manifest.json"))

    # 单独 split 导出
    for split_name, df in frames.items():
        split_dir = os.path.join(out_dir, split_name)
        err_grid = parse_err_grid(args.err_grid, df)
        export_one_group(df, split_dir, split_name, q=float(args.q), err_grid=err_grid, show_grid=bool(args.grid))

    # 合并导出：overall_all
    all_df = pd.concat([frames[k] for k in frames.keys()], axis=0, ignore_index=True)
    all_err_grid = parse_err_grid(args.err_grid, all_df)
    export_one_group(all_df, os.path.join(out_dir, "overall_all"), "overall_all", q=float(args.q), err_grid=all_err_grid, show_grid=bool(args.grid))

    # 顶层总表
    top_rows = []
    for name, df in list(frames.items()) + [("overall_all", all_df)]:
        row = summarize_prediction_df(df, q=float(args.q))
        row["label"] = name
        top_rows.append(row)
    pd.DataFrame(top_rows).to_csv(os.path.join(out_dir, "all_group_summary.csv"), index=False)
    save_json({"groups": top_rows}, os.path.join(out_dir, "all_group_summary.json"))

    print(f"[OK] finished. output_dir={out_dir}")


if __name__ == "__main__":
    main()
