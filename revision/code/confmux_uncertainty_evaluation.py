#!/usr/bin/env python3
"""Rigorous post-hoc uncertainty evaluation for ConfMux-DTA.

Required prediction columns are y_true, y_pred, sigma, and affinity_type.
The point predictor and sigma model must already be frozen. The script either
splits one prediction file into calibration/evaluation subsets or accepts
separate calibration and evaluation files. It compares:

1. normalized split conformal: |y-yhat| / sigma(x) (proposed),
2. standard global split conformal: |y-yhat| (established baseline), and
3. endpoint-Mondrian split conformal: |y-yhat| calibrated by affinity type.

All conformal quantiles use the finite-sample rank correction
ceil((n_cal + 1) * (1 - alpha)).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


METHOD_NORMALIZED = "Normalized split conformal (proposed)"
METHOD_GLOBAL = "Global split conformal (baseline)"
METHOD_MONDRIAN = "Endpoint-Mondrian split conformal"
METHODS = (METHOD_NORMALIZED, METHOD_GLOBAL, METHOD_MONDRIAN)
ENDPOINT_ORDER = ("pIC50", "pKi", "pKd")


def parse_levels(text: str) -> List[float]:
    levels = sorted({float(x.strip()) for x in text.split(",") if x.strip()})
    if not levels or any((x <= 0.0 or x >= 1.0) for x in levels):
        raise ValueError("Confidence levels must lie strictly between 0 and 1")
    return levels


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return value or "dataset"


def read_predictions(path: Path, y_col: str, pred_col: str,
                     sigma_col: str, endpoint_col: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    required = [y_col, pred_col, sigma_col]
    missing = [name for name in required if name not in header]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}; columns={header}")
    usecols = required + ([endpoint_col] if endpoint_col in header else [])
    frame = pd.read_csv(path, usecols=usecols)
    frame = frame.rename(columns={y_col: "y_true", pred_col: "y_pred", sigma_col: "sigma"})
    if endpoint_col in frame.columns:
        frame = frame.rename(columns={endpoint_col: "endpoint"})
    else:
        frame["endpoint"] = "Overall"
    for col in ("y_true", "y_pred", "sigma"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["endpoint"] = frame["endpoint"].fillna("Unknown").astype(str)
    frame["source_row"] = np.arange(len(frame), dtype=np.int64)
    valid = (
        np.isfinite(frame["y_true"].to_numpy(float))
        & np.isfinite(frame["y_pred"].to_numpy(float))
        & np.isfinite(frame["sigma"].to_numpy(float))
        & (frame["sigma"].to_numpy(float) > 0)
    )
    removed = int((~valid).sum())
    frame = frame.loc[valid].reset_index(drop=True)
    if len(frame) < 200:
        raise ValueError(f"Too few valid prediction rows in {path}: {len(frame)}")
    frame.attrs["removed_invalid_rows"] = removed
    frame.attrs["source_path"] = str(path.resolve())
    frame.attrs["source_sha256"] = sha256_file(path)
    return frame


def make_strata(frame: pd.DataFrame) -> pd.Series:
    y = frame["y_true"]
    try:
        bins = pd.qcut(y, q=10, labels=False, duplicates="drop").astype(str)
        strata = frame["endpoint"].astype(str) + "|" + bins
        counts = strata.value_counts()
        if counts.min() >= 2:
            return strata
    except Exception:
        pass
    endpoint = frame["endpoint"].astype(str)
    if endpoint.value_counts().min() >= 2:
        return endpoint
    return pd.Series(np.repeat("all", len(frame)), index=frame.index)


def split_calibration_evaluation(frame: pd.DataFrame, calibration_fraction: float,
                                 seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.1 <= calibration_fraction <= 0.9:
        raise ValueError("calibration_fraction must be between 0.1 and 0.9")
    idx = np.arange(len(frame), dtype=np.int64)
    idx_cal, idx_eval = train_test_split(
        idx,
        train_size=float(calibration_fraction),
        random_state=int(seed),
        shuffle=True,
        stratify=make_strata(frame),
    )
    if np.intersect1d(idx_cal, idx_eval).size:
        raise RuntimeError("Calibration/evaluation overlap detected")
    return (
        frame.iloc[np.sort(idx_cal)].reset_index(drop=True),
        frame.iloc[np.sort(idx_eval)].reset_index(drop=True),
    )


def finite_sample_quantile(scores: np.ndarray, confidence: float) -> Tuple[float, int]:
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        raise ValueError("At least two finite calibration scores are required")
    rank = int(math.ceil((n + 1) * float(confidence)))
    rank = min(max(rank, 1), n)
    q = float(np.partition(values, rank - 1)[rank - 1])
    return q, rank


def fit_quantiles(calibration: pd.DataFrame, levels: Sequence[float]) -> Dict:
    residual = np.abs(calibration["y_true"].to_numpy(float) - calibration["y_pred"].to_numpy(float))
    sigma = calibration["sigma"].to_numpy(float)
    normalized = residual / np.maximum(sigma, 1e-12)
    endpoints = list(dict.fromkeys(list(ENDPOINT_ORDER) + calibration["endpoint"].unique().tolist()))
    fitted: Dict[str, Dict[str, Dict]] = {}
    for level in levels:
        key = f"{level:.6f}"
        q_norm, rank_norm = finite_sample_quantile(normalized, level)
        q_global, rank_global = finite_sample_quantile(residual, level)
        endpoint_q = {}
        for endpoint in endpoints:
            mask = calibration["endpoint"].to_numpy(str) == endpoint
            if int(mask.sum()) >= 2:
                q_ep, rank_ep = finite_sample_quantile(residual[mask], level)
                endpoint_q[endpoint] = {
                    "q": q_ep,
                    "rank": rank_ep,
                    "n": int(mask.sum()),
                }
        fitted[key] = {
            METHOD_NORMALIZED: {"q": q_norm, "rank": rank_norm, "n": len(normalized)},
            METHOD_GLOBAL: {"q": q_global, "rank": rank_global, "n": len(residual)},
            METHOD_MONDRIAN: {
                "fallback_q": q_global,
                "endpoint_q": endpoint_q,
                "n": len(residual),
            },
        }
    return fitted


def half_widths(frame: pd.DataFrame, method: str, fit: Mapping) -> np.ndarray:
    if method == METHOD_NORMALIZED:
        return float(fit["q"]) * frame["sigma"].to_numpy(float)
    if method == METHOD_GLOBAL:
        return np.full(len(frame), float(fit["q"]), dtype=float)
    fallback = float(fit["fallback_q"])
    mapping = {name: float(obj["q"]) for name, obj in fit["endpoint_q"].items()}
    return frame["endpoint"].map(mapping).fillna(fallback).to_numpy(float)


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n))) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> float:
    width = hi - lo
    penalty_lo = (2.0 / alpha) * (lo - y) * (y < lo)
    penalty_hi = (2.0 / alpha) * (y - hi) * (y > hi)
    return float(np.mean(width + penalty_lo + penalty_hi))


def evaluate_subset(y: np.ndarray, pred: np.ndarray, width: np.ndarray,
                    confidence: float) -> Dict[str, float]:
    half = width / 2.0
    lo, hi = pred - half, pred + half
    covered = (y >= lo) & (y <= hi)
    n = int(len(y))
    successes = int(covered.sum())
    ci_lo, ci_hi = wilson_interval(successes, n)
    empirical = successes / n if n else float("nan")
    alpha = 1.0 - float(confidence)
    return {
        "N": n,
        "Covered_N": successes,
        "Empirical_coverage": empirical,
        "Coverage_gap": empirical - float(confidence),
        "Coverage_Wilson95_lo": ci_lo,
        "Coverage_Wilson95_hi": ci_hi,
        "Mean_width": float(np.mean(width)),
        "Median_width": float(np.median(width)),
        "P90_width": float(np.quantile(width, 0.90)),
        "RMSE": float(np.sqrt(np.mean((y - pred) ** 2))),
        "MAE": float(np.mean(np.abs(y - pred))),
        "Interval_score": interval_score(y, lo, hi, alpha),
    }


def evaluate_dataset(name: str, frame: pd.DataFrame, levels: Sequence[float],
                     fitted: Mapping) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[Tuple[str, float], np.ndarray]]:
    summary_rows: List[Dict] = []
    width_rows: List[Dict] = []
    widths_by_method_level: Dict[Tuple[str, float], np.ndarray] = {}
    y_all = frame["y_true"].to_numpy(float)
    p_all = frame["y_pred"].to_numpy(float)
    endpoint_values = list(dict.fromkeys(list(ENDPOINT_ORDER) + frame["endpoint"].unique().tolist()))
    subsets = [("Overall", np.ones(len(frame), dtype=bool))]
    subsets.extend((endpoint, frame["endpoint"].to_numpy(str) == endpoint) for endpoint in endpoint_values)

    for level in levels:
        fit_level = fitted[f"{level:.6f}"]
        for method in METHODS:
            half = half_widths(frame, method, fit_level[method])
            width = 2.0 * half
            widths_by_method_level[(method, float(level))] = width
            for endpoint, mask in subsets:
                if int(mask.sum()) < 2:
                    continue
                metrics = evaluate_subset(y_all[mask], p_all[mask], width[mask], level)
                summary_rows.append({
                    "Dataset": name,
                    "Method": method,
                    "Endpoint": endpoint,
                    "Nominal_coverage": float(level),
                    **metrics,
                })
            qvals = np.quantile(width, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
            width_rows.append({
                "Dataset": name,
                "Method": method,
                "Nominal_coverage": float(level),
                "N": len(width),
                **{label: float(value) for label, value in zip(
                    ("P01", "P05", "P10", "P25", "P50", "P75", "P90", "P95", "P99"), qvals
                )},
                "Mean": float(np.mean(width)),
                "SD": float(np.std(width)),
                "Min": float(np.min(width)),
                "Max": float(np.max(width)),
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(width_rows), widths_by_method_level


def select_level(levels: Sequence[float], target: float = 0.90) -> float:
    return min(levels, key=lambda value: abs(float(value) - target))


def uncertainty_deciles(name: str, frame: pd.DataFrame, fitted: Mapping,
                         level: float) -> pd.DataFrame:
    half = half_widths(frame, METHOD_NORMALIZED, fitted[f"{level:.6f}"][METHOD_NORMALIZED])
    width = 2.0 * half
    try:
        bins = pd.qcut(frame["sigma"], q=10, labels=False, duplicates="drop")
    except ValueError:
        bins = pd.Series(np.zeros(len(frame), dtype=int), index=frame.index)
    rows = []
    y = frame["y_true"].to_numpy(float)
    pred = frame["y_pred"].to_numpy(float)
    sigma = frame["sigma"].to_numpy(float)
    for value in sorted(pd.Series(bins).dropna().unique()):
        mask = np.asarray(bins == value)
        metrics = evaluate_subset(y[mask], pred[mask], width[mask], level)
        rows.append({
            "Dataset": name,
            "Uncertainty_decile": int(value) + 1,
            "Mean_sigma": float(np.mean(sigma[mask])),
            **metrics,
        })
    return pd.DataFrame(rows)


def uncertainty_quality(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether sigma ranks realized absolute errors."""
    rows: List[Dict] = []
    subsets = [("Overall", np.ones(len(frame), dtype=bool))]
    subsets.extend(
        (endpoint, frame["endpoint"].to_numpy(str) == endpoint)
        for endpoint in list(dict.fromkeys(list(ENDPOINT_ORDER) + frame["endpoint"].unique().tolist()))
    )
    sigma_all = frame["sigma"].to_numpy(float)
    abs_err_all = np.abs(frame["y_true"].to_numpy(float) - frame["y_pred"].to_numpy(float))
    for endpoint, mask in subsets:
        if int(mask.sum()) < 3:
            continue
        sigma = sigma_all[mask]
        abs_err = abs_err_all[mask]
        pearson = float(np.corrcoef(sigma, abs_err)[0, 1])
        sigma_rank = pd.Series(sigma).rank(method="average").to_numpy(float)
        error_rank = pd.Series(abs_err).rank(method="average").to_numpy(float)
        spearman = float(np.corrcoef(sigma_rank, error_rank)[0, 1])
        rows.append({
            "Dataset": name,
            "Endpoint": endpoint,
            "N": int(mask.sum()),
            "Pearson_sigma_vs_abs_error": pearson,
            "Spearman_sigma_vs_abs_error": spearman,
            "Mean_sigma": float(np.mean(sigma)),
            "Mean_absolute_error": float(np.mean(abs_err)),
        })
    return pd.DataFrame(rows)


def configure_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    })


def plot_calibration(summary: pd.DataFrame, out_path: Path, dataset: str,
                     endpoint: str = "Overall") -> None:
    configure_plot_style()
    sub = summary[(summary["Dataset"] == dataset) & (summary["Endpoint"] == endpoint)]
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    colors = {METHOD_NORMALIZED: "#1f77b4", METHOD_GLOBAL: "#d62728", METHOD_MONDRIAN: "#2ca02c"}
    for method in METHODS:
        part = sub[sub["Method"] == method].sort_values("Nominal_coverage")
        ax.plot(part["Nominal_coverage"], part["Empirical_coverage"], marker="o", ms=3.5,
                lw=1.3, label=method, color=colors[method])
        ax.fill_between(part["Nominal_coverage"].to_numpy(float),
                        part["Coverage_Wilson95_lo"].to_numpy(float),
                        part["Coverage_Wilson95_hi"].to_numpy(float),
                        alpha=0.12, color=colors[method], linewidth=0)
    lower = max(0.45, float(sub["Nominal_coverage"].min()) - 0.03)
    ax.plot([lower, 1.0], [lower, 1.0], ls="--", lw=1.0, color="black", label="Ideal")
    ax.set(xlim=(lower, 1.0), ylim=(lower, 1.0), xlabel="Nominal coverage",
           ylabel="Empirical coverage", title=f"{dataset}: {endpoint}")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_endpoint_calibration(summary: pd.DataFrame, out_path: Path, dataset: str) -> None:
    configure_plot_style()
    endpoints = [ep for ep in ENDPOINT_ORDER if ((summary["Dataset"] == dataset) & (summary["Endpoint"] == ep)).any()]
    if not endpoints:
        return
    colors = {METHOD_NORMALIZED: "#1f77b4", METHOD_GLOBAL: "#d62728", METHOD_MONDRIAN: "#2ca02c"}
    fig, axes = plt.subplots(1, len(endpoints), figsize=(4.0 * len(endpoints), 3.6), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, endpoint in zip(axes, endpoints):
        sub = summary[(summary["Dataset"] == dataset) & (summary["Endpoint"] == endpoint)]
        for method in METHODS:
            part = sub[sub["Method"] == method].sort_values("Nominal_coverage")
            ax.plot(part["Nominal_coverage"], part["Empirical_coverage"], marker="o", ms=3,
                    lw=1.2, color=colors[method], label=method)
        lower = max(0.45, float(sub["Nominal_coverage"].min()) - 0.03)
        ax.plot([lower, 1.0], [lower, 1.0], ls="--", lw=0.9, color="black")
        ax.set_title(endpoint)
        ax.set_xlim(lower, 1.0)
        ax.set_ylim(lower, 1.0)
        ax.grid(alpha=0.2)
        ax.set_xlabel("Nominal coverage")
    axes[0].set_ylabel("Empirical coverage")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_width_ecdf(widths: Mapping[Tuple[str, float], np.ndarray], out_path: Path,
                    dataset: str, level: float) -> None:
    configure_plot_style()
    colors = {METHOD_NORMALIZED: "#1f77b4", METHOD_GLOBAL: "#d62728", METHOD_MONDRIAN: "#2ca02c"}
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    all_widths = []
    for method in METHODS:
        values = np.asarray(widths[(method, float(level))], float)
        all_widths.append(values)
        ordered = np.sort(values)
        cdf = np.arange(1, len(ordered) + 1) / len(ordered)
        ax.plot(ordered, cdf, lw=1.4, color=colors[method], label=method)
    xmax = float(np.quantile(np.concatenate(all_widths), 0.995))
    ax.set(xlim=(0, max(xmax, 1e-6)), ylim=(0, 1.01), xlabel="Full interval width (pX)",
           ylabel="Empirical cumulative probability",
           title=f"{dataset}: interval-width distribution at {level:.0%}")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_deciles(table: pd.DataFrame, out_path: Path, dataset: str) -> None:
    configure_plot_style()
    if table.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))
    x = table["Uncertainty_decile"]
    axes[0].plot(x, table["MAE"], marker="o", label="MAE", color="#1f77b4")
    axes[0].plot(x, table["RMSE"], marker="s", label="RMSE", color="#d62728")
    axes[0].set(xlabel="Predicted-uncertainty decile", ylabel="Observed error (pX)", title="Error ranking")
    axes[0].legend(frameon=False)
    axes[1].plot(x, table["Mean_width"], marker="o", color="#2ca02c", label="Mean interval width")
    axes[1].axhline(table["Mean_width"].mean(), color="black", ls="--", lw=0.8)
    axes[1].set(xlabel="Predicted-uncertainty decile", ylabel="Full interval width (pX)", title="Adaptive width")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.set_xticks(x)
    fig.suptitle(dataset, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_external(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"--external must be NAME=PATH, got {spec!r}")
    name, raw_path = spec.split("=", 1)
    return name.strip(), Path(raw_path).expanduser()


def write_json(path: Path, obj: Mapping) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path,
                        help="Combined frozen-model prediction file to split into calibration/evaluation")
    parser.add_argument("--calibration-predictions", type=Path)
    parser.add_argument("--evaluation-predictions", type=Path)
    parser.add_argument("--external", action="append", default=[], metavar="NAME=PATH",
                        help="Additional untouched dataset; calibration quantiles remain fixed")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--levels", default="0.50,0.60,0.70,0.80,0.90,0.95,0.975,0.99")
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--y-col", default="y_true")
    parser.add_argument("--pred-col", default="y_pred")
    parser.add_argument("--sigma-col", default="sigma")
    parser.add_argument("--endpoint-col", default="affinity_type")
    args = parser.parse_args()

    combined_mode = args.predictions is not None
    separate_mode = args.calibration_predictions is not None or args.evaluation_predictions is not None
    if combined_mode == separate_mode:
        parser.error("Use either --predictions OR both --calibration-predictions and --evaluation-predictions")
    if separate_mode and (args.calibration_predictions is None or args.evaluation_predictions is None):
        parser.error("Both separate prediction files are required")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    levels = parse_levels(args.levels)
    sources: Dict[str, Dict] = {}

    if combined_mode:
        combined = read_predictions(args.predictions, args.y_col, args.pred_col, args.sigma_col, args.endpoint_col)
        calibration, evaluation = split_calibration_evaluation(
            combined, args.calibration_fraction, args.seed
        )
        sources["combined"] = {
            "path": combined.attrs["source_path"],
            "sha256": combined.attrs["source_sha256"],
            "valid_n": len(combined),
            "removed_invalid_rows": combined.attrs["removed_invalid_rows"],
        }
    else:
        calibration = read_predictions(args.calibration_predictions, args.y_col, args.pred_col,
                                       args.sigma_col, args.endpoint_col)
        evaluation = read_predictions(args.evaluation_predictions, args.y_col, args.pred_col,
                                      args.sigma_col, args.endpoint_col)
        sources["calibration"] = {
            "path": calibration.attrs["source_path"], "sha256": calibration.attrs["source_sha256"],
            "valid_n": len(calibration), "removed_invalid_rows": calibration.attrs["removed_invalid_rows"],
        }
        sources["evaluation"] = {
            "path": evaluation.attrs["source_path"], "sha256": evaluation.attrs["source_sha256"],
            "valid_n": len(evaluation), "removed_invalid_rows": evaluation.attrs["removed_invalid_rows"],
        }

    fitted = fit_quantiles(calibration, levels)
    datasets: Dict[str, pd.DataFrame] = {"Internal independent evaluation": evaluation}
    for spec in args.external:
        name, path = parse_external(spec)
        if name in datasets:
            raise ValueError(f"Duplicate dataset name: {name}")
        external = read_predictions(path, args.y_col, args.pred_col, args.sigma_col, args.endpoint_col)
        datasets[name] = external
        sources[f"external:{name}"] = {
            "path": external.attrs["source_path"], "sha256": external.attrs["source_sha256"],
            "valid_n": len(external), "removed_invalid_rows": external.attrs["removed_invalid_rows"],
        }

    summaries = []
    width_summaries = []
    decile_tables = []
    quality_tables = []
    main_level = select_level(levels, 0.90)
    for name, frame in datasets.items():
        summary, width_summary, widths = evaluate_dataset(name, frame, levels, fitted)
        summaries.append(summary)
        width_summaries.append(width_summary)
        deciles = uncertainty_deciles(name, frame, fitted, main_level)
        decile_tables.append(deciles)
        quality_tables.append(uncertainty_quality(name, frame))
        stem = slug(name)
        plot_calibration(summary, args.out_dir / f"calibration_curve_{stem}.png", name)
        plot_endpoint_calibration(summary, args.out_dir / f"calibration_by_endpoint_{stem}.png", name)
        plot_width_ecdf(widths, args.out_dir / f"interval_width_ecdf_{stem}.png", name, main_level)
        plot_uncertainty_deciles(deciles, args.out_dir / f"error_by_uncertainty_decile_{stem}.png", name)

    coverage = pd.concat(summaries, ignore_index=True)
    widths = pd.concat(width_summaries, ignore_index=True)
    deciles = pd.concat(decile_tables, ignore_index=True)
    quality = pd.concat(quality_tables, ignore_index=True)
    coverage.to_csv(args.out_dir / "coverage_summary_all_levels.csv", index=False)
    widths.to_csv(args.out_dir / "interval_width_distribution_summary.csv", index=False)
    deciles.to_csv(args.out_dir / "uncertainty_decile_analysis.csv", index=False)
    quality.to_csv(args.out_dir / "uncertainty_quality_summary.csv", index=False)

    if combined_mode:
        membership = pd.concat([
            calibration[["source_row", "endpoint"]].assign(subset="calibration"),
            evaluation[["source_row", "endpoint"]].assign(subset="evaluation"),
        ], ignore_index=True).sort_values("source_row")
        membership.to_csv(args.out_dir / "calibration_evaluation_membership.csv", index=False)

    main_levels = [value for value in (0.80, 0.90, 0.95) if any(abs(value - x) < 1e-9 for x in levels)]
    main_table = coverage[
        (coverage["Endpoint"] == "Overall")
        & coverage["Nominal_coverage"].isin(main_levels)
    ].copy()
    main_table.to_csv(args.out_dir / "paper_uncertainty_comparison.csv", index=False)
    endpoint_table = coverage[
        (coverage["Dataset"] == "Internal independent evaluation")
        & (coverage["Endpoint"].isin(ENDPOINT_ORDER))
        & coverage["Nominal_coverage"].isin(main_levels)
    ].copy()
    endpoint_table.to_csv(args.out_dir / "paper_uncertainty_by_endpoint.csv", index=False)

    manifest = {
        "protocol": "frozen predictor and sigma; disjoint conformal calibration and empirical evaluation",
        "seed": int(args.seed),
        "calibration_fraction": float(args.calibration_fraction) if combined_mode else None,
        "confidence_levels": levels,
        "main_reported_levels": main_levels,
        "methods": list(METHODS),
        "finite_sample_quantile": "ceil((n_cal + 1) * confidence), capped at n_cal",
        "n_calibration": int(len(calibration)),
        "calibration_endpoint_counts": {
            str(key): int(value) for key, value in calibration["endpoint"].value_counts().items()
        },
        "evaluation_sizes": {name: int(len(frame)) for name, frame in datasets.items()},
        "evaluation_endpoint_counts": {
            name: {str(key): int(value) for key, value in frame["endpoint"].value_counts().items()}
            for name, frame in datasets.items()
        },
        "calibration_source_row_sha256": hashlib.sha256(
            np.ascontiguousarray(calibration["source_row"].to_numpy(np.int64)).tobytes()
        ).hexdigest(),
        "evaluation_source_row_sha256": hashlib.sha256(
            np.ascontiguousarray(evaluation["source_row"].to_numpy(np.int64)).tobytes()
        ).hexdigest(),
        "sources": sources,
        "interpretation": (
            "Finite-sample marginal coverage requires a predictor and score function fixed before calibration "
            "and exchangeability between calibration and future observations. External distribution shift can "
            "invalidate nominal coverage and is evaluated empirically without recalibration."
        ),
    }
    write_json(args.out_dir / "uncertainty_evaluation_manifest.json", manifest)
    write_json(args.out_dir / "conformal_quantiles.json", fitted)
    print(f"UNCERTAINTY_EVALUATION_COMPLETE out_dir={args.out_dir}")
    print(f"n_calibration={len(calibration)} n_internal_evaluation={len(evaluation)}")
    print(main_table[["Dataset", "Method", "Nominal_coverage", "Empirical_coverage", "Mean_width", "Interval_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
