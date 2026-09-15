#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--out_pdf", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.points)
    met = pd.read_csv(args.metrics).iloc[0]

    y = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(float)
    p = pd.to_numeric(df["y_pred_calibrated"], errors="coerce").to_numpy(float)

    ok = np.isfinite(y) & np.isfinite(p)
    y = y[ok]
    p = p[ok]

    if len(y) != int(met["sample_count"]):
        raise ValueError("Point count does not match metrics table")

    # Independent checks of the directly reproducible metrics
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot

    if abs(rmse - float(met["rmse"])) > 1e-10:
        raise ValueError("RMSE mismatch")
    if abs(r2 - float(met["r2"])) > 1e-10:
        raise ValueError("R2 mismatch")

    lo = float(min(y.min(), p.min()))
    hi = float(max(y.max(), p.max()))
    pad = 0.04 * (hi - lo)
    lo -= pad
    hi += pad

    fig, ax = plt.subplots(figsize=(3.6, 3.6))

    ax.scatter(
        y, p,
        s=5,
        alpha=0.25,
        edgecolors="none",
        rasterized=True
    )

    ax.plot(
        [lo, hi], [lo, hi],
        linestyle="--",
        linewidth=1.0
    )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel("Observed pKd")
    ax.set_ylabel("Predicted pKd")
  #  ax.set_title("DAVIS benchmark")

    txt = (
        f"N = {int(met['sample_count']):,}\n"
        f"R² = {float(met['r2']):.3f}\n"
        f"RMSE = {float(met['rmse']):.3f}\n"
        f"CI = {float(met['ci']):.3f}\n"
        f"Rm² = {float(met['r2m']):.3f}"
    )

    ax.text(
        0.04, 0.96, txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8
    )

    ax.tick_params(labelsize=8)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()

    fig.savefig(args.out_png, dpi=600, bbox_inches="tight")

    if args.out_pdf:
        fig.savefig(args.out_pdf, bbox_inches="tight")

    plt.close(fig)


if __name__ == "__main__":
    main()
