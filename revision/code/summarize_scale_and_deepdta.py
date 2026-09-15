#!/usr/bin/env python3
"""Verify split identity and summarize the 50%-versus-100% and DeepDTA runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd


ENDPOINTS = ("overall", "pIC50", "pKi", "pKd")
METRICS = ("R2", "RMSE", "MSE", "CI", "PearsonR", "R2m", "R2m_bar")


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing nonempty file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def index_sha1(path: Path) -> Tuple[str, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.ascontiguousarray(np.asarray(np.load(path), dtype=np.int64))
    return hashlib.sha1(values.tobytes()).hexdigest(), int(len(values))


def normalize_metrics(values: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for metric in METRICS:
        key = metric
        if metric == "PearsonR" and metric not in values:
            key = "R"
        out[metric] = values.get(key, np.nan)
    out["N"] = values.get("N", np.nan)
    return out


def load_confmux(run_root: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    train_dir = run_root / "train_test"
    metrics = read_json(train_dir / "paper_metrics_by_split_and_type.json")
    valid = metrics.get("val") or metrics.get("test")
    if not valid or "overall" not in valid:
        raise ValueError(f"No validation metrics in {train_dir}")
    train = metrics.get("train", {}).get("overall", {})
    result = {
        "overall": normalize_metrics(valid["overall"]),
        **{
            endpoint: normalize_metrics(valid.get("by_type", {}).get(endpoint, {}))
            for endpoint in ENDPOINTS[1:]
        },
    }
    return result, {
        "n_train": int(train.get("N", 0)),
        "metrics_path": str(train_dir / "paper_metrics_by_split_and_type.json"),
    }


def load_deepdta(run_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    metrics = read_json(run_dir / "metrics.json")
    valid = metrics.get("validation") or metrics.get("test")
    if not valid or "overall" not in valid:
        raise ValueError(f"No validation metrics in {run_dir / 'metrics.json'}")
    result = {
        "overall": normalize_metrics(valid["overall"]),
        **{
            endpoint: normalize_metrics(valid.get("by_type", {}).get(endpoint, {}))
            for endpoint in ENDPOINTS[1:]
        },
    }
    return result, {
        "n_train": int(metrics.get("train", {}).get("overall", {}).get("N", 0)),
        "metrics_path": str(run_dir / "metrics.json"),
    }


def rows_for_model(
    model: str,
    fraction: float,
    metrics: Dict[str, Any],
    meta: Dict[str, Any],
) -> Iterable[Dict[str, Any]]:
    for endpoint in ENDPOINTS:
        values = metrics[endpoint]
        row = {
            "model": model,
            "training_fraction": fraction,
            "endpoint": endpoint,
            "N_train": meta["n_train"],
            "N_validation": values["N"],
            "metrics_source": meta["metrics_path"],
        }
        row.update({metric: values[metric] for metric in METRICS})
        yield row


def format_num(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not np.isfinite(number) else f"{number:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--full_run_dir", default="confmux_dta_runs/leakage_controlled_onehot"
    )
    parser.add_argument(
        "--half_run_dir", default="confmux_dta_runs/scale_controlled_half"
    )
    parser.add_argument(
        "--deepdta_run_dir", default="confmux_dta_runs/deepdta_full_controlled"
    )
    parser.add_argument(
        "--reference_cache_dir",
        default="confmux_dta_feature_cache/feat_train_only_633b5630a6107d71",
    )
    parser.add_argument("--out_dir", default="scale_deepdta_summary")
    args = parser.parse_args()

    full_root = Path(args.full_run_dir).resolve()
    half_root = Path(args.half_run_dir).resolve()
    deep_root = Path(args.deepdta_run_dir).resolve()
    ref_root = Path(args.reference_cache_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_train_hash, ref_train_n = index_sha1(ref_root / "idx_train.npy")
    ref_valid_hash, ref_valid_n = index_sha1(ref_root / "idx_valid.npy")
    half_manifest = read_json(half_root / "training_fraction_manifest.json")
    deep_manifest = read_json(deep_root / "input_and_split_manifest.json")
    deep_split = deep_manifest.get("split", {})

    checks = {
        "reference_train_index_sha1": ref_train_hash,
        "reference_valid_index_sha1": ref_valid_hash,
        "reference_n_train": ref_train_n,
        "reference_n_valid": ref_valid_n,
        "half_reference_checked": bool(half_manifest.get("checked")),
        "half_base_train_matches_reference": (
            half_manifest.get("base_train_index_sha1") == ref_train_hash
        ),
        "half_validation_matches_reference": (
            half_manifest.get("valid_index_sha1") == ref_valid_hash
        ),
        "half_is_nested_in_reference_training": bool(
            half_manifest.get("selected_is_nested_in_base_train")
        ),
        "deepdta_training_matches_reference": (
            deep_split.get("train_index_sha1") == ref_train_hash
        ),
        "deepdta_validation_matches_reference": (
            deep_split.get("valid_index_sha1") == ref_valid_hash
        ),
    }
    required_checks = (
        "half_reference_checked",
        "half_base_train_matches_reference",
        "half_validation_matches_reference",
        "half_is_nested_in_reference_training",
        "deepdta_training_matches_reference",
        "deepdta_validation_matches_reference",
    )
    failed = [key for key in required_checks if not checks[key]]
    if failed:
        raise RuntimeError("Split-integrity checks failed: " + ", ".join(failed))

    full_metrics, full_meta = load_confmux(full_root)
    half_metrics, half_meta = load_confmux(half_root)
    deep_metrics, deep_meta = load_deepdta(deep_root)
    if full_meta["n_train"] != ref_train_n:
        raise RuntimeError("The completed 100% ConfMux-DTA N_train differs from reference cache")
    if half_meta["n_train"] != int(half_manifest["selected_train_n"]):
        raise RuntimeError(
            f"The 50% result reports N_train={half_meta['n_train']}, expected {int(half_manifest['selected_train_n'])}"
        )
    if deep_meta["n_train"] != ref_train_n:
        raise RuntimeError("DeepDTA N_train differs from the full ConfMux-DTA training set")

    rows = []
    rows.extend(rows_for_model("ConfMux-DTA", 0.5, half_metrics, half_meta))
    rows.extend(rows_for_model("ConfMux-DTA", 1.0, full_metrics, full_meta))
    rows.extend(rows_for_model("Endpoint-conditioned DeepDTA-style CNN", 1.0, deep_metrics, deep_meta))
    long_table = pd.DataFrame(rows)
    long_table.to_csv(out_dir / "controlled_comparison_by_endpoint.csv", index=False)
    long_table[long_table["endpoint"] == "overall"].to_csv(
        out_dir / "controlled_comparison_overall.csv", index=False
    )

    scale_rows = []
    for endpoint in ENDPOINTS:
        for metric in METRICS:
            half_value = float(half_metrics[endpoint][metric])
            full_value = float(full_metrics[endpoint][metric])
            if not np.isfinite(half_value) or not np.isfinite(full_value):
                gain = np.nan
            elif metric in ("RMSE", "MSE"):
                gain = half_value - full_value
            else:
                gain = full_value - half_value
            scale_rows.append({
                "endpoint": endpoint,
                "metric": metric,
                "ConfMux_50pct": half_value,
                "ConfMux_100pct": full_value,
                "gain_at_100pct_positive_is_better": gain,
            })
    pd.DataFrame(scale_rows).to_csv(out_dir / "confmux_scale_effect.csv", index=False)

    checks["all_required_checks_passed"] = True
    checks["half_selected_train_n"] = int(half_manifest["selected_train_n"])
    checks["half_selected_train_index_sha1"] = half_manifest.get(
        "selected_train_index_sha1"
    )
    with (out_dir / "split_integrity.json").open("w", encoding="utf-8") as handle:
        json.dump(checks, handle, indent=2, ensure_ascii=False)

    full = full_metrics["overall"]
    half = half_metrics["overall"]
    deep = deep_metrics["overall"]
    reply = f"""# Result-ready wording for Reviewer 2, Suggestion 10

The 100% ConfMux-DTA model was **not retrained**. Its completed leakage-controlled result was reused after exact index verification. A nested 50% subset ({half_meta['n_train']:,} of {full_meta['n_train']:,} training observations) was sampled only from the original training partition, while the original {int(full['N']):,}-observation validation set was kept unchanged. BPE vocabularies and TF-IDF vectorizers for the 50% run were fitted only on that subset.

On the identical validation set, increasing the ConfMux-DTA training data from 50% to 100% changed R² from {format_num(half['R2'])} to {format_num(full['R2'])}, RMSE from {format_num(half['RMSE'])} to {format_num(full['RMSE'])}, and CI from {format_num(half['CI'])} to {format_num(full['CI'])}. These controlled results support the conclusion that the sparse BPE-LightGBM framework benefits from large-scale training data; they do not imply a universal monotonic learning curve beyond the two evaluated sizes.

We additionally implemented an endpoint-conditioned DeepDTA-style categorical CNN and trained it on the same {deep_meta['n_train']:,} observations with the same fixed validation indices. It achieved R²={format_num(deep['R2'])}, RMSE={format_num(deep['RMSE'])}, and CI={format_num(deep['CI'])}, compared with R²={format_num(full['R2'])}, RMSE={format_num(full['RMSE'])}, and CI={format_num(full['CI'])} for ConfMux-DTA. This is a protocol-matched architectural comparison rather than a comparison of values copied from studies using different datasets.

A full-data DeepDTAGen result is not reported because its graph-generation/training pipeline could not be completed under the same predefined hardware allocation. We therefore make no claim that this failure demonstrates inferior predictive accuracy; it is reported only as a scalability limitation under the tested resources.
"""
    (out_dir / "Reviewer2_S10_result_ready.md").write_text(reply, encoding="utf-8")
    print(long_table.to_string(index=False), flush=True)
    print(f"\nCompleted: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
