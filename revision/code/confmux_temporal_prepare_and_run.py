#!/usr/bin/env python3
"""Prepare and evaluate the ConfMux-DTA temporal hold-out for Reviewer 2 S8.

The input ``bindingdb_diff_only_in_new.csv`` is a wide table containing
``pIC50``, ``pKi`` and ``pKd`` columns.  This script:

1. expands the wide table into endpoint-conditioned long records;
2. predicts every record with an already exported ConfMux-DTA ``predict_bundle``;
3. writes a canonical temporal prediction file with ``y_true`` and ``y_pred``;
4. reports overall and endpoint-specific metrics without fitting anything to
   the temporal labels; and
5. optionally launches ``confmux_temporal_shift_audit.py`` using the matching
   training and random-validation predictions.

Prediction is batched and resumable.  Completed batches are kept under a
run-signature directory, so a rerun skips only batches generated from the same
input, model bundle and batch size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ENDPOINTS: Sequence[Tuple[str, str]] = (
    ("pIC50", "pIC50"),
    ("pKi", "pKi"),
    ("pKd", "pKd"),
)

EXPECTED_ENDPOINT_COUNTS = {"pIC50": 119407, "pKi": 19814, "pKd": 5369}
EXPECTED_TOTAL = 144590
PREDICTION_CANDIDATES = (
    "pred_pX",
    "pred_final",
    "y_pred",
    "prediction",
    "predicted_pX",
    "pred",
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def file_state(path: Path) -> Dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def require_nonempty(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty {label}: {path}")


def load_bundle_manifest(bundle_dir: Path) -> Tuple[dict, Path]:
    required = [
        "predict_cli.py",
        "predict_manifest.json",
        "best_model.txt",
        "vectorizer_ligand.pkl",
        "vectorizer_protein.pkl",
        "smiles_bpe_tokenizer.json",
        "protein_bpe_tokenizer.json",
    ]
    for filename in required:
        require_nonempty(bundle_dir / filename, f"bundle file {filename}")
    manifest_path = bundle_dir / "predict_manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    return manifest, manifest_path


def resolve_input_columns(manifest: dict) -> Tuple[str, str, str]:
    columns = manifest.get("columns", {}) or {}
    smiles_col = str(columns.get("smiles", "smiles"))
    protein_col = str(columns.get("protein", "protein"))
    affinity = manifest.get("affinity_unified", {}) or {}
    affinity_col = str(
        affinity.get("affinity_type_col", affinity.get("keep_type_col_name", "affinity_type"))
    )
    return smiles_col, protein_col, affinity_col


def infer_prediction_column(frame: pd.DataFrame, manifest: dict) -> str:
    declared = manifest.get("pred_col")
    candidates = ([str(declared)] if declared else []) + list(PREDICTION_CANDIDATES)
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(
        "The bundle output has no recognized prediction column. "
        f"Expected one of {candidates}; got {frame.columns.tolist()}"
    )


def source_metadata_columns(columns: Iterable[str]) -> List[str]:
    preferred = [
        "standard_inchi_key",
        "chembl_molecule_chembl_id",
        "uniprot_id",
        "chembl_target_chembl_id",
        "group_size",
        "pH_n",
        "pH_median",
        "T_n",
        "T_median",
        "env_penalty_mean",
    ]
    available = set(columns)
    return [column for column in preferred if column in available]


def expand_wide_chunk(
    wide: pd.DataFrame,
    source_offset: int,
    model_smiles_col: str,
    model_protein_col: str,
    affinity_col: str,
) -> pd.DataFrame:
    for required in ["smiles", "protein", "pIC50", "pKi", "pKd"]:
        if required not in wide.columns:
            raise ValueError(f"Temporal CSV is missing required column: {required}")

    source_row = np.arange(source_offset, source_offset + len(wide), dtype=np.int64)
    smiles = wide["smiles"].fillna("").astype(str).str.strip()
    protein = wide["protein"].fillna("").astype(str).str.strip()
    base_valid = smiles.ne("") & protein.ne("")
    metadata = source_metadata_columns(wide.columns)
    blocks: List[pd.DataFrame] = []

    for endpoint, value_column in ENDPOINTS:
        values = pd.to_numeric(wide[value_column], errors="coerce")
        keep = base_valid & np.isfinite(values)
        if not bool(keep.any()):
            continue
        positions = np.flatnonzero(keep.to_numpy())
        block = pd.DataFrame(
            {
                model_smiles_col: smiles.iloc[positions].to_numpy(),
                model_protein_col: protein.iloc[positions].to_numpy(),
                "affinity_type": endpoint,
                "y_true": values.iloc[positions].to_numpy(dtype=np.float64),
                "source_row": source_row[positions],
                "_temporal_uid": [f"{endpoint}:{int(i)}" for i in source_row[positions]],
            }
        )
        if affinity_col != "affinity_type":
            block[affinity_col] = endpoint
        for column in metadata:
            block[column] = wide.iloc[positions][column].to_numpy()
        blocks.append(block)

    if not blocks:
        return pd.DataFrame()
    return pd.concat(blocks, ignore_index=True)


def validate_completed_part(
    part_csv: Path,
    part_meta: Path,
    expected_rows: int,
    uid_first: str,
    uid_last: str,
) -> bool:
    if not part_csv.is_file() or part_csv.stat().st_size == 0 or not part_meta.is_file():
        return False
    try:
        meta = json.loads(part_meta.read_text(encoding="utf-8"))
        if int(meta.get("n_rows", -1)) != int(expected_rows):
            return False
        if meta.get("uid_first") != uid_first or meta.get("uid_last") != uid_last:
            return False
        header = pd.read_csv(part_csv, nrows=0).columns
        return {"smiles_norm", "protein_clean", "affinity_type", "y_true", "y_pred"}.issubset(header)
    except Exception:
        return False


def run_bundle_batch(
    batch: pd.DataFrame,
    batch_index: int,
    bundle_dir: Path,
    manifest: dict,
    parts_dir: Path,
) -> Path:
    part_csv = parts_dir / f"temporal_part_{batch_index:05d}.csv"
    part_meta = parts_dir / f"temporal_part_{batch_index:05d}.json"
    uid_first = str(batch["_temporal_uid"].iloc[0])
    uid_last = str(batch["_temporal_uid"].iloc[-1])
    if validate_completed_part(part_csv, part_meta, len(batch), uid_first, uid_last):
        log(f"Batch {batch_index:05d}: resume existing {len(batch):,} predictions")
        return part_csv

    input_csv = parts_dir / f"model_input_{batch_index:05d}.csv"
    raw_output = parts_dir / f"model_output_{batch_index:05d}.csv"
    atomic_csv(batch, input_csv)

    # Passing a deliberately nonexistent calibration path disables optional
    # interval generation.  Point predictions still use the fixed bundle
    # manifest/model, and no temporal labels are used for calibration or refit.
    disabled_calibration = parts_dir / "__no_temporal_recalibration__.json"
    command = [
        sys.executable,
        str((bundle_dir / "predict_cli.py").resolve()),
        "--in",
        str(input_csv.resolve()),
        "--out",
        str(raw_output.resolve()),
        "--calib_json",
        str(disabled_calibration.resolve()),
    ]
    log(f"Batch {batch_index:05d}: predicting {len(batch):,} endpoint records")
    process = subprocess.run(command, text=True, capture_output=True)
    if process.returncode != 0:
        raise RuntimeError(
            f"predict_cli.py failed for batch {batch_index} (exit={process.returncode})\n"
            f"STDOUT tail:\n{process.stdout[-8000:]}\nSTDERR tail:\n{process.stderr[-8000:]}"
        )
    require_nonempty(raw_output, "bundle prediction output")

    predicted = pd.read_csv(raw_output, low_memory=False)
    if len(predicted) != len(batch):
        raise ValueError(
            f"Batch {batch_index}: input/output row mismatch: {len(batch)} vs {len(predicted)}"
        )
    if "_temporal_uid" not in predicted.columns:
        raise ValueError("predict_cli.py did not preserve _temporal_uid; cannot align labels safely")
    if predicted["_temporal_uid"].astype(str).tolist() != batch["_temporal_uid"].astype(str).tolist():
        raise ValueError("predict_cli.py changed row order or identifiers; refusing positional merge")

    pred_col = infer_prediction_column(predicted, manifest)
    if "smiles_norm" not in predicted.columns or "protein_clean" not in predicted.columns:
        raise ValueError("Bundle output is missing smiles_norm or protein_clean")

    canonical = pd.DataFrame(
        {
            "smiles_norm": predicted["smiles_norm"],
            "protein_clean": predicted["protein_clean"],
            "affinity_type": batch["affinity_type"].to_numpy(),
            "y_true": pd.to_numeric(batch["y_true"], errors="coerce").to_numpy(),
            "y_pred": pd.to_numeric(predicted[pred_col], errors="coerce").to_numpy(),
            "source_row": batch["source_row"].to_numpy(),
            "_temporal_uid": batch["_temporal_uid"].to_numpy(),
        }
    )
    for column in source_metadata_columns(batch.columns):
        canonical[column] = batch[column].to_numpy()
    if "valid_input" in predicted.columns:
        canonical["valid_input"] = predicted["valid_input"].to_numpy()

    atomic_csv(canonical, part_csv)
    atomic_json(
        {
            "batch_index": int(batch_index),
            "n_rows": int(len(canonical)),
            "n_finite_predictions": int(np.isfinite(canonical["y_pred"]).sum()),
            "uid_first": uid_first,
            "uid_last": uid_last,
            "prediction_column_from_bundle": pred_col,
        },
        part_meta,
    )
    input_csv.unlink(missing_ok=True)
    raw_output.unlink(missing_ok=True)
    return part_csv


def paired_arrays(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    y_true = pd.to_numeric(frame["y_true"], errors="coerce").to_numpy(dtype=np.float64)
    y_pred = pd.to_numeric(frame["y_pred"], errors="coerce").to_numpy(dtype=np.float64)
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[keep], y_pred[keep]


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = len(y_true)
    if n < 2:
        return float("nan")
    order = np.argsort(y_true, kind="mergesort")
    y_true = y_true[order]
    y_pred = y_pred[order]
    _, ranks = np.unique(y_pred, return_inverse=True)
    ranks = ranks + 1
    bit = np.zeros(int(ranks.max()) + 1, dtype=np.int64)

    def add(index: int) -> None:
        while index < len(bit):
            bit[index] += 1
            index += index & -index

    def prefix(index: int) -> int:
        total = 0
        while index > 0:
            total += int(bit[index])
            index -= index & -index
        return total

    permissible = 0.0
    concordant = 0.0
    previous = 0
    start = 0
    while start < n:
        end = start + 1
        while end < n and y_true[end] == y_true[start]:
            end += 1
        for rank in ranks[start:end]:
            less = prefix(int(rank) - 1)
            equal = prefix(int(rank)) - less
            concordant += less + 0.5 * equal
            permissible += previous
        for rank in ranks[start:end]:
            add(int(rank))
        previous += end - start
        start = end
    return float(concordant / permissible) if permissible > 0 else float("nan")


def metric_pack(frame: pd.DataFrame) -> Dict[str, float]:
    y_true, y_pred = paired_arrays(frame)
    if len(y_true) == 0:
        return {name: float("nan") for name in ["n", "r2", "rmse", "mae", "pearson", "ci", "rm2"]}
    residual = y_true - y_pred
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    pearson = (
        float(np.corrcoef(y_true, y_pred)[0, 1])
        if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0
        else float("nan")
    )
    denom = float(np.sum(y_pred ** 2))
    if denom > 0 and ss_tot > 0 and np.isfinite(r2):
        slope0 = float(np.sum(y_true * y_pred) / denom)
        r02 = 1.0 - float(np.sum((y_true - slope0 * y_pred) ** 2)) / ss_tot
        rm2 = float(r2 * (1.0 - math.sqrt(abs(r2 - r02))))
    else:
        rm2 = float("nan")
    return {
        "n": int(len(y_true)),
        "r2": float(r2),
        "rmse": rmse,
        "mae": mae,
        "pearson": pearson,
        "ci": concordance_index(y_true, y_pred),
        "rm2": rm2,
        "mean_error_pred_minus_true": float(np.mean(y_pred - y_true)),
    }


def write_metrics(frame: pd.DataFrame, out_dir: Path, expected_r2: float, expected_rmse: float) -> None:
    rows = []
    overall = metric_pack(frame)
    rows.append({"endpoint": "overall", **overall})
    for endpoint, _ in ENDPOINTS:
        rows.append(
            {"endpoint": endpoint, **metric_pack(frame[frame["affinity_type"] == endpoint])}
        )
    metrics = pd.DataFrame(rows)
    atomic_csv(metrics, out_dir / "temporal_prediction_metrics.csv")

    check = {
        "expected_paper_metrics": {"r2": expected_r2, "rmse": expected_rmse},
        "observed_metrics": overall,
        "absolute_difference": {
            "r2": abs(float(overall["r2"]) - expected_r2),
            "rmse": abs(float(overall["rmse"]) - expected_rmse),
        },
        "interpretation": (
            "This is a reproduction check only. Differences can indicate that the supplied "
            "predict_bundle is not the exact model used for the manuscript's original temporal table."
        ),
    }
    atomic_json(check, out_dir / "temporal_metric_reproduction_check.json")
    log("Temporal raw metrics:\n" + metrics.to_string(index=False))


def run_audit(
    audit_script: Path,
    train_predictions: Path,
    validation_predictions: Path,
    temporal_predictions: Path,
    audit_out: Path,
    cpus: int,
) -> None:
    for path, label in [
        (audit_script, "audit script"),
        (train_predictions, "training predictions"),
        (validation_predictions, "validation predictions"),
        (temporal_predictions, "temporal predictions"),
    ]:
        require_nonempty(path, label)
    command = [
        sys.executable,
        str(audit_script.resolve()),
        "--train_predictions",
        str(train_predictions.resolve()),
        "--validation_predictions",
        str(validation_predictions.resolve()),
        "--temporal_predictions",
        str(temporal_predictions.resolve()),
        "--out_dir",
        str(audit_out.resolve()),
        "--cache_dir",
        str((audit_out / "cache").resolve()),
        "--cpus",
        str(max(1, int(cpus))),
    ]
    log("Launching Reviewer 2 Suggestion 8 temporal-shift audit")
    process = subprocess.run(command)
    if process.returncode != 0:
        raise RuntimeError(f"Temporal-shift audit failed with exit code {process.returncode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal_csv", required=True, type=Path)
    parser.add_argument("--bundle_dir", required=True, type=Path)
    parser.add_argument("--train_predictions", required=True, type=Path)
    parser.add_argument("--validation_predictions", required=True, type=Path)
    parser.add_argument("--audit_script", type=Path, default=Path("confmux_temporal_shift_audit.py"))
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--batch_rows", type=int, default=25000)
    parser.add_argument("--cpus", type=int, default=max(1, min(32, os.cpu_count() or 1)))
    parser.add_argument("--expected_total", type=int, default=EXPECTED_TOTAL)
    parser.add_argument("--expected_r2", type=float, default=0.600)
    parser.add_argument("--expected_rmse", type=float, default=0.919)
    parser.add_argument("--prepare_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_nonempty(args.temporal_csv, "temporal CSV")
    manifest, manifest_path = load_bundle_manifest(args.bundle_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model_smiles_col, model_protein_col, affinity_col = resolve_input_columns(manifest)
    signature_payload = {
        "temporal_csv": file_state(args.temporal_csv),
        "predict_manifest_sha256": sha256_file(manifest_path),
        "best_model": file_state(args.bundle_dir / "best_model.txt"),
        "predict_cli": file_state(args.bundle_dir / "predict_cli.py"),
        "vectorizer_ligand": file_state(args.bundle_dir / "vectorizer_ligand.pkl"),
        "vectorizer_protein": file_state(args.bundle_dir / "vectorizer_protein.pkl"),
        "smiles_tokenizer": file_state(args.bundle_dir / "smiles_bpe_tokenizer.json"),
        "protein_tokenizer": file_state(args.bundle_dir / "protein_bpe_tokenizer.json"),
        "batch_rows": int(args.batch_rows),
        "script_version": "temporal_prepare_v3",
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    parts_dir = args.out_dir / "prediction_cache" / signature
    parts_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(signature_payload, parts_dir / "run_signature.json")

    header = pd.read_csv(args.temporal_csv, nrows=0).columns.tolist()
    required_columns = {"smiles", "protein", "pIC50", "pKi", "pKd"}
    missing = sorted(required_columns - set(header))
    if missing:
        raise ValueError(f"Temporal CSV missing columns: {missing}; header={header}")

    part_paths: List[Path] = []
    observed_counts = {endpoint: 0 for endpoint, _ in ENDPOINTS}
    source_offset = 0
    for batch_index, wide in enumerate(
        pd.read_csv(args.temporal_csv, chunksize=args.batch_rows, low_memory=False)
    ):
        batch = expand_wide_chunk(
            wide,
            source_offset,
            model_smiles_col,
            model_protein_col,
            affinity_col,
        )
        source_offset += len(wide)
        if batch.empty:
            continue
        for endpoint, count in batch["affinity_type"].value_counts().items():
            observed_counts[str(endpoint)] += int(count)
        part_paths.append(
            run_bundle_batch(batch, batch_index, args.bundle_dir, manifest, parts_dir)
        )

    observed_total = int(sum(observed_counts.values()))
    identity = {
        "wide_rows": int(source_offset),
        "long_rows": observed_total,
        "endpoint_counts": observed_counts,
        "expected_long_rows": int(args.expected_total),
        "expected_endpoint_counts": EXPECTED_ENDPOINT_COUNTS,
    }
    atomic_json(identity, args.out_dir / "temporal_dataset_identity.json")
    if args.expected_total > 0 and observed_total != args.expected_total:
        raise ValueError(
            f"Temporal dataset identity mismatch: expected {args.expected_total:,} long rows, "
            f"observed {observed_total:,}; counts={observed_counts}"
        )
    if args.expected_total == EXPECTED_TOTAL and observed_counts != EXPECTED_ENDPOINT_COUNTS:
        raise ValueError(
            f"Endpoint count mismatch: expected {EXPECTED_ENDPOINT_COUNTS}, observed {observed_counts}"
        )

    log(f"Combining {len(part_paths)} completed prediction batches")
    combined = pd.concat(
        [pd.read_csv(path, low_memory=False) for path in part_paths],
        ignore_index=True,
    )
    if combined["_temporal_uid"].duplicated().any():
        raise ValueError("Duplicate _temporal_uid values found after batch concatenation")
    combined = combined.sort_values(
        ["source_row", "affinity_type"], kind="mergesort"
    ).reset_index(drop=True)
    temporal_predictions = args.out_dir / "temporal_predictions_for_audit.csv"
    atomic_csv(combined, temporal_predictions)
    write_metrics(combined, args.out_dir, args.expected_r2, args.expected_rmse)

    finite_predictions = int(np.isfinite(pd.to_numeric(combined["y_pred"], errors="coerce")).sum())
    if finite_predictions != len(combined):
        log(
            f"WARNING: {len(combined) - finite_predictions:,} records have non-finite predictions; "
            "they are retained for traceability and excluded from metrics."
        )

    if not args.prepare_only:
        run_audit(
            args.audit_script,
            args.train_predictions,
            args.validation_predictions,
            temporal_predictions,
            args.out_dir / "shift_audit",
            args.cpus,
        )
    log(f"Completed temporal S8 pipeline: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
