"""Rebuild corrected tables without training, calibration, or external services."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from release_metrics import regression_metrics

S4 = "historical_reproducibility/supp_table_s4"
S2 = "historical_reproducibility/suppfig_s2"


def csv_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dump_csv(path, rows, columns=None):
    rows = list(rows)
    if not rows:
        raise ValueError("Refusing an empty result table: " + str(path))
    columns = columns or list(rows[0])
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def dump_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def davis(root):
    labels = list(csv_rows(root / S2 / "input/DAVIS.csv.gz"))
    predictions = list(csv_rows(root / S2 / "input/DAVIS_pred_minimal.csv.gz"))
    history = list(csv_rows(root / S2 / "input/DAVIS_S2_points.csv.gz"))
    if not len(labels) == len(predictions) == len(history) == 30060:
        raise ValueError("DAVIS row-count mismatch")
    pred_by_id = {}
    for row in predictions:
        key = row["_pair_uid"]
        if key in pred_by_id:
            raise ValueError("Duplicate DAVIS prediction ID: " + key)
        pred_by_id[key] = float(row["pred_pX"])
    if set(pred_by_id) != {"DAVIS_" + str(i) for i in range(len(labels))}:
        raise ValueError("DAVIS IDs do not match deposited row identifiers")
    points = []
    for i, (row, old) in enumerate(zip(labels, history)):
        y = float(row["pKd"])
        if y != float(old["y_true"]):
            raise ValueError("DAVIS label ordering changed at row " + str(i))
        points.append({"_pair_uid": "DAVIS_" + str(i), "y_true": y,
                       "y_pred": pred_by_id["DAVIS_" + str(i)]})
    y, p = [r["y_true"] for r in points], [r["y_pred"] for r in points]
    corrected = [float(r["y_pred_calibrated"]) for r in history]
    frozen_metrics = regression_metrics(y, p)
    posthoc_metrics = regression_metrics(y, corrected)
    # Audit an ALREADY archived mapping; never fit a mapping for primary predictions.
    a, b = 0.8695953388019565, 0.6184351873043372
    deviation = max(abs(a * raw + b - old) for raw, old in zip(p, corrected))
    if deviation > 1e-10:
        raise ValueError("Archived DAVIS calibration provenance does not match")
    return points, {
        "primary_protocol": "frozen_predictions_no_DAVIS_label_refit",
        "calibration_fit_in_this_rebuild": False,
        "overlap_audited": False,
        "primary_frozen": frozen_metrics,
        "historical_posthoc_same_evaluation_labels": posthoc_metrics,
        "archived_posthoc_mapping": {"slope": a, "intercept": b,
                                    "maximum_replay_difference": deviation},
        "interpretation": "Historical post-hoc values are descriptive only, not independent external predictive performance."
    }


def s4_table(root):
    old_path = root / S4 / "input/Supplementary_Table_S4_source_original_20260914.csv"
    if not old_path.is_file():
        old_path = root / S4 / "input/Supplementary_Table_S4_source.csv"
    requested = list(csv_rows(old_path))
    fields = ("standard_inchi_key", "uniprot_id", "affinity_type")
    def key(row):
        return tuple(row[k] for k in fields)
    selected_keys = [key(r) for r in requested]
    if len(selected_keys) != 5 or len(set(selected_keys)) != 5:
        raise ValueError("S4 must identify five unique compound/complete-target/endpoint keys")
    pool = {}
    for row in csv_rows(root / S4 / "input/cand_balanced_for_review.csv"):
        k = key(row)
        if k not in selected_keys:
            continue
        if k in pool and pool[k] != row:
            raise ValueError("Ambiguous S4 candidate key: " + repr(k))
        pool[k] = row
    output = []
    for k in selected_keys:
        if k not in pool:
            raise ValueError("Missing S4 candidate: " + repr(k))
        row = pool[k]
        lo, hi = float(row["interval_lo_calibrated"]), float(row["interval_hi_calibrated"])
        prediction, observed = float(row["y_pred_calibrated"]), float(row["y_true"])
        raw_width = float(row["err_bound"])
        half = (hi - lo) / 2
        vals = [lo, hi, prediction, observed, raw_width, half]
        if not all(math.isfinite(v) for v in vals) or min(raw_width, half) <= 0:
            raise ValueError("Non-finite or nonpositive S4 width")
        if abs((hi + lo) / 2 - prediction) > 1e-10:
            raise ValueError("S4 interval midpoint != calibrated prediction")
        if abs(abs(prediction - observed) - float(row["abs_error_calibrated"])) > 1e-10:
            raise ValueError("S4 absolute-error mismatch")
        out = {name: row[name] for name in requested[0] if name != "err_bound"}
        out["err_bound_raw"] = raw_width
        out["err_bound_calibrated"] = half
        out["interval_scale"] = "historical_external_posthoc_calibrated"
        out["evaluation_label_use"] = "used_in_historical_posthoc_fit_and_example_selection"
        output.append(out)
    return output


def fig5_table(root):
    rows = list(csv_rows(root / "historical_reproducibility/fig5/input/valid_predictions_for_paper.csv"))
    if len(rows) != 406650:
        raise ValueError("Figure 5 requires the historical VALIDATION set (406650 rows)")
    out = []
    for threshold in [0.8, 1.0, 1.2, 1.5]:
        selected = [r for r in rows if float(r["err_bound"]) <= threshold]
        metrics = regression_metrics([float(r["y_true"]) for r in selected],
                                     [float(r["y_pred"]) for r in selected], include_ci=False)
        out.append({"half_width_threshold": threshold, "accepted_count": len(selected),
                    "validation_count": len(rows), "accepted_fraction": len(selected)/len(rows),
                    "rmse": metrics["rmse"], "r2": metrics["r2"]})
    return out


def build(root, output):
    output.mkdir(parents=True, exist_ok=True)
    points, metrics = davis(root)
    dump_csv(output / "DAVIS_frozen_points.csv", points)
    dump_json(output / "DAVIS_evaluation_protocols.json", metrics)
    metric_rows = []
    for label, key in [("frozen_no_external_label_refit", "primary_frozen"),
                       ("historical_same_set_posthoc_not_independent", "historical_posthoc_same_evaluation_labels")]:
        metric_rows.append({"protocol": label, **metrics[key]})
    dump_csv(output / "Table5_DAVIS_protocol_comparison.csv", metric_rows)
    dump_csv(output / "Supplementary_Table_S4_posthoc_corrected.csv", s4_table(root))
    dump_csv(output / "Table4_Figure5_validation_check.csv", fig5_table(root))
    source_paths = [
        S2 + "/input/DAVIS.csv.gz", S2 + "/input/DAVIS_pred_minimal.csv.gz",
        S2 + "/input/DAVIS_S2_points.csv.gz", S4 + "/input/cand_balanced_for_review.csv",
        "historical_reproducibility/fig5/input/valid_predictions_for_paper.csv",
    ]
    dump_json(output / "derivation_manifest.json", {
        "schema": "confmux.revision_correction.v1",
        "fits_model": False, "fits_calibration": False, "drops_records": False,
        "source_sha256": {s: file_hash(root / s) for s in source_paths},
        "davis_frozen_join": "DAVIS_i row identity; cross-checked against archived y_true",
        "s4_join": ["standard_inchi_key", "complete_uniprot_id", "affinity_type"],
        "s4_selection": "Existing five author-selected examples; no new selection or performance claim",
        "s4_half_width": "(interval_hi_calibrated - interval_lo_calibrated) / 2",
        "historical_validation_n": 406650,
        "controlled_revision_validation_n": 406647,
    })
    print(json.dumps({"outputs": str(output), "davis_frozen": metrics["primary_frozen"]}, ensure_ascii=False))
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    root, out = args.root.resolve(), args.out_dir.resolve()
    if out == root or root.is_relative_to(out):
        raise ValueError("Choose a separate output subdirectory, not a source root/ancestor")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Output directory is not empty; use a new directory: " + str(out))
    build(root, out)


if __name__ == "__main__":
    main()

