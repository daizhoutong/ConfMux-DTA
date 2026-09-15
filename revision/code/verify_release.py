"""Verify this release locally with standard-library Python; never train or submit jobs."""
import argparse
import array
import ast
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import traceback

from rebuild_release_outputs import build, csv_rows, file_hash, dump_json
from release_metrics import regression_metrics, concordance_index
import run_release

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, tol=1e-10):
    require(math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b)) <= tol,
            "Numerical mismatch: {!r} != {!r} (tol={!r})".format(a, b, tol))


def compare_values(a, b, where=""):
    if isinstance(a, dict):
        require(isinstance(b, dict) and a.keys() == b.keys(), "Dictionary keys changed: " + where)
        for k in a:
            compare_values(a[k], b[k], where + "/" + k)
    elif isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), "List changed: " + where)
        for i, (x, y) in enumerate(zip(a, b)):
            compare_values(x, y, where + "/" + str(i))
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        close(a, b)
    else:
        require(a == b, "Value changed: " + where)


def load_indices(path):
    with open(path, "rb") as f:
        require(f.read(6) == b"\x93NUMPY", "Not an NPY file: " + str(path))
        version = tuple(f.read(2))
        require(version in ((1, 0), (2, 0), (3, 0)), "Unsupported NPY format")
        fmt = "<H" if version == (1, 0) else "<I"
        length = struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
        header = ast.literal_eval(f.read(length).decode("latin1").strip())
        require(header["descr"] in ("<i8", "=i8"), "Expected int64 indices")
        require(len(header["shape"]) == 1 and not header["fortran_order"], "Unexpected index shape")
        values = array.array("q")
        values.frombytes(f.read())
        if sys.byteorder != "little":
            values.byteswap()
        require(len(values) == header["shape"][0], "Truncated NPY indices")
        return values


def verify_hashes(root):
    count = 0
    seen = set()
    for line in (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel = line.split(None, 1)
        rel = rel.strip().lstrip("*")
        p = (root / rel).resolve()
        require(p.is_relative_to(root), "Unsafe checksum path: " + rel)
        require(p not in seen, "Duplicate checksum entry: " + rel)
        seen.add(p)
        require(p.is_file(), "Missing release file: " + rel)
        require(file_hash(p) == expected.lower(), "SHA256 mismatch: " + rel)
        count += 1
    require(count >= 166, "Incomplete checksum inventory")
    return {"files_verified": count, "all_match": True}


def verify_splits(root):
    train = load_indices(root / "manifests/full_controlled_split/idx_train.npy")
    valid = load_indices(root / "manifests/full_controlled_split/idx_valid.npy")
    require((len(train), len(valid)) == (1626585, 406647), "Controlled split counts differ")
    n = len(train) + len(valid)
    marks = bytearray(n)
    for subset, code in [(train, 1), (valid, 2)]:
        for value in subset:
            require(0 <= value < n and marks[value] == 0, "Duplicate/overlapping/out-of-range split index")
            marks[value] = code
    require(all(marks), "Controlled split does not cover every eligible row")
    base = load_indices(root / "manifests/half_scale_split/idx_base_train.npy")
    half_valid = load_indices(root / "manifests/half_scale_split/idx_valid.npy")
    require(base == train and half_valid == valid, "Half-scale reference split differs")
    half = load_indices(root / "manifests/half_scale_split/idx_train.npy")
    require(len(half) == 813292, "Half-scale size mismatch")
    for value in half:
        require(0 <= value < n and marks[value] == 1, "Half-scale index is invalid, repeated or outside training")
        marks[value] = 3
    return {"controlled_train": len(train), "controlled_valid": len(valid),
            "half_train": len(half), "nested_and_disjoint": True}, valid


def metric_unit_tests():
    # Includes observation ties, prediction ties, reversal and constant predictions.
    cases = [
        ([1, 2, 3], [1, 2, 3]), ([1, 2, 3], [3, 2, 1]),
        ([1, 1, 2, 3, 3], [1, 2, 2, 0, 3]), ([1, 2, 3, 4], [2, 2, 2, 2]),
        ([0, 0, 1, 1, 2, 3], [3, 1, 1, 2, 2, 3]),
    ]
    for y, p in cases:
        total = score = 0
        for i in range(len(y)):
            for j in range(i):
                if y[i] == y[j]:
                    continue
                total += 1
                score += 0.5 if p[i] == p[j] else float((p[i]-p[j])*(y[i]-y[j]) > 0)
        close(concordance_index(y, p), score/total, 0)
    close(regression_metrics([1, 2, 3], [1, 2, 3])["r2"], 1, 0)
    try:
        regression_metrics([1, 2], [1, float("nan")])
    except ValueError:
        pass
    else:
        raise ValueError("Non-finite metrics did not fail")
    return {"ci_cases_vs_bruteforce": len(cases), "finite_and_perfect_fit_checks": True}


def verify_outputs(root):
    reference = root / "results/revision_fixed"
    with tempfile.TemporaryDirectory(prefix="confmux_recompute_") as tmp:
        out = Path(tmp)
        result = build(root, out)
        names = sorted(p.name for p in out.iterdir() if p.is_file())
        for name in names:
            ref = reference / name
            require(ref.is_file(), "Missing corrected reference output: " + name)
            if name.endswith(".json"):
                compare_values(json.loads((out/name).read_text(encoding="utf-8")),
                               json.loads(ref.read_text(encoding="utf-8")), name)
            else:
                # Deterministic CSV re-export must be byte-identical.
                require((out/name).read_bytes() == ref.read_bytes(), "CSV rerun mismatch: " + name)
        s4_source = root / "historical_reproducibility/supp_table_s4/input/Supplementary_Table_S4_source.csv"
        require(s4_source.read_bytes() == (out/"Supplementary_Table_S4_posthoc_corrected.csv").read_bytes(),
                "S4 publication source differs from corrected exporter")
    close(result["primary_frozen"]["r2"], 0.5654588893342065)
    close(result["primary_frozen"]["rmse"], 0.5913741474795985)
    close(result["primary_frozen"]["ci"], 0.8709186245491355)
    close(result["historical_posthoc_same_evaluation_labels"]["r2"], 0.5928998061538723)
    return {"deterministic_outputs": len(names), "csv_byte_identical": True,
            "davis_frozen": result["primary_frozen"]}


def verify_entry_points(root):
    plans = []
    for task in run_release.TASKS:
        a = run_release.parser().parse_args(["--task", task, "--root", str(root), "--dry-run"])
        command, output, required = run_release.recipe(a)
        require(Path(command[2]).is_file(), "Recipe script missing")
        plans.append({"task": task, "script": str(Path(command[2]).relative_to(root)),
                      "argv_checked_against_actual_parser": True})
    for name in ["code/submit_deepdta_full_controlled.sbatch",
                 "code/submit_deepdtagen_speed_benchmark.sbatch",
                 "code/run_confmux_temporal_s8_local.sh", "PATH_CONFIG.example.sh"]:
        text = (root / name).read_text(encoding="utf-8")
        require("/data/person/" not in text and "/home/lhy/" not in text, "Private deployment default: " + name)
        require(b"\r" not in (root / name).read_bytes(), "CRLF in shell entry point: " + name)
    return plans


def verify_uncertainty(root):
    base = root / "results/uncertainty_rigorous"
    membership = {}
    for r in csv_rows(base / "evaluation/calibration_evaluation_membership.csv"):
        i = int(r["source_row"])
        require(i not in membership, "Duplicate uncertainty membership row")
        require(r["subset"] in ("calibration", "evaluation"), "Unknown uncertainty subset")
        membership[i] = (r["subset"], r["endpoint"])
    require(len(membership) == 406647, "Uncertainty membership count mismatch")
    cal, evaluation = [], []
    seen = set()
    for r in csv_rows(base / "sigma_head/valid_predictions_with_sigma.csv.gz"):
        i = int(r["source_row"])
        require(i not in seen and i in membership, "Invalid uncertainty prediction identity")
        seen.add(i)
        subset, endpoint = membership[i]
        require(endpoint == r["affinity_type"], "Uncertainty endpoint mapping mismatch")
        y, p, sigma = (float(r[k]) for k in ("y_true", "y_pred", "sigma"))
        require(all(math.isfinite(v) for v in (y, p, sigma)) and sigma > 0, "Invalid sigma/label")
        (cal if subset == "calibration" else evaluation).append((abs(y-p), sigma, endpoint))
    require(len(cal) == 203323 and len(evaluation) == 203324, "Uncertainty split counts differ")
    quantiles = json.loads((base / "evaluation/conformal_quantiles.json").read_text())
    table = list(csv_rows(base / "evaluation/paper_uncertainty_comparison.csv"))
    norm_scores = sorted(e/s for e, s, _ in cal)
    global_scores = sorted(e for e, _, _ in cal)
    per_endpoint = {ep: sorted(e for e, _, t in cal if t == ep) for ep in ("pIC50", "pKi", "pKd")}
    output = []
    for level in (0.8, 0.9, 0.95):
        qinfo = quantiles[f"{level:.6f}"]
        rank = min(len(cal), math.ceil((len(cal)+1)*level))
        nq, gq = norm_scores[rank-1], global_scores[rank-1]
        close(nq, qinfo["Normalized split conformal (proposed)"]["q"])
        close(gq, qinfo["Global split conformal (baseline)"]["q"])
        eq = {}
        for ep, scores in per_endpoint.items():
            eq[ep] = scores[min(len(scores), math.ceil((len(scores)+1)*level))-1]
            close(eq[ep], qinfo["Endpoint-Mondrian split conformal"]["endpoint_q"][ep]["q"])
        for method in ("Normalized split conformal (proposed)", "Global split conformal (baseline)", "Endpoint-Mondrian split conformal"):
            covered = sum(e <= (nq*s if method.startswith("Normalized") else gq if method.startswith("Global") else eq[ep])
                          for e, s, ep in evaluation)
            matches = [r for r in table if r["Method"] == method and r["Endpoint"] == "Overall"
                       and float(r["Nominal_coverage"]) == level]
            require(len(matches) == 1, "Uncertainty reference table row ambiguous")
            require(covered == int(matches[0]["Covered_N"]), "Uncertainty covered count differs")
            output.append({"method": method, "level": level, "covered": covered, "n": len(evaluation)})
    return {"calibration_n": len(cal), "evaluation_n": len(evaluation), "checks": output}


def verify_prediction_metrics(root, valid_indices):
    output = {}
    for label, rel, expected_n, r2, rmse in [
        ("deepdta", "results/deepdta_full_controlled/valid_predictions_for_paper.csv.gz", 406647, 0.7561716909538263, 0.7615340655753229),
        ("temporal_frozen", "results/temporal_shift/temporal_predictions_for_audit.csv", 144590, 0.6004856421795948, 0.918632244099824),
    ]:
        y, pred = [], []
        ids = set()
        for ordinal, row in enumerate(csv_rows(root / rel)):
            if label == "deepdta":
                key = int(row["row_index"])
                require(ordinal < len(valid_indices) and key == valid_indices[ordinal], "DeepDTA validation ordering changed")
            else:
                key = row["_temporal_uid"]
            require(key not in ids, "Duplicate prediction identity: " + label)
            ids.add(key)
            y.append(float(row["y_true"]))
            pred.append(float(row["y_pred"]))
        require(len(y) == expected_n, "Prediction count differs: " + label)
        m = regression_metrics(y, pred, include_ci=False)
        # Stored float32 DeepDTA predictions are decimal-serialized; do not claim bitwise metric equality.
        tol = 1e-6 if label == "deepdta" else 1e-10
        close(m["r2"], r2, tol)
        close(m["rmse"], rmse, tol)
        output[label] = m
    return output


def verify_s8(root):
    hist = root / "historical_reproducibility/s8_historical_inputs/input"
    output = {}
    files = [
        ("train", "s8_historical_random_train_reference.csv.gz", 1626597, {"pIC50":1197379,"pKi":366336,"pKd":62882}),
        ("validation", "s8_historical_random_validation_predictions.csv.gz", 406650, {"pIC50":298960,"pKi":91873,"pKd":15817}),
    ]
    for label, name, expected_n, expected_endpoints in files:
        n = 0
        counts = Counter()
        y, pred = [], []
        for row in csv_rows(hist / name):
            require(row["smiles_norm"] and row["protein_clean"], "Missing historical S8 structure/sequence")
            observed = float(row["y_true"])
            require(math.isfinite(observed), "Invalid historical S8 label")
            n += 1
            counts[row["affinity_type"]] += 1
            if label == "validation":
                y.append(observed)
                pred.append(float(row["y_pred"]))
            if n % 250000 == 0:
                print(f"[S8 {label}] {n:,}/{expected_n:,}", flush=True)
        require(n == expected_n and dict(counts) == expected_endpoints, "S8 historical counts differ: " + label)
        output[label] = {"n": n, "endpoints": dict(counts)}
        if y:
            m = regression_metrics(y, pred, include_ci=False)
            close(m["r2"], 0.8137929255102422)
            close(m["rmse"], 0.6648962792913985)
            output[label]["metrics"] = m
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--report", type=Path)
    p.add_argument("--full", action="store_true", help="Also stream all S8 records and recheck frozen predictions/uncertainty")
    a = p.parse_args()
    root = a.root.resolve()
    report = {"schema": "confmux.release_verification.v1", "status": "running",
              "full": a.full, "python": sys.version, "platform": sys.platform,
              "checks": {}, "not_run": ["model retraining", "historical bundle inference", "Slurm/GPU jobs"],
              "maximum_numeric_tolerance": {"default": 1e-10, "decimal_float32_DeepDTA": 1e-6}}
    started = time.monotonic()
    try:
        for name, fn in [
            ("file_hashes", lambda: verify_hashes(root)),
            ("metric_unit_tests", metric_unit_tests),
            ("corrected_outputs", lambda: verify_outputs(root)),
            ("entry_points", lambda: verify_entry_points(root)),
        ]:
            print("[check] " + name, flush=True)
            report["checks"][name] = fn()
        parsed = 0
        for path in root.rglob("*.py"):
            if any(part in ("reproduction_runs", "__pycache__") for part in path.parts):
                continue
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            parsed += 1
        report["checks"]["python_syntax"] = {"files": parsed}
        print("[check] fixed_splits", flush=True)
        report["checks"]["fixed_splits"], valid = verify_splits(root)
        if a.full:
            for name, fn in [
                ("frozen_prediction_metrics", lambda: verify_prediction_metrics(root, valid)),
                ("uncertainty", lambda: verify_uncertainty(root)),
                ("historical_S8", lambda: verify_s8(root)),
            ]:
                print("[check] " + name, flush=True)
                report["checks"][name] = fn()
        else:
            report["not_run"] += ["full S8 stream", "frozen prediction metric recalculation", "uncertainty recalculation"]
        report["status"] = "passed"
    except Exception as e:
        report["status"] = "failed"
        report["error"] = type(e).__name__ + ": " + str(e)
        traceback.print_exc()
    report["elapsed_seconds"] = round(time.monotonic()-started, 3)
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        dump_json(a.report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
