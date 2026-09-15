"""Portable, explicit launch recipes. Dry-run is the default; no implicit training."""
import argparse
import ast
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("full", "no_onehot", "char_lgbm", "endpoint_pIC50", "endpoint_pKi",
         "endpoint_pKd", "ridge", "half_scale", "deepdta", "deepdtagen",
         "uncertainty", "sigma_head", "temporal_audit", "temporal_predict")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--data-file", default=os.environ.get("CONFMUX_DATA", ""))
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--threads", type=int, default=int(os.environ.get("CONFMUX_THREADS", "4")))
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--reference-cache", default=os.environ.get("REFERENCE_CACHE_DIR", ""))
    p.add_argument("--deepdtagen-root", default=os.environ.get("DEEPDTAGEN_ROOT", ""))
    p.add_argument("--pt-file", default=os.environ.get("DEEPDTAGEN_BINDINGDB_PT", os.environ.get("PT_FILE", "")))
    p.add_argument("--tokenizer-file", default=os.environ.get("TOKENIZER_FILE", ""))
    p.add_argument("--bundle-dir", default=os.environ.get("CONFMUX_HISTORICAL_BUNDLE", ""))
    p.add_argument("--full-run-dir", default=os.environ.get("CONFMUX_FULL_RUN", ""))
    p.add_argument("--feature-cache", default=os.environ.get("CONFMUX_FEATURE_CACHE", ""))
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--resume", action="store_true")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    p.add_argument("--check-inputs", action="store_true")
    return p


def recipe(a):
    root = a.root.resolve()
    task = a.task
    output = (a.out_dir or root / "reproduction_runs" / task).resolve()
    core = root / "code/confmux_dta_train_predict_leakage_controlled.py"
    scale = root / "code/confmux_dta_train_predict_scale_controlled.py"
    required = []
    def input_path(value, label):
        if not value:
            required.append({"label": label, "path": None})
            return "<SET_" + label.upper() + ">"
        path = Path(value).expanduser().resolve()
        required.append({"label": label, "path": str(path)})
        return str(path)
    def script(name):
        p = root / "code" / name
        required.append({"label": "script", "path": str(p)})
        return str(p)
    common = ["--seed", "5313", "--num_threads", str(a.threads),
              "--total_iters", "50000", "--stage_iters", "2000"]
    if task in ("full", "no_onehot", "half_scale"):
        argv = [script(scale.name if task == "half_scale" else core.name), "train_test",
                "--data_file", input_path(a.data_file, "data_file"), "--unified",
                "--no_type_onehot" if task == "no_onehot" else "--type_onehot",
                "--split_mode", "random", "--test_size", "0.20", "--force_device", "cpu",
                *common, "--optimize_by", "rmse", "--calib_method", "train_linear",
                "--no_selective", "--no_backup", "--run_dir", str(output)]
        if task == "half_scale":
            argv += ["--train_fraction", "0.5", "--train_subsample_seed", "20260910",
                     "--reference_cache_dir", input_path(a.reference_cache or str(root / "manifests/full_controlled_split"), "reference_indices"),
                     "--no_sigma_uncert"]
        if a.resume:
            argv += ["--resume"]
    elif task == "char_lgbm" or task.startswith("endpoint_"):
        argv = [script("confmux_major_design_ablation.py"),
                "char_lgbm" if task == "char_lgbm" else "endpoint_lgbm",
                "--core", str(core), "--data_file", input_path(a.data_file, "data_file"),
                "--run_dir", str(output), *common]
        required.append({"label": "core_script", "path": str(core)})
        if task.startswith("endpoint_"):
            argv += ["--endpoint", task.split("_", 1)[1]]
    elif task == "ridge":
        argv = [script("confmux_bpe_ridge_ablation.py"), "--core", str(core),
                "--data_file", input_path(a.data_file, "data_file"), "--run_dir", str(output),
                "--seed", "5313", "--num_threads", str(a.threads),
                "--alpha", "1.0", "--tol", "1e-4", "--max_iter", "1000"]
        required.append({"label": "core_script", "path": str(core)})
    elif task == "deepdta":
        ref = Path(a.reference_cache) if a.reference_cache else root / "manifests/full_controlled_split"
        for f in ("idx_train.npy", "idx_valid.npy"):
            required.append({"label": f, "path": str(ref.resolve() / f)})
        argv = [script("deepdta_full_controlled.py"), "--data_file", input_path(a.data_file, "data_file"),
                "--confmux_source", str(scale), "--reference_cache_dir", str(ref.resolve()),
                "--work_dir", str(output), "--cache_dir", str(output / "encoding_cache"),
                "--seed", "5313", "--split_mode", "random", "--test_size", "0.2",
                "--device", a.device, "--max_smiles_len", "100", "--max_protein_len", "1000",
                "--embedding_dim", "128", "--filters", "32", "--smiles_kernel", "4",
                "--protein_kernel", "8", "--batch_size", "1024", "--eval_batch_size", "2048",
                "--workers", str(min(8, a.threads)), "--learning_rate", "0.001",
                "--max_epochs", "100", "--patience", "15", "--min_delta", "0.00001",
                "--grad_clip", "5", "--amp"]
        required.append({"label": "split_source", "path": str(scale)})
        if a.resume:
            argv += ["--resume"]
    elif task == "deepdtagen":
        repo = input_path(a.deepdtagen_root, "deepdtagen_root")
        if a.deepdtagen_root:
            for f in ("model.py", "utils.py", "FetterGrad.py"):
                required.append({"label": f, "path": str(Path(repo) / f)})
        argv = [script("deepdtagen_speed_benchmark.py"), "--repo-root", repo,
                "--pt-file", input_path(a.pt_file, "pt_file"),
                "--batch-size", str(a.batch_size), "--warmup-steps", "10",
                "--measure-steps", "100", "--eval-measure-steps", "30", "--num-workers", "0",
                "--project-train-rows", "1626585", "--project-valid-rows", "406647",
                "--eval-every", "20", "--project-epochs", "50", "100", "500",
                "--output-json", str(output / "deepdtagen_speed.json"),
                "--output-csv", str(output / "deepdtagen_speed.csv")]
        if a.tokenizer_file:
            argv += ["--tokenizer-file", input_path(a.tokenizer_file, "tokenizer_file")]
    elif task == "uncertainty":
        argv = [script("confmux_uncertainty_evaluation.py"), "--predictions",
                input_path(str(root / "results/uncertainty_rigorous/sigma_head/valid_predictions_with_sigma.csv.gz"), "sigma_predictions"),
                "--out-dir", str(output), "--seed", "20260908", "--calibration-fraction", "0.5",
                "--levels", "0.50,0.60,0.70,0.80,0.90,0.95,0.975,0.99"]
    elif task == "sigma_head":
        full = Path(a.full_run_dir) if a.full_run_dir else root / "reproduction_runs/full"
        argv = [script("confmux_sigma_head_only.py"), "--core-script", str(core),
                "--config", input_path(str(full / "config_used.json"), "full_config"),
                "--feature-cache", input_path(a.feature_cache, "feature_cache"),
                "--train-predictions", input_path(str(full / "train_predictions_for_paper.csv"), "full_train_predictions"),
                "--valid-predictions", input_path(str(full / "valid_predictions_for_paper.csv"), "full_valid_predictions"),
                "--out-dir", str(output), "--threads", str(a.threads)]
        required.append({"label": "core_script", "path": str(core)})
    else:
        hist = root / "historical_reproducibility/s8_historical_inputs/input"
        train = input_path(str(hist / "s8_historical_random_train_reference.csv.gz"), "historical_train")
        valid = input_path(str(hist / "s8_historical_random_validation_predictions.csv.gz"), "historical_valid")
        if task == "temporal_audit":
            argv = [script("confmux_temporal_shift_audit.py"), "--train_predictions", train,
                    "--validation_predictions", valid, "--temporal_predictions",
                    input_path(str(root / "results/temporal_shift/temporal_predictions_for_audit.csv"), "frozen_temporal_predictions"),
                    "--out_dir", str(output), "--cpus", str(a.threads), "--seed", "5313"]
        else:
            argv = [script("confmux_temporal_prepare_and_run.py"), "--temporal_csv",
                    input_path(str(root / "data/temporal/bindingdb_diff_only_in_new.csv"), "temporal_data"),
                    "--bundle_dir", input_path(a.bundle_dir, "historical_bundle"),
                    "--train_predictions", train, "--validation_predictions", valid,
                    "--audit_script", script("confmux_temporal_shift_audit.py"),
                    "--out_dir", str(output), "--cpus", str(a.threads)]
    command = [sys.executable, "-u", *argv]
    known_options = set()
    for node in ast.walk(ast.parse(Path(argv[0]).read_text(encoding="utf-8-sig"))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            known_options.update(n.value for n in node.args if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith("--"))
    unknown = [word for word in argv if word.startswith("--") and word not in known_options]
    if unknown:
        raise ValueError("Recipe has unsupported CLI options: " + repr(unknown))
    return command, output, required


def main():
    a = parser().parse_args()
    if a.threads < 1:
        raise ValueError("--threads must be >= 1")
    command, output, required = recipe(a)
    missing = [r for r in required if not r["path"] or not Path(r["path"]).exists()]
    plan = {"task": a.task, "command": command, "cwd": str(a.root.resolve()),
            "out_dir": str(output), "required_inputs": required, "missing_inputs": missing,
            "executed": a.execute}
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if (a.execute or a.check_inputs) and missing:
        raise FileNotFoundError("Required inputs missing; see plan above. No job was launched.")
    if not a.execute:
        return
    if output.exists() and any(output.iterdir()) and not a.resume:
        raise FileExistsError("Use a new output directory, or explicit --resume where supported.")
    if a.resume and a.task not in ("full", "no_onehot", "half_scale", "deepdta"):
        raise ValueError("--resume is not supported for this task")
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = str(a.threads)
    environment["PYTHONUNBUFFERED"] = "1"
    if a.task == "deepdtagen":
        compat = str(a.root.resolve() / "code/deepdtagen_compat")
        environment["PYTHONPATH"] = compat + os.pathsep + environment.get("PYTHONPATH", "")
    child = subprocess.Popen(command, cwd=a.root.resolve(), env=environment)
    previous = {}
    def forward(signum, frame):
        if child.poll() is None:
            child.send_signal(signum)
    for name in ('SIGUSR1', 'SIGTERM'):
        if hasattr(signal, name):
            sig = getattr(signal, name)
            previous[sig] = signal.signal(sig, forward)
    try:
        status = child.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    raise SystemExit(status)


if __name__ == "__main__":
    main()
