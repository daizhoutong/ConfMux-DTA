#!/usr/bin/env python3
"""Measure DeepDTAGen end-to-end GPU step throughput and project full-data time.

This script deliberately performs only a short benchmark.  It executes the
official DeepDTAGen forward pass, both FetterGrad backward objectives, and the
optimizer step.  It never writes a model checkpoint and does not modify the
input PT snapshot.

The released BindingDB PT snapshot can be used as a representative stream of
real batches.  Runtime is projected to a larger training/validation row count.
If a full-data PT snapshot is already available, pass it with --pt-file.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import pickle
import random
import socket
import sys
import time
from pathlib import Path
from typing import Any, Mapping


BINDINGDB_TOKEN_VOCAB = (
    "<sos>", "<eos>", "<pad>", "<mask>", "<sep>", "<unk>",
    "<t_6>", "<t_7>", "<t_8>", "<t_9>", "<t_10>", "<t_11>",
    "<t_12>", "<t_13>", "<t_14>", "<t_15>", "<t_16>", "<t_17>",
    "<t_18>", "<t_19>", "<t_20>", "<t_21>", "<t_22>", "<t_23>",
    "<t_24>", "<t_25>", "<t_26>", "<t_27>", "<t_28>", "<t_29>",
    "<t_30>", "<t_31>", "#", "(", ")", "-", ".", "/", "1", "2",
    "3", "4", "5", "6", "7", "8", "=", "B", "C", "F", "I", "N",
    "O", "P", "S", "\\", "c", "n", "o", "s", "Br", "Cl", "[2H]",
    "[3H]", "[4H]", "[B-]", "[C+]", "[C-]", "[C@]", "[I-]", "[N+]",
    "[N-]", "[O-]", "[P+]", "[P@]", "[PH]", "[S+]", "[Se]", "[Si]",
    "[c+]", "[n+]", "[n-]", "[nH]", "[o+]", "[se]", "[18F]", "[BH-]",
    "[Br-]", "[C@@]", "[C@H]", "[CH+]", "[CH-]", "[Cl-]", "[N@+]",
    "[NH+]", "[Na+]", "[OH+]", "[P@@]", "[S@@]", "[nH+]", "[125I]",
    "[C@@H]", "[NH2+]", "[NH3+]", "[S@@+]", "[SiH2]", "[N@@H+]",
)


class EmbeddedTokenizer:
    def __init__(self) -> None:
        self.vocabs = list(BINDINGDB_TOKEN_VOCAB)
        self.i2s = {i: token for i, token in enumerate(self.vocabs)}
        self.s2i = {token: i for i, token in self.i2s.items()}

    def __len__(self) -> int:
        return len(self.vocabs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--pt-file", type=Path, default=None)
    parser.add_argument("--tokenizer-file", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=100)
    parser.add_argument("--eval-warmup-steps", type=int, default=5)
    parser.add_argument("--eval-measure-steps", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--project-train-rows", type=int, default=1_626_585)
    parser.add_argument("--project-valid-rows", type=int, default=406_647)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--project-epochs", type=int, nargs="+", default=[50, 100, 500])
    parser.add_argument("--seed", type=int, default=4221)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--skip-batch-cindex",
        action="store_true",
        help="Skip the small per-training-batch CI used by the official loop.",
    )
    return parser.parse_args()


def find_first(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file() and path.stat().st_size > 0:
            return path.resolve()
    return None


def discover_pt(repo_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"PT snapshot not found or empty: {path}")
        return path
    preferred = find_first([
        repo_root / "data" / "processed" / "bindingdb_train.pt",
        repo_root / "data" / "bindingdb_train.pt",
    ])
    if preferred is not None:
        return preferred
    candidates = sorted(repo_root.rglob("*bindingdb*train*.pt"))
    candidates = [p.resolve() for p in candidates if p.is_file() and p.stat().st_size > 0]
    if not candidates:
        raise FileNotFoundError(
            f"No non-empty BindingDB training PT snapshot was found below {repo_root}"
        )
    return candidates[0]


def discover_tokenizer(repo_root: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Tokenizer pickle not found or empty: {path}")
        return path
    preferred = find_first([
        repo_root / "data" / "bindingdb_tokenizer.pkl",
        repo_root / "bindingdb_tokenizer.pkl",
    ])
    if preferred is not None:
        return preferred
    candidates = sorted(repo_root.rglob("*bindingdb*token*.pkl"))
    return candidates[0].resolve() if candidates else None


def trusted_torch_load(torch: Any, path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def unpack_snapshot(loaded: Any) -> tuple[Any, Mapping[str, Any]]:
    if isinstance(loaded, tuple) and len(loaded) >= 2 and isinstance(loaded[1], Mapping):
        return loaded[0], loaded[1]
    if isinstance(loaded, Mapping) and "data" in loaded and "slices" in loaded:
        return loaded["data"], loaded["slices"]
    slices = getattr(loaded, "slices", None)
    data = getattr(loaded, "_data", None)
    if data is None:
        data = getattr(loaded, "data", None)
    if data is not None and isinstance(slices, Mapping):
        return data, slices
    raise TypeError(
        "Expected a collated PyTorch-Geometric snapshot containing (data, slices); "
        f"received {type(loaded)!r}"
    )


def slice_len(slices: Mapping[str, Any]) -> int:
    counts = [len(value) - 1 for value in slices.values()]
    if not counts or min(counts) <= 0:
        raise ValueError("Invalid or empty PT slice dictionary")
    common = max(set(counts), key=counts.count)
    return int(common)


def human_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "not available"
    days, rem = divmod(seconds, 86400.0)
    hours, rem = divmod(rem, 3600.0)
    minutes, secs = divmod(rem, 60.0)
    if days >= 1:
        return f"{days:.1f} days ({int(days)}d {int(hours):02d}h)"
    if hours >= 1:
        return f"{hours:.2f} hours"
    if minutes >= 1:
        return f"{minutes:.2f} minutes"
    return f"{secs:.2f} seconds"


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(f"DeepDTAGen repository not found: {repo_root}")
    required_code = [repo_root / "model.py", repo_root / "utils.py", repo_root / "FetterGrad.py"]
    missing_code = [str(path) for path in required_code if not path.is_file()]
    if missing_code:
        raise FileNotFoundError("Missing official DeepDTAGen source file(s): " + ", ".join(missing_code))

    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch_geometric.data import InMemoryDataset
    try:
        from torch_geometric.loader import DataLoader
    except ImportError:
        from torch_geometric.data import DataLoader

    from FetterGrad import FetterGrad
    from model import DeepDTAGen
    from utils import get_cindex

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this benchmark must run inside an A800 Slurm allocation")

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True

    pt_file = discover_pt(repo_root, args.pt_file)
    tokenizer_file = discover_tokenizer(repo_root, args.tokenizer_file)

    print(f"[input] repository={repo_root}", flush=True)
    print(f"[input] PT={pt_file}", flush=True)
    print(f"[input] PT_size_GiB={pt_file.stat().st_size / 2**30:.3f}", flush=True)
    print(f"[hardware] host={socket.gethostname()}", flush=True)
    print(f"[hardware] gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(f"[software] python={sys.version.split()[0]} torch={torch.__version__} cuda={torch.version.cuda}", flush=True)

    class SnapshotDataset(InMemoryDataset):
        def __init__(self, data: Any, slices: Mapping[str, Any]):
            super().__init__(root=None, transform=None, pre_transform=None, pre_filter=None)
            self.data = data
            self.slices = slices

    load_start = time.perf_counter()
    loaded = trusted_torch_load(torch, pt_file)
    collated_data, slices = unpack_snapshot(loaded)
    source_rows = slice_len(slices)
    dataset = SnapshotDataset(collated_data, slices)
    load_seconds = time.perf_counter() - load_start
    if len(dataset) != source_rows:
        raise RuntimeError(f"Dataset length mismatch: len(dataset)={len(dataset)}, slices={source_rows}")
    print(f"[input] source_rows={source_rows} load_seconds={load_seconds:.3f}", flush=True)

    if tokenizer_file is not None:
        with tokenizer_file.open("rb") as handle:
            tokenizer = pickle.load(handle)
        tokenizer_source = str(tokenizer_file)
    else:
        tokenizer = EmbeddedTokenizer()
        tokenizer_source = "embedded released BindingDB 107-token vocabulary"
    print(f"[input] tokenizer={tokenizer_source} vocab_size={len(tokenizer)}", flush=True)

    first = dataset[0]
    required_fields = ("x", "edge_index", "y", "target", "target_seq", "c_size")
    missing_fields = [name for name in required_fields if getattr(first, name, None) is None]
    if missing_fields:
        raise ValueError("PT row lacks required field(s): " + ", ".join(missing_fields))
    token_max = int(first.target_seq.max().item())
    if token_max >= len(tokenizer):
        raise ValueError(f"target_seq token id {token_max} exceeds vocabulary size {len(tokenizer)}")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": True,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(dataset, **loader_kwargs)
    if len(train_loader) == 0:
        raise ValueError("The source PT snapshot is smaller than one complete batch")

    eval_loader_kwargs = dict(loader_kwargs)
    eval_loader_kwargs["shuffle"] = False
    eval_loader = DataLoader(dataset, **eval_loader_kwargs)

    model = DeepDTAGen(tokenizer).to(device)
    optimizer = FetterGrad(optim.Adam(model.parameters(), lr=args.learning_rate))
    mse_f = nn.MSELoss()
    model_parameters = int(sum(p.numel() for p in model.parameters()))
    trainable_parameters = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(
        f"[model] parameters={model_parameters:,} trainable={trainable_parameters:,} "
        f"batch_size={args.batch_size}",
        flush=True,
    )

    def next_batch(iterator: Any, loader: Any) -> tuple[Any, Any]:
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            return next(iterator), iterator

    def train_one(data: Any) -> tuple[float, float, float, float]:
        optimizer.zero_grad()
        data = data.to(device, non_blocking=True)
        prediction, _new_drug, lm_loss, kl_loss = model(data)
        mse_loss = mse_f(prediction, data.y.view(-1, 1).float())
        if not args.skip_batch_cindex:
            get_cindex(
                prediction.detach().float().cpu().numpy(),
                data.y.view(-1, 1).detach().float().cpu().numpy(),
            )
        total_loss = kl_loss * 0.001 + mse_loss + lm_loss
        optimizer.ft_backward([total_loss, mse_loss])
        optimizer.step()
        return (
            float(total_loss.detach().item()),
            float(mse_loss.detach().item()),
            float(lm_loss.detach().item()),
            float(kl_loss.detach().item()),
        )

    model.train()
    train_iter = iter(train_loader)
    print(f"[warmup] starting {args.warmup_steps} exact training steps", flush=True)
    for step in range(args.warmup_steps):
        batch, train_iter = next_batch(train_iter, train_loader)
        train_one(batch)
        if (step + 1) % max(1, args.progress_every) == 0 or step + 1 == args.warmup_steps:
            print(f"[warmup] {step + 1}/{args.warmup_steps}", flush=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    measure_start = time.perf_counter()
    last_progress = measure_start
    final_losses = None
    print(f"[measure-train] starting {args.measure_steps} exact training steps", flush=True)
    for step in range(args.measure_steps):
        batch, train_iter = next_batch(train_iter, train_loader)
        final_losses = train_one(batch)
        completed = step + 1
        if completed % max(1, args.progress_every) == 0 or completed == args.measure_steps:
            torch.cuda.synchronize()
            now = time.perf_counter()
            elapsed = now - measure_start
            interval = now - last_progress
            print(
                f"[measure-train] {completed}/{args.measure_steps} "
                f"elapsed={elapsed:.2f}s mean_step={elapsed/completed:.4f}s "
                f"last_interval={interval:.2f}s",
                flush=True,
            )
            last_progress = now
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - measure_start
    peak_allocated_gib = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved_gib = torch.cuda.max_memory_reserved() / 2**30

    model.eval()
    eval_iter = iter(eval_loader)
    print(f"[warmup-eval] starting {args.eval_warmup_steps} forward-only steps", flush=True)
    with torch.inference_mode():
        for step in range(args.eval_warmup_steps):
            batch, eval_iter = next_batch(eval_iter, eval_loader)
            model(batch.to(device, non_blocking=True))
    torch.cuda.synchronize()
    eval_start = time.perf_counter()
    print(f"[measure-eval] starting {args.eval_measure_steps} forward-only steps", flush=True)
    with torch.inference_mode():
        for step in range(args.eval_measure_steps):
            batch, eval_iter = next_batch(eval_iter, eval_loader)
            model(batch.to(device, non_blocking=True))
            completed = step + 1
            if completed % max(1, args.progress_every) == 0 or completed == args.eval_measure_steps:
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - eval_start
                print(
                    f"[measure-eval] {completed}/{args.eval_measure_steps} "
                    f"elapsed={elapsed:.2f}s mean_step={elapsed/completed:.4f}s",
                    flush=True,
                )
    torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - eval_start

    train_step_seconds = train_seconds / args.measure_steps
    eval_step_seconds = eval_seconds / args.eval_measure_steps
    train_rows_per_second = args.batch_size / train_step_seconds
    eval_rows_per_second = args.batch_size / eval_step_seconds
    train_batches_per_epoch = math.ceil(args.project_train_rows / args.batch_size)
    valid_batches_per_pass = math.ceil(args.project_valid_rows / args.batch_size)
    projected_train_epoch_seconds = train_batches_per_epoch * train_step_seconds
    projected_valid_pass_seconds = valid_batches_per_pass * eval_step_seconds

    projections: dict[str, Any] = {}
    for epochs in args.project_epochs:
        validation_passes = epochs // args.eval_every if args.eval_every > 0 else 0
        train_only = projected_train_epoch_seconds * epochs
        validation_forward = projected_valid_pass_seconds * validation_passes
        total = train_only + validation_forward
        projections[str(epochs)] = {
            "epochs": epochs,
            "validation_passes": validation_passes,
            "training_seconds": train_only,
            "validation_forward_seconds": validation_forward,
            "total_seconds": total,
            "total_hours": total / 3600.0,
            "total_days": total / 86400.0,
            "human": human_duration(total),
            "minimum_24h_jobs": math.ceil(total / (23.0 * 3600.0)),
        }

    result = {
        "format": "deepdtagen-a800-speed-benchmark-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "complete",
        "scope": {
            "training_step": "Official full multitask forward + MSE/LM/KL + two-objective FetterGrad backward + Adam step",
            "validation_step": "Full-model forward only",
            "excluded": [
                "Conversion of the 2,033,232-row CSV into PyTorch-Geometric PT",
                "Checkpoint serialization",
                "Official cumulative per-batch validation CI/Rm2/AUPR implementation",
            ],
            "warning": (
                "The official validation loop repeatedly recomputes pairwise CI on all accumulated rows. "
                "That implementation is not scalable to 406,647 validation rows and must be replaced "
                "by a single end-of-pass or memory-efficient metric calculation for a full-data run."
            ),
        },
        "hardware": {
            "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
            "peak_allocated_gib": peak_allocated_gib,
            "peak_reserved_gib": peak_reserved_gib,
        },
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "input": {
            "repo_root": str(repo_root),
            "pt_file": str(pt_file),
            "pt_size_bytes": pt_file.stat().st_size,
            "source_rows": source_rows,
            "pt_load_seconds": load_seconds,
            "tokenizer_source": tokenizer_source,
            "vocab_size": len(tokenizer),
        },
        "configuration": {
            "batch_size": args.batch_size,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "eval_warmup_steps": args.eval_warmup_steps,
            "eval_measure_steps": args.eval_measure_steps,
            "num_workers": args.num_workers,
            "learning_rate": args.learning_rate,
            "batch_cindex_included": not args.skip_batch_cindex,
            "project_train_rows": args.project_train_rows,
            "project_valid_rows": args.project_valid_rows,
            "eval_every": args.eval_every,
            "model_parameters": model_parameters,
            "trainable_parameters": trainable_parameters,
        },
        "measurements": {
            "train_total_seconds": train_seconds,
            "train_step_seconds": train_step_seconds,
            "train_rows_per_second": train_rows_per_second,
            "eval_total_seconds": eval_seconds,
            "eval_step_seconds": eval_step_seconds,
            "eval_rows_per_second": eval_rows_per_second,
            "final_losses": {
                "total": final_losses[0] if final_losses else None,
                "mse": final_losses[1] if final_losses else None,
                "lm": final_losses[2] if final_losses else None,
                "kl": final_losses[3] if final_losses else None,
            },
        },
        "full_data_projection": {
            "train_batches_per_epoch": train_batches_per_epoch,
            "valid_batches_per_pass": valid_batches_per_pass,
            "training_seconds_per_epoch": projected_train_epoch_seconds,
            "training_hours_per_epoch": projected_train_epoch_seconds / 3600.0,
            "validation_forward_seconds_per_pass": projected_valid_pass_seconds,
            "projections": projections,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temp_json = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temp_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp_json, args.output_json)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    temp_csv = args.output_csv.with_suffix(args.output_csv.suffix + ".tmp")
    with temp_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epochs", "validation_passes", "training_hours",
                "validation_forward_hours", "total_hours", "total_days",
                "minimum_24h_jobs",
            ],
        )
        writer.writeheader()
        for epochs in args.project_epochs:
            row = projections[str(epochs)]
            writer.writerow({
                "epochs": epochs,
                "validation_passes": row["validation_passes"],
                "training_hours": row["training_seconds"] / 3600.0,
                "validation_forward_hours": row["validation_forward_seconds"] / 3600.0,
                "total_hours": row["total_hours"],
                "total_days": row["total_days"],
                "minimum_24h_jobs": row["minimum_24h_jobs"],
            })
    os.replace(temp_csv, args.output_csv)

    print("===== SPEED TEST COMPLETE =====", flush=True)
    print(f"train_step_seconds={train_step_seconds:.6f}", flush=True)
    print(f"train_rows_per_second={train_rows_per_second:.3f}", flush=True)
    print(f"projected_training_hours_per_epoch={projected_train_epoch_seconds/3600.0:.3f}", flush=True)
    print(f"projected_validation_forward_hours_per_pass={projected_valid_pass_seconds/3600.0:.3f}", flush=True)
    for epochs in args.project_epochs:
        row = projections[str(epochs)]
        print(
            f"projected_{epochs}_epochs={row['total_days']:.3f}_days "
            f"minimum_23h_chunks={row['minimum_24h_jobs']}",
            flush=True,
        )
    print(f"peak_gpu_allocated_gib={peak_allocated_gib:.3f}", flush=True)
    print(f"result_json={args.output_json}", flush=True)
    print(f"result_csv={args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
