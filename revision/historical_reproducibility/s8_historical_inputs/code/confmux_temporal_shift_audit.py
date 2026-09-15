#!/usr/bin/env python3
"""Post-hoc temporal distribution-shift audit for ConfMux-DTA.

This script reuses completed train/random-validation/temporal predictions.  It
does not fit or alter the DTA model.  It quantifies four prespecified domains:

1. compound overlap, Bemis--Murcko scaffold novelty, and physicochemical shift;
2. exact target overlap, target length, and nearest training-target 3-mer
   TF-IDF cosine similarity (a neighborhood score, not sequence identity);
3. affinity-type composition and endpoint-specific error;
4. endpoint-conditioned pX distribution shift.

It also performs one-factor composition standardization and a descriptive
multivariable error-association analysis.  Neither is a causal decomposition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

try:
    from rdkit import Chem, rdBase
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold

    rdBase.DisableLog("rdApp.*")
    HAVE_RDKIT = True
except Exception:
    Chem = None
    Crippen = Descriptors = Lipinski = rdMolDescriptors = MurckoScaffold = None
    HAVE_RDKIT = False


ENDPOINT_ORDER = ["pIC50", "pKi", "pKd"]
STAGE_CACHE_VERSION = "temporal_shift_audit_v2"
DESCRIPTOR_COLUMNS = [
    "mw",
    "clogp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "ring_count",
    "heavy_atoms",
    "formal_charge",
    "smiles_length",
]

ALIASES: Dict[str, List[str]] = {
    "compound": [
        "smiles_norm",
        "canonical_smiles",
        "canonical_smiles_norm",
        "smiles",
        "SMILES",
    ],
    "target": [
        "protein_clean",
        "protein_sequence",
        "target_sequence",
        "protein",
        "sequence",
    ],
    "endpoint": [
        "affinity_type",
        "endpoint",
        "endpoint_type",
        "measure_type",
        "metric",
    ],
    "y_true": [
        "y_true",
        "affinity_value",
        "observed_px",
        "observed_pX",
        "px_true",
        "label",
    ],
    "y_pred": [
        "y_pred",
        "y_pred_raw",
        "pred_pX",
        "pred_final",
        "prediction",
        "predicted_px",
        "predicted_pX",
        "pred",
    ],
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def atomic_pickle_dump(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def pickle_load(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def cache_run_key(paths: Sequence[Path], parameters: Mapping[str, object]) -> str:
    files = []
    for path in paths:
        stat = path.stat()
        files.append(
            {
                "path": str(path.resolve()),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    payload = {
        "cache_version": STAGE_CACHE_VERSION,
        "files": files,
        "parameters": dict(parameters),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def endpoint_from_text(value: object) -> Optional[str]:
    s = str(value).strip().upper().replace("_", "").replace("-", "")
    if s.startswith("P"):
        s = s[1:]
    if "IC50" in s:
        return "pIC50"
    if s == "KI" or s.endswith("KI"):
        return "pKi"
    if s == "KD" or s.endswith("KD"):
        return "pKd"
    return None


def infer_endpoint_from_path(path: Path) -> Optional[str]:
    return endpoint_from_text(path.name)


def resolve_columns(path: Path, require_prediction: bool) -> Dict[str, Optional[str]]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    lower = {str(c).lower(): str(c) for c in header}
    found: Dict[str, Optional[str]] = {}
    for canonical, candidates in ALIASES.items():
        value = None
        for candidate in candidates:
            if candidate in header:
                value = candidate
                break
            if candidate.lower() in lower:
                value = lower[candidate.lower()]
                break
        found[canonical] = value

    required = ["compound", "target", "y_true"]
    if require_prediction:
        required.append("y_pred")
    missing = [name for name in required if found[name] is None]
    if missing:
        raise ValueError(
            f"{path}: missing required logical columns {missing}; header={header}"
        )
    if found["endpoint"] is None and infer_endpoint_from_path(path) is None:
        raise ValueError(
            f"{path}: no endpoint column and endpoint cannot be inferred from filename"
        )
    return found


def iter_normalized_chunks(
    paths: Sequence[Path],
    require_prediction: bool,
    chunksize: int,
) -> Iterable[pd.DataFrame]:
    for path in paths:
        mapping = resolve_columns(path, require_prediction=require_prediction)
        actual_columns = [v for v in mapping.values() if v is not None]
        fixed_endpoint = infer_endpoint_from_path(path)
        for raw in pd.read_csv(
            path,
            usecols=actual_columns,
            chunksize=chunksize,
            low_memory=False,
            on_bad_lines="skip",
        ):
            rename = {v: k for k, v in mapping.items() if v is not None}
            df = raw.rename(columns=rename)
            if "endpoint" not in df:
                df["endpoint"] = fixed_endpoint
            df["endpoint"] = df["endpoint"].map(endpoint_from_text)
            df["compound"] = df["compound"].fillna("").astype(str).str.strip()
            df["target"] = (
                df["target"]
                .fillna("")
                .astype(str)
                .str.upper()
                .str.replace(r"[^A-Z]", "", regex=True)
            )
            df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
            if require_prediction:
                df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
                finite = np.isfinite(df["y_true"]) & np.isfinite(df["y_pred"])
            else:
                finite = np.isfinite(df["y_true"])
            keep = (
                finite
                & df["endpoint"].isin(ENDPOINT_ORDER)
                & df["compound"].ne("")
                & df["target"].ne("")
            )
            cols = ["compound", "target", "endpoint", "y_true"]
            if require_prediction:
                cols.append("y_pred")
            out = df.loc[keep, cols].copy()
            if len(out):
                yield out


def pair_hashes(df: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(
        df[["compound", "target"]], index=False, categorize=True
    ).to_numpy(dtype=np.uint64, copy=False)


def scan_reference(
    paths: Sequence[Path], chunksize: int
) -> Tuple[set, set, set, Counter, Dict[str, np.ndarray], int]:
    compounds: set = set()
    targets: set = set()
    pairs: set = set()
    endpoint_counts: Counter = Counter()
    y_parts: Dict[str, List[np.ndarray]] = defaultdict(list)
    n_total = 0

    for chunk_id, df in enumerate(
        iter_normalized_chunks(paths, require_prediction=False, chunksize=chunksize), 1
    ):
        n_total += len(df)
        compounds.update(df["compound"].unique().tolist())
        targets.update(df["target"].unique().tolist())
        pairs.update(pair_hashes(df).tolist())
        endpoint_counts.update(df["endpoint"].value_counts().to_dict())
        for endpoint, values in df.groupby("endpoint", observed=True)["y_true"]:
            y_parts[str(endpoint)].append(values.to_numpy(dtype=np.float64))
        if chunk_id % 5 == 0:
            log(
                "Reference scan: "
                f"rows={n_total:,}, compounds={len(compounds):,}, "
                f"targets={len(targets):,}, pairs={len(pairs):,}"
            )

    y_by_endpoint = {
        endpoint: np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
        for endpoint, parts in y_parts.items()
    }
    return compounds, targets, pairs, endpoint_counts, y_by_endpoint, n_total


def load_evaluation(paths: Sequence[Path], chunksize: int) -> pd.DataFrame:
    parts = list(
        iter_normalized_chunks(paths, require_prediction=True, chunksize=chunksize)
    )
    if not parts:
        raise ValueError(f"No usable prediction records in {[str(p) for p in paths]}")
    df = pd.concat(parts, ignore_index=True)
    df["error"] = df["y_pred"] - df["y_true"]
    df["abs_error"] = df["error"].abs()
    df["squared_error"] = df["error"] ** 2
    return df


def stable_sample(values: Sequence[str], cap: int, seed: int) -> List[str]:
    values = list(values)
    if len(values) <= cap:
        return values
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(values), size=cap, replace=False)
    return [values[int(i)] for i in indices]


def molecule_information(task: Tuple[str, bool]) -> Tuple:
    smiles, need_descriptors = task
    if not HAVE_RDKIT:
        return (smiles, "", False, False) + (np.nan,) * len(DESCRIPTOR_COLUMNS)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("invalid molecule")
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        acyclic = scaffold == ""
        if need_descriptors:
            descriptors = (
                float(Descriptors.MolWt(mol)),
                float(Crippen.MolLogP(mol)),
                float(rdMolDescriptors.CalcTPSA(mol)),
                float(Lipinski.NumHDonors(mol)),
                float(Lipinski.NumHAcceptors(mol)),
                float(Lipinski.NumRotatableBonds(mol)),
                float(rdMolDescriptors.CalcNumRings(mol)),
                float(mol.GetNumHeavyAtoms()),
                float(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
                float(len(smiles)),
            )
        else:
            descriptors = (np.nan,) * len(DESCRIPTOR_COLUMNS)
        return (smiles, scaffold, True, acyclic) + descriptors
    except Exception:
        return (smiles, "", False, False) + (np.nan,) * len(DESCRIPTOR_COLUMNS)


def compute_molecule_maps(
    compounds: Sequence[str],
    descriptor_set: set,
    cpus: int,
    label: str,
    block_cache_dir: Optional[Path] = None,
    block_size: int = 50000,
) -> Tuple[Dict[str, str], Dict[str, bool], Dict[str, bool], pd.DataFrame]:
    compounds = list(compounds)
    scaffold_map: Dict[str, str] = {}
    valid_map: Dict[str, bool] = {}
    acyclic_map: Dict[str, bool] = {}
    descriptor_rows: List[Tuple] = []
    done = 0
    if block_cache_dir is not None:
        block_cache_dir.mkdir(parents=True, exist_ok=True)

    pool = None
    if cpus > 1 and HAVE_RDKIT:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=cpus)
    try:
        for block_start in range(0, len(compounds), block_size):
            block = compounds[block_start : block_start + block_size]
            block_path = None
            cached = None
            if block_cache_dir is not None:
                block_path = block_cache_dir / f"block_{block_start:09d}_{len(block):06d}.pkl"
                if block_path.is_file() and block_path.stat().st_size > 0:
                    try:
                        cached = pickle_load(block_path)
                        log(f"{label}: resumed {block_path.name}")
                    except Exception:
                        cached = None

            if cached is None:
                tasks = ((s, s in descriptor_set) for s in block)
                if pool is not None:
                    iterator = pool.imap_unordered(
                        molecule_information, tasks, chunksize=256
                    )
                else:
                    iterator = map(molecule_information, tasks)
                block_scaffolds: Dict[str, str] = {}
                block_valid: Dict[str, bool] = {}
                block_acyclic: Dict[str, bool] = {}
                block_descriptors: List[Tuple] = []
                for result in iterator:
                    smiles, scaffold, valid, acyclic, *descriptors = result
                    block_scaffolds[smiles] = scaffold
                    block_valid[smiles] = bool(valid)
                    block_acyclic[smiles] = bool(acyclic)
                    if smiles in descriptor_set:
                        block_descriptors.append((smiles,) + tuple(descriptors))
                cached = (
                    block_scaffolds,
                    block_valid,
                    block_acyclic,
                    block_descriptors,
                )
                if block_path is not None:
                    atomic_pickle_dump(cached, block_path)

            block_scaffolds, block_valid, block_acyclic, block_descriptors = cached
            scaffold_map.update(block_scaffolds)
            valid_map.update(block_valid)
            acyclic_map.update(block_acyclic)
            descriptor_rows.extend(block_descriptors)
            done += len(block)
            log(f"{label}: RDKit compounds={done:,}/{len(compounds):,}")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    descriptors_df = pd.DataFrame(
        descriptor_rows, columns=["compound"] + DESCRIPTOR_COLUMNS
    ).set_index("compound")
    return scaffold_map, valid_map, acyclic_map, descriptors_df


def target_similarity_maps(
    train_targets: Sequence[str],
    query_targets: Sequence[str],
    cpus: int,
    query_chunk: int,
    cache_path: Optional[Path] = None,
) -> Dict[str, float]:
    train_targets = sorted(set(train_targets))
    query_targets = sorted(set(query_targets))
    train_set = set(train_targets)
    result: Dict[str, float] = {}
    if cache_path is not None and cache_path.is_file() and cache_path.stat().st_size > 0:
        try:
            result = dict(pickle_load(cache_path))
            log(f"Target similarity: resumed {len(result):,} cached target scores")
        except Exception:
            result = {}
    result.update({target: 1.0 for target in query_targets if target in train_set})
    unseen = [
        target
        for target in query_targets
        if target not in train_set and target not in result
    ]
    if not unseen:
        return result

    log(
        f"Target 3-mer TF-IDF: training targets={len(train_targets):,}, "
        f"unseen query targets={len(unseen):,}"
    )
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 3),
        lowercase=False,
        min_df=1,
        max_features=8192,
        norm="l2",
        dtype=np.float32,
    )
    x_train = vectorizer.fit_transform(train_targets)
    nn = NearestNeighbors(
        n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=max(1, cpus)
    )
    nn.fit(x_train)
    for start in range(0, len(unseen), query_chunk):
        batch = unseen[start : start + query_chunk]
        x_query = vectorizer.transform(batch)
        distances, _ = nn.kneighbors(x_query, return_distance=True)
        similarities = np.clip(1.0 - distances[:, 0], 0.0, 1.0)
        result.update(zip(batch, similarities.astype(float).tolist()))
        if cache_path is not None:
            atomic_pickle_dump(result, cache_path)
        log(f"Target similarity: {min(start + len(batch), len(unseen)):,}/{len(unseen):,}")
    return result


def training_px_spec(y_by_endpoint: Mapping[str, np.ndarray]) -> Dict[str, Dict]:
    specs: Dict[str, Dict] = {}
    for endpoint in ENDPOINT_ORDER:
        values = np.asarray(y_by_endpoint.get(endpoint, []), dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        q05, q25, q50, q75, q95 = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
        iqr = max(float(q75 - q25), 1e-8)
        specs[endpoint] = {
            "q05": float(q05),
            "q25": float(q25),
            "median": float(q50),
            "q75": float(q75),
            "q95": float(q95),
            "iqr": iqr,
        }
    return specs


def px_bin(value: float, spec: Mapping[str, float]) -> str:
    if value < spec["q05"]:
        return "below_train_Q05"
    if value < spec["q25"]:
        return "train_Q05_Q25"
    if value < spec["median"]:
        return "train_Q25_Q50"
    if value < spec["q75"]:
        return "train_Q50_Q75"
    if value <= spec["q95"]:
        return "train_Q75_Q95"
    return "above_train_Q95"


def enrich_evaluation(
    df: pd.DataFrame,
    train_compounds: set,
    train_scaffolds: set,
    train_targets: set,
    train_pairs: set,
    scaffold_map: Mapping[str, str],
    valid_map: Mapping[str, bool],
    acyclic_map: Mapping[str, bool],
    descriptors: pd.DataFrame,
    similarity_map: Mapping[str, float],
    px_specs: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    out = df.copy()
    out["compound_seen"] = out["compound"].isin(train_compounds)
    out["scaffold"] = out["compound"].map(scaffold_map).fillna("")
    out["compound_valid"] = out["compound"].map(valid_map).fillna(False)
    out["compound_acyclic"] = out["compound"].map(acyclic_map).fillna(False)
    out["scaffold_seen"] = out["scaffold"].isin(train_scaffolds) & out["scaffold"].ne("")
    conditions = [
        out["compound_seen"],
        ~out["compound_valid"],
        out["compound_acyclic"],
        out["scaffold_seen"],
    ]
    choices = [
        "exact_compound_seen",
        "invalid_structure",
        "unseen_acyclic",
        "unseen_known_scaffold",
    ]
    out["compound_novelty"] = np.select(
        conditions, choices, default="unseen_novel_scaffold"
    )

    out["target_seen"] = out["target"].isin(train_targets)
    out["target_similarity_3mer"] = out["target"].map(similarity_map).astype(float)
    out["target_length"] = out["target"].str.len().astype(float)
    out["target_novelty"] = np.select(
        [
            out["target_seen"],
            out["target_similarity_3mer"] >= 0.90,
            out["target_similarity_3mer"] >= 0.70,
        ],
        [
            "exact_target_seen",
            "unseen_high_3mer_similarity",
            "unseen_moderate_3mer_similarity",
        ],
        default="unseen_low_3mer_similarity",
    )
    out["pair_seen"] = np.fromiter(
        (int(h) in train_pairs for h in pair_hashes(out)),
        dtype=bool,
        count=len(out),
    )

    for column in DESCRIPTOR_COLUMNS:
        out[column] = out["compound"].map(descriptors[column])

    out["px_train_bin"] = [
        px_bin(float(y), px_specs[str(endpoint)])
        for y, endpoint in zip(out["y_true"], out["endpoint"])
    ]
    out["px_distance_iqr"] = [
        abs(float(y) - px_specs[str(endpoint)]["median"])
        / px_specs[str(endpoint)]["iqr"]
        for y, endpoint in zip(out["y_true"], out["endpoint"])
    ]
    return out


def metric_pack(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[finite], y_pred[finite]
    if not len(y_true):
        return {
            "n": 0,
            "mae": np.nan,
            "mse": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "mean_signed_error": np.nan,
        }
    error = y_pred - y_true
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(error))),
        "mse": float(np.mean(error ** 2)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "mean_signed_error": float(np.mean(error)),
    }


def make_performance_table(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict] = []
    for dataset, df in datasets.items():
        rows.append({"dataset": dataset, "subset": "overall", **metric_pack(df.y_true, df.y_pred)})
        for endpoint in ENDPOINT_ORDER:
            sub = df[df["endpoint"] == endpoint]
            rows.append(
                {
                    "dataset": dataset,
                    "subset": endpoint,
                    **metric_pack(sub.y_true, sub.y_pred),
                }
            )
    return pd.DataFrame(rows)


def make_composition_table(
    datasets: Mapping[str, pd.DataFrame], factors: Sequence[str]
) -> pd.DataFrame:
    rows: List[Dict] = []
    for dataset, df in datasets.items():
        for factor in factors:
            counts = df[factor].astype(str).value_counts(dropna=False)
            total = int(counts.sum())
            for level, n in counts.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "factor": factor,
                        "level": str(level),
                        "n": int(n),
                        "proportion": float(n / total) if total else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def make_strata_table(
    datasets: Mapping[str, pd.DataFrame], factors: Sequence[str]
) -> pd.DataFrame:
    rows: List[Dict] = []
    for dataset, df in datasets.items():
        for factor in factors:
            for level, sub in df.groupby(factor, dropna=False, observed=True):
                rows.append(
                    {
                        "dataset": dataset,
                        "factor": factor,
                        "level": str(level),
                        **metric_pack(sub.y_true, sub.y_pred),
                    }
                )
    return pd.DataFrame(rows)


def categorical_shift_table(
    composition: pd.DataFrame,
    comparisons: Sequence[Tuple[str, str]],
) -> pd.DataFrame:
    rows: List[Dict] = []
    for reference, comparison in comparisons:
        factors = sorted(composition["factor"].unique())
        for factor in factors:
            a = composition[
                (composition.dataset == reference) & (composition.factor == factor)
            ].set_index("level")["proportion"]
            b = composition[
                (composition.dataset == comparison) & (composition.factor == factor)
            ].set_index("level")["proportion"]
            levels = sorted(set(a.index) | set(b.index))
            pa = np.array([a.get(level, 0.0) for level in levels], dtype=float)
            pb = np.array([b.get(level, 0.0) for level in levels], dtype=float)
            js = float(jensenshannon(pa, pb, base=2.0) ** 2)
            tv = float(0.5 * np.abs(pa - pb).sum())
            rows.append(
                {
                    "reference": reference,
                    "comparison": comparison,
                    "factor": factor,
                    "jensen_shannon_divergence_bits": js,
                    "total_variation_distance": tv,
                }
            )
    return pd.DataFrame(rows)


def numeric_shift(
    reference_values: Sequence[float],
    comparison_values: Sequence[float],
    variable: str,
    subset: str,
    reference_name: str,
    comparison_name: str,
    sampling_basis: str,
) -> Dict:
    a = np.asarray(reference_values, dtype=float)
    b = np.asarray(comparison_values, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if not len(a) or not len(b):
        return {
            "reference": reference_name,
            "comparison": comparison_name,
            "variable": variable,
            "subset": subset,
            "sampling_basis": sampling_basis,
            "n_reference": len(a),
            "n_comparison": len(b),
        }
    pooled_sd = math.sqrt((float(np.var(a)) + float(np.var(b))) / 2.0)
    smd = (float(np.mean(b)) - float(np.mean(a))) / pooled_sd if pooled_sd > 0 else np.nan
    ks = ks_2samp(a, b, alternative="two-sided", mode="auto")
    return {
        "reference": reference_name,
        "comparison": comparison_name,
        "variable": variable,
        "subset": subset,
        "sampling_basis": sampling_basis,
        "n_reference": int(len(a)),
        "n_comparison": int(len(b)),
        "reference_mean": float(np.mean(a)),
        "comparison_mean": float(np.mean(b)),
        "reference_sd": float(np.std(a)),
        "comparison_sd": float(np.std(b)),
        "reference_median": float(np.median(a)),
        "comparison_median": float(np.median(b)),
        "standardized_mean_difference": float(smd),
        "wasserstein_distance": float(wasserstein_distance(a, b)),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
    }


def make_numeric_shift_table(
    validation: pd.DataFrame,
    temporal: pd.DataFrame,
    training_descriptors: pd.DataFrame,
    validation_descriptors: pd.DataFrame,
    temporal_descriptors: pd.DataFrame,
    y_by_endpoint: Mapping[str, np.ndarray],
    train_targets: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict] = []
    rows.append(
        numeric_shift(
            validation.y_true,
            temporal.y_true,
            "pX",
            "overall",
            "random_validation",
            "temporal",
            "records",
        )
    )
    for endpoint in ENDPOINT_ORDER:
        va = validation[validation.endpoint == endpoint]
        te = temporal[temporal.endpoint == endpoint]
        rows.append(
            numeric_shift(
                va.y_true,
                te.y_true,
                "pX",
                endpoint,
                "random_validation",
                "temporal",
                "records",
            )
        )
    train_all_y = np.concatenate(
        [np.asarray(y_by_endpoint.get(endpoint, []), dtype=float) for endpoint in ENDPOINT_ORDER]
    )
    rows.append(
        numeric_shift(
            train_all_y,
            temporal.y_true,
            "pX",
            "overall",
            "training",
            "temporal",
            "records",
        )
    )
    for endpoint in ENDPOINT_ORDER:
        te = temporal[temporal.endpoint == endpoint]
        rows.append(
            numeric_shift(
                y_by_endpoint.get(endpoint, np.empty(0)),
                te.y_true,
                "pX",
                endpoint,
                "training",
                "temporal",
                "records",
            )
        )
    for variable in DESCRIPTOR_COLUMNS:
        rows.append(
            numeric_shift(
                validation_descriptors[variable],
                temporal_descriptors[variable],
                variable,
                "overall",
                "random_validation",
                "temporal",
                "unique_compounds",
            )
        )
        rows.append(
            numeric_shift(
                training_descriptors[variable],
                temporal_descriptors[variable],
                variable,
                "overall",
                "training",
                "temporal",
                "unique_compounds; training uniformly capped",
            )
        )
    va_targets = validation[["target", "target_length", "target_similarity_3mer"]].drop_duplicates("target")
    te_targets = temporal[["target", "target_length", "target_similarity_3mer"]].drop_duplicates("target")
    for variable in ["target_length", "target_similarity_3mer"]:
        rows.append(
            numeric_shift(
                va_targets[variable],
                te_targets[variable],
                variable,
                "overall",
                "random_validation",
                "temporal",
                "unique_targets",
            )
        )
    rows.append(
        numeric_shift(
            [len(target) for target in train_targets],
            te_targets["target_length"],
            "target_length",
            "overall",
            "training",
            "temporal",
            "unique_targets",
        )
    )
    return pd.DataFrame(rows)


def standardization_table(
    validation: pd.DataFrame,
    temporal: pd.DataFrame,
    factors: Sequence[str],
) -> pd.DataFrame:
    val_metrics = metric_pack(validation.y_true, validation.y_pred)
    temp_metrics = metric_pack(temporal.y_true, temporal.y_pred)
    rows: List[Dict] = []
    for factor in factors:
        val_weights = validation[factor].astype(str).value_counts(normalize=True)
        temp_group_mse = temporal.assign(_level=temporal[factor].astype(str)).groupby(
            "_level", observed=True
        )["squared_error"].mean()
        common = sorted(set(val_weights.index) & set(temp_group_mse.index))
        covered_weight = float(val_weights.reindex(common).sum())
        if not common or covered_weight <= 0:
            continue
        weights = val_weights.reindex(common).fillna(0.0) / covered_weight
        standardized_mse = float((weights * temp_group_mse.reindex(common)).sum())
        standardized_rmse = math.sqrt(max(standardized_mse, 0.0))
        mse_gap = float(temp_metrics["mse"] - val_metrics["mse"])
        removed = float(temp_metrics["mse"] - standardized_mse)
        rows.append(
            {
                "factor": factor,
                "random_validation_rmse": val_metrics["rmse"],
                "observed_temporal_rmse": temp_metrics["rmse"],
                "temporal_rmse_standardized_to_validation_mix": standardized_rmse,
                "rmse_reduction_after_standardization": float(temp_metrics["rmse"] - standardized_rmse),
                "observed_mse_gap": mse_gap,
                "mse_gap_removed_by_one_factor_standardization": removed,
                "percent_of_observed_mse_gap": float(100.0 * removed / mse_gap) if abs(mse_gap) > 1e-12 else np.nan,
                "validation_weight_covered": covered_weight,
                "interpretation": "descriptive one-factor standardization; estimates are not additive or causal",
            }
        )
    return pd.DataFrame(rows).sort_values(
        "mse_gap_removed_by_one_factor_standardization", ascending=False
    )


def error_association_analysis(
    validation: pd.DataFrame,
    temporal: pd.DataFrame,
    seed: int,
    sample_cap: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    cap_each = max(1000, sample_cap // 2)

    def sample(df: pd.DataFrame, label: str) -> pd.DataFrame:
        if len(df) > cap_each:
            out = df.sample(n=cap_each, random_state=seed)
        else:
            out = df.copy()
        out = out.copy()
        out["audit_dataset"] = label
        return out

    data = pd.concat(
        [sample(validation, "random_validation"), sample(temporal, "temporal")],
        ignore_index=True,
    )
    continuous = DESCRIPTOR_COLUMNS + [
        "target_similarity_3mer",
        "target_length",
        "px_distance_iqr",
        "y_true",
        "compound_seen",
        "scaffold_seen",
        "target_seen",
        "pair_seen",
    ]
    categorical = ["endpoint", "compound_novelty", "target_novelty", "px_train_bin"]
    base = data[continuous].copy()
    for column in continuous:
        base[column] = pd.to_numeric(base[column], errors="coerce")
        finite_column = base.loc[np.isfinite(base[column]), column]
        median = finite_column.median() if len(finite_column) else 0.0
        base[column] = base[column].fillna(0.0 if not np.isfinite(median) else median)
    dummies = pd.get_dummies(data[categorical], prefix=categorical, dtype=float)
    x = pd.concat([base.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    y = data["squared_error"].to_numpy(dtype=float)
    indices = np.arange(len(data))
    idx_train, idx_test = train_test_split(indices, test_size=0.25, random_state=seed)
    model = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=220,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=1.0,
        random_state=seed,
    )
    model.fit(x.iloc[idx_train].to_numpy(dtype=np.float32), y[idx_train])

    temporal_test = idx_test[data.iloc[idx_test]["audit_dataset"].to_numpy() == "temporal"]
    if len(temporal_test) < 100:
        temporal_test = idx_test
    x_test = x.iloc[temporal_test].to_numpy(dtype=np.float32)
    y_test = y[temporal_test]
    baseline_pred = model.predict(x_test)
    baseline_mae = float(mean_absolute_error(y_test, baseline_pred))
    baseline_r2 = float(r2_score(y_test, baseline_pred))

    column_names = list(x.columns)
    groups = {
        "compound_characteristics": [
            i
            for i, name in enumerate(column_names)
            if name in DESCRIPTOR_COLUMNS
            or name in {"compound_seen", "scaffold_seen"}
            or name.startswith("compound_novelty_")
        ],
        "target_characteristics": [
            i
            for i, name in enumerate(column_names)
            if name in {"target_similarity_3mer", "target_length", "target_seen"}
            or name.startswith("target_novelty_")
        ],
        "affinity_type": [
            i for i, name in enumerate(column_names) if name.startswith("endpoint_")
        ],
        "pX_distribution": [
            i
            for i, name in enumerate(column_names)
            if name in {"px_distance_iqr", "y_true"} or name.startswith("px_train_bin_")
        ],
        "compound_target_pair_overlap": [
            i for i, name in enumerate(column_names) if name == "pair_seen"
        ],
    }
    rows: List[Dict] = []
    repeats = 5
    for group, columns in groups.items():
        increases: List[float] = []
        if not columns:
            continue
        for _ in range(repeats):
            permuted = x_test.copy()
            order = rng.permutation(len(permuted))
            permuted[:, columns] = permuted[order][:, columns]
            permuted_pred = model.predict(permuted)
            increases.append(float(mean_absolute_error(y_test, permuted_pred) - baseline_mae))
        rows.append(
            {
                "factor_group": group,
                "permutation_increase_in_MAE_of_squared_error_mean": float(np.mean(increases)),
                "permutation_increase_in_MAE_of_squared_error_sd": float(np.std(increases)),
                "n_permuted_columns": len(columns),
                "n_temporal_evaluation_records": int(len(y_test)),
            }
        )
    table = pd.DataFrame(rows).sort_values(
        "permutation_increase_in_MAE_of_squared_error_mean", ascending=False
    )
    positive = table["permutation_increase_in_MAE_of_squared_error_mean"].clip(lower=0.0)
    table["normalized_positive_importance"] = (
        positive / positive.sum() if positive.sum() > 0 else 0.0
    )
    diagnostics = {
        "sample_n": int(len(data)),
        "temporal_test_n": int(len(y_test)),
        "baseline_MAE_of_squared_error": baseline_mae,
        "baseline_R2_for_squared_error": baseline_r2,
        "method": "post-hoc HistGradientBoosting grouped permutation importance",
        "interpretation": "descriptive error association, not causal attribution",
    }
    return table, diagnostics


def plot_audit(
    composition: pd.DataFrame,
    strata: pd.DataFrame,
    numeric_shifts: pd.DataFrame,
    standardization: pd.DataFrame,
    importance: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    colors = {"random_validation": "#4C78A8", "temporal": "#E45756"}

    def composition_panel(ax, factor: str, title: str) -> None:
        sub = composition[composition.factor == factor]
        levels = list(dict.fromkeys(sub.level.tolist()))
        x = np.arange(len(levels))
        width = 0.36
        for offset, dataset in [(-width / 2, "random_validation"), (width / 2, "temporal")]:
            values = (
                sub[sub.dataset == dataset]
                .set_index("level")["proportion"]
                .reindex(levels)
                .fillna(0.0)
                .to_numpy()
            )
            ax.bar(x + offset, values, width, label=dataset.replace("_", " "), color=colors[dataset])
        ax.set_xticks(x)
        ax.set_xticklabels(levels, rotation=28, ha="right", fontsize=8)
        ax.set_ylabel("Proportion")
        ax.set_title(title)

    composition_panel(axes[0, 0], "endpoint", "A  Affinity-type composition")
    composition_panel(axes[0, 1], "compound_novelty", "B  Compound novelty")
    composition_panel(axes[0, 2], "target_novelty", "C  Target novelty")

    px = numeric_shifts[(numeric_shifts.variable == "pX") & (numeric_shifts.subset != "overall")]
    x = np.arange(len(px))
    axes[1, 0].bar(x - 0.18, px.reference_mean, 0.36, color=colors["random_validation"], label="random validation")
    axes[1, 0].bar(x + 0.18, px.comparison_mean, 0.36, color=colors["temporal"], label="temporal")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(px.subset)
    axes[1, 0].set_ylabel("Mean observed pX")
    axes[1, 0].set_title("D  Endpoint-conditioned label shift")

    key = strata[
        (strata.dataset == "temporal")
        & strata.factor.isin(["compound_novelty", "target_novelty"])
        & (strata.n >= 100)
    ].copy()
    if key.empty:
        key = strata[
            (strata.dataset == "temporal")
            & strata.factor.isin(["compound_novelty", "target_novelty"])
        ].copy()
    key["display"] = key.factor.str.replace("_novelty", "", regex=False) + ": " + key.level
    key = key.sort_values("rmse", ascending=True)
    axes[1, 1].barh(key.display, key.rmse, color="#72B7B2")
    axes[1, 1].set_xlabel("Temporal RMSE")
    axes[1, 1].tick_params(axis="y", labelsize=7)
    axes[1, 1].set_title("E  Error across novelty strata")

    std = standardization.copy()
    std = std.sort_values("mse_gap_removed_by_one_factor_standardization")
    axes[1, 2].barh(
        std.factor.str.replace("_", " ", regex=False),
        std.percent_of_observed_mse_gap,
        color="#F58518",
        alpha=0.82,
        label="one-factor standardization",
    )
    axes[1, 2].axvline(0, color="black", linewidth=0.8)
    axes[1, 2].set_xlabel("Observed MSE gap removed (%)")
    axes[1, 2].set_title("F  Descriptive composition analysis")

    axes[0, 0].legend(loc="best", fontsize=8, frameon=False)
    fig.suptitle("ConfMux-DTA temporal generalization audit", fontsize=14, y=0.995)
    fig.text(
        0.5,
        0.007,
        "3-mer similarity, composition standardization, and error importance are descriptive; no causal attribution is implied.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def file_fingerprint(path: Path) -> Dict:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime": float(stat.st_mtime),
        "sha256_first_1MiB": digest.hexdigest(),
    }


def write_markdown_summary(
    output: Path,
    performance: pd.DataFrame,
    standardization: pd.DataFrame,
    importance: pd.DataFrame,
    composition: pd.DataFrame,
) -> None:
    perf = performance.set_index(["dataset", "subset"])
    va = perf.loc[("random_validation", "overall")]
    te = perf.loc[("temporal", "overall")]
    best_std = standardization.iloc[0] if len(standardization) else None
    best_imp = importance.iloc[0] if len(importance) else None

    endpoint_comp = composition[composition.factor == "endpoint"].pivot(
        index="level", columns="dataset", values="proportion"
    )
    lines = [
        "# Reviewer 2, Suggestion 8: temporal shift audit",
        "",
        "## Performance gap",
        "",
        f"- Random-validation RMSE: {va.rmse:.4f}; temporal RMSE: {te.rmse:.4f}.",
        f"- Random-validation R²: {va.r2:.4f}; temporal R²: {te.r2:.4f}.",
        "",
        "## Prespecified distribution domains",
        "",
        "Affinity-type proportions (random validation versus temporal):",
        "",
    ]
    for endpoint in ENDPOINT_ORDER:
        if endpoint in endpoint_comp.index:
            row = endpoint_comp.loc[endpoint]
            lines.append(
                f"- {endpoint}: {100 * row.get('random_validation', np.nan):.2f}% versus "
                f"{100 * row.get('temporal', np.nan):.2f}%."
            )
    lines += ["", "## Descriptive attribution", ""]
    if best_std is not None:
        lines.append(
            f"- The largest one-factor composition-standardization change was for "
            f"`{best_std.factor}`: {best_std.percent_of_observed_mse_gap:.2f}% of the "
            "observed MSE gap. This estimate is not additive or causal."
        )
    if best_imp is not None:
        lines.append(
            f"- The strongest multivariable association with temporal squared error was "
            f"`{best_imp.factor_group}` (normalized positive permutation importance "
            f"{best_imp.normalized_positive_importance:.3f})."
        )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "Exact compound/target overlap and 3-mer neighborhood similarity are operational "
        "measures of representation-space novelty. The 3-mer score is not sequence "
        "identity. One-factor standardization and grouped permutation importance quantify "
        "descriptive associations; correlated factors prevent a causal or additive "
        "decomposition. Unrecorded assay format, laboratory, censoring, and protocol "
        "differences remain possible contributors.",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_predictions", nargs="+", required=True, type=Path)
    parser.add_argument("--validation_predictions", nargs="+", required=True, type=Path)
    parser.add_argument("--temporal_predictions", nargs="+", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--cpus", type=int, default=max(1, min(32, os.cpu_count() or 1)))
    parser.add_argument("--chunksize", type=int, default=100000)
    parser.add_argument("--descriptor_train_cap", type=int, default=200000)
    parser.add_argument("--error_model_sample_cap", type=int, default=240000)
    parser.add_argument("--target_query_chunk", type=int, default=512)
    parser.add_argument("--seed", type=int, default=5313)
    parser.add_argument(
        "--allow_no_rdkit",
        action="store_true",
        help="Debugging only; scientific run must have RDKit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not HAVE_RDKIT and not args.allow_no_rdkit:
        raise ImportError(
            "RDKit is required for the scientific audit. Activate the ConfMux-DTA "
            "environment or use --allow_no_rdkit only for a non-scientific smoke test."
        )
    for group in [args.train_predictions, args.validation_predictions, args.temporal_predictions]:
        for path in group:
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing or empty input: {path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.out_dir / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cpus = max(1, int(args.cpus))

    all_inputs = list(args.train_predictions) + list(args.validation_predictions) + list(args.temporal_predictions)
    run_key = cache_run_key(
        all_inputs,
        {
            "seed": args.seed,
            "descriptor_train_cap": args.descriptor_train_cap,
            "target_query_chunk": args.target_query_chunk,
            "rdkit_available": HAVE_RDKIT,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )
    stage_cache = cache_dir / f"audit_{run_key}"
    stage_cache.mkdir(parents=True, exist_ok=True)
    log(f"Resumable stage cache: {stage_cache}")

    log("Stage 1/7: scan training-reference predictions")
    reference_cache = stage_cache / "reference_state.pkl"
    if reference_cache.is_file() and reference_cache.stat().st_size > 0:
        (
            train_compounds,
            train_targets,
            train_pairs,
            endpoint_counts,
            y_by_endpoint,
            n_train,
        ) = pickle_load(reference_cache)
        log("Stage 1/7 resumed from reference_state.pkl")
    else:
        (
            train_compounds,
            train_targets,
            train_pairs,
            endpoint_counts,
            y_by_endpoint,
            n_train,
        ) = scan_reference(args.train_predictions, args.chunksize)
        atomic_pickle_dump(
            (
                train_compounds,
                train_targets,
                train_pairs,
                endpoint_counts,
                y_by_endpoint,
                n_train,
            ),
            reference_cache,
        )
    px_specs = training_px_spec(y_by_endpoint)
    if set(px_specs) != set(ENDPOINT_ORDER):
        raise ValueError(f"Training data do not contain all endpoints: {sorted(px_specs)}")

    log("Stage 2/7: load fixed random-validation and temporal predictions")
    evaluation_cache = stage_cache / "minimal_evaluation_tables.pkl"
    if evaluation_cache.is_file() and evaluation_cache.stat().st_size > 0:
        validation, temporal = pickle_load(evaluation_cache)
        log("Stage 2/7 resumed from minimal_evaluation_tables.pkl")
    else:
        validation = load_evaluation(args.validation_predictions, args.chunksize)
        temporal = load_evaluation(args.temporal_predictions, args.chunksize)
        atomic_pickle_dump((validation, temporal), evaluation_cache)

    log("Stage 3/7: calculate compound scaffolds and descriptors")
    train_descriptor_sample = set(
        stable_sample(sorted(train_compounds), args.descriptor_train_cap, args.seed)
    )
    train_scaffold_map, train_valid_map, train_acyclic_map, train_descriptors = compute_molecule_maps(
        sorted(train_compounds),
        train_descriptor_sample,
        cpus,
        "Training compounds",
        block_cache_dir=stage_cache / "training_compound_blocks",
    )
    train_scaffolds = {
        scaffold
        for smiles, scaffold in train_scaffold_map.items()
        if scaffold and train_valid_map.get(smiles, False) and not train_acyclic_map.get(smiles, False)
    }
    eval_compounds = sorted(set(validation.compound) | set(temporal.compound))
    eval_descriptor_set = set(eval_compounds)
    eval_scaffold_map, eval_valid_map, eval_acyclic_map, eval_descriptors = compute_molecule_maps(
        eval_compounds,
        eval_descriptor_set,
        cpus,
        "Evaluation compounds",
        block_cache_dir=stage_cache / "evaluation_compound_blocks",
    )

    log("Stage 4/7: calculate target overlap and 3-mer neighborhood similarity")
    eval_targets = sorted(set(validation.target) | set(temporal.target))
    similarity_map = target_similarity_maps(
        train_targets,
        eval_targets,
        cpus,
        args.target_query_chunk,
        cache_path=stage_cache / "target_similarity_map.pkl",
    )

    log("Stage 5/7: enrich evaluation records and calculate tables")
    validation = enrich_evaluation(
        validation,
        train_compounds,
        train_scaffolds,
        train_targets,
        train_pairs,
        eval_scaffold_map,
        eval_valid_map,
        eval_acyclic_map,
        eval_descriptors,
        similarity_map,
        px_specs,
    )
    temporal = enrich_evaluation(
        temporal,
        train_compounds,
        train_scaffolds,
        train_targets,
        train_pairs,
        eval_scaffold_map,
        eval_valid_map,
        eval_acyclic_map,
        eval_descriptors,
        similarity_map,
        px_specs,
    )
    datasets = {"random_validation": validation, "temporal": temporal}
    factors = ["endpoint", "compound_novelty", "target_novelty", "pair_seen", "px_train_bin"]
    performance = make_performance_table(datasets)
    composition = make_composition_table(datasets, factors)
    strata = make_strata_table(datasets, factors)
    categorical_shifts = categorical_shift_table(
        composition, [("random_validation", "temporal")]
    )
    validation_unique_descriptors = eval_descriptors.loc[
        eval_descriptors.index.intersection(validation.compound.unique())
    ]
    temporal_unique_descriptors = eval_descriptors.loc[
        eval_descriptors.index.intersection(temporal.compound.unique())
    ]
    numeric_shifts = make_numeric_shift_table(
        validation,
        temporal,
        train_descriptors,
        validation_unique_descriptors,
        temporal_unique_descriptors,
        y_by_endpoint,
        train_targets,
    )
    standardization = standardization_table(
        validation,
        temporal,
        ["endpoint", "compound_novelty", "target_novelty", "pair_seen", "px_train_bin"],
    )

    log("Stage 6/7: descriptive multivariable error-association analysis")
    importance, error_model_diagnostics = error_association_analysis(
        validation, temporal, args.seed, args.error_model_sample_cap
    )

    log("Stage 7/7: write reviewer-ready outputs")
    outputs = {
        "paper_temporal_performance.csv": performance,
        "paper_temporal_composition.csv": composition,
        "paper_temporal_error_strata.csv": strata,
        "paper_temporal_categorical_shift.csv": categorical_shifts,
        "paper_temporal_numeric_shift.csv": numeric_shifts,
        "paper_temporal_factor_standardization.csv": standardization,
        "paper_temporal_grouped_importance.csv": importance,
    }
    for filename, table in outputs.items():
        table.to_csv(args.out_dir / filename, index=False)

    train_reference = {
        "n_rows": int(n_train),
        "n_unique_compounds": int(len(train_compounds)),
        "n_unique_nonempty_scaffolds": int(len(train_scaffolds)),
        "n_unique_targets": int(len(train_targets)),
        "n_unique_pair_hashes": int(len(train_pairs)),
        "endpoint_counts": dict(endpoint_counts),
        "px_training_quantiles": px_specs,
        "descriptor_reference_sample_n": int(len(train_descriptors)),
    }
    manifest = {
        "analysis": "ConfMux-DTA temporal distribution-shift audit",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "inputs": {
            "train_predictions": [file_fingerprint(p) for p in args.train_predictions],
            "validation_predictions": [file_fingerprint(p) for p in args.validation_predictions],
            "temporal_predictions": [file_fingerprint(p) for p in args.temporal_predictions],
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "rdkit_available": HAVE_RDKIT,
        },
        "parameters": vars(args),
        "resumable_stage_cache": str(stage_cache.resolve()),
        "training_reference": train_reference,
        "evaluation_counts": {
            "random_validation": int(len(validation)),
            "temporal": int(len(temporal)),
        },
        "error_model_diagnostics": error_model_diagnostics,
        "methodological_notes": [
            "The target 3-mer TF-IDF cosine value is a neighborhood similarity score, not sequence identity.",
            "Empty Bemis-Murcko scaffolds are classified as acyclic and are not treated as one shared scaffold.",
            "One-factor standardization estimates are descriptive, non-additive, and non-causal.",
            "Grouped permutation importance measures error association and is not a causal attribution.",
            "Unrecorded assay, laboratory, protocol, and censoring differences cannot be evaluated here.",
        ],
    }
    with (args.out_dir / "analysis_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, default=str)
    plot_audit(
        composition,
        strata,
        numeric_shifts,
        standardization,
        importance,
        args.out_dir / "Supplementary_Fig_S6_temporal_shift_audit.png",
    )
    write_markdown_summary(
        args.out_dir / "Reviewer2_Suggestion8_result_summary.md",
        performance,
        standardization,
        importance,
        composition,
    )

    log(f"Completed: {args.out_dir.resolve()}")
    for filename in sorted(p.name for p in args.out_dir.iterdir() if p.is_file()):
        log(f"Output: {filename}")


if __name__ == "__main__":
    main()
