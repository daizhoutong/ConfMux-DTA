#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConfMux-DTA BindingDB preprocessing script.

This script prepares BindingDB-derived affinity records for the ConfMux-DTA
sequence-only unified drug-target affinity modeling pipeline. It reads a
BindingDB full TSV file, standardizes ligand SMILES, extracts target sequences
and UniProt identifiers, parses Ki/Kd/IC50 values with exact/approximate/
censored qualifiers, applies record-level reliability weights, summarizes
environmental assay metadata, and aggregates repeated measurements into a
model-ready table.

Main processing steps
---------------------
1. Read BindingDB TSV files in chunks.
2. Standardize SMILES with RDKit and optionally generate standard InChIKeys.
3. Collapse redundant multi-chain target entries only when chains appear to be
   equivalent; otherwise discard the record to avoid mixing distinct complexes.
4. Parse Ki, Kd, IC50, and optional EC50 strings into numeric nM values, sign
   labels, and endpoint-specific reliability weights.
5. Apply pH/temperature-based environmental penalties to endpoint weights.
6. Filter invalid records, including missing affinity measurements, invalid
   SMILES, empty target sequences, out-of-range affinities, and incompatible
   multi-chain targets.
7. Aggregate replicated records by
   (standardized SMILES, UniProt ID, protein sequence, metric-kind combination)
   using endpoint-specific weighted medians.
8. Drop groups with severe within-endpoint disagreement in p-space.
9. Export pKi, pKd, pIC50, optional pEC50, endpoint weights, sign-audit fields,
   environmental summaries, and optional ChEMBL cross-reference identifiers.

Note
----
The ConfMux-DTA manuscript uses the unified pIC50/pKi/pKd long table for model
training. EC50 is retained here only as an optional preprocessing output for
auditing or future extension and is not used by the main published unified model
unless explicitly enabled downstream.

Example
-------
python confmux_dta_bindingdb_preprocess.py BindingDB_All.tsv preprocessed/confmux_dta_bindingdb_agg

The command above writes preprocessed/confmux_dta_bindingdb_agg.csv, which is the default input file
used by confmux_dta_train_predict.py.
"""


import re
import sys
import os
import math
import sqlite3
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

tqdm.pandas()

import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# -------------------------- Logging --------------------------
logging.basicConfig(
    filename='bindingdb_process_errors.log',
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

# -------------------------- Configuration --------------------------
CHEMBL_DB_PATH_DEFAULT = "/home/lhy/cpi/chembl_36/chembl_36_sqlite/chembl_36.db"

_cfg_ = {
    'pH_range': [5.5, 8.5],
    'T_range': [33, 45],

    'Kd/Ki_range': [1e-5, 1e15],       # nM units allowed
    'Kd/Ki_percentile': 50,            # weighted median percentile for Ki/Kd

    'IC50_range': [1e-5, 1e15],
    'IC50_percentile': 50,

    'EC50_range': [1e-5, 1e15],
    'EC50_percentile': 50,

    'protein_len_range': [20, 25600],
    'max_chains': 10,
    'chunk_size': 100000,
    'max_workers': 30,

    'env_penalty_factor': 0.8,         # penalty multiplier for out-of-range env
    'spread_log10_tol': 4.0,           # drop a group if any endpoint spread exceeds this p-space range
    'do_smiles_standardize': True,     # enable RDKit-based SMILES standardization

    # --- Optional ChEMBL cross-reference mapping ---
    'use_chembl_mapping': True,
    'chembl_db_path': CHEMBL_DB_PATH_DEFAULT,
    'sqlite_in_batch': 900,            # keep below SQLite's common 999-parameter IN limit
}

BALANCED_WEIGHTS = {
    '=' : 1.00,   # exact value
    '≈' : 0.95,   # approximate or estimated value
    '<' : 1.3,    # upper-bound concentration; may indicate stronger activity
    '>' : 0.7,    # lower-bound concentration; treated as less informative
    '<=': 1.15,   # weak upper-bound evidence
    '>=': 0.85    # weak lower-bound evidence
}

_SIGN_ORDER = ['=', '≈', '<=', '<', '>=', '>', 'missing']

def check_weight_symmetry():
    print("[INFO] Weight-symmetry check:")
    inverse = {'<': '>', '>': '<', '<=': '>=', '>=': '<='}
    for k, w in BALANCED_WEIGHTS.items():
        if k in ('=', '≈'):
            continue
        if k in inverse:
            w2 = BALANCED_WEIGHTS[inverse[k]]
            product = w * w2
            dev = abs(product - 1.0)
            ok = "OK" if dev < 0.05 else "WARN"
            print(f"  {ok}: {k}({w}) x {inverse[k]}({w2}) = {product:.3f}; deviation={dev:.3f}")
    print("  Target: reciprocal upper/lower-bound weights should multiply to approximately 1.")

# -------------------------- RDKit standardization --------------------------
def _std_modules():
    """Import rdMolStandardize lazily for compatibility across RDKit builds."""
    try:
        from rdkit.Chem import rdMolStandardize
        return rdMolStandardize
    except Exception:
        try:
            from rdkit.Chem.MolStandardize import rdMolStandardize
            return rdMolStandardize
        except Exception:
            return None

_STD = _std_modules()

def standardize_smiles_once(s):
    """
    Standardize one SMILES string using RDKit.

    No length-based truncation is applied. If standardization fails, the
    original string is returned and later filtered by RDKit parseability.
    """
    try:
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return s
        s = str(s).strip()
        if not s:
            return s

        m = Chem.MolFromSmiles(s)
        if m is None:
            return s

        if _STD is not None:
            try:
                m = _STD.Cleanup(m)
            except Exception:
                pass
            try:
                lfc = _STD.LargestFragmentChooser(preferOrganic=True)
                m = lfc.choose(m)
            except Exception:
                pass
            try:
                m = _STD.Normalizer().normalize(m)
            except Exception:
                pass
            try:
                m = _STD.Reionizer().reionize(m)
            except Exception:
                pass
            try:
                m = _STD.Uncharger().uncharge(m)
            except Exception:
                pass

        can = Chem.MolToSmiles(m, isomericSmiles=True, canonical=True)
        return can if can else s
    except Exception:
        return s

# -------------------------- InChIKey generation for ChEMBL mapping --------------------------
def _get_rdkit_inchi():
    try:
        from rdkit.Chem import inchi as rd_inchi
        return rd_inchi
    except Exception:
        return None

_RD_INCHI = _get_rdkit_inchi()
_WARNED_NO_INCHI = False

def smiles_to_inchikey(smiles: str):
    global _WARNED_NO_INCHI
    if pd.isna(smiles):
        return np.nan
    if _RD_INCHI is None:
        if not _WARNED_NO_INCHI:
            print("[WARN] RDKit InChI support is unavailable; standard_inchi_key and ChEMBL molecule mapping will be skipped.")
            _WARNED_NO_INCHI = True
        return np.nan
    try:
        m = Chem.MolFromSmiles(str(smiles))
        if m is None:
            return np.nan
        return _RD_INCHI.MolToInchiKey(m)
    except Exception:
        return np.nan

# -------------------------- I/O and basic cleaning --------------------------
def detect_chain_columns(sample_df):
    chain_pat = re.compile(r'BindingDB Target Chain Sequence \d+', re.IGNORECASE)
    uni_pat   = re.compile(r'UniProt.*Primary ID of Target Chain \d+', re.IGNORECASE)
    chain_cols = [c for c in sample_df.columns if chain_pat.match(c)]
    uniprot_cols = [c for c in sample_df.columns if uni_pat.match(c)]
    logging.info(f"Detected chain seq columns: {chain_cols}")
    logging.info(f"Detected UniProt ID columns: {uniprot_cols}")
    return chain_cols, uniprot_cols

def clean_smiles(s):
    """Remove BindingDB suffixes after the pipe character without length truncation."""
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    return s.split('|')[0].strip() if '|' in s else s

def _process_multichain(seq_joined: str) -> str:
    """
    Handle multi-chain target sequence fields.

    Redundant chains are collapsed only when all chains have the same length
    and matching three-residue N- and C-terminal segments. Otherwise the record
    is discarded to avoid representing heterogeneous complexes as one target.
    """
    if pd.isna(seq_joined) or not isinstance(seq_joined, str):
        return ""
    seq_joined = seq_joined.strip()
    if not seq_joined:
        return ""
    chains = [c.strip() for c in seq_joined.split('|') if c.strip()]
    if len(chains) <= 1:
        return chains[0] if chains else ""
    lens  = {len(c) for c in chains}
    head3 = {c[:3] for c in chains}
    tail3 = {c[-3:] for c in chains}
    if len(lens)==1 and len(head3)==1 and len(tail3)==1:
        return chains[0]
    else:
        return ""

def read_bindingdb(fname):
    """
    Read a BindingDB TSV file in chunks and return the core curation columns.

    Returned columns include ligand SMILES, processed protein sequence,
    UniProt identifiers, Ki/Kd/IC50/optional EC50 fields, pH, temperature,
    chain count, organism, and BindingDB link fields when available.
    """
    print(f'[INFO] Reading TSV: {fname} (chunk size: {_cfg_["chunk_size"]})...')
    pdv = pd.__version__.split('.')
    is_pandas20plus = int(pdv[0]) >= 2 or (int(pdv[0])==1 and int(pdv[1])>=4)
    encoding = 'utf-8'

    # Inspect a small sample first to detect available BindingDB columns.
    try:
        if is_pandas20plus:
            sample_df = pd.read_csv(
                fname, sep='\t', nrows=200, encoding=encoding, encoding_errors='replace',
                escapechar='\\', quoting=3,
                on_bad_lines=lambda x: logging.warning(f"Skipping bad sample line: {x[:120]}..."),
                engine='python'
            )
        else:
            sample_df = pd.read_csv(
                fname, sep='\t', nrows=200, encoding=encoding, encoding_errors='replace',
                escapechar='\\', quoting=3, error_bad_lines=False, warn_bad_lines=True
            )
    except Exception:
        sample_df = pd.read_csv(
            fname, sep='\t', nrows=200, encoding='latin-1',
            escapechar='\\', quoting=3,
            on_bad_lines=(lambda x: logging.warning(f"Skipping bad sample line: {x[:120]}...")) if is_pandas20plus else None,
            engine='python' if is_pandas20plus else None
        )
        logging.warning("Fell back to latin-1 encoding for sample detection")

    available = set(sample_df.columns)
    chain_cols, uni_cols = detect_chain_columns(sample_df)

    # Fallback for common BindingDB sequence-column naming.
    if not chain_cols:
        if 'BindingDB Target Chain Sequence 1' in available:
            chain_cols = ['BindingDB Target Chain Sequence 1']
        else:
            logging.warning("No chain sequence columns detected and fallback column not found.")

    fixed_map_all = {
        'Ligand SMILES': 'smiles',
        'Ki (nM)': 'Ki',
        'Kd (nM)': 'Kd',
        'IC50 (nM)': 'IC50',
        'EC50 (nM)': 'EC50',
        'pH': 'pH',
        'Temp (C)': 'T',
        'Number of Protein Chains in Target (>1 implies a multichain complex)': 'n_chains',
        'Target Source Organism According to Curator or DataSource': 'organism',
        'Link to Target in BindingDB': 'bdb_link',
    }

    fixed_map = {k: v for k, v in fixed_map_all.items() if k in available}
    if 'Ligand SMILES' not in fixed_map:
        logging.error("Required column missing: Ligand SMILES")
        return pd.DataFrame()

    # Keep only chain and UniProt columns that actually exist in the file.
    chain_cols = [c for c in chain_cols if c in available]
    uni_cols   = [c for c in uni_cols if c in available]

    usecols_map = fixed_map.copy()
    for i, c in enumerate(chain_cols[:_cfg_['max_chains']]):
        usecols_map[c] = f'seq_{i+1}'
    for i, c in enumerate(uni_cols[:_cfg_['max_chains']]):
        usecols_map[c] = f'uniprot_{i+1}'

    read_kwargs = dict(
        sep='\t',
        usecols=list(usecols_map.keys()),
        chunksize=_cfg_['chunk_size'],
        encoding=encoding,
        encoding_errors='replace',
        escapechar='\\',
        quoting=3,
        engine='python'
    )
    if is_pandas20plus:
        read_kwargs['on_bad_lines'] = lambda x: logging.warning(f"Skipping bad line: {x[:120]}...")
    else:
        read_kwargs['error_bad_lines'] = False
        read_kwargs['warn_bad_lines']  = True

    chunks = []
    total_rows = 0
    total_smiles_std = 0
    total_smiles_removed = 0
    total_multichain_removed = 0

    try:
        reader = pd.read_csv(fname, **read_kwargs)
    except UnicodeDecodeError:
        logging.warning("UTF-8 decoding failed - using latin-1")
        read_kwargs['encoding'] = 'latin-1'
        reader = pd.read_csv(fname, **read_kwargs)

    for idx, ch in enumerate(reader):
        total_rows += len(ch)
        ch = ch.dropna(how='all').rename(columns=usecols_map)

        # 1) Basic SMILES cleanup.
        ch['smiles'] = ch['smiles'].apply(clean_smiles)

        # 2) RDKit-based SMILES standardization.
        if _cfg_['do_smiles_standardize']:
            before_smiles = ch['smiles'].copy()
            ch['smiles'] = ch['smiles'].apply(standardize_smiles_once)
            total_smiles_std += int((before_smiles != ch['smiles']).sum())

        # 3) Remove empty SMILES after cleanup/standardization.
        before = len(ch)
        ch = ch[ch['smiles'].notna() & (ch['smiles'].astype(str).str.len() > 0)]
        total_smiles_removed += (before - len(ch))

        # 4) Join target-chain sequence columns.
        seq_cols = [c for c in ch.columns if c.startswith('seq_')]
        if seq_cols:
            ch['protein_joined'] = ch[seq_cols].apply(
                lambda row: '|'.join([str(s).strip() for s in row if pd.notna(s) and str(s).strip()]),
                axis=1
            )
            ch = ch.drop(columns=seq_cols)

            # 5) Collapse redundant multi-chain targets or discard incompatible complexes.
            before_multichain = len(ch)
            ch['protein'] = ch['protein_joined'].apply(_process_multichain)
            ch = ch[ch['protein'] != ""]
            total_multichain_removed += (before_multichain - len(ch))
            ch = ch.drop(columns=['protein_joined'])
        else:
            ch['protein'] = ""

        # 6) Merge available UniProt identifiers.
        uni_cols_now = [c for c in ch.columns if c.startswith('uniprot_')]
        if uni_cols_now:
            ch['uniprot_id'] = ch[uni_cols_now].apply(
                lambda row: '|'.join(
                    [str(s).strip() for s in row if pd.notna(s) and str(s).strip() and str(s).strip() != 'nan']
                ),
                axis=1
            ).replace('', np.nan)
        else:
            ch['uniprot_id'] = np.nan
        ch = ch.drop(columns=uni_cols_now, errors='ignore')

        # 7) Keep canonical output columns and fill missing optional fields.
        keep_cols = [
            'smiles', 'protein', 'uniprot_id',
            'Ki', 'Kd', 'IC50', 'EC50',
            'pH', 'T', 'n_chains', 'organism', 'bdb_link'
        ]
        for c in keep_cols:
            if c not in ch.columns:
                ch[c] = np.nan if c != 'protein' else ""
        ch = ch[keep_cols]

        chunks.append(ch)
        print(f"  [INFO] Chunk {idx}: total_in={total_rows}, kept={len(ch)}")

    if not chunks:
        logging.error("No valid data found in the TSV file")
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)

    print("\n[INFO] Data loaded:")
    print(f"   total rows read: {total_rows}")
    print(f"   after basic SMILES/protein filtering: {len(df)}")
    print(f"   standardized SMILES changed: {total_smiles_std}")
    print(f"   removed due to empty SMILES (post-std): {total_smiles_removed}")
    print(f"   removed due to incompatible multichain protein: {total_multichain_removed}")
    print(f"   UniProt coverage: {df['uniprot_id'].notna().sum()}/{len(df)} "
          f"({df['uniprot_id'].notna().mean()*100:.1f}%)")
    return df

# -------------------------- Numeric parsing and endpoint weights --------------------------
def safe_extract_float(s):
    if pd.isna(s):
        return np.nan
    try:
        txt = str(s).strip().replace(",", "")
        if txt.lower() in ("", "nan", "none", "null"):
            return np.nan
        m = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', txt)
        return float(m[0]) if m else np.nan
    except Exception:
        logging.warning(f"Could not extract float from: {s}")
        return np.nan

def convert_other_floats(df):
    for col in ['pH', 'T', 'n_chains']:
        if col in df.columns:
            df[col] = df[col].apply(safe_extract_float)
    return df

def parse_apprx_value_with_balanced_weights(val):
    """
    Parse an affinity string into (numeric_value, base_weight, sign_type).

    Missing tokens such as 'nan', 'None', and empty strings are treated as
    missing values.
    """
    if pd.isna(val):
        return (np.nan, np.nan, 'missing')
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return (np.nan, np.nan, 'missing')

    if s.startswith('>='):
        v = safe_extract_float(s[2:])
        return (v, BALANCED_WEIGHTS['>='], '>=') if np.isfinite(v) else (np.nan, np.nan, 'missing')
    if s.startswith('<='):
        v = safe_extract_float(s[2:])
        return (v, BALANCED_WEIGHTS['<='], '<=') if np.isfinite(v) else (np.nan, np.nan, 'missing')
    if s.startswith('>'):
        v = safe_extract_float(s[1:])
        return (v, BALANCED_WEIGHTS['>'], '>') if np.isfinite(v) else (np.nan, np.nan, 'missing')
    if s.startswith('<'):
        v = safe_extract_float(s[1:])
        return (v, BALANCED_WEIGHTS['<'], '<') if np.isfinite(v) else (np.nan, np.nan, 'missing')

    if ('~' in s) or ('≈' in s) or ('about' in s.lower()):
        v = safe_extract_float(s)
        return (v, BALANCED_WEIGHTS['≈'], '≈') if np.isfinite(v) else (np.nan, np.nan, 'missing')

    v = safe_extract_float(s)
    return (v, BALANCED_WEIGHTS['='], '=') if np.isfinite(v) else (np.nan, np.nan, 'missing')

def compute_env_penalty(row):
    pen = 1.0
    if 'pH' in row and pd.notna(row['pH']):
        if row['pH'] < _cfg_['pH_range'][0] or row['pH'] > _cfg_['pH_range'][1]:
            pen *= _cfg_['env_penalty_factor']
    if 'T' in row and pd.notna(row['T']):
        if row['T'] < _cfg_['T_range'][0] or row['T'] > _cfg_['T_range'][1]:
            pen *= _cfg_['env_penalty_factor']
    return pen

# -------------------------- Environmental metadata summarization --------------------------
def _summarize_numeric(series: pd.Series):
    s = pd.to_numeric(series, errors='coerce').dropna()
    if len(s) == 0:
        return (0, np.nan, np.nan, np.nan)
    return (int(len(s)), float(s.min()), float(s.median()), float(s.max()))

def _count_out_of_range(series: pd.Series, lo, hi):
    s = pd.to_numeric(series, errors='coerce').dropna()
    if len(s) == 0:
        return 0
    return int(((s < lo) | (s > hi)).sum())

def _compact_set(series: pd.Series, round_digits=2, max_items=16):
    s = pd.to_numeric(series, errors='coerce').dropna()
    if len(s) == 0:
        return ""
    vals = sorted(set(s.round(round_digits).tolist()))
    if len(vals) <= max_items:
        return "|".join([str(v) for v in vals])
    return f"nuniq={len(vals)};min={vals[0]};max={vals[-1]}"

# -------------------------- Summary and audit helpers --------------------------
def analyze_inequality_distribution(df, stage):
    print(f"\n[INFO] {stage} inequality-sign distribution:")
    total = len(df)
    kinds = ['=', '≈', '<', '>', '<=', '>=', 'missing']
    for col in ['Ki', 'Kd', 'IC50', 'EC50']:
        tcol = f'{col}_type'
        if tcol not in df.columns:
            continue
        vc = df[tcol].value_counts()
        print(f"  -- {col} --")
        for k in kinds:
            n = int(vc.get(k, 0))
            if n > 0:
                print(f"     {k:>7}: {n} records ({n/total*100:.1f}%)")

def analyze_weight_distribution(df, stage):
    print(f"\n[INFO] {stage} weight distribution:")
    cols = [c for c in ['weight_Ki','weight_Kd','weight_IC50','weight_EC50'] if c in df.columns]
    if not cols:
        print("  (no weight columns)")
        return
    for c in cols:
        s = pd.to_numeric(df[c], errors='coerce').dropna()
        if len(s)==0:
            continue
        print(f"  -- {c} --")
        print(f"     min={s.min():.3f} max={s.max():.3f} mean={s.mean():.3f} n={len(s)}")

def analyze_affinity_distribution(df, stage):
    print(f"\n[INFO] {stage} affinity-data distribution:")
    total = len(df)
    for col in ['Ki', 'Kd', 'IC50', 'EC50']:
        if col in df.columns:
            n = df[col].notna().sum()
            print(f"   {col}: {n} records ({n/total*100:.1f}%)")
    for col in ['pKi', 'pKd', 'pIC50', 'pEC50']:
        if col in df.columns:
            n = df[col].notna().sum()
            print(f"   {col}: {n} records ({n/total*100:.1f}%)")

# -------------------------- Filtering --------------------------
def filter_df(name, df, mask, filtered_rows_list=None, reason=None):
    before = len(df)
    removed_rows = df[mask].copy() if mask.any() else pd.DataFrame()
    df = df[~mask].copy()
    after = len(df)
    if removed_rows.size and filtered_rows_list is not None and reason:
        removed_rows['filter_reason'] = reason
        filtered_rows_list.append(removed_rows)
    print(f"   {name} filter: {before-after} rows removed ({before} -> {after})")
    return df.reset_index(drop=True)

def filter_invalid_data(df, save_filtered_path=None):
    print(f"\n[INFO] Filtering process:")
    flist = []

    # 1) Keep records with at least one Ki/Kd/IC50/EC50 value.
    has_const = df['Ki'].notna() | df['Kd'].notna() | df['IC50'].notna() | df['EC50'].notna()
    df = filter_df("No binding constants", df, ~has_const, flist, "No Ki/Kd/IC50/EC50")

    # 2) Require RDKit-parseable SMILES.
    def is_valid_smiles(sm):
        if pd.isna(sm):
            return False
        try:
            return Chem.MolFromSmiles(str(sm)) is not None
        except Exception:
            return False
    valid_smiles = df['smiles'].apply(is_valid_smiles)
    df = filter_df("Invalid SMILES", df, ~valid_smiles, flist, "Invalid SMILES")

    # 3) Require a non-empty protein sequence.
    valid_protein = df['protein'] != ''
    df = filter_df("Empty protein", df, ~valid_protein, flist, "Empty protein")

    # 4) Apply numeric range checks while allowing endpoint-wise missing values.
    if 'Ki' in df.columns:
        v = df['Ki'].isna() | ((df['Ki']>=_cfg_['Kd/Ki_range'][0]) & (df['Ki']<=_cfg_['Kd/Ki_range'][1]))
        df = filter_df("Ki out of range", df, ~v, flist, "Ki out of range")
    if 'Kd' in df.columns:
        v = df['Kd'].isna() | ((df['Kd']>=_cfg_['Kd/Ki_range'][0]) & (df['Kd']<=_cfg_['Kd/Ki_range'][1]))
        df = filter_df("Kd out of range", df, ~v, flist, "Kd out of range")
    if 'IC50' in df.columns:
        v = df['IC50'].isna() | ((df['IC50']>=_cfg_['IC50_range'][0]) & (df['IC50']<=_cfg_['IC50_range'][1]))
        df = filter_df("IC50 out of range", df, ~v, flist, "IC50 out of range")
    if 'EC50' in df.columns:
        v = df['EC50'].isna() | ((df['EC50']>=_cfg_['EC50_range'][0]) & (df['EC50']<=_cfg_['EC50_range'][1]))
        df = filter_df("EC50 out of range", df, ~v, flist, "EC50 out of range")

    # Optionally export filtered-out rows for audit.
    if save_filtered_path and flist:
        allf = pd.concat(flist, ignore_index=True).drop_duplicates(
            subset=['smiles','uniprot_id','protein','filter_reason']
        )
        path = f"{save_filtered_path}_filtered_rows.tsv"
        allf.to_csv(path, sep='\t', index=False)
        print(f"[INFO] Saved {len(allf)} filtered rows to {path}")

    return df

# -------------------------- Aggregation --------------------------
def weighted_percentile(data, weights, perc):
    data = np.array(data); weights = np.array(weights)
    mask = ~np.isnan(data) & ~np.isnan(weights)
    data = data[mask]; weights = weights[mask]
    if len(data)==0:
        return np.nan
    ix = np.argsort(data)
    ds = data[ix]; ws = weights[ix]
    s = np.sum(ws)
    if not np.isfinite(s) or s <= 0:
        return np.nan
    cdf = np.cumsum(ws) / s
    return np.interp(perc/100, cdf, ds)

def _calc_spread_log10_nm(vals_nm):
    """
    Calculate the p-space spread for a group of positive nM affinity values.

    The returned spread is max(-log10(M)) - min(-log10(M)), where M = nM / 1e9.
    """
    good = [v for v in vals_nm if (v is not None and not np.isnan(v) and v>0)]
    if len(good) < 2:
        return 0.0
    good = np.array(good, dtype=float)
    pvals = -np.log10(good / 1e9)
    return float(np.max(pvals) - np.min(pvals))

def _summarize_signs(series):
    s = series.dropna().astype(str).str.strip()
    if len(s) == 0:
        return ("", "")
    vc = s.value_counts()
    maxcnt = int(vc.max())
    candidates = list(vc[vc == maxcnt].index)

    sign_mode = None
    for k in _SIGN_ORDER:
        if k in candidates:
            sign_mode = k
            break
    if sign_mode is None:
        sign_mode = str(candidates[0])

    uniq = set(s.tolist())
    sign_set = "|".join([k for k in _SIGN_ORDER if k in uniq])
    return (sign_set, sign_mode)

def _format_signed_value(sign, value_nm):
    if value_nm is None or (isinstance(value_nm, float) and np.isnan(value_nm)):
        return ""
    if not sign or sign == "missing":
        sign = "="
    try:
        return f"{sign}{float(value_nm):.6g}"
    except Exception:
        return f"{sign}{value_nm}"

def aggregate_constants(df):
    """
    Aggregate repeated affinity records.

    Grouping key:
        (smiles, uniprot_id, protein, metric_kind)

    For each endpoint present in a group, the function calculates the p-space
    spread and discards the whole group if any endpoint exceeds the configured
    conflict threshold. Endpoint values are summarized by weighted medians, and
    endpoint weights are propagated as max(single-record weight) *
    (1 + log10(number of records for that endpoint)).

    The output also keeps sign-audit fields and group-level environmental
    summaries.
    """
    if len(df)==0:
        return pd.DataFrame()

    groups_obj = df.groupby(['smiles','uniprot_id','protein','metric_kind'], dropna=False)
    groups_items = list(groups_obj.groups.items())

    print("\n[INFO] Aggregation process:")
    print(f"   Unique groups (with metric_kind sets): {len(groups_items)}")

    agg_list_parallel = []

    def agg_single(key_and_idx):
        (smiles, uid, prot, mkind), idx_arr = key_and_idx
        g = df.iloc[idx_arr]
        group_size = len(g)

        # Conflict detection in p-space for each endpoint.
        metric_set = [x.strip() for x in str(mkind).split('|') if x.strip()]
        for metric in metric_set:
            if metric == 'Ki':
                vals_nm = g['Ki'].dropna().values
            elif metric == 'Kd':
                vals_nm = g['Kd'].dropna().values
            elif metric == 'IC50':
                vals_nm = g['IC50'].dropna().values
            elif metric == 'EC50':
                vals_nm = g['EC50'].dropna().values
            else:
                continue
            spread = _calc_spread_log10_nm(vals_nm)
            if spread > _cfg_['spread_log10_tol']:
                return None

        # Endpoint-specific weighted medians.
        ki_vals = g['Ki'].dropna().values
        ki_w    = g.loc[g['Ki'].notna(), 'weight_Ki'].values if 'weight_Ki' in g.columns else np.ones(len(ki_vals))
        ki_agg  = weighted_percentile(ki_vals, ki_w, _cfg_['Kd/Ki_percentile']) if len(ki_vals)>0 else np.nan

        kd_vals = g['Kd'].dropna().values
        kd_w    = g.loc[g['Kd'].notna(), 'weight_Kd'].values if 'weight_Kd' in g.columns else np.ones(len(kd_vals))
        kd_agg  = weighted_percentile(kd_vals, kd_w, _cfg_['Kd/Ki_percentile']) if len(kd_vals)>0 else np.nan

        ic_vals = g['IC50'].dropna().values
        ic_w    = g.loc[g['IC50'].notna(), 'weight_IC50'].values if 'weight_IC50' in g.columns else np.ones(len(ic_vals))
        ic_agg  = weighted_percentile(ic_vals, ic_w, _cfg_['IC50_percentile']) if len(ic_vals)>0 else np.nan

        ec_vals = g['EC50'].dropna().values
        ec_w    = g.loc[g['EC50'].notna(), 'weight_EC50'].values if 'weight_EC50' in g.columns else np.ones(len(ec_vals))
        ec_agg  = weighted_percentile(ec_vals, ec_w, _cfg_['EC50_percentile']) if len(ec_vals)>0 else np.nan

        # Propagated weight: max single-record weight x (1 + log10(n_metric)).
        nKi = int(g['Ki'].notna().sum())
        nKd = int(g['Kd'].notna().sum())
        nIc = int(g['IC50'].notna().sum())
        nEc = int(g['EC50'].notna().sum())

        if nKi > 0 and 'weight_Ki' in g.columns:
            base = float(np.nanmax(g.loc[g['Ki'].notna(), 'weight_Ki'].values))
            wKi_final = base * (1.0 + math.log10(nKi))
        else:
            wKi_final = np.nan

        if nKd > 0 and 'weight_Kd' in g.columns:
            base = float(np.nanmax(g.loc[g['Kd'].notna(), 'weight_Kd'].values))
            wKd_final = base * (1.0 + math.log10(nKd))
        else:
            wKd_final = np.nan

        if nIc > 0 and 'weight_IC50' in g.columns:
            base = float(np.nanmax(g.loc[g['IC50'].notna(), 'weight_IC50'].values))
            wIc_final = base * (1.0 + math.log10(nIc))
        else:
            wIc_final = np.nan

        if nEc > 0 and 'weight_EC50' in g.columns:
            base = float(np.nanmax(g.loc[g['EC50'].notna(), 'weight_EC50'].values))
            wEc_final = base * (1.0 + math.log10(nEc))
        else:
            wEc_final = np.nan

        # Convert nM values to p-space.
        pKi   = -np.log10(ki_agg/1e9) if (not np.isnan(ki_agg) and ki_agg>0) else np.nan
        pKd   = -np.log10(kd_agg/1e9) if (not np.isnan(kd_agg) and kd_agg>0) else np.nan
        pIC50 = -np.log10(ic_agg/1e9) if (not np.isnan(ic_agg) and ic_agg>0) else np.nan
        pEC50 = -np.log10(ec_agg/1e9) if (not np.isnan(ec_agg) and ec_agg>0) else np.nan

        # Preserve sign information for audit.
        Ki_sign_set, Ki_sign = ("", "")
        Kd_sign_set, Kd_sign = ("", "")
        IC50_sign_set, IC50_sign = ("", "")
        EC50_sign_set, EC50_sign = ("", "")

        if 'Ki_type' in g.columns and nKi > 0:
            Ki_sign_set, Ki_sign = _summarize_signs(g.loc[g['Ki'].notna(), 'Ki_type'])
        if 'Kd_type' in g.columns and nKd > 0:
            Kd_sign_set, Kd_sign = _summarize_signs(g.loc[g['Kd'].notna(), 'Kd_type'])
        if 'IC50_type' in g.columns and nIc > 0:
            IC50_sign_set, IC50_sign = _summarize_signs(g.loc[g['IC50'].notna(), 'IC50_type'])
        if 'EC50_type' in g.columns and nEc > 0:
            EC50_sign_set, EC50_sign = _summarize_signs(g.loc[g['EC50'].notna(), 'EC50_type'])

        Ki_str   = _format_signed_value(Ki_sign, ki_agg)
        Kd_str   = _format_signed_value(Kd_sign, kd_agg)
        IC50_str = _format_signed_value(IC50_sign, ic_agg)
        EC50_str = _format_signed_value(EC50_sign, ec_agg)

        # Group-level environmental metadata summaries.
        pH_n, pH_min, pH_med, pH_max = _summarize_numeric(g.get('pH', pd.Series([], dtype=float)))
        T_n,  T_min,  T_med,  T_max  = _summarize_numeric(g.get('T',  pd.Series([], dtype=float)))
        pH_out_n = _count_out_of_range(g.get('pH', pd.Series([], dtype=float)), _cfg_['pH_range'][0], _cfg_['pH_range'][1])
        T_out_n  = _count_out_of_range(g.get('T',  pd.Series([], dtype=float)), _cfg_['T_range'][0],  _cfg_['T_range'][1])
        pH_set = _compact_set(g.get('pH', pd.Series([], dtype=float)), round_digits=2, max_items=16)
        T_set  = _compact_set(g.get('T',  pd.Series([], dtype=float)), round_digits=1, max_items=16)

        ep = pd.to_numeric(g.get('env_penalty', pd.Series([], dtype=float)), errors='coerce').dropna()
        if len(ep) == 0:
            env_penalty_n = 0
            env_penalty_min = env_penalty_mean = env_penalty_max = np.nan
            env_penalty_lt1_n = 0
            env_penalty_set = ""
        else:
            env_penalty_n = int(len(ep))
            env_penalty_min = float(ep.min())
            env_penalty_mean = float(ep.mean())
            env_penalty_max = float(ep.max())
            env_penalty_lt1_n = int((ep < 1.0 - 1e-12).sum())
            env_penalty_set = _compact_set(ep, round_digits=3, max_items=12)

        return {
            'smiles': smiles,
            'uniprot_id': uid,
            'protein': prot,
            'group_size': group_size,

            # Group-level environmental metadata summary columns.
            'pH_n': pH_n, 'pH_min': pH_min, 'pH_median': pH_med, 'pH_max': pH_max,
            'pH_out_n': pH_out_n, 'pH_set': pH_set,
            'T_n': T_n, 'T_min': T_min, 'T_median': T_med, 'T_max': T_max,
            'T_out_n': T_out_n, 'T_set': T_set,
            'env_penalty_n': env_penalty_n,
            'env_penalty_min': env_penalty_min,
            'env_penalty_mean': env_penalty_mean,
            'env_penalty_max': env_penalty_max,
            'env_penalty_lt1_n': env_penalty_lt1_n,
            'env_penalty_set': env_penalty_set,

            # Aggregated affinity values in nM.
            'Ki': ki_agg, 'Kd': kd_agg, 'IC50': ic_agg, 'EC50': ec_agg,
            'Ki_str': Ki_str, 'Kd_str': Kd_str, 'IC50_str': IC50_str, 'EC50_str': EC50_str,

            # p-space values.
            'pKi': pKi, 'pKd': pKd, 'pIC50': pIC50, 'pEC50': pEC50,

            # Weights and replicate counts.
            'weight_Ki': wKi_final,
            'weight_Kd': wKd_final,
            'weight_IC50': wIc_final,
            'weight_EC50': wEc_final,
            'n_Ki': nKi, 'n_Kd': nKd, 'n_IC50': nIc, 'n_EC50': nEc,

            # Sign-audit fields.
            'Ki_sign_set': Ki_sign_set, 'Ki_sign': Ki_sign,
            'Kd_sign_set': Kd_sign_set, 'Kd_sign': Kd_sign,
            'IC50_sign_set': IC50_sign_set, 'IC50_sign': IC50_sign,
            'EC50_sign_set': EC50_sign_set, 'EC50_sign': EC50_sign,
        }

    with ThreadPoolExecutor(max_workers=_cfg_['max_workers']) as pool:
        futures = [pool.submit(agg_single, item) for item in groups_items]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Parallel aggregation"):
            res = fut.result()
            if res is not None:
                agg_list_parallel.append(res)

    res_df = pd.DataFrame(agg_list_parallel)
    print(f"[DONE] Aggregation completed after conflict filtering: {len(res_df)} rows")
    return res_df

# -------------------------- Optional ChEMBL SQLite lookups --------------------------
def _batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def chembl_query_inchikey_to_molecule_ids(db_path, inchikeys):
    """
    inchikeys: list of unique standard InChIKeys.
    return: dict[inchikey] -> sorted list[molecule_chembl_id]
    """
    if not inchikeys:
        return {}
    if not os.path.exists(db_path):
        print(f"[WARN] ChEMBL DB not found: {db_path} (skip molecule mapping)")
        return {}

    q_tpl = """
    SELECT cs.standard_inchi_key, md.molecule_chembl_id
    FROM compound_structures cs
    JOIN molecule_dictionary md ON cs.molregno = md.molregno
    WHERE cs.standard_inchi_key IN ({placeholders})
    """
    out = {}
    con = None
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        bs = int(_cfg_['sqlite_in_batch'])

        for batch in tqdm(list(_batched(inchikeys, bs)), desc="ChEMBL molecule lookup (by InChIKey)"):
            ph = ",".join(["?"] * len(batch))
            q = q_tpl.format(placeholders=ph)
            cur.execute(q, batch)
            rows = cur.fetchall()
            for ik, mid in rows:
                if ik is None or mid is None:
                    continue
                out.setdefault(ik, set()).add(mid)

        out2 = {k: sorted(list(v)) for k, v in out.items()}
        print(f"[INFO] ChEMBL molecule mapping hits: {len(out2)}/{len(inchikeys)} InChIKeys")
        return out2
    except Exception as e:
        logging.warning(f"ChEMBL inchikey->molecule query failed: {e}", exc_info=True)
        print(f"[WARN] ChEMBL molecule mapping query failed: {e}")
        return {}
    finally:
        try:
            if con is not None:
                con.close()
        except Exception:
            pass

def chembl_query_uniprot_to_target_ids(db_path, uniprots):
    """
    uniprots: list of unique UniProt accessions.
    return: dict[uniprot] -> sorted list[target_chembl_id]
    """
    if not uniprots:
        return {}
    if not os.path.exists(db_path):
        print(f"[WARN] ChEMBL DB not found: {db_path} (skip target mapping)")
        return {}

    q_tpl = """
    SELECT cs.accession AS uniprot_id, td.target_chembl_id
    FROM component_sequences cs
    JOIN target_components tc ON tc.component_id = cs.component_id
    JOIN target_dictionary td ON td.tid = tc.tid
    WHERE cs.accession IN ({placeholders})
    """
    out = {}
    con = None
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        bs = int(_cfg_['sqlite_in_batch'])

        for batch in tqdm(list(_batched(uniprots, bs)), desc="ChEMBL target lookup (by UniProt)"):
            ph = ",".join(["?"] * len(batch))
            q = q_tpl.format(placeholders=ph)
            cur.execute(q, batch)
            rows = cur.fetchall()
            for u, tid in rows:
                if u is None or tid is None:
                    continue
                out.setdefault(u, set()).add(tid)

        out2 = {k: sorted(list(v)) for k, v in out.items()}
        print(f"[INFO] ChEMBL target mapping hits: {len(out2)}/{len(uniprots)} UniProts")
        return out2
    except Exception as e:
        logging.warning(f"ChEMBL uniprot->target query failed: {e}", exc_info=True)
        print(f"[WARN] ChEMBL target mapping query failed: {e}")
        return {}
    finally:
        try:
            if con is not None:
                con.close()
        except Exception:
            pass

def _split_tokens_pipe(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return []
    return [t.strip() for t in s.split('|') if t.strip() and t.strip().lower() != 'nan']

def _map_multi_tokens_to_pipe(token_field, mapping_dict):
    """
    token_field: a pipe-delimited field such as "P12345|Q99999" or a single token.
    mapping_dict: token -> list[str]
    return: a stable, sorted, pipe-delimited string after mapping and de-duplication.
    """
    toks = _split_tokens_pipe(token_field)
    if not toks:
        return ""
    acc = set()
    for t in toks:
        vals = mapping_dict.get(t)
        if vals:
            for v in vals:
                acc.add(v)
    return "|".join(sorted(acc)) if acc else ""

# -------------------------- Main workflow --------------------------
def run(src, tgt):
    try:
        check_weight_symmetry()

        bdb = read_bindingdb(src)
        if len(bdb)==0:
            logging.error("No valid data after reading TSV - exiting")
            return

        # Preserve raw endpoint strings for qualifier parsing.
        for c in ['Ki','Kd','IC50','EC50']:
            if c in bdb.columns:
                bdb[f'{c}_raw'] = bdb[c]

        print("\n[INFO] Parsing affinity qualifiers and assigning endpoint-specific weights...")

        def parse_and_assign(colname):
            parsed = bdb[f'{colname}_raw'].apply(parse_apprx_value_with_balanced_weights)
            bdb[colname]             = parsed.apply(lambda x: x[0])
            bdb[f'{colname}_type']   = parsed.apply(lambda x: x[2])
            # If an endpoint value is missing, its weight is set to NaN so it does not affect endpoint-level summaries.
            w = parsed.apply(lambda x: x[1])
            w = w.where(bdb[colname].notna(), np.nan)
            bdb[f'weight_{colname}'] = w

        if 'Ki_raw'   in bdb.columns: parse_and_assign('Ki')
        if 'Kd_raw'   in bdb.columns: parse_and_assign('Kd')
        if 'IC50_raw' in bdb.columns: parse_and_assign('IC50')
        if 'EC50_raw' in bdb.columns: parse_and_assign('EC50')

        bdb = convert_other_floats(bdb)

        # Apply environmental penalties to endpoint-specific weights.
        bdb['env_penalty'] = bdb.apply(compute_env_penalty, axis=1)
        for wcol, vcol in [('weight_Ki','Ki'), ('weight_Kd','Kd'), ('weight_IC50','IC50'), ('weight_EC50','EC50')]:
            if wcol in bdb.columns:
                # Apply the penalty only to records where the endpoint value exists.
                m = bdb[vcol].notna() & bdb[wcol].notna()
                bdb.loc[m, wcol] = bdb.loc[m, wcol] * bdb.loc[m, 'env_penalty']

        analyze_weight_distribution(bdb, "Raw data weights after environmental penalty")
        analyze_inequality_distribution(bdb, "Raw data")
        analyze_affinity_distribution(bdb, "Raw data")

        # Filtering.
        bdb_f = filter_invalid_data(bdb, save_filtered_path=tgt)
        if len(bdb_f)==0:
            logging.error("No valid data after filtering - exiting")
            return

        analyze_weight_distribution(bdb_f, "Filtered data weights")
        analyze_inequality_distribution(bdb_f, "Filtered data")
        analyze_affinity_distribution(bdb_f, "Filtered data")

        # Construct metric_kind in a fixed endpoint order: Ki, Kd, IC50, EC50.
        def infer_metric_kind_row(row):
            kinds = []
            if not pd.isna(row.get('Ki')):
                kinds.append('Ki')
            if not pd.isna(row.get('Kd')):
                kinds.append('Kd')
            if not pd.isna(row.get('IC50')):
                kinds.append('IC50')
            if not pd.isna(row.get('EC50')):
                kinds.append('EC50')
            return "|".join(kinds) if kinds else "NA"

        bdb_f['metric_kind'] = bdb_f.apply(infer_metric_kind_row, axis=1)

        # Aggregation.
        df_agg = aggregate_constants(bdb_f)
        if len(df_agg)==0:
            logging.error("No valid data after aggregation - exiting")
            return

        # Generate standard InChIKeys after aggregation to avoid repeated computation.
        print("\n[INFO] Computing standard_inchi_key from aggregated SMILES...")
        df_agg['standard_inchi_key'] = df_agg['smiles'].apply(smiles_to_inchikey)

        # Optional ChEMBL mapping by on-demand SQLite queries.
        df_agg['chembl_molecule_chembl_id'] = ""
        df_agg['chembl_target_chembl_id'] = ""
        if _cfg_.get('use_chembl_mapping', True):
            db_path = _cfg_.get('chembl_db_path', CHEMBL_DB_PATH_DEFAULT)
            if os.path.exists(db_path):
                # molecule mapping
                inchikeys = sorted(set([x for x in df_agg['standard_inchi_key'].dropna().astype(str).tolist() if x.strip()]))
                inchi2mols = chembl_query_inchikey_to_molecule_ids(db_path, inchikeys)

                # Target mapping; uniprot_id can contain pipe-delimited accessions.
                uniq_uniprots = set()
                for v in df_agg['uniprot_id'].tolist():
                    for t in _split_tokens_pipe(v):
                        uniq_uniprots.add(t)
                uniq_uniprots = sorted(list(uniq_uniprots))
                uni2tgts = chembl_query_uniprot_to_target_ids(db_path, uniq_uniprots)

                df_agg['chembl_molecule_chembl_id'] = df_agg['standard_inchi_key'].map(
                    lambda k: "|".join(inchi2mols.get(str(k), [])) if pd.notna(k) else ""
                )
                df_agg['chembl_target_chembl_id'] = df_agg['uniprot_id'].map(
                    lambda u: _map_multi_tokens_to_pipe(u, uni2tgts)
                )
            else:
                print(f"[WARN] ChEMBL DB not found at: {db_path} (skip ChEMBL mapping columns)")

        analyze_weight_distribution(df_agg, "Aggregated weights: max single-record weight x replicate confidence")
        analyze_affinity_distribution(df_agg, "Aggregated affinities")

        out = f"{tgt}.csv"
        out_dir = os.path.dirname(os.path.abspath(out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df_agg.to_csv(out, index=False, columns=[
            # Recommended merge-key candidates.
            'standard_inchi_key',
            'chembl_molecule_chembl_id',

            # Original/core identifiers.
            'smiles', 'uniprot_id', 'chembl_target_chembl_id', 'protein', 'group_size',

            # Environmental audit columns.
            'pH_n', 'pH_min', 'pH_median', 'pH_max', 'pH_out_n', 'pH_set',
            'T_n',  'T_min',  'T_median',  'T_max',  'T_out_n',  'T_set',
            'env_penalty_n', 'env_penalty_min', 'env_penalty_mean', 'env_penalty_max',
            'env_penalty_lt1_n', 'env_penalty_set',

            # Numeric values and sign-preserving readable fields.
            'Ki', 'Kd', 'IC50', 'EC50',
            'Ki_str', 'Kd_str', 'IC50_str', 'EC50_str',

            # p-space values.
            'pKi', 'pKd', 'pIC50', 'pEC50',

            # Weights and replicate counts.
            'weight_Ki', 'weight_Kd', 'weight_IC50', 'weight_EC50',
            'n_Ki', 'n_Kd', 'n_IC50', 'n_EC50',

            # Sign-audit fields.
            'Ki_sign_set', 'Ki_sign',
            'Kd_sign_set', 'Kd_sign',
            'IC50_sign_set', 'IC50_sign',
            'EC50_sign_set', 'EC50_sign',
        ])

        print("\n[DONE] Final results")
        print(f"   Saved {len(df_agg)} rows to: {out}")
        print("   Recommended merge key: (standard_inchi_key, uniprot_id)")

    except Exception as e:
        logging.critical(f"Fatal error in main process: {str(e)}", exc_info=True)
        print("\n[ERROR] Processing failed. Check bindingdb_process_errors.log for details.")
        print(f"   Error: {e}")

# -------------------------- CLI --------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python confmux_dta_bindingdb_preprocess.py <input_tsv> <output_prefix_without_csv>")
        print("Example: python confmux_dta_bindingdb_preprocess.py BindingDB_All.tsv preprocessed/confmux_dta_bindingdb_agg")
        sys.exit(1)

    print("="*70)
    print("[INFO] ConfMux-DTA BindingDB preprocessing: Ki/Kd/IC50 with optional EC50, sign audit, endpoint weights, environmental audit, and optional ChEMBL mapping")
    print("="*70)
    print("[INFO] Records are aggregated only within the same metric-kind combination, e.g. Ki|IC50|EC50.")
    print(f"[INFO] A group is discarded if any endpoint exceeds {_cfg_['spread_log10_tol']} log10 units in p-space spread.")
    print("[INFO] Out-of-range pH/temperature values reduce endpoint weights but do not directly remove records.")
    print("[INFO] Replicate support: endpoint weight = max(single-record weight) x (1 + log10(n_metric)).")
    print("[INFO] Output includes environmental summaries, standard_inchi_key, and optional ChEMBL molecule/target identifiers.")
    print("="*70)

    src = sys.argv[1]
    tgt = sys.argv[2]
    run(src, tgt)