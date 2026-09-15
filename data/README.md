# Data and model access

This directory contains retrieval instructions and a small download helper, not the datasets themselves. Run the commands below from the repository root. Python >=3.9 is sufficient for the helper.

## Select one resource

```bash
python data/get_data.py --list
python data/get_data.py revision --check
python data/get_data.py revision --extract
```

`--check` reads public metadata only. Omit it to download. Other resource names are `bindingdb`, `davis` and `model`; downloads are explicit and never automatically expanded to all datasets.

The full revision ZIP is about 1.02 GB and expands to about 1.28 GB (decimal GB). Allow at least 3.5 GB of free disk space for the download, verification and extraction. It already includes the saved predictions needed for verification. You do not need the older large deposits merely to check those saved results.

## Locations

| Resource | Downloaded file | Extraction/use |
|---|---|---|
| revision | `data/downloads/ConfMux_revision_code_and_data.zip` | `data/extracted/revision/` with `--extract` |
| bindingdb | `data/downloads/confmux_dta_bindingdb_agg.csv` | Read directly; no extraction |
| davis | `data/downloads/DAVIS.csv` | Read directly; no extraction |
| model | `data/downloads/predict_bundle.zip` | `data/extracted/model/` with `--extract`; locate `predict_manifest.json` |

Use `--output-dir` or `--extract-dir` for another location. Existing downloads must match their checksums. The helper never overwrites an existing extraction directory. After a successful extraction, reuse that directory instead of extracting it again.

## Manual download

- Revision deposit: https://doi.org/10.5281/zenodo.22747289
- Public revision file: https://zenodo.org/records/22747289/files/ConfMux_revision_code_and_data.zip?download=1
- Canonical BindingDB data: https://doi.org/10.5281/zenodo.19903083
- DAVIS input: https://doi.org/10.5281/zenodo.19904695
- Prediction bundle: https://doi.org/10.5281/zenodo.20153878

The revision deposit is published. Use the public DOI/record page, not a logged-in `/draft/` API link. If a public endpoint becomes unavailable, check the DOI page and retry later; the helper never requests login credentials or substitutes a private draft URL.

For an already downloaded revision ZIP, including one with its previous long filename:

```bash
python data/get_data.py revision --archive /path/to/revision.zip --extract
```

This offline path checks the pinned SHA256 in `sources.json`, then extracts without executing code. Renaming a ZIP does not change its hash. If the ZIP contents are changed, its identity and verification must be updated deliberately; the helper does not accept a mismatch silently.

## After extraction

```bash
cd data/extracted/revision
python code/verify_release.py --full --report ../ConfMux_verification.json
```

Consult the extracted archive's `RUN_COMMANDS.md` for reconstruction and training. The archive verifier needs the full fixed manifests/results, so it must not be run from the code-only GitHub `revision/` directory.

Downloaded datasets, model objects, outputs and local path files are excluded by the root `.gitignore`. Do not upload them to GitHub. Pickle/joblib/PT model files should only be loaded from trusted deposits in the documented environment.
