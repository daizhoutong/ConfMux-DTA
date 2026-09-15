# ConfMux-DTA

ConfMux-DTA predicts pIC50, pKi and pKd from compound SMILES and protein amino-acid sequences. It combines BPE/TF-IDF features, endpoint encoding and LightGBM, with uncertainty outputs for screening prioritization. No protein 3D structure is required.

This repository contains source code and instructions. Large datasets, model files and saved prediction results are distributed through Zenodo, not Git.

## Start here

| Goal | Entry point | Download needed |
|---|---|---|
| Predict new compound-target pairs | [Prediction](#prediction) | Existing prediction bundle |
| Check the revised paper's saved results | [Revision result verification](#revision-result-verification) | One complete revision ZIP |
| Retrain the controlled revision models | [Controlled revision training](#controlled-revision-training) | Revision ZIP and the inputs named for that experiment |
| Run the original training pipeline | [Original pipeline](#original-pipeline) | Canonical processed BindingDB data |

**Published revision archive:** [10.5281/zenodo.22747289](https://doi.org/10.5281/zenodo.22747289), file `ConfMux_revision_code_and_data.zip` (approximately 1.02 GB). This is one combined code-and-data download, not a collection of scripts that must be downloaded separately. [Download the revision ZIP](https://zenodo.org/records/22747289/files/ConfMux_revision_code_and_data.zip?download=1).

## Repository layout

```text
README.md
confmux_dta_bindingdb_preprocess.py   Original preprocessing entry point
confmux_dta_train_predict.py         Original training and prediction entry point
requirements.txt                    Original core dependencies
data/
    README.md                       What to download and where to put it
    get_data.py                     Selective download and checksum verification
    sources.json                    Versioned resource identifiers
revision/
    readme.md                       Revision workflow entry point
    code/                           Revision code copied from the fixed archive
    environment/                    Recorded environments and reconstruction notes
    historical_reproducibility/     Historical plotting/analysis code only
    RUN_COMMANDS.md                  Detailed commands for the complete archive
    PATH_CONFIG.example.sh
    reproducibility_map.tsv
    ARCHIVE_FILES_SHA256.json        Identities of files copied from the archive
```

Downloaded files go under `data/downloads/` and `data/extracted/`, both excluded from Git. The `revision/` directory is a browsable source companion; its datasets, fixed split arrays, model artifacts and results remain in the Zenodo ZIP. Run archive verification from the complete extracted archive, not the code-only `revision/` directory.

## Data and model downloads

| Resource | DOI | Needed for |
|---|---|---|
| Revision code, fixed splits, inputs and saved results | [22747289](https://doi.org/10.5281/zenodo.22747289) | Revision verification and controlled experiments |
| Canonical processed BindingDB data | [19903083](https://doi.org/10.5281/zenodo.19903083) | Model training |
| DAVIS input data | [19904695](https://doi.org/10.5281/zenodo.19904695) | A new DAVIS inference run |
| Previously deposited prediction bundle | [20153878](https://doi.org/10.5281/zenodo.20153878) | Prediction on new pairs |

List resources without downloading anything:

```bash
python data/get_data.py --list
```

Download only the resource you need, for example:

```bash
python data/get_data.py revision --extract
```

The helper uses public Zenodo URLs, verifies checksums and rejects unsafe ZIP paths. It does not log in, use private draft links, load model objects or start training. Reusing a downloaded file verifies its checksum; extraction requires a new destination. See [data/README.md](data/README.md) for manual downloads and local-archive use.

## Revision result verification

After downloading and extracting the revision ZIP:

```bash
cd data/extracted/revision
python code/verify_release.py --full --report ../ConfMux_verification.json
```

This requires Python 3.9 or newer and only the standard library. The corrected archive passed saved-result verification on Linux with Python 3.11.16. Checks cover file identities, split accounting and the specified metric/table reconstruction. They do not rerun full training, frozen-model inference or GPU benchmarking.

To rebuild corrected tables from included saved predictions:

```bash
python code/rebuild_release_outputs.py --out-dir reproduction_runs/corrected_tables
```

Use a new or empty output directory. Current corrected outputs belong to `results/revision_fixed/`; historical reference outputs are retained separately and must not replace the corrected figures/tables. The current S2 figure command and its optional dependencies are in `RUN_COMMANDS.md`.

## Environments

Use separate environments for different workflows; a successful standard-library verification is not evidence that an old prediction bundle works with arbitrary newer dependencies.

- **Saved-result verification:** Python >=3.9, no additional packages.
- **Original core pipeline and deposited prediction bundle:** retain the original dependency specification and the bundle's manifest. The historical README recommended Python 3.9.12. Install the root `requirements.txt` in an isolated environment; do not replace it with revision dependencies without compatibility testing.
- **Revision training:** follow [revision/environment/ENVIRONMENTS.md](revision/environment/ENVIRONMENTS.md) and the task-specific requirements. The supplied main requirements are a recorded revision environment, not a claim that every platform has been tested.
- **Historical plots, DeepDTA and DeepDTAGen:** their environments differ. Some files are reconstruction specifications rather than complete historical lockfiles; this is documented explicitly. GPU timing depends on matching hardware and settings.

Original core environment example:

```bash
conda create -n confmux_dta python=3.9.12
conda activate confmux_dta
python -m pip install -r requirements.txt
```

Conda is optional; an isolated compatible Python environment is sufficient. Install an appropriate RDKit build if it is unavailable through pip on your platform. Full training is CPU- and memory-intensive. Match thread counts to the resources allocated to your run; the examples below use four threads, not a universal timing recommendation.

## Prediction

Download the existing model once:

```bash
python data/get_data.py model --extract
```

Locate `predict_manifest.json` under `data/extracted/model/`; its containing directory is the bundle directory. Prepare your own input CSV with `smiles` and `protein` columns. An optional `affinity_type` column can contain `pIC50`, `pKi` or `pKd`.

```bash
python confmux_dta_train_predict.py predict \
  --bundle_dir /path/to/predict_bundle \
  --in input_pairs.csv \
  --out prediction_output.csv \
  --affinity_type pIC50 \
  --err_max 0.8
```

Pass the directory containing `predict_manifest.json`, not the JSON file itself. This directory must also contain the model, tokenizers, vectorizers and related metadata. Output names are defined by the bundle manifest; the unified prediction is typically `pred_pX`. Uncertainty bounds and Gaussian-derived screening scores are not guarantees of an individual prediction's accuracy. This revision does not replace the previously deposited prediction bundle.

## Original pipeline

To train from the canonical processed data:

```bash
python data/get_data.py bindingdb
python confmux_dta_train_predict.py train_test \
  --data_file data/downloads/confmux_dta_bindingdb_agg.csv \
  --split_mode random --unified \
  --total_iters 50000 --stage_iters 2000 --num_threads 4
```

The original entry point also supports `--split_mode compound`, `protein` and `pair`. Outputs are written beneath `confmux_dta_runs/`. These examples invoke the original pipeline; they are not the controlled revision protocol and must not be relabelled as its experiments.

To rebuild processed data from a locally obtained raw BindingDB TSV:

```bash
python confmux_dta_bindingdb_preprocess.py data/BindingDB_All.tsv preprocessed/confmux_dta_bindingdb_agg
```

The processed CSV is sufficient for most users. Raw-data preprocessing standardizes inputs and aggregates affinity records; optional external resources must follow the preprocessing script's own configuration.

## Controlled revision training

First download the canonical data and the full revision archive. From the extracted archive root, follow `RUN_COMMANDS.md`. In a Linux/Bash shell, copy `PATH_CONFIG.example.sh` to `PATH_CONFIG.local.sh`, edit the canonical CSV path and task-specific Python executables, load that configuration, then inspect a dry run:

```bash
source PATH_CONFIG.local.sh
"$PY_MAIN" code/run_release.py --task full --dry-run --check-inputs
```

Only an explicit `--execute` starts a task. The launcher covers the controlled full model, endpoint/design ablations, the 50% scale control, DeepDTA, uncertainty analysis and temporal analysis. Some tasks require additional external inputs. DeepDTAGen is an optional resource benchmark with separately identified third-party prerequisites, not a completed matched full-dataset accuracy comparison.

The canonical dataset has 2,033,247 records; routine model-input checks leave 2,033,232 eligible records for controlled revision experiments. Their fixed split is 1,626,585 training / 406,647 validation. Historical analyses using 406,650 validation rows are identified separately. Do not mix their memberships or metrics.

## Interpretation of corrected outputs

Primary DAVIS results use frozen predictions without fitting to DAVIS labels. Historical same-set post-hoc results are retained for traceability, not reported as independent external performance. Supplementary Table 4 contains historical, error-selected post-hoc examples; calibrated half-width is `(U - L) / 2`, distinct from the raw `err_bound` field. The complete archive documents these distinctions and their source files.

## Citation and versioning

Cite the manuscript when its publication details are available, the [revision archive DOI](https://doi.org/10.5281/zenodo.22747289), and the data/model deposits actually used. The revision code in this upload matches the corrected archive; the new README and download helper are a lightweight access layer. Record the Git commit used for a new experiment. Preserve applicable third-party code and data terms.
