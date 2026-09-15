# Reproduction commands for the corrected release

Run from the extracted supplement root. Original labels, predictions, trained
models and fixed split files are retained. Rebuilding figures/tables does not
train or fit a calibration.

## 1. Verify the supplied package (no additional Python packages)

python code/verify_release.py --full --report ../ConfMux_verification.json

Python 3.9 or newer is required. This verifies hashes, actual CLI option names,
fixed split integrity, corrected tables, frozen DAVIS/DeepDTA/temporal metrics,
conformal coverage and all historical S8 rows. It never loads pickle/PT models,
contacts a service, submits a job or runs training.

A passed check is not a claim of full training reproduction or independent
external generalization. See environment/ENVIRONMENTS.md and MANUSCRIPT_SYNC.md.

## 2. Rebuild corrected tables only

python code/rebuild_release_outputs.py --out-dir reproduction_runs/corrected_tables

The output directory must be new or empty. Inputs are the included frozen
predictions and candidate pool. Outputs are:
- DAVIS_frozen_points.csv
- DAVIS_evaluation_protocols.json
- Table5_DAVIS_protocol_comparison.csv
- Supplementary_Table_S4_posthoc_corrected.csv
- Table4_Figure5_validation_check.csv
- derivation_manifest.json

The S4 exporter matches InChIKey + complete target ID + endpoint. It does not
select new examples or require the absent full merged_calibrated.csv.

## 3. Rebuild the current S2 figure

Install the optional plotting dependency in a chosen environment:
python -m pip install -r environment/requirements_release_plot.txt

python code/plot_release_davis.py --points results/revision_fixed/DAVIS_frozen_points.csv --metrics results/revision_fixed/DAVIS_evaluation_protocols.json --out-pdf reproduction_runs/Supplementary_Fig_S2_frozen.pdf

Optional 600-dpi PNG export, if Poppler is available:
pdftoppm -r 600 -singlefile -png reproduction_runs/Supplementary_Fig_S2_frozen.pdf reproduction_runs/Supplementary_Fig_S2_frozen

Use this frozen figure in the manuscript. The historical calibrated S2 renderer
and figures are archived, not the current main evaluation.

## 4. Configure expensive or environment-dependent reproduction

cp -n PATH_CONFIG.example.sh PATH_CONFIG.local.sh
# Edit paths in PATH_CONFIG.local.sh, then:
source PATH_CONFIG.local.sh
cd "$CONFMUX_SUPPLEMENT_ROOT"
mkdir -p logs

Set PY_MAIN, PY_DEEPDTA, PY_DEEPDTAGEN and PY_HIST to the appropriate Python
executables. These can differ. Set CONFMUX_DATA to the separately deposited
canonical CSV. Other external inputs are needed only for their named tasks.
CONFMUX_THREADS defaults to 4; use an allocation-appropriate value. CPU time
is not expected to match historical 80/96-core or A800 timing on other hardware.

The launcher defaults to DRY RUN and prints the exact argv and missing inputs:
"$PY_MAIN" code/run_release.py --task full --dry-run

Check input paths without launching:
"$PY_MAIN" code/run_release.py --task full --dry-run --check-inputs

No task executes unless --execute is supplied. Failed tasks are not auto-retried.
Use a new --out-dir for a new experiment. --resume is explicit and supported
only for full, no_onehot, half_scale and deepdta.

## 5. Main and ablation runs

Choose ONE task, inspect its dry-run first, then explicitly execute it.
Available tasks:
- full: unified one-hot full model, seed 5313, random 80/20, train-linear calibration
- no_onehot: same base protocol without endpoint one-hot
- char_lgbm: character n-gram LightGBM
- endpoint_pIC50, endpoint_pKi, endpoint_pKd: endpoint-specific fits
- ridge: BPE Ridge, alpha=1, tol=1e-4, max_iter=1000
- half_scale: nested 50% training sample, subset seed 20260910; same validation

Example:
"$PY_MAIN" code/run_release.py --task half_scale --dry-run --check-inputs
"$PY_MAIN" code/run_release.py --task half_scale --execute

The full controlled split is 1,626,585 train / 406,647 validation.
Half-scale keeps 813,292 training records. Its reference directory is the shipped
manifests/full_controlled_split/, not a private server feature cache.
Original measured configuration is preserved under manifests/half_scale_split/.

## 6. DeepDTA

"$PY_DEEPDTA" code/run_release.py --task deepdta --dry-run --check-inputs

To submit deliberately to Slurm, after configuring paths and creating logs/:
sbatch --partition=gpu --gres=gpu:1 code/submit_deepdta_full_controlled.sbatch

The partition/GPU name and resources can be overridden with sbatch options.
Submit from the supplement root or export CONFMUX_SUPPLEMENT_ROOT.
The wrapper uses the exported root/SLURM_SUBMIT_DIR, not Slurm's spool path.
The exact fixed indices are included. The reimplementation is endpoint-conditioned
PyTorch DeepDTA-style, not a byte-identical legacy TensorFlow reproduction.
Checkpoints resume; automatic requeue is OFF unless AUTO_REQUEUE=1 is explicit.

## 7. Uncertainty

Re-evaluate the already supplied sigma predictions without training:
"$PY_MAIN" code/run_release.py --task uncertainty --dry-run --check-inputs
"$PY_MAIN" code/run_release.py --task uncertainty --execute

This uses seed 20260908 and the original 0.5 split (203,323 calibration /
203,324 evaluation records). The full release verifier independently recomputes
the 80/90/95% quantiles and covered counts from the shipped membership file.

Only if retraining the sigma head is desired:
- First run full and keep its config_used.json, train/valid predictions and cache.
- Set CONFMUX_FULL_RUN and CONFMUX_FEATURE_CACHE to that SAME completed run/cache.
- "$PY_MAIN" code/run_release.py --task sigma_head --dry-run --check-inputs
- "$PY_MAIN" code/run_release.py --task sigma_head --execute

The original full-validation descriptive uncertainty analysis and the later
disjoint calibration/evaluation audit are distinct. Model selection had already
used the broader validation partition; do not call this a pristine untouched test.

## 8. Historical S8 / temporal analysis

Using included frozen predictions (no new inference):
"$PY_HIST" code/run_release.py --task temporal_audit --dry-run --check-inputs
"$PY_HIST" code/run_release.py --task temporal_audit --execute

The historical S8 environment is recorded under historical_reproducibility/
s8_historical_inputs/environment/. It is not the newer main-training environment.
Historical train / validation counts are 1,626,597 / 406,650.

For optional regeneration from the separately deposited historical bundle:
set CONFMUX_HISTORICAL_BUNDLE, use that bundle's compatible environment, then:
"$PY_HIST" code/run_release.py --task temporal_predict --dry-run --check-inputs
"$PY_HIST" code/run_release.py --task temporal_predict --execute

The frozen temporal predictions were not recalibrated on temporal labels.
Do not confuse them with the external-label post-hoc S4 illustrations.

## 9. Historical Figure 5

The validation input and historical plotter are retained in
historical_reproducibility/fig5/. Its original RUN_COMMAND.txt records the
historical command. For portable reproduction from the package root:

mkdir -p reproduction_runs/fig5
ln -s "$(pwd)/historical_reproducibility/fig5/input/valid_predictions_for_paper.csv" reproduction_runs/fig5/valid_predictions_for_paper.csv
"$PY_HIST" historical_reproducibility/fig5/code/posthoc_best_model_full_plots_with_CI_R2m.py --run_dir reproduction_runs/fig5 --out_subdir plots --use_splits val

No --err_grid is supplied: this preserves the historical quantile-derived curve.
The dependency-free table rebuild in section 2 already checks the four manuscript
thresholds against the SAME 406,650 validation records. It does not use all-data
training predictions.

## 10. Optional DeepDTAGen benchmark

Set DEEPDTAGEN_ROOT to the directory with model.py, utils.py and FetterGrad.py.
Set DEEPDTAGEN_BINDINGDB_PT to the recorded third-party training file.
Set TOKENIZER_FILE only if the official layout needs an explicit tokenizer path.
See manifests/deepdtagen_speed_a800/THIRD_PARTY_SOURCE.md and the local file hashes.
Third-party model/data are not automatically downloaded.

"$PY_DEEPDTAGEN" code/run_release.py --task deepdtagen --dry-run --check-inputs
BATCH_SIZE=128 sbatch --partition=gpu --gres=gpu:a800:1 code/submit_deepdtagen_speed_benchmark.sbatch

The wrapper uses code/ paths and enables the included compatibility shim.
It measures 10 warm-up and 100 training steps plus 30 evaluation steps, not full
predictive retraining. Original batch 128 and projection counts are preserved.

## Integrity after deliberate edits

Use code/seal_release.py only when authoring a NEW release copy, after verifying
every intended edit. It recalculates inventories; it is NOT a repair for failed
checksums and must not be run to conceal a failed verification.
