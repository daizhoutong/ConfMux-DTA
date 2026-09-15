# Revision reproducibility code

This is the code-only companion to the corrected ConfMux-DTA revision archive:
[10.5281/zenodo.22747289](https://doi.org/10.5281/zenodo.22747289), `ConfMux_revision_code_and_data.zip`.

`code/`, `environment/`, the historical analysis scripts, and the command/map files are copied byte-for-byte from that archive. Their SHA256 values are recorded in `ARCHIVE_FILES_SHA256.json`. Datasets, fixed split arrays, saved predictions, trained models and result figures are deliberately omitted from GitHub.

## Reproduce from one download

From the repository root:

```bash
python data/get_data.py revision --extract
cd data/extracted/revision
python code/verify_release.py --full --report ../ConfMux_verification.json
```

The full archive preserves all original relative paths. **Run `RUN_COMMANDS.md` examples from that extracted root**, not from this incomplete source companion. The original archive README remains inside the download. There is no need to copy datasets individually into this GitHub directory.

## Code map

- `code/confmux_dta_train_predict_leakage_controlled.py`: controlled full model.
- `code/confmux_major_design_ablation.py`, `confmux_bpe_ridge_ablation.py`: major-design and Ridge controls.
- `code/confmux_dta_train_predict_scale_controlled.py`: nested 50% training-scale control.
- `code/deepdta_full_controlled.py`: matched DeepDTA-style comparator.
- `code/confmux_sigma_head_only.py`, `confmux_uncertainty_evaluation.py`: uncertainty analysis.
- `code/confmux_temporal_prepare_and_run.py`, `confmux_temporal_shift_audit.py`: temporal evaluation and descriptive shift audit.
- `code/rebuild_release_outputs.py`, `plot_release_davis.py`, `verify_release.py`: corrected outputs and saved-result verification.
- `code/run_release.py`: dry-run-first task launcher; only `--execute` runs a task.
- `historical_reproducibility/`: historical plotting/analysis scripts, clearly separate from the current corrected outputs.

See [RUN_COMMANDS.md](RUN_COMMANDS.md), [environment/ENVIRONMENTS.md](environment/ENVIRONMENTS.md) and [reproducibility_map.tsv](reproducibility_map.tsv) for detailed scope. No training, inference or Slurm job runs when you merely download this code.
