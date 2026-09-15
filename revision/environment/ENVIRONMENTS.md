# Environment scope

## Verification and data-table reconstruction
Python >=3.9, standard library only. No model object is unpickled or executed.
The portable verifier is intentionally separate from model-training dependencies.

The current S2 PDF renderer uses reportlab==4.4.9. Poppler renders the optional
PNG; this is a figure-production dependency, not a model change.

## Main revision
requirements_confmux_revision.txt is the author-supplied recorded environment.
It was not upgraded or reinstalled during this repair. Historical results remain
associated with their recorded manifests; local data checks do not prove a
successful fresh installation or full training run.

## DeepDTA
The run manifest explicitly records PyTorch 2.7.1+cu126 and CUDA 12.6.
requirements_deepdta.txt now references the main requirements and torch==2.7.1
as a candidate reconstruction. A complete independent historical pip freeze was
not supplied. Do not describe this reconstruction as an independently recovered
complete freeze. Install a CUDA 12.6 wheel using the official PyTorch wheel source
when matching the recorded GPU environment, or a platform-appropriate wheel for
a new run. GPU timing is comparable only with matching hardware/settings.

## Historical S8 and plotting
requirements_temporal_audit.txt pins the documented historical S8 versions:
Python 3.9.12; NumPy 1.26.1; pandas 2.1.4; SciPy 1.12.0;
scikit-learn 1.3.2; Matplotlib 3.5.2; RDKit 2022.09.5.
PyPI spells the same RDKit release as 2022.9.5.

Use an isolated historical environment, not the newer main runtime. This list
covers the shift-audit/plotting scripts. To rerun the separately deposited frozen
prediction bundle, follow that bundle's own dependency specification as well.
Do not claim package-table recomputation reproduces bundle inference.

## DeepDTAGen
Keep the recorded third-party source hashes and dependencies. The optional
requirements are not a complete historical lockfile. Existing benchmark logs
record the measured A800 environment. No new full benchmark was run by this fix.

## Server receipt
The one-command installer writes verification_report.json, including the actual
Python version, platform, elapsed time, passed checks and work explicitly NOT run.
It does not install packages, access the network, or submit Slurm jobs.

