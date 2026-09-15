# Manuscript and response synchronization

This repair updates code/data/figures, not the Word manuscript or response.
Apply the following bounded changes together. Do not change other experiment
counts or models to make them resemble these historical results.

## Table 5: DAVIS row (frozen predictions)
Use the row with protocol=frozen_no_external_label_refit in
results/revision_fixed/Table5_DAVIS_protocol_comparison.csv.

N=30,060
R2=0.5654588893342064 (display 0.565)
RMSE=0.5913741474795986 (display 0.591)
CI=0.8709186245491355 (display 0.871)
Rm2=0.4897356738523223 (display 0.490)
Absolute error <=1 pK: 27,047/30,060 = 89.9767% (display 90.0%)

Rm2 follows the historical implementation's predictive-R2 convention, not
squared Pearson r. Preserve the method definition consistently.

## Supplementary Figure S2
Replace the image with results/revision_fixed/Supplementary_Fig_S2_frozen.pdf
(or its 600-dpi PNG). Its numbers are derived from the same CSV as the Table 5 row.

Suggested caption:
"Supplementary Figure S2. Observed and predicted pKd values for the DAVIS
cross-dataset benchmark (n=30,060). Predictions were obtained from the frozen
model without fitting a calibration to DAVIS labels. The dashed line denotes
identity. Overlap with the original training corpus was not independently
audited; this benchmark is not claimed to be sample-disjoint."

## External-evaluation Methods / Results
Suggested clarification:
"We report the deposited frozen DAVIS predictions without refitting a
calibration on the evaluation labels. Historical same-set post-hoc calibration
results are retained only as a clearly labelled descriptive analysis and are
not used as the primary external predictive-performance estimate."

If retaining old R2=0.593/RMSE=0.572, label them as same-evaluation-set post-hoc
calibration. The disclosure about unverified training-data overlap does not
replace disclosure that evaluation labels were used for calibration.

## Supplementary Table S4
Use results/revision_fixed/Supplementary_Table_S4_posthoc_corrected.csv.
The displayed calibrated half-widths are 0.846, 0.530, 0.751, 1.309, 1.437.
Do not replace them with the separately retained raw half-widths.

Suggested note:
"These five examples were selected for illustration from a historical
externally post-calibrated candidate table. The reported interval half-width
equals (U-L)/2 on the calibrated scale; the pre-calibration width is retained
separately in the source data. External labels contributed to the historical
post-hoc calibration and error-based example selection. These examples do not
estimate independent external predictive performance."

The frozen temporal aggregate (R2=0.600486, RMSE=0.918632) is a DIFFERENT pipeline
and is unchanged. Do not relabel that aggregate as having the S4 post-hoc fit.

## Response amendment
"We rechecked the provenance of the external-evaluation outputs. We have
separated frozen DAVIS predictions from historical same-set post-hoc calibration
and now report the frozen-prediction metrics consistently in Table 5 and
Supplementary Figure S2. We also corrected the Supplementary Table S4 source
field mapping so that its half-width is on the same calibrated scale as its
interval endpoints, and explicitly identified those rows as selected historical
post-hoc illustrations. The revised reproducibility package includes a single
export path and automated numerical and file-integrity checks."

Use this response wording only after synchronizing the manuscript/caption/table.
Code-package validation alone does not establish Word-document consistency.

