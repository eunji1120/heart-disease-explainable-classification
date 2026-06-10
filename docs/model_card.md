# Model Card — Explainable Heart Disease Risk Classification

> ⚠️ Educational prototype. **Not** intended for clinical diagnosis, screening,
> or treatment recommendation.

## 1. Model purpose

Demonstrate a reproducible health-data classification workflow on the UCI
Cleveland Heart Disease dataset: from raw load through data quality,
modelling, calibration, SHAP explainability, and dashboard reporting.

## 2. Dataset

| Property | Value |
| --- | --- |
| Source | UCI Heart Disease — Cleveland subset (`processed.cleveland.data`) |
| Cohort size | 303 records |
| Collection period | 1981 – 1984 (Detrano et al., Cleveland Clinic Foundation) |
| Features used | 13 (8 numeric + ordinal, 5 categorical) |
| Original target | `num` — 5-level severity 0–4 |
| Missing values in source | `ca` × 4 rows, `thal` × 2 rows; no other column |
| Imputation | `ca` median (0), `thal` mode (`normal`); per-row audit flags persisted |

Full variable definitions: [data_dictionary.md](data_dictionary.md).
Full DQ batch: [data_quality_report.md](data_quality_report.md)
(35 checks, all passing).

## 3. Target

The 5-level `num` field is binarized to a presence/absence label following
the dataset's documented convention:

```
target_binary = 0 if num == 0 else 1
```

Class balance after binarization: **164 (54.1%) negative, 139 (45.9%) positive**.
Resampling was not applied — classes are nearly balanced and stratified
splitting preserves the ratio in every fold.

## 4. Model lineup

Six classifiers were trained with **10-fold stratified cross-validation**
(seed = 42) inside a `sklearn` Pipeline (`SimpleImputer` → numeric
`StandardScaler` / categorical `OneHotEncoder` → estimator):

| model_run_id | Model | Role |
| ---: | --- | --- |
| 5 | `DummyClassifier(most_frequent)` | Absolute floor baseline |
| **6** | **`LogisticRegression` (L2)** | **Interpretable linear baseline → selected model** |
| 7 | `LogisticRegression` (L1) | Sparse linear / implicit feature selection |
| 8 | `RandomForestClassifier` (`n_estimators=300`, `min_samples_leaf=2`) | Non-linear ensemble |
| 9 | `HistGradientBoostingClassifier` (`early_stopping=True`) | Gradient boosting |
| 10 | `CalibratedClassifierCV` wrapping (9), isotonic, cv = 5 | Probability-calibrated GBM |

Overfit guards: explicit `min_samples_leaf` on the forest, `early_stopping`
on the GBM, L1 penalty on LR-L1, isotonic calibration on the GBM ensemble.
All metrics are computed from **out-of-fold predictions**, never on data
the model trained on.

## 5. Intended use

- Educational demonstration of an end-to-end health-data analytics workflow.
- Model card / dashboard artefact for portfolio review.
- Source for SHAP-based explainability examples.

## 6. **Not** intended use

- Clinical decision-making, diagnosis, screening, triage, or referral.
- Risk-scoring of any real patient cohort.
- Population-level prevalence estimation (this sample was referred for
  angiography — disease prevalence in source is ≈ 5× general-population).
- Generalization to non-Cleveland cohorts (Hungarian, Switzerland, and
  Long Beach VA subsets are deliberately out of scope).

## 7. Performance

All metrics computed from 10-fold stratified out-of-fold predictions
(n = 303). The selected model is **`logistic_regression`** (highest OOF
ROC-AUC, lowest Brier, best F1).

| Model | Accuracy | ROC AUC | PR AUC | Precision | Recall | F1 | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **logistic_regression** | **0.842** | **0.909** | **0.902** | 0.853 | **0.791** | **0.821** | **0.118** |
| logistic_regression_l1 | 0.835 | 0.908 | 0.899 | 0.847 | 0.784 | 0.813 | 0.119 |
| random_forest | 0.832 | 0.908 | 0.897 | 0.840 | 0.784 | 0.813 | 0.124 |
| hist_gradient_boosting | 0.802 | 0.888 | 0.872 | 0.806 | 0.748 | 0.776 | 0.138 |
| hist_gradient_boosting_calibrated | 0.809 | 0.886 | 0.871 | 0.812 | 0.755 | 0.782 | 0.137 |
| dummy (baseline) | 0.541 | 0.500 | 0.459 | 0.000 | 0.000 | 0.000 | 0.459 |

Per-fold variance (boxplot view in `notebooks/figures/09e_cv_metric_stability.png`)
is similar across the 5 real models — confidence intervals overlap, and the
LR ↔ RF ↔ LR-L1 ranking should not be over-interpreted at n = 303.

### Confusion matrix (logistic regression, threshold = 0.5)

|  | Predicted no disease | Predicted disease |
| --- | ---: | ---: |
| **Actual no disease (164)** | 145 (TN) | 19 (FP) |
| **Actual disease (139)** | 29 (FN) | 110 (TP) |

### Operating-point sensitivity

The default threshold = 0.5 yields recall ≈ 0.79. The threshold sweep
(`notebooks/figures/09f_threshold_sensitivity.png`) shows the trade-off
explicitly so a deployment team could pick a recall floor (e.g. 0.85)
with the resulting precision cost made visible.

## 8. Calibration

Calibration matters for any health risk-classifier because the predicted
probability — not just the class label — is what stakeholders consume.

| Model | Brier score | Visible deviation from diagonal? |
| --- | ---: | --- |
| logistic_regression | **0.118** | Minimal — well-calibrated |
| logistic_regression_l1 | 0.119 | Minimal |
| random_forest | 0.124 | Minor |
| hist_gradient_boosting | 0.138 | Visible at high-probability bins |
| hist_gradient_boosting_calibrated | 0.137 | Slight improvement, but limited by n = 303 |

See `notebooks/figures/09d_calibration_curves.png`.

Caveat: with 303 records distributed across 10 probability bins, each bin
holds ~30 records, so calibration estimates carry wide confidence
intervals. The figure should be read as **indicative**, not definitive.

## 9. Explainability

SHAP values were computed **out-of-fold** for the two top-performing
families (logistic regression via `LinearExplainer`, random forest via
`TreeExplainer`). One-hot-encoded SHAP contributions were summed back to
the 13 original features so the attribution matches the data dictionary.

### Global feature ranking (mean(|SHAP|))

| Rank | LR | RF |
| ---: | --- | --- |
| 1 | `ca` | `ca` |
| 2 | `cp` | `cp` |
| 3 | `thal` | `thal` |
| 4 | `sex` | `oldpeak` |
| 5 | `slope` | `thalach` |
| 6 | `exang` | `sex` |
| 7 | `oldpeak` | `exang` |
| 8 | `thalach` | `slope` |

The two model families **agree on the top three drivers** (number of
fluoroscopy-coloured vessels, chest-pain type, and thalassemia result),
which matches the cardiology literature on coronary-artery-disease risk
factors. Disagreement on positions 4 – 8 is small and matches the
EDA-derived correlations.

`chol` and `fbs` are at the bottom of every ranking — the EDA already
flagged cholesterol as a weak independent signal, and SHAP confirms it.

### Patient-level explanation

`shap_patient_level` stores one row per (model, patient, feature) with
the signed SHAP value, the source `feature_value`, and a direction tag.
This is the back-end for the dashboard's per-record drill-through page.
Example waterfalls for a correctly-identified disease case (TP) and a
missed case (FN) are saved under `notebooks/figures/10c_*`.

## 10. Limitations

1. **Sample size (n = 303).** Confidence intervals on every metric are
   wide. The LR ↔ RF ↔ LR-L1 ranking should not be interpreted as
   definitive — they are essentially tied within fold-level variance.
2. **Historical data.** Collected at Cleveland Clinic in the early 1980s;
   diagnostic conventions, demographics, and clinical practice have
   changed materially since.
3. **Selection bias.** The source cohort was *referred for coronary
   angiography*. Disease prevalence in the sample (≈ 46%) is far above the
   general population, so predicted probabilities cannot be used for
   population-level risk scoring without recalibration.
4. **No external validation.** Hungarian / Switzerland / Long Beach VA
   subsets are deliberately out of scope due to schema and missingness
   differences. True external validation would require a modern,
   harmonized cohort.
5. **Target simplification.** The 5-level `num` field is collapsed to a
   binary label, discarding ordinal information about disease severity.
6. **Demographic coverage.** Only `age` and `sex` available — no race,
   ethnicity, socio-economic status, or comorbidity detail. Fairness
   analyses by protected attributes are not possible from this dataset.

## 11. Ethical and governance notes

- No direct patient identifiers are present in the processed source file.
- This project does **not** constitute a medical device under any
  jurisdiction.
- The published interactive dashboard inherits the same disclaimers and
  links back to this model card.
- If this workflow were to be applied to real PHI in production:
  data-use agreement, REB/IRB review, access controls on the MySQL
  back-end, audit logging, bias and fairness analysis on protected
  attributes, and a clinician sign-off on the operating threshold would
  all be required.

## 12. Reproducibility

| Item | Value |
| --- | --- |
| `random_seed` | 42 (across all `model_runs`) |
| CV scheme | `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)` |
| Pipeline | `ColumnTransformer` defined in `src/features.py` |
| Source-of-truth code | `src/train.py`, `src/evaluate.py`, `src/explain.py` |
| `git_sha` per run | persisted in `model_runs.git_sha` |
| Hyperparameters per run | persisted as JSON in `model_runs.hyperparameters` |
| Python version | 3.9.6 |
| Key package versions | see `requirements.txt`; pinned in `.venv/` |

To re-run end-to-end: see `README.md` → *How to run*.

## 13. Future improvements

- External validation on a modern, harmonized cohort.
- Threshold tuning against a stakeholder-defined recall target.
- Sub-group performance breakdown by `age_band` and `sex`.
- Replace `chol` with a derived ratio (LDL/HDL) when richer panel data
  becomes available.
- Replace `CalibratedClassifierCV` with Beta calibration once n is large
  enough to support it reliably.
