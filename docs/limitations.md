# Model Card and Limitations Report (Skeleton)

This document will be filled in as the project progresses. It serves as both
the project's **model card** (model intent, evaluation, caveats) and its
**limitations report**.

> ⚠️ This model is an educational prototype. It is **not** intended for clinical
> diagnosis, screening, or treatment recommendation.

## 1. Intended Use

- **In-scope use:** demonstrating a reproducible health data analytics
  workflow — data quality review, classification modelling, explainability,
  and dashboard reporting on the UCI Cleveland Heart Disease dataset.
- **Out-of-scope use:** any clinical decision-making, screening of real
  patients, generalization to populations outside the original Cleveland
  cohort (mid-1980s, single US institution, referred-for-angiography sample).

## 2. Data

- **Source:** UCI Heart Disease dataset, Cleveland subset (`processed.cleveland.data`).
- **Sample size:** 303 records.
- **Period and setting:** Cleveland Clinic Foundation, data collected through
  1988. Patients in this dataset were referred for coronary angiography, so
  the sample is **not** representative of the general population.
- **Missingness:** 4 missing in `ca`, 2 missing in `thal`. Imputation strategy
  documented in [data_dictionary.md](data_dictionary.md).
- **Known dataset issues:** historical, single-institution, possible coding
  drift between contributing sites, no demographic detail beyond `age` and
  `sex`.

## 3. Modelling Approach

*(To be filled in: model families considered, hyperparameter search strategy,
CV scheme, final model choice and why.)*

## 4. Performance

*(To be filled in after Step 8.)*

- Cross-validated metrics: ROC AUC, PR AUC, accuracy, sensitivity, specificity
  at threshold X.
- Calibration: reliability diagram and Brier score, with the explicit caveat
  that 303 rows is too small for a strict reliability assessment.
- Confusion matrix at the chosen operating point.

## 5. Explainability

*(To be filled in after Step 9.)*

- Global SHAP summary across folds.
- Local SHAP explanations for representative correctly-classified and
  misclassified cases.
- Comparison of SHAP rankings against published clinical literature on
  coronary artery disease risk factors.

## 6. Limitations

To be expanded; the major ones are already clear:

1. **Sample size.** 303 rows. Confidence intervals on every metric are wide;
   a single train/test split would be uninformative. Even with stratified
   k-fold CV, calibration metrics should be read as indicative, not definitive.
2. **Historical data.** Collected in the 1980s. Clinical practice, diagnostic
   coding, and patient demographics have shifted substantially.
3. **Selection bias.** Patients were referred for coronary angiography. The
   prevalence of disease in this sample (~46%) is far above the general
   population. Probability outputs from this model would be miscalibrated for
   any population-level use.
4. **Geographic and demographic narrowness.** Single US institution; no
   race/ethnicity, socio-economic, or comorbidity detail beyond what is in the
   14 columns.
5. **No external validation.** Even Hungarian / Switzerland / Long Beach VA
   subsets are out of scope here because of schema and missingness issues;
   true external validation would require a modern, harmonized cohort.
6. **Target simplification.** The 5-level severity scale (`num` 0–4) is
   collapsed to binary, which discards information about disease severity.

## 7. Ethical and Governance Notes

- No identifiable patient information is used. The processed Cleveland file
  has no names, IDs, or dates beyond `age`.
- This project does not constitute a medical device under any jurisdiction.
- If this workflow were to be applied to real PHI in a production setting,
  additional controls would be required: data use agreement, IRB / REB
  review, access controls on the MySQL backend, audit logging, bias and
  fairness analysis on protected attributes, and clinical-expert sign-off on
  the operating threshold.

## 8. Reproducibility

*(To be filled in: random seeds, package versions, how to re-run the pipeline.)*
