# Explainable Heart Disease Risk Classification Pipeline

End-to-end health data classification and explainability workflow using the UCI
Cleveland Heart Disease dataset, with results delivered through a MySQL-backed
Power BI dashboard.

## Project Goal

Develop a reproducible health data classification workflow that:

1. Classifies whether a patient record indicates presence of heart disease
   (binary target derived from the original 5-level diagnosis).
2. Evaluates model performance beyond accuracy (ROC AUC, PR AUC, calibration,
   stratified cross-validation).
3. Explains model predictions globally and locally using SHAP.
4. Delivers results to non-technical stakeholders through a Power BI dashboard
   reading from a MySQL backend.

## Use Case

| Aspect                    | Description                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------------ |
| Clinical question         | Can a model classify presence vs absence of heart disease from clinical and diagnostic variables? |
| Target user (in scenario) | Health data analyst / population health analytics team                                     |
| Decision context          | Educational risk-classification prototype, not a clinical decision tool                    |
| Output                    | Classification model, SHAP explanations, Power BI dashboard, model card                    |

## Privacy and Governance Note

This project uses a public, de-identified educational dataset
([UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)).
No direct patient identifiers are present in the processed Cleveland file.
The model is **not** intended for clinical diagnosis or treatment
recommendation. Its purpose is to demonstrate a reproducible health data
analytics workflow including data quality review, classification modelling,
explainability, and dashboard reporting.

## Scope Decisions

- **Cleveland only** — the Hungarian, Switzerland, and Long Beach VA files use
  inconsistent schemas and have substantially more missingness. Combining them
  would weaken data quality without adding analytic value for this prototype.
- **Binary target** — the original `num` field ranges 0–4 (severity). Following
  the dataset's documented convention, this project predicts presence
  (`num >= 1`) vs absence (`num == 0`).
- **Small-sample evaluation** — with only 303 rows, the project uses stratified
  k-fold cross-validation for all reported metrics. Single train/test splits
  would be too noisy to be informative.

## Project Structure

```
.
├── data/
│   ├── raw/                  # Original UCI files (processed.cleveland.data)
│   └── processed/            # Cleaned outputs from the pipeline
├── docs/
│   ├── data_dictionary.md    # Variable definitions and treatment plan
│   └── limitations.md        # Model card / limitations report (skeleton)
├── notebooks/                # EDA and analysis notebooks
├── src/                      # Reusable pipeline code (imputation, models, SHAP)
├── sql/                      # MySQL schema and load scripts
├── dashboards/               # Power BI files (.pbix)
└── heart+disease/            # Original UCI archive (kept for reference)
```

## Workflow

1. Data governance and scope (this document)
2. Data dictionary ([docs/data_dictionary.md](docs/data_dictionary.md))
3. MySQL raw and cleaned tables
4. Data quality checks
5. Exploratory analysis
6. Feature engineering and preprocessing (inside a sklearn Pipeline to avoid
   CV leakage)
7. Classification models (logistic regression, random forest, gradient boosting)
8. Model evaluation and calibration
9. SHAP explainability (global and local)
10. Model output tables in MySQL
11. Power BI dashboard
12. Model card and limitations report ([docs/limitations.md](docs/limitations.md))

## Status

Step 1 (project scope, data dictionary, limitations skeleton) — in progress.
