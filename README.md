# Explainable Heart Disease Risk Classification Pipeline

End-to-end health-data classification workflow using the UCI Cleveland Heart
Disease dataset: a MySQL-backed pipeline, six cross-validated `sklearn`
models with calibration analysis, three layers of SHAP explainability, and
a Tableau Public dashboard for stakeholder review.

> ⚠️ **Educational prototype. Not for clinical decision-making.**

| | |
| --- | --- |
| Live dashboard | _(will be added after Tableau Public is published)_ |
| Model card | [docs/model_card.md](docs/model_card.md) |
| Data dictionary | [docs/data_dictionary.md](docs/data_dictionary.md) |
| DQ report | [docs/data_quality_report.md](docs/data_quality_report.md) |
| Tableau build guide | [docs/tableau_guide.md](docs/tableau_guide.md) |

---

## 1. Project Overview

This project builds a reproducible health-data classification workflow that:

1. Classifies whether a patient record indicates presence of heart disease
   (binary target derived from a 5-level diagnosis severity field).
2. Evaluates **both discrimination and calibration** — because a health-risk
   classifier is consumed for predicted probability, not just the class
   label.
3. Explains predictions at **three levels**: global SHAP feature importance,
   summary-level direction of effect, and patient-level drill-down.
4. Delivers results to non-technical stakeholders through a Tableau Public
   dashboard backed by CSV exports from a MySQL schema designed for full
   model-lineage auditability.

## 2. Health Data Context

| Aspect | Description |
| --- | --- |
| Clinical question | Can a model classify presence vs absence of heart disease from 13 clinical/diagnostic variables? |
| Target user (scenario) | Health-data / population-health analytics team |
| Decision context | Educational prototype, **not** a clinical decision tool |
| Output | Classification model, SHAP explanations, Tableau dashboard, model card |

## 3. Data Source

UCI Heart Disease dataset, **Cleveland subset** (`processed.cleveland.data`,
303 records). Citation:

> Detrano R., Janosi A., Steinbrunn W., Pfisterer M., Schmid J., Sandhu S.,
> Guppy K., Lee S., Froelicher V. (1989). *International application of a
> new probability algorithm for the diagnosis of coronary artery disease.*
> American Journal of Cardiology, 64, 304–310.

Source archive: <https://archive.ics.uci.edu/dataset/45/heart+disease>.

## 4. Privacy and Governance Note

This project uses a public, de-identified educational dataset. No direct
patient identifiers are present in the processed Cleveland file. The model
is **not** intended for clinical diagnosis or treatment recommendation. Its
purpose is to demonstrate a reproducible health-data analytics workflow
including data quality review, classification modelling, explainability,
and dashboard reporting.

The Tableau Public dashboard inherits these disclaimers and links back to
the model card on every page.

## 5. Target Definition

The original `num` field is a 5-level severity label (0 = no disease, 1–4 =
increasing severity). Following the dataset's documented convention, the
modelling target is binarized:

```
target_binary = 0 if num == 0 else 1
```

- `target_binary = 0` → no presence of heart disease (`num == 0`)
- `target_binary = 1` → presence of heart disease (`num ∈ {1, 2, 3, 4}`)

Resulting class balance: **164 (54.1%) negative, 139 (45.9%) positive**.

## 6. Data Dictionary

Full per-column documentation including expected ranges, encoding strategy,
and imputation policy: [docs/data_dictionary.md](docs/data_dictionary.md).

## 7. Data Quality Review

Six standard health-data DQ check families run on every load — missing
values, duplicates, value validity, category validity, clinical-range
plausibility, and Tukey IQR outlier counts. Results persisted to MySQL
`data_quality_summary` and rendered as a Markdown report:

→ [docs/data_quality_report.md](docs/data_quality_report.md) — 35 checks, 35 passing.

Imputation is audited per row: `cleaned_patient_records.ca_was_imputed` and
`thal_was_imputed` mark exactly which records had values filled in.

## 8. MySQL Data Architecture

| Table | Role |
| --- | --- |
| `raw_heart_disease` | Verbatim source rows, `?` preserved as string |
| `cleaned_patient_records` | Imputed, human-readable ENUM categories, target + audit flags |
| `data_quality_summary` | One row per (column, check_type, batch run_timestamp) |
| `model_runs` | One row per training run; **stores `git_sha`, seed, hyperparameters JSON** |
| `model_training_results` | Per-fold + aggregate metrics in long format |
| `model_predictions` | One row per (run, patient) out-of-fold prediction |
| `shap_global_importance` | Mean(\|SHAP\|) per (run, feature) |
| `shap_patient_level` | (run, patient, feature) with signed SHAP + `feature_value` |

Three views (`vw_dashboard_patient_summary`, `vw_dashboard_model_comparison`,
`vw_dashboard_shap_summary`) front the operational tables for downstream
BI consumption.

Schema definitions: [sql/01_schema.sql](sql/01_schema.sql) and
[sql/02_add_feature_value.sql](sql/02_add_feature_value.sql).

## 9. Modelling Workflow

`sklearn` Pipeline (`SimpleImputer` → numeric `StandardScaler` / categorical
`OneHotEncoder` → estimator) wrapped in `StratifiedKFold(n_splits=10,
random_state=42)`. Six classifiers compared (see [model card](docs/model_card.md)
§4). The preprocessing pipeline lives in [src/features.py](src/features.py);
the training loop in [src/train.py](src/train.py).

Out-of-fold predictions and per-fold metrics are written back to MySQL on
every run — **no overwrites**, every training run leaves a permanent audit
trail.

## 10. Model Evaluation

Both **discrimination** (ROC AUC, PR AUC) and **calibration** (reliability
diagram, Brier score) are evaluated, since predicted-probability quality
matters as much as class label in a health-risk context.

Selected model: **`logistic_regression`** — OOF ROC AUC **0.909**, Brier
**0.118**, recall 0.79 at threshold 0.5.

Full numbers, confusion matrix, calibration curves, per-fold variance, and
threshold-sensitivity sweep: [docs/model_card.md](docs/model_card.md) §7-8
and figures in [notebooks/figures/](notebooks/figures/) (`09a–09f`).

## 11. SHAP Explainability

Three layers, in line with the project policy that a strong-performing
model still needs interpretability before being consumed in a health
analytics setting:

1. **Global feature importance** — `shap_global_importance` table + bar
   chart (`10a_shap_global_*`).
2. **Direction-of-effect summary** — beeswarm plot per model
   (`10b_shap_summary_*`).
3. **Patient-level drill-down** — `shap_patient_level` table (with
   `feature_value` column for BI), waterfall examples for one correctly
   classified and one missed case (`10c_shap_waterfall_*`).

LR and RF agree on the top three drivers (`ca`, `cp`, `thal`) — concordant
with the cardiology literature on coronary-artery-disease risk factors.

## 12. Tableau Dashboard

Five-page Tableau Public dashboard fed by eight CSV exports from MySQL.
Pages: Project Overview → Cohort / Data Profile → Model Performance →
SHAP Global Explainability → Individual Record Explanation (drill-through).

- CSVs: [data/outputs/tableau/](data/outputs/tableau/) (generated by
  `python -m src.export_for_tableau`)
- Build guide: [docs/tableau_guide.md](docs/tableau_guide.md)

## 13. Key Findings

- **Logistic regression wins on a 303-row dataset.** Linear, regularized,
  well-calibrated, and ties on ROC AUC with the more complex tree models —
  an important lesson about model complexity vs sample size.
- **Top drivers match cardiology literature.** Number of fluoroscopy-
  coloured vessels (`ca`), chest-pain type (`cp`), and thalassemia result
  (`thal`) lead both linear and tree SHAP rankings.
- **Cholesterol is a weak independent signal here.** Both EDA correlations
  (≈ 0.09 with target) and SHAP rankings place it near the bottom — a
  finding consistent with modern CAD literature.
- **Probability calibration matters and was met.** LR's Brier of 0.118 and
  a near-diagonal reliability diagram mean the predicted probability can
  be reported, not just the class label.

## 14. Limitations

Summarized below; full version in [docs/model_card.md](docs/model_card.md) §10.

1. **n = 303** — confidence intervals on every metric are wide.
2. **Historical (1980s) Cleveland Clinic data** — clinical practice and
   demographics have shifted.
3. **Selection bias** — cohort was referred for coronary angiography;
   prevalence ≈ 5× general population.
4. **No external validation** — Hungarian / Switzerland / VA subsets
   deliberately out of scope.
5. **Binary target** discards severity information.
6. **No fairness analysis possible** — only `age` and `sex` available, no
   race / ethnicity / socio-economic detail.

## 15. Future Improvements

- External validation on a modern, harmonized cohort.
- Stakeholder-driven threshold tuning (current 0.5 is a default, not a
  recommendation).
- Sub-group performance breakdown by `age_band` and `sex`.
- Replace `chol` with LDL/HDL ratio if richer lipid panel becomes
  available.

---

## Repo layout

```
.
├── README.md
├── requirements.txt
├── .env.example                — copy to .env to wire up MySQL
│
├── data/
│   ├── raw/                    — UCI source files
│   ├── processed/              — derived artefacts
│   └── outputs/tableau/        — 8 CSVs for the dashboard
│
├── docs/
│   ├── data_dictionary.md
│   ├── data_quality_report.md  — generated from MySQL
│   ├── model_card.md
│   └── tableau_guide.md
│
├── sql/
│   ├── 01_schema.sql           — 8 tables + 3 BI views
│   └── 02_add_feature_value.sql — SHAP table migration
│
├── src/
│   ├── db.py                   — SQLAlchemy engine factory
│   ├── load_raw.py             — raw CSV → raw_heart_disease
│   ├── dq_and_clean.py         — DQ checks + cleaned_patient_records
│   ├── dq_report.py            — MySQL → docs/data_quality_report.md
│   ├── eda.py                  — 6 EDA figures
│   ├── features.py             — ColumnTransformer
│   ├── train.py                — 6-model 10-fold CV trainer
│   ├── evaluate.py             — 6 evaluation figures
│   ├── explain.py              — SHAP global + patient-level + figures
│   └── export_for_tableau.py   — MySQL → 8 dashboard CSVs
│
├── notebooks/
│   └── figures/                — EDA, evaluation, SHAP PNGs
│
└── scripts/
    └── uninstall_oracle_mysql.sh — one-time environment cleanup
```

## How to run

```bash
# 0. Create + activate venv, install deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Wire up MySQL (8.3+) and create the project DB/user
mysql -u root -p <<'EOF'
CREATE DATABASE heart_disease_project
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'heart_project_user'@'localhost' IDENTIFIED BY '<your-password>';
GRANT ALL PRIVILEGES ON heart_disease_project.* TO 'heart_project_user'@'localhost';
FLUSH PRIVILEGES;
EOF

# 2. Configure credentials
cp .env.example .env       # then edit .env with your password

# 3. Build the schema
mysql -u heart_project_user -p heart_disease_project < sql/01_schema.sql
mysql -u heart_project_user -p heart_disease_project < sql/02_add_feature_value.sql

# 4. Run the pipeline
python -m src.load_raw            # raw_heart_disease  (303 rows)
python -m src.dq_and_clean        # cleaned_patient_records + data_quality_summary
python -m src.dq_report           # docs/data_quality_report.md
python -m src.eda                 # 6 EDA figures
python -m src.train               # 6 models × 10 folds → MySQL
python -m src.evaluate            # 6 evaluation figures
python -m src.explain             # SHAP tables + 8 figures
python -m src.export_for_tableau  # 8 CSVs for the dashboard
```

## Skills demonstrated

| Skill | Where in this project |
| --- | --- |
| Python | data pipeline, EDA, modelling, SHAP, exports |
| pandas / numpy | data wrangling, audit flags, threshold sweeps |
| scikit-learn | Pipeline, ColumnTransformer, 6-model comparison, calibration |
| SHAP | global + patient-level explainability, LR + RF explainers |
| MySQL | raw / cleaned / model-output / SHAP tables + BI views |
| SQL | dashboard views, joins, threshold-grid aggregation |
| Tableau | stakeholder-facing 5-page dashboard |
| Health-data sense | privacy note, cohort definition, target design, DQ report |
| Communication | README, model card, limitations, build guide |
| Reproducibility | structured repo, `.env`-driven config, seed-locked CV, audit columns, `git_sha` per `model_run` |
