# Data Dictionary — Cleveland Heart Disease Dataset

**Source:** UCI Machine Learning Repository — Heart Disease dataset
(`processed.cleveland.data`).
**Rows:** 303.
**Columns used:** 14 (13 features + 1 target). The original raw archive has 76
columns; published experiments use this 14-column subset.
**Missing value marker in source:** `?` (the `.names` file mentions `-9.0`, but
that applies to the unprocessed files; the `processed.cleveland.data` file uses `?`).
**Missing values present:** 4 in `ca`, 2 in `thal`. No other columns have
missing values.

## Variables

| #  | Name       | Description                                                          | Type        | Codes / Range                                                                                 | Missing | Treatment plan                                                  |
| -- | ---------- | -------------------------------------------------------------------- | ----------- | --------------------------------------------------------------------------------------------- | ------- | --------------------------------------------------------------- |
| 1  | `age`      | Age in years                                                         | Numeric     | 29–77 (observed)                                                                              | 0       | Keep as continuous; range check.                                |
| 2  | `sex`      | Sex                                                                  | Binary      | 1 = male, 0 = female                                                                          | 0       | Keep as 0/1.                                                    |
| 3  | `cp`       | Chest pain type                                                      | Categorical | 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic               | 0       | One-hot encode (no natural ordering across the four types).     |
| 4  | `trestbps` | Resting blood pressure (mm Hg) on admission                          | Numeric     | Clinical range ~80–200                                                                        | 0       | Keep as continuous; range check.                                |
| 5  | `chol`     | Serum cholesterol (mg/dl)                                            | Numeric     | Clinical range ~100–600                                                                       | 0       | Keep as continuous; range check.                                |
| 6  | `fbs`      | Fasting blood sugar > 120 mg/dl                                      | Binary      | 1 = true, 0 = false                                                                           | 0       | Keep as 0/1.                                                    |
| 7  | `restecg`  | Resting electrocardiographic results                                 | Categorical | 0 = normal, 1 = ST-T wave abnormality, 2 = probable / definite left ventricular hypertrophy   | 0       | One-hot encode.                                                 |
| 8  | `thalach`  | Maximum heart rate achieved                                          | Numeric     | Clinical range ~60–220                                                                        | 0       | Keep as continuous; range check.                                |
| 9  | `exang`    | Exercise-induced angina                                              | Binary      | 1 = yes, 0 = no                                                                               | 0       | Keep as 0/1.                                                    |
| 10 | `oldpeak`  | ST depression induced by exercise relative to rest                   | Numeric     | Typically 0.0–6.0                                                                             | 0       | Keep as continuous.                                             |
| 11 | `slope`    | Slope of the peak exercise ST segment                                | Categorical | 1 = upsloping, 2 = flat, 3 = downsloping                                                      | 0       | One-hot encode. (Has weak ordinal interpretation but small cardinality — one-hot is safer.) |
| 12 | `ca`       | Number of major vessels (0–3) colored by fluoroscopy                 | Ordinal     | 0, 1, 2, 3                                                                                    | 4       | **Median imputation** (ordinal count). Keep as numeric.         |
| 13 | `thal`     | Thalassemia status                                                   | Categorical | 3 = normal, 6 = fixed defect, 7 = reversable defect                                           | 2       | **Mode imputation** (categorical with non-meaningful codes), then one-hot encode. |
| 14 | `num`      | Diagnosis of heart disease (severity)                                | Target      | 0 = no presence, 1–4 = increasing severity                                                    | 0       | See target design below.                                        |

### Imputation note

Imputers are fit **inside the sklearn Pipeline** during cross-validation, so the
imputed values for each fold are computed only from that fold's training data.
This prevents the test fold from leaking into the imputation step.

## Target Design

The original `num` is a 5-level severity label. Following the dataset's
documented convention, the modelling target is binarized:

```
target_binary = 0 if num == 0 else 1
```

- `target_binary = 0` → no presence of heart disease (`num == 0`)
- `target_binary = 1` → presence of heart disease (`num ∈ {1, 2, 3, 4}`)

### Class distribution (Cleveland)

| `num` | Count | `target_binary` |
| ----- | ----- | --------------- |
| 0     | 164   | 0               |
| 1     | 55    | 1               |
| 2     | 36    | 1               |
| 3     | 35    | 1               |
| 4     | 13    | 1               |

After binarization: **164 negative (54.1%) vs 139 positive (45.9%)** — roughly
balanced, so resampling techniques (SMOTE etc.) are not needed. Stratified
splits will preserve this ratio.

## Encoding Summary

| Variable    | Encoding strategy        |
| ----------- | ------------------------ |
| `age`, `trestbps`, `chol`, `thalach`, `oldpeak` | Standard scaling (for linear models) / pass-through (for tree models) |
| `sex`, `fbs`, `exang` | Already 0/1, pass-through  |
| `cp`, `restecg`, `slope`, `thal` | One-hot encode |
| `ca` | Median-imputed, treated as numeric ordinal |

## References

- UCI Heart Disease dataset: https://archive.ics.uci.edu/dataset/45/heart+disease
- `heart+disease/heart-disease.names` (in this repository) — original
  attribute documentation.
- Detrano, R. et al. (1989). *International application of a new probability
  algorithm for the diagnosis of coronary artery disease.* American Journal of
  Cardiology, 64, 304–310.
