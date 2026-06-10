# Data Quality Report

- **Source dataset:** `processed.cleveland.data` (UCI Heart Disease, Cleveland subset)
- **DQ batch run timestamp (UTC):** 2026-06-10 00:46:00
- **Total checks:** 35
- **Passed:** 35
- **Failed (action required):** 0

This report is generated from `data_quality_summary` in MySQL by `src/dq_report.py`. To refresh: rerun `src.dq_and_clean` then `src.dq_report`.

## Summary by check type

| Check type | # checks | Passed | Failed |
| --- | ---: | ---: | ---: |
| Row count | 1 | 1 | 0 |
| Duplicate rows | 1 | 1 | 0 |
| Missing values | 14 | 14 | 0 |
| Category validity | 7 | 7 | 0 |
| Value validity | 2 | 2 | 0 |
| Clinical range | 5 | 5 | 0 |
| IQR outliers | 5 | 5 | 0 |

## Row count

| Quality check | Result | Status | Action |
| --- | --- | :---: | --- |
| Table | 303 records | Passed | Matches expected (303 records) |

## Duplicate rows

| Quality check | Result | Status | Action |
| --- | --- | :---: | --- |
| Table | 0 duplicates | Passed | No duplicates |

## Missing values

| Quality check | Result | Status | Action |
| --- | --- | :---: | --- |
| `age` | 0 missing | Passed | No missing |
| `ca` | 4 missing | Passed | Imputed for patient_ids [167, 193, 288, 303]; audit flag set |
| `chol` | 0 missing | Passed | No missing |
| `cp` | 0 missing | Passed | No missing |
| `exang` | 0 missing | Passed | No missing |
| `fbs` | 0 missing | Passed | No missing |
| `num` | 0 missing | Passed | No missing |
| `oldpeak` | 0 missing | Passed | No missing |
| `restecg` | 0 missing | Passed | No missing |
| `sex` | 0 missing | Passed | No missing |
| `slope` | 0 missing | Passed | No missing |
| `thal` | 2 missing | Passed | Imputed for patient_ids [88, 267]; audit flag set |
| `thalach` | 0 missing | Passed | No missing |
| `trestbps` | 0 missing | Passed | No missing |

## Category validity

| Quality check | Result | Status | Action |
| --- | --- | :---: | --- |
| `cp` | 0 unexpected (Unexpected values: none) | Passed | All values within expected set |
| `exang` | 0 unexpected (Unexpected values: none) | Passed | All values within expected set |
| `fbs` | 0 unexpected (Unexpected values: none) | Passed | All values within expected set |
| `num` | 0 unexpected (Unexpected values: none) | Passed | All values within expected set |
| `restecg` | 0 unexpected (Unexpected values: none) | Passed | All values within expected set |
| `sex` | 0 unexpected (Unexpected values: none) | Passed | All values within expected set |
| `slope` | 0 unexpected (Unexpected values: none) | Passed | All values within expected set |

## Value validity

| Quality check | Result | Status | Action |
| --- | --- | :---: | --- |
| `ca` | 0 unexpected (Values not in {0,1,2,3,?}) | Passed | All values within expected set |
| `thal` | 0 unexpected (Values not in {3,6,7,?}) | Passed | All values within expected set |

## Clinical range

| Quality check | Result | Status | Action |
| --- | --- | :---: | --- |
| `age` | 0 out of range (Outside [18, 100]) | Passed | All values within clinically plausible range |
| `chol` | 0 out of range (Outside [80, 700]) | Passed | All values within clinically plausible range |
| `oldpeak` | 0 out of range (Outside [0, 7]) | Passed | All values within clinically plausible range |
| `thalach` | 0 out of range (Outside [50, 230]) | Passed | All values within clinically plausible range |
| `trestbps` | 0 out of range (Outside [70, 220]) | Passed | All values within clinically plausible range |

## IQR outliers

| Quality check | Result | Status | Action |
| --- | --- | :---: | --- |
| `age` | 0 flagged (Outside [28.5, 80.5] (Tukey 1.5*IQR)) | Passed | No 1.5×IQR outliers |
| `chol` | 5 flagged (Outside [115.0, 371.0] (Tukey 1.5*IQR)) | Passed | Retained — values are clinically plausible extremes (no removal; tree models are robust) |
| `oldpeak` | 5 flagged (Outside [-2.4, 4.0] (Tukey 1.5*IQR)) | Passed | Retained — values are clinically plausible extremes (no removal; tree models are robust) |
| `thalach` | 1 flagged (Outside [84.8, 214.8] (Tukey 1.5*IQR)) | Passed | Retained — values are clinically plausible extremes (no removal; tree models are robust) |
| `trestbps` | 9 flagged (Outside [90.0, 170.0] (Tukey 1.5*IQR)) | Passed | Retained — values are clinically plausible extremes (no removal; tree models are robust) |

## Imputation audit

Per the `ca_was_imputed` / `thal_was_imputed` audit columns on `cleaned_patient_records`:

- **`ca`** — median imputation applied to 4 rows (patient_ids: [167, 193, 288, 303])
- **`thal`** — mode imputation applied to 2 rows (patient_ids: [88, 267])

Both imputers will also be present inside the modelling `sklearn` Pipeline as a safety net, so any future input row with a missing `ca` / `thal` is handled identically.
