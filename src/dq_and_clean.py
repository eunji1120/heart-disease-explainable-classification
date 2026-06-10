"""Data quality checks + build cleaned_patient_records.

Reads raw_heart_disease, runs DQ checks (writing one row per check to
data_quality_summary), then applies imputation + label decoding + binary target
and writes the result to cleaned_patient_records.

Idempotent: truncates cleaned_patient_records before re-inserting, and appends a
fresh run_timestamp batch to data_quality_summary so historical checks are kept.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sqlalchemy import text

from src.db import get_engine


# Decoding maps — translate numeric source values to human-readable categories.
# These mirror docs/data_dictionary.md.

SEX_MAP = {1: "male", 0: "female"}
CP_MAP = {
    1: "typical_angina",
    2: "atypical_angina",
    3: "non_anginal_pain",
    4: "asymptomatic",
}
RESTECG_MAP = {0: "normal", 1: "st_t_abnormality", 2: "lv_hypertrophy"}
SLOPE_MAP = {1: "upsloping", 2: "flat", 3: "downsloping"}
THAL_MAP = {3: "normal", 6: "fixed_defect", 7: "reversable_defect"}

# Expected category sets (used in DQ checks)
EXPECTED_VALUES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "num": {0, 1, 2, 3, 4},
}

# Plausible clinical ranges (used in DQ checks; failures are flagged, not blocked)
RANGES = {
    "age": (18, 100),
    "trestbps": (70, 220),
    "chol": (80, 700),
    "thalach": (50, 230),
    "oldpeak": (0, 7),
}


# DQ check accumulator

@dataclass
class Check:
    column_name: str
    check_type: str
    check_value: Optional[float]
    check_detail: Optional[str]
    passed: bool


def _collect_dq(df: pd.DataFrame) -> list[Check]:
    checks: list[Check] = []

    # Row count
    checks.append(Check("__table__", "row_count", float(len(df)),
                        f"Expected 303 (UCI Cleveland)", passed=(len(df) == 303)))

    # Duplicate rows over the 14 source columns
    dup_cols = [c for c in df.columns if c != "patient_id"]
    dup_count = int(df.duplicated(subset=dup_cols).sum())
    checks.append(Check("__table__", "duplicate_rows", float(dup_count),
                        "Duplicates on the 14 source columns", passed=(dup_count == 0)))

    # Missing values per column ('?' marker). Skip patient_id and the
    # MySQL-managed created_at timestamp — only the 14 source columns matter.
    SKIP_COLS = {"patient_id", "created_at"}
    for col in df.columns:
        if col in SKIP_COLS:
            continue
        miss = int((df[col].astype(str) == "?").sum())
        checks.append(Check(col, "missing_count", float(miss),
                            "Count of '?' values", passed=True))

    # Unexpected categories
    for col, expected in EXPECTED_VALUES.items():
        # Treat as float-coerced int (source stores as '1.0', etc.)
        seen = set(pd.to_numeric(df[col], errors="coerce").dropna().astype(int).unique())
        unexpected = seen - expected
        checks.append(Check(col, "unexpected_category", float(len(unexpected)),
                            f"Unexpected values: {sorted(unexpected) if unexpected else 'none'}",
                            passed=(len(unexpected) == 0)))

    # Range checks (numeric only)
    for col, (lo, hi) in RANGES.items():
        vals = pd.to_numeric(df[col], errors="coerce")
        oor = int(((vals < lo) | (vals > hi)).sum())
        checks.append(Check(col, "out_of_range", float(oor),
                            f"Outside [{lo}, {hi}]", passed=(oor == 0)))

    # ca: should be 0,1,2,3 or '?'
    ca_vals = df["ca"].astype(str)
    bad_ca = int((~ca_vals.isin({"0.0", "1.0", "2.0", "3.0", "?"})).sum())
    checks.append(Check("ca", "unexpected_value", float(bad_ca),
                        "Values not in {0,1,2,3,?}", passed=(bad_ca == 0)))

    # thal: should be 3,6,7 or '?'
    thal_vals = df["thal"].astype(str)
    bad_thal = int((~thal_vals.isin({"3.0", "6.0", "7.0", "?"})).sum())
    checks.append(Check("thal", "unexpected_value", float(bad_thal),
                        "Values not in {3,6,7,?}", passed=(bad_thal == 0)))

    # Outliers via Tukey's IQR rule (informational — passed=True always; in the
    # DQ report the action is "review", not "block").
    for col in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        q1, q3 = vals.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        n_out = int(((vals < lo) | (vals > hi)).sum())
        checks.append(Check(col, "outlier_iqr", float(n_out),
                            f"Outside [{lo:.1f}, {hi:.1f}] (Tukey 1.5*IQR)",
                            passed=True))

    return checks


def _write_dq(checks: list[Check], engine, run_ts: dt.datetime) -> None:
    rows = [
        dict(
            run_timestamp=run_ts,
            column_name=c.column_name,
            check_type=c.check_type,
            check_value=c.check_value,
            check_detail=c.check_detail,
            passed=int(c.passed),
        )
        for c in checks
    ]
    pd.DataFrame(rows).to_sql("data_quality_summary", engine,
                              if_exists="append", index=False)



# Cleaning pipeline

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["patient_id"] = df["patient_id"].astype(int)

    # Numeric pass-through with type coercion
    out["age"] = pd.to_numeric(df["age"]).astype(int)
    out["trestbps"] = pd.to_numeric(df["trestbps"]).astype(int)
    out["chol"] = pd.to_numeric(df["chol"]).astype(int)
    out["thalach"] = pd.to_numeric(df["thalach"]).astype(int)
    out["oldpeak"] = pd.to_numeric(df["oldpeak"]).round(1)

    # Binary pass-through
    out["fbs"] = pd.to_numeric(df["fbs"]).astype(int)
    out["exang"] = pd.to_numeric(df["exang"]).astype(int)

    # Categorical decoding to ENUM labels
    out["sex"] = pd.to_numeric(df["sex"]).astype(int).map(SEX_MAP)
    out["cp"] = pd.to_numeric(df["cp"]).astype(int).map(CP_MAP)
    out["restecg"] = pd.to_numeric(df["restecg"]).astype(int).map(RESTECG_MAP)
    out["slope"] = pd.to_numeric(df["slope"]).astype(int).map(SLOPE_MAP)

    # ca: median imputation (ordinal count 0–3)
    ca_raw = pd.to_numeric(df["ca"], errors="coerce")
    ca_was_imputed = ca_raw.isna()
    ca_median = int(ca_raw.median())
    out["ca"] = ca_raw.fillna(ca_median).astype(int)
    out["ca_was_imputed"] = ca_was_imputed.astype(int)

    # thal: mode imputation (categorical, codes not ordered)
    thal_raw = pd.to_numeric(df["thal"], errors="coerce")
    thal_was_imputed = thal_raw.isna()
    thal_mode = int(thal_raw.mode().iloc[0])
    out["thal"] = thal_raw.fillna(thal_mode).astype(int).map(THAL_MAP)
    out["thal_was_imputed"] = thal_was_imputed.astype(int)

    # Target
    out["num"] = pd.to_numeric(df["num"]).astype(int)
    out["target_binary"] = (out["num"] >= 1).astype(int)

    column_order = [
        "patient_id", "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
        "target_binary", "ca_was_imputed", "thal_was_imputed",
    ]
    return out[column_order]


def run() -> dict:
    engine = get_engine()
    raw = pd.read_sql("SELECT * FROM raw_heart_disease ORDER BY patient_id", engine)

    run_ts = dt.datetime.utcnow().replace(microsecond=0)

    # DQ
    checks = _collect_dq(raw)
    _write_dq(checks, engine, run_ts)

    # Cleaning
    cleaned = _clean(raw)
    with engine.begin() as conn:
        # DELETE rather than TRUNCATE — cleaned_patient_records is referenced by
        # model_predictions and shap_patient_level via FK, and TRUNCATE is
        # forbidden on a referenced table even when the referencing tables are
        # empty.
        conn.execute(text("DELETE FROM cleaned_patient_records"))
        cleaned.to_sql("cleaned_patient_records", conn,
                       if_exists="append", index=False)

    with engine.connect() as conn:
        n_clean = conn.execute(text("SELECT COUNT(*) FROM cleaned_patient_records")).scalar_one()
        n_pos = conn.execute(text("SELECT SUM(target_binary) FROM cleaned_patient_records")).scalar_one()
        n_ca_imp = conn.execute(text("SELECT SUM(ca_was_imputed) FROM cleaned_patient_records")).scalar_one()
        n_thal_imp = conn.execute(text("SELECT SUM(thal_was_imputed) FROM cleaned_patient_records")).scalar_one()

    summary = {
        "dq_run_timestamp": run_ts.isoformat(),
        "dq_checks_written": len(checks),
        "dq_checks_failed": sum(1 for c in checks if not c.passed),
        "cleaned_rows": int(n_clean),
        "cleaned_positives": int(n_pos),
        "ca_imputed": int(n_ca_imp),
        "thal_imputed": int(n_thal_imp),
    }
    return summary


if __name__ == "__main__":
    s = run()
    print("==> DQ + clean complete")
    for k, v in s.items():
        print(f"  {k}: {v}")
