"""Render docs/data_quality_report.md from the latest data_quality_summary batch.

The report is built from the most recent run_timestamp in data_quality_summary
plus the imputation audit columns on cleaned_patient_records, so it always
reflects the actual checks that ran (no hand-written claims that can drift).
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pandas as pd
from sqlalchemy import text

from src.db import PROJECT_ROOT, get_engine

REPORT_PATH = PROJECT_ROOT / "docs" / "data_quality_report.md"


# Mapping from raw check_type to a (human header, action-when-pass, action-when-fail) triple.
ACTION_MAP = {
    "row_count":           ("Row count",          "Matches expected (303 records)",                   "Investigate row drift"),
    "duplicate_rows":      ("Duplicate rows",     "No duplicates",                                    "Deduplicate"),
    "missing_count":       ("Missing values",     "No missing",                                       "Imputed + audit flag set"),
    "unexpected_category": ("Category validity",  "All values within expected set",                   "Review new category"),
    "out_of_range":        ("Clinical range",     "All values within clinically plausible range",     "Review out-of-range record"),
    "unexpected_value":    ("Value validity",     "All values within expected set",                   "Review unexpected value"),
    "outlier_iqr":         ("IQR outliers",       "No 1.5×IQR outliers",                              "Review (informational, not blocking)"),
}

CHECK_ORDER = [
    "row_count",
    "duplicate_rows",
    "missing_count",
    "unexpected_category",
    "unexpected_value",
    "out_of_range",
    "outlier_iqr",
]


def _load_latest_batch(engine) -> tuple[pd.DataFrame, pd.Timestamp]:
    with engine.connect() as conn:
        latest = conn.execute(text(
            "SELECT MAX(run_timestamp) FROM data_quality_summary"
        )).scalar_one()
    if latest is None:
        raise RuntimeError("data_quality_summary is empty — run src.dq_and_clean first.")
    df = pd.read_sql(
        text("""
            SELECT column_name, check_type, check_value, check_detail, passed
            FROM data_quality_summary
            WHERE run_timestamp = :ts
            ORDER BY check_type, column_name
        """),
        engine,
        params={"ts": latest},
    )
    return df, pd.Timestamp(latest)


def _load_imputation_audit(engine) -> dict[str, list[int]]:
    sql = text("""
        SELECT patient_id, ca_was_imputed, thal_was_imputed
        FROM cleaned_patient_records
        WHERE ca_was_imputed = 1 OR thal_was_imputed = 1
        ORDER BY patient_id
    """)
    df = pd.read_sql(sql, engine)
    return {
        "ca": df.loc[df["ca_was_imputed"] == 1, "patient_id"].tolist(),
        "thal": df.loc[df["thal_was_imputed"] == 1, "patient_id"].tolist(),
    }


def _result_cell(check_type: str, column_name: str,
                 check_value: float | None, check_detail: str | None) -> str:
    if check_type == "row_count":
        return f"{int(check_value)} records"
    if check_type == "duplicate_rows":
        return f"{int(check_value)} duplicates"
    if check_type == "missing_count":
        return f"{int(check_value)} missing"
    if check_type in ("unexpected_category", "unexpected_value"):
        return f"{int(check_value)} unexpected ({check_detail})"
    if check_type == "out_of_range":
        return f"{int(check_value)} out of range ({check_detail})"
    if check_type == "outlier_iqr":
        return f"{int(check_value)} flagged ({check_detail})"
    return f"{check_value}"


def _action_cell(check_type: str, column_name: str, passed: bool,
                 check_value: float | None,
                 imputed_ids: dict[str, list[int]]) -> str:
    pass_action, fail_action = ACTION_MAP[check_type][1], ACTION_MAP[check_type][2]
    if check_type == "missing_count" and column_name in imputed_ids and imputed_ids[column_name]:
        # Special case: imputed columns are described with the patient_ids touched
        ids = imputed_ids[column_name]
        return f"Imputed for patient_ids {ids}; audit flag set"
    if check_type == "outlier_iqr":
        # Outliers are informational. Pass status is always True, but the action
        # text depends on whether any outliers were actually flagged.
        if check_value and check_value > 0:
            return ("Retained — values are clinically plausible extremes "
                    "(no removal; tree models are robust)")
        return pass_action
    return pass_action if passed else fail_action


def build_report() -> str:
    engine = get_engine()
    df, run_ts = _load_latest_batch(engine)
    imputed_ids = _load_imputation_audit(engine)

    total = len(df)
    n_passed = int(df["passed"].sum())
    n_failed = total - n_passed

    lines: list[str] = []
    lines.append("# Data Quality Report")
    lines.append("")
    lines.append(f"- **Source dataset:** `processed.cleveland.data` (UCI Heart Disease, Cleveland subset)")
    lines.append(f"- **DQ batch run timestamp (UTC):** {run_ts.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Total checks:** {total}")
    lines.append(f"- **Passed:** {n_passed}")
    lines.append(f"- **Failed (action required):** {n_failed}")
    lines.append("")
    lines.append("This report is generated from `data_quality_summary` in MySQL by "
                 "`src/dq_report.py`. To refresh: rerun `src.dq_and_clean` then "
                 "`src.dq_report`.")
    lines.append("")

    # Summary table by check type
    lines.append("## Summary by check type")
    lines.append("")
    lines.append("| Check type | # checks | Passed | Failed |")
    lines.append("| --- | ---: | ---: | ---: |")
    grouped = df.groupby("check_type")["passed"].agg(["count", "sum"])
    for ct in CHECK_ORDER:
        if ct not in grouped.index:
            continue
        n = int(grouped.loc[ct, "count"])
        n_p = int(grouped.loc[ct, "sum"])
        n_f = n - n_p
        lines.append(f"| {ACTION_MAP[ct][0]} | {n} | {n_p} | {n_f} |")
    lines.append("")

    # Detail tables per check type
    for ct in CHECK_ORDER:
        sub = df[df["check_type"] == ct]
        if sub.empty:
            continue
        lines.append(f"## {ACTION_MAP[ct][0]}")
        lines.append("")
        lines.append("| Quality check | Result | Status | Action |")
        lines.append("| --- | --- | :---: | --- |")
        for _, row in sub.iterrows():
            col = row["column_name"]
            result = _result_cell(ct, col, row["check_value"], row["check_detail"])
            passed = bool(row["passed"])
            status = "Passed" if passed else "Failed"
            action = _action_cell(ct, col, passed, row["check_value"], imputed_ids)
            label = "Table" if col == "__table__" else f"`{col}`"
            lines.append(f"| {label} | {result} | {status} | {action} |")
        lines.append("")

    # Imputation audit
    lines.append("## Imputation audit")
    lines.append("")
    lines.append("Per the `ca_was_imputed` / `thal_was_imputed` audit columns on "
                 "`cleaned_patient_records`:")
    lines.append("")
    lines.append(f"- **`ca`** — median imputation applied to {len(imputed_ids['ca'])} rows "
                 f"(patient_ids: {imputed_ids['ca']})")
    lines.append(f"- **`thal`** — mode imputation applied to {len(imputed_ids['thal'])} rows "
                 f"(patient_ids: {imputed_ids['thal']})")
    lines.append("")
    lines.append("Both imputers will also be present inside the modelling "
                 "`sklearn` Pipeline as a safety net, so any future input row "
                 "with a missing `ca` / `thal` is handled identically.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    report = build_report()
    REPORT_PATH.write_text(report)
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)} ({len(report):,} bytes)")


if __name__ == "__main__":
    main()
