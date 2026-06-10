"""Load processed.cleveland.data into raw_heart_disease (verbatim).

Idempotent: truncates the table before re-inserting. patient_id is the 1-based
row index in the source file (1..303). Missing values ('?') are preserved as
strings in the ca and thal columns; everything else is inserted as-is and MySQL
casts to DECIMAL on store.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from src.db import PROJECT_ROOT, get_engine

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "processed.cleveland.data"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]


def load() -> int:
    df = pd.read_csv(RAW_FILE, header=None, names=COLUMNS, dtype=str)
    df.insert(0, "patient_id", range(1, len(df) + 1))

    engine = get_engine()
    with engine.begin() as conn:
        # DELETE rather than TRUNCATE because raw_heart_disease is referenced
        # by a foreign key from cleaned_patient_records.
        conn.execute(text("DELETE FROM raw_heart_disease"))
        df.to_sql("raw_heart_disease", conn, if_exists="append", index=False)

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM raw_heart_disease")).scalar_one()
    return int(count)


if __name__ == "__main__":
    n = load()
    print(f"Loaded {n} rows into raw_heart_disease")
