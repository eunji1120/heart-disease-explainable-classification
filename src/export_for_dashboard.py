"""Step 11 — Export the dashboard-ready datasets to CSV.

The Streamlit dashboard cannot connect to a local MySQL instance once deployed
to Streamlit Cloud, so we extract the 8 datasets it needs into UTF-8 CSV
files under ``data/outputs/dashboard/``. Each CSV is shaped for one or two
specific dashboard sheets so no further data wrangling is needed.

Outputs (all flat, single-sheet CSVs):

  cohort_cleaned.csv          — one row per patient (the 303 cleaned records)
  model_metrics_aggregate.csv — long format: (model_type, metric_name, metric_value)
  model_metrics_per_fold.csv  — long format with `fold` column for boxplots
  predictions.csv             — flat join of model_predictions + patient features
  model_curves.csv            — ROC and PR curve points for every model
  threshold_grid.csv          — best-model metrics over a threshold sweep
  shap_global.csv             — global mean(|SHAP|) per (model, feature)
  shap_patient.csv            — patient × feature × model SHAP with feature_value

Re-run anytime to refresh from the latest MySQL state.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_curve,
)
from sqlalchemy import text

from src.db import PROJECT_ROOT, get_engine

OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "dashboard"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _latest_run_ids(engine) -> pd.DataFrame:
    sql = text("""
        SELECT mr.model_run_id, mr.model_type, mr.run_timestamp
        FROM model_runs mr
        JOIN (
            SELECT model_type, MAX(model_run_id) AS max_id
            FROM model_runs GROUP BY model_type
        ) latest
          ON latest.model_type = mr.model_type
         AND latest.max_id     = mr.model_run_id
        ORDER BY mr.run_timestamp
    """)
    return pd.read_sql(sql, engine)


def _save(df: pd.DataFrame, name: str) -> Path:
    path = OUT_DIR / name
    df.to_csv(path, index=False, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def export_cohort(engine) -> Path:
    df = pd.read_sql(
        "SELECT patient_id, age, sex, cp, trestbps, chol, fbs, restecg, "
        "thalach, exang, oldpeak, slope, ca, thal, num, target_binary, "
        "ca_was_imputed, thal_was_imputed "
        "FROM cleaned_patient_records ORDER BY patient_id",
        engine,
    )
    df["target_label"] = df["target_binary"].map({0: "no disease", 1: "disease"})
    df["age_band"] = pd.cut(
        df["age"], bins=[0, 39, 49, 59, 69, 120],
        labels=["<40", "40-49", "50-59", "60-69", "70+"],
    ).astype(str)
    return _save(df, "cohort_cleaned.csv")


def export_metrics_aggregate(engine, latest: pd.DataFrame) -> Path:
    ids = tuple(int(i) for i in latest["model_run_id"])
    sql = text("""
        SELECT mr.model_type, mtr.metric_name, mtr.metric_value
        FROM model_training_results mtr
        JOIN model_runs mr ON mr.model_run_id = mtr.model_run_id
        WHERE mtr.fold IS NULL AND mtr.model_run_id IN :ids
        ORDER BY mr.model_type, mtr.metric_name
    """).bindparams(ids=ids)
    df = pd.read_sql(sql, engine)
    return _save(df, "model_metrics_aggregate.csv")


def export_metrics_per_fold(engine, latest: pd.DataFrame) -> Path:
    ids = tuple(int(i) for i in latest["model_run_id"])
    sql = text("""
        SELECT mr.model_type, mtr.fold, mtr.metric_name, mtr.metric_value
        FROM model_training_results mtr
        JOIN model_runs mr ON mr.model_run_id = mtr.model_run_id
        WHERE mtr.fold IS NOT NULL AND mtr.model_run_id IN :ids
        ORDER BY mr.model_type, mtr.metric_name, mtr.fold
    """).bindparams(ids=ids)
    df = pd.read_sql(sql, engine)
    return _save(df, "model_metrics_per_fold.csv")


def export_predictions(engine, latest: pd.DataFrame) -> Path:
    ids = tuple(int(i) for i in latest["model_run_id"])
    sql = text("""
        SELECT mr.model_type,
               mp.patient_id, mp.fold,
               mp.predicted_class, mp.predicted_probability, mp.true_label,
               c.age, c.sex, c.cp, c.chol, c.thalach, c.oldpeak, c.ca, c.thal,
               c.target_binary
        FROM model_predictions mp
        JOIN model_runs mr             ON mr.model_run_id = mp.model_run_id
        JOIN cleaned_patient_records c ON c.patient_id    = mp.patient_id
        WHERE mp.model_run_id IN :ids
        ORDER BY mr.model_type, mp.patient_id
    """).bindparams(ids=ids)
    df = pd.read_sql(sql, engine)
    df["predicted_label"] = df["predicted_class"].map({0: "no disease", 1: "disease"})
    df["true_label_str"] = df["true_label"].map({0: "no disease", 1: "disease"})
    df["outcome"] = np.where(
        (df["true_label"] == 1) & (df["predicted_class"] == 1), "true positive",
        np.where((df["true_label"] == 0) & (df["predicted_class"] == 0), "true negative",
        np.where((df["true_label"] == 0) & (df["predicted_class"] == 1), "false positive",
                                                                          "false negative")))
    return _save(df, "predictions.csv")


def export_model_curves(predictions_path: Path) -> Path:
    df = pd.read_csv(predictions_path)
    rows = []
    for model_type, sub in df.groupby("model_type"):
        if model_type == "dummy":
            continue
        y = sub["true_label"].to_numpy()
        p = sub["predicted_probability"].to_numpy()
        # ROC
        fpr, tpr, _ = roc_curve(y, p)
        for f, t in zip(fpr, tpr):
            rows.append({"model_type": model_type, "curve": "ROC",
                         "x_value": float(f), "y_value": float(t)})
        # PR
        prec, rec, _ = precision_recall_curve(y, p)
        for r, pr in zip(rec, prec):
            rows.append({"model_type": model_type, "curve": "PR",
                         "x_value": float(r), "y_value": float(pr)})
    return _save(pd.DataFrame(rows), "model_curves.csv")


def export_threshold_grid(predictions_path: Path, best_model: str) -> Path:
    df = pd.read_csv(predictions_path)
    sub = df[df["model_type"] == best_model]
    y = sub["true_label"].to_numpy()
    p = sub["predicted_probability"].to_numpy()
    rows = []
    for t in np.arange(0.05, 1.00, 0.05):
        yhat = (p >= t).astype(int)
        tp = int(((yhat == 1) & (y == 1)).sum())
        fp = int(((yhat == 1) & (y == 0)).sum())
        tn = int(((yhat == 0) & (y == 0)).sum())
        fn = int(((yhat == 0) & (y == 1)).sum())
        rows.append({
            "model_type": best_model,
            "threshold": round(float(t), 2),
            "accuracy":  round(accuracy_score(y, yhat), 4),
            "precision": round(precision_score(y, yhat, zero_division=0), 4),
            "recall":    round(recall_score(y, yhat, zero_division=0), 4),
            "f1":        round(f1_score(y, yhat, zero_division=0), 4),
            "true_positive":  tp,
            "false_positive": fp,
            "true_negative":  tn,
            "false_negative": fn,
        })
    return _save(pd.DataFrame(rows), "threshold_grid.csv")


def export_shap_global(engine, latest: pd.DataFrame) -> Path:
    ids = tuple(int(i) for i in latest["model_run_id"])
    sql = text("""
        SELECT mr.model_type, sgi.feature_name, sgi.mean_abs_shap, sgi.rank_position
        FROM shap_global_importance sgi
        JOIN model_runs mr ON mr.model_run_id = sgi.model_run_id
        WHERE sgi.model_run_id IN :ids
        ORDER BY mr.model_type, sgi.rank_position
    """).bindparams(ids=ids)
    return _save(pd.read_sql(sql, engine), "shap_global.csv")


def export_shap_patient(engine, latest: pd.DataFrame) -> Path:
    ids = tuple(int(i) for i in latest["model_run_id"])
    sql = text("""
        SELECT mr.model_type, spl.patient_id, spl.feature_name, spl.feature_value,
               spl.shap_value,
               CASE WHEN spl.shap_value > 0 THEN 'increases risk'
                    WHEN spl.shap_value < 0 THEN 'decreases risk'
                    ELSE 'neutral' END AS direction,
               mp.predicted_class, mp.predicted_probability, mp.true_label
        FROM shap_patient_level spl
        JOIN model_runs mr        ON mr.model_run_id   = spl.model_run_id
        JOIN model_predictions mp ON mp.model_run_id   = spl.model_run_id
                                 AND mp.patient_id    = spl.patient_id
        WHERE spl.model_run_id IN :ids
        ORDER BY mr.model_type, spl.patient_id, ABS(spl.shap_value) DESC
    """).bindparams(ids=ids)
    df = pd.read_sql(sql, engine)
    df["predicted_label"] = df["predicted_class"].map({0: "no disease", 1: "disease"})
    df["true_label_str"] = df["true_label"].map({0: "no disease", 1: "disease"})
    return _save(df, "shap_patient.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _pick_best_model(engine, latest: pd.DataFrame) -> str:
    ids = tuple(int(i) for i in latest["model_run_id"])
    sql = text("""
        SELECT mr.model_type, mtr.metric_value AS roc_auc
        FROM model_training_results mtr
        JOIN model_runs mr ON mr.model_run_id = mtr.model_run_id
        WHERE mtr.fold IS NULL AND mtr.metric_name = 'roc_auc'
          AND mtr.model_run_id IN :ids
    """).bindparams(ids=ids)
    df = pd.read_sql(sql, engine)
    df = df[df["model_type"] != "dummy"]
    return df.sort_values("roc_auc", ascending=False).iloc[0]["model_type"]


def main() -> None:
    engine = get_engine()
    latest = _latest_run_ids(engine)
    best = _pick_best_model(engine, latest)
    print(f"Latest runs: {latest['model_type'].tolist()}")
    print(f"Best model (by OOF ROC AUC): {best}\n")

    paths = []
    paths.append(export_cohort(engine))
    paths.append(export_metrics_aggregate(engine, latest))
    paths.append(export_metrics_per_fold(engine, latest))
    paths.append(export_predictions(engine, latest))
    paths.append(export_model_curves(paths[-1]))
    paths.append(export_threshold_grid(paths[-2], best))
    paths.append(export_shap_global(engine, latest))
    paths.append(export_shap_patient(engine, latest))

    print("CSVs written:")
    for p in paths:
        size_kb = p.stat().st_size / 1024
        print(f"  {p.relative_to(PROJECT_ROOT)} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
