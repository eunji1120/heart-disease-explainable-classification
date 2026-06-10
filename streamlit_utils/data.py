"""Cached data loaders + cached derived computations.

Heavy work (calibration curves, subgroup metrics, threshold sweeps) is wrapped
in @st.cache_data so it runs once per session, not once per interaction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "outputs" / "dashboard"


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


@st.cache_data
def load_cohort() -> pd.DataFrame:
    return _csv("cohort_cleaned.csv")


@st.cache_data
def load_metrics_aggregate() -> pd.DataFrame:
    return _csv("model_metrics_aggregate.csv")


@st.cache_data
def load_metrics_per_fold() -> pd.DataFrame:
    return _csv("model_metrics_per_fold.csv")


@st.cache_data
def load_predictions() -> pd.DataFrame:
    return _csv("predictions.csv")


@st.cache_data
def load_curves() -> pd.DataFrame:
    return _csv("model_curves.csv")


@st.cache_data
def load_threshold_grid() -> pd.DataFrame:
    return _csv("threshold_grid.csv")


@st.cache_data
def load_shap_global() -> pd.DataFrame:
    return _csv("shap_global.csv")


@st.cache_data
def load_shap_patient() -> pd.DataFrame:
    return _csv("shap_patient.csv")


@st.cache_data
def best_model_name() -> str:
    agg = load_metrics_aggregate()
    auc = agg[(agg["metric_name"] == "roc_auc") & (agg["model_type"] != "dummy")]
    return auc.sort_values("metric_value", ascending=False).iloc[0]["model_type"]


# ---------------------------------------------------------------------------
# Cached derived computations — these used to recompute on every interaction
# ---------------------------------------------------------------------------
@st.cache_data
def calibration_data() -> pd.DataFrame:
    """One row per (model, bin) with prob_pred, prob_true, brier — for the
    reliability diagram. Runs once, not on every slider move."""
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    preds = load_predictions()
    rows = []
    for model_type, sub in preds.groupby("model_type"):
        if model_type == "dummy":
            continue
        y = sub["true_label"].to_numpy()
        p = sub["predicted_probability"].to_numpy()
        prob_true, prob_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        brier = brier_score_loss(y, p)
        for pt, pp in zip(prob_true, prob_pred):
            rows.append({"model_type": model_type, "prob_pred": float(pp),
                         "prob_true": float(pt), "brier": float(brier)})
    return pd.DataFrame(rows)


@st.cache_data
def subgroup_metrics(model_type: str) -> pd.DataFrame:
    """ROC AUC + recall per (subgroup_type, subgroup level) for one model."""
    from sklearn.metrics import recall_score, roc_auc_score

    preds = load_predictions()
    cohort = load_cohort()
    sub = preds[preds["model_type"] == model_type].merge(
        cohort[["patient_id", "age_band"]], on="patient_id", how="left"
    )
    rows = []
    for grouping in ["sex", "age_band", "cp"]:
        for level, gsub in sub.groupby(grouping):
            if gsub["true_label"].nunique() < 2 or len(gsub) < 10:
                continue
            auc = roc_auc_score(gsub["true_label"], gsub["predicted_probability"])
            yhat = (gsub["predicted_probability"] >= 0.5).astype(int)
            rec = recall_score(gsub["true_label"], yhat, zero_division=0)
            rows.append({
                "subgroup_type": grouping,
                "subgroup": str(level),
                "n": len(gsub),
                "roc_auc": round(auc, 3),
                "recall_at_0.5": round(rec, 3),
            })
    return pd.DataFrame(rows)


@st.cache_data
def threshold_sweep(model_type: str) -> pd.DataFrame:
    """Precompute precision/recall/F1/confusion-cell counts across a threshold
    grid for one model — so the slider only does a lookup, not a recompute."""
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    preds = load_predictions()
    sub = preds[preds["model_type"] == model_type]
    y = sub["true_label"].to_numpy()
    p = sub["predicted_probability"].to_numpy()
    thresholds = np.arange(0.05, 1.00, 0.05)
    rows = []
    for t in thresholds:
        yhat = (p >= t).astype(int)
        tp = int(((yhat == 1) & (y == 1)).sum())
        fp = int(((yhat == 1) & (y == 0)).sum())
        tn = int(((yhat == 0) & (y == 0)).sum())
        fn = int(((yhat == 0) & (y == 1)).sum())
        rows.append({
            "threshold": round(float(t), 2),
            "accuracy":  round(accuracy_score(y, yhat), 4),
            "precision": round(precision_score(y, yhat, zero_division=0), 4),
            "recall":    round(recall_score(y, yhat, zero_division=0), 4),
            "f1":        round(f1_score(y, yhat, zero_division=0), 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })
    return pd.DataFrame(rows)


@st.cache_data
def model_leaderboard() -> pd.DataFrame:
    """Long → wide aggregate metrics, one row per model with ROC AUC, F1, Brier."""
    agg = load_metrics_aggregate()
    wide = agg.pivot(index="model_type", columns="metric_name",
                     values="metric_value").round(3).reset_index()
    keep = ["model_type", "roc_auc", "pr_auc", "accuracy", "precision",
            "recall", "f1", "brier_score"]
    return wide[[c for c in keep if c in wide.columns]].sort_values("roc_auc", ascending=False)


@st.cache_data
def key_insights() -> dict:
    """Pre-baked insight numbers — used by Overview's insight callouts."""
    cohort = load_cohort()
    shap = load_shap_global()
    best = best_model_name()
    metrics = load_metrics_aggregate()
    best_row = metrics[metrics["model_type"] == best].set_index("metric_name")["metric_value"]

    # Top 3 risk drivers (LR)
    lr = shap[shap["model_type"] == "logistic_regression"].sort_values("rank_position")
    top3 = lr.head(3)["feature_name"].tolist()

    # Disease prevalence by sex
    by_sex = cohort.groupby("sex")["target_binary"].mean().to_dict()

    # Age band with highest risk
    risk_by_band = cohort.groupby("age_band")["target_binary"].mean()
    high_band = risk_by_band.idxmax()
    high_rate = risk_by_band.max()

    # Imputation audit
    n_imputed = int(((cohort["ca_was_imputed"] == 1) | (cohort["thal_was_imputed"] == 1)).sum())

    return {
        "best_model": best,
        "roc_auc": float(best_row["roc_auc"]),
        "brier": float(best_row["brier_score"]),
        "recall": float(best_row["recall"]),
        "top3_features": top3,
        "male_prev": float(by_sex.get("male", 0)),
        "female_prev": float(by_sex.get("female", 0)),
        "highest_risk_band": high_band,
        "highest_risk_rate": float(high_rate),
        "n_imputed": n_imputed,
    }
