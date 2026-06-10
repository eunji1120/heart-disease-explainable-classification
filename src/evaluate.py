"""Step 9 — Model evaluation figures for the model card and Power BI.

Reads out-of-fold predictions and per-fold metrics from MySQL (so it never
re-trains; evaluation reflects exactly the stored model_run) and writes:

  notebooks/figures/
    09a_confusion_matrices.png      — confusion matrix per model
    09b_roc_curves.png              — ROC curves for all models on one axis
    09c_pr_curves.png               — Precision-Recall curves
    09d_calibration_curves.png      — Reliability diagrams + Brier scores
    09e_cv_metric_stability.png     — Per-fold boxplots for ROC AUC, PR AUC, F1
    09f_threshold_sensitivity.png   — Best model's TPR/FPR/precision/recall vs threshold

By policy this project evaluates both *discrimination* (ROC / PR AUC) and
*calibration* (reliability diagram, Brier score). For a health-risk
classifier, the quality of the predicted probability matters as much as the
class label, so accuracy alone would be misleading.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc, brier_score_loss, confusion_matrix, precision_recall_curve,
    precision_score, recall_score, roc_curve,
)
from sqlalchemy import text

from src.db import PROJECT_ROOT, get_engine

FIG_DIR = PROJECT_ROOT / "notebooks" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "dummy":                              "#888888",
    "logistic_regression":                "#1F77B4",
    "logistic_regression_l1":             "#17BECF",
    "random_forest":                      "#2CA02C",
    "hist_gradient_boosting":             "#D62728",
    "hist_gradient_boosting_calibrated":  "#9467BD",
}


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
def _latest_run_per_model(engine) -> pd.DataFrame:
    sql = text("""
        SELECT mr.model_run_id, mr.model_type, mr.run_timestamp
        FROM model_runs mr
        JOIN (
            SELECT model_type, MAX(model_run_id) AS max_id
            FROM model_runs GROUP BY model_type
        ) latest ON latest.model_type = mr.model_type
                AND latest.max_id     = mr.model_run_id
        ORDER BY mr.run_timestamp
    """)
    return pd.read_sql(sql, engine)


def _load_predictions(engine, model_run_ids: list[int]) -> pd.DataFrame:
    sql = text("""
        SELECT mp.model_run_id, mr.model_type, mp.patient_id, mp.fold,
               mp.predicted_class, mp.predicted_probability, mp.true_label
        FROM model_predictions mp
        JOIN model_runs mr ON mr.model_run_id = mp.model_run_id
        WHERE mp.model_run_id IN :ids
        ORDER BY mp.model_run_id, mp.patient_id
    """).bindparams(ids=tuple(model_run_ids))
    return pd.read_sql(sql, engine)


def _load_per_fold_metrics(engine, model_run_ids: list[int]) -> pd.DataFrame:
    sql = text("""
        SELECT mr.model_type, mtr.model_run_id, mtr.fold, mtr.metric_name, mtr.metric_value
        FROM model_training_results mtr
        JOIN model_runs mr ON mr.model_run_id = mtr.model_run_id
        WHERE mtr.fold IS NOT NULL AND mtr.model_run_id IN :ids
    """).bindparams(ids=tuple(model_run_ids))
    return pd.read_sql(sql, engine)


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def _style():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"


def _color(model_type: str) -> str:
    return PALETTE.get(model_type, "#444444")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_confusion(preds: pd.DataFrame) -> Path:
    models = sorted(preds["model_type"].unique())
    n = len(models)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.7 * nrows))
    for ax, model_type in zip(axes.flat, models):
        sub = preds[preds["model_type"] == model_type]
        cm = confusion_matrix(sub["true_label"], sub["predicted_class"], labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    cbar=False, square=True, linewidths=0.4,
                    xticklabels=["pred no", "pred yes"],
                    yticklabels=["true no", "true yes"])
        ax.set_title(model_type, fontsize=10)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.suptitle("Confusion matrices (out-of-fold predictions, threshold = 0.5)",
                 y=1.02)
    fig.tight_layout()
    path = FIG_DIR / "09a_confusion_matrices.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_roc(preds: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for model_type, sub in preds.groupby("model_type"):
        y = sub["true_label"].to_numpy()
        p = sub["predicted_probability"].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{model_type} (AUC = {roc_auc:.3f})",
                color=_color(model_type), linewidth=1.6)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — out-of-fold predictions")
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    path = FIG_DIR / "09b_roc_curves.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_pr(preds: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for model_type, sub in preds.groupby("model_type"):
        y = sub["true_label"].to_numpy()
        p = sub["predicted_probability"].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y, p)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, label=f"{model_type} (AUC = {pr_auc:.3f})",
                color=_color(model_type), linewidth=1.6)
    baseline = float(preds["true_label"].mean())
    ax.axhline(baseline, color="k", linestyle="--", linewidth=0.8, alpha=0.5,
               label=f"prevalence = {baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves — out-of-fold predictions")
    ax.legend(loc="lower left", frameon=True, fontsize=8)
    path = FIG_DIR / "09c_pr_curves.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_calibration(preds: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for model_type, sub in preds.groupby("model_type"):
        if model_type == "dummy":
            continue   # constant predicted_proba makes calibration meaningless
        y = sub["true_label"].to_numpy()
        p = sub["predicted_probability"].to_numpy()
        prob_true, prob_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        brier = brier_score_loss(y, p)
        ax.plot(prob_pred, prob_true, marker="o", linewidth=1.6,
                color=_color(model_type),
                label=f"{model_type} (Brier = {brier:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5,
            label="perfect calibration")
    ax.set_xlabel("Mean predicted probability (per bin)")
    ax.set_ylabel("Empirical positive rate (per bin)")
    ax.set_title("Calibration curves — predicted probability quality")
    ax.legend(loc="upper left", frameon=True, fontsize=8)
    path = FIG_DIR / "09d_calibration_curves.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_cv_stability(per_fold: pd.DataFrame) -> Path:
    metrics = ["roc_auc", "pr_auc", "f1"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.2),
                             sharey=False)
    for ax, metric in zip(axes, metrics):
        sub = per_fold[per_fold["metric_name"] == metric]
        order = (sub.groupby("model_type")["metric_value"].median()
                    .sort_values().index.tolist())
        sns.boxplot(
            data=sub, x="model_type", y="metric_value", ax=ax,
            order=order, hue="model_type",
            palette={m: _color(m) for m in order},
            width=0.55, linewidth=1.0, fliersize=3, legend=False,
        )
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
    fig.suptitle("Per-fold CV stability (10 folds per model)", y=1.02)
    fig.tight_layout()
    path = FIG_DIR / "09e_cv_metric_stability.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_threshold_sensitivity(preds: pd.DataFrame, best_model: str) -> Path:
    sub = preds[preds["model_type"] == best_model]
    y = sub["true_label"].to_numpy()
    p = sub["predicted_probability"].to_numpy()
    thresholds = np.linspace(0.05, 0.95, 19)
    tpr = []
    fpr = []
    prec = []
    rec = []
    for t in thresholds:
        yhat = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        tpr.append(tp / (tp + fn) if (tp + fn) else 0.0)
        fpr.append(fp / (fp + tn) if (fp + tn) else 0.0)
        prec.append(precision_score(y, yhat, zero_division=0))
        rec.append(recall_score(y, yhat, zero_division=0))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, rec, label="Recall (sensitivity)", color="#C43B3B", marker="o", linewidth=1.4)
    ax.plot(thresholds, prec, label="Precision", color="#3B82C4", marker="s", linewidth=1.4)
    ax.plot(thresholds, fpr, label="False positive rate", color="#D29C30", marker="^", linewidth=1.4)
    ax.axvline(0.5, color="k", linestyle="--", linewidth=0.8, alpha=0.5,
               label="default threshold = 0.5")
    ax.set_xlabel("Decision threshold on P(disease)")
    ax.set_ylabel("Rate")
    ax.set_title(f"Threshold sensitivity — {best_model}\n"
                 "Operating-point choice for a clinical setting "
                 "(higher recall → fewer missed cases)")
    ax.legend(loc="lower left", frameon=True, fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    path = FIG_DIR / "09f_threshold_sensitivity.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _pick_best_model(per_fold: pd.DataFrame) -> str:
    auc_means = (per_fold[per_fold["metric_name"] == "roc_auc"]
                 .groupby("model_type")["metric_value"].mean())
    return auc_means.idxmax()


def main() -> None:
    _style()
    engine = get_engine()
    latest = _latest_run_per_model(engine)
    ids = latest["model_run_id"].tolist()
    preds = _load_predictions(engine, ids)
    per_fold = _load_per_fold_metrics(engine, ids)
    best = _pick_best_model(per_fold)

    paths = [
        fig_confusion(preds),
        fig_roc(preds),
        fig_pr(preds),
        fig_calibration(preds),
        fig_cv_stability(per_fold),
        fig_threshold_sensitivity(preds, best),
    ]
    print(f"Best model by mean per-fold ROC AUC: {best}")
    print("Evaluation figures saved:")
    for p in paths:
        print(f"  {p.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
