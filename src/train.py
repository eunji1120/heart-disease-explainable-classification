"""Train and persist 4 classification models with 10-fold stratified CV.

Each model run gets one row in model_runs and:
  * per-fold metrics in model_training_results (fold = 1..10)
  * aggregate metrics in model_training_results (fold = NULL, computed from
    out-of-fold predictions)
  * one row per patient in model_predictions (the held-out fold's prediction)

Models compared:
  * dummy (most_frequent) — absolute baseline
  * logistic_regression — interpretable linear baseline
  * random_forest — non-linear ensemble baseline
  * hist_gradient_boosting — sklearn-native gradient boosting

Use ``python -m src.train`` to train all four. Each call inserts new model_run
rows; previous runs are kept for trend/history (no overwrites — health
analytics audit standard).
"""
from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sqlalchemy import text

from src.db import PROJECT_ROOT, get_engine
from src.features import CATEGORICAL_COLS, NUMERIC_COLS, make_preprocessor

SEED = 42
N_SPLITS = 10
THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
def _build_models() -> dict[str, BaseEstimator]:
    """Six classifiers covering: floor baseline, two linear variants (L2/L1),
    a non-linear ensemble, a gradient-boosted ensemble, and a probability-
    calibrated GBM. Each is fit inside the same ColumnTransformer pipeline.

    Overfit-guard choices:
      * `logistic_regression` already L2-regularized (sklearn default C=1.0).
      * `logistic_regression_l1` uses L1 for explicit feature selection — the
        penalty itself prevents overfit on small data.
      * `random_forest` uses `min_samples_leaf=2` to forbid single-row leaves.
      * `hist_gradient_boosting` uses `early_stopping=True` so training halts
        once the internal validation loss stops improving.
      * `hist_gradient_boosting_calibrated` wraps the GBM in
        `CalibratedClassifierCV` (isotonic, cv=5) — the calibrator itself is
        a regularizer; main benefit is a sharper Brier score for clinical
        risk-scoring.
    """
    return {
        "dummy": DummyClassifier(strategy="most_frequent", random_state=SEED),
        "logistic_regression": LogisticRegression(
            max_iter=2000, random_state=SEED, solver="lbfgs",
        ),
        "logistic_regression_l1": LogisticRegression(
            max_iter=2000, random_state=SEED, solver="liblinear",
            penalty="l1", C=1.0,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=SEED, n_jobs=-1,
            min_samples_leaf=2,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            random_state=SEED, max_iter=200, early_stopping=True,
        ),
        "hist_gradient_boosting_calibrated": CalibratedClassifierCV(
            HistGradientBoostingClassifier(
                random_state=SEED, max_iter=200, early_stopping=True,
            ),
            method="isotonic", cv=5,
        ),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def _load_data() -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    engine = get_engine()
    cols = ", ".join(["patient_id", "target_binary"] + NUMERIC_COLS + CATEGORICAL_COLS)
    df = pd.read_sql(
        f"SELECT {cols} FROM cleaned_patient_records ORDER BY patient_id",
        engine,
    )
    patient_ids = df["patient_id"].to_numpy()
    y = df["target_binary"].astype(int)
    X = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    return X, y, patient_ids


def _safe_metric(fn, *args, **kwargs) -> float:
    """Return NaN instead of raising when a metric is undefined (e.g. AUC on a
    single-class fold)."""
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return float("nan")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     y_proba: np.ndarray) -> dict[str, float]:
    return {
        "accuracy":   _safe_metric(accuracy_score, y_true, y_pred),
        "precision":  _safe_metric(precision_score, y_true, y_pred, zero_division=0),
        "recall":     _safe_metric(recall_score, y_true, y_pred, zero_division=0),
        "f1":         _safe_metric(f1_score, y_true, y_pred, zero_division=0),
        "roc_auc":    _safe_metric(roc_auc_score, y_true, y_proba),
        "pr_auc":     _safe_metric(average_precision_score, y_true, y_proba),
        "brier_score": _safe_metric(brier_score_loss, y_true, y_proba),
    }


def _serialize_hparams(est: BaseEstimator) -> str:
    params = est.get_params(deep=False)
    return json.dumps(params, default=str, sort_keys=True)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
@dataclass
class TrainOutput:
    model_run_id: int
    aggregate_metrics: dict[str, float]
    per_fold_metrics: list[dict[str, float]]


def _persist_run(engine, model_type: str, estimator: BaseEstimator,
                 git_sha: str | None, notes: str | None) -> int:
    with engine.begin() as conn:
        result = conn.execute(text("""
            INSERT INTO model_runs
                (model_type, random_seed, git_sha, hyperparameters, cv_folds, notes)
            VALUES
                (:model_type, :random_seed, :git_sha, :hp, :cv_folds, :notes)
        """), {
            "model_type": model_type,
            "random_seed": SEED,
            "git_sha": git_sha,
            "hp": _serialize_hparams(estimator),
            "cv_folds": N_SPLITS,
            "notes": notes,
        })
        return int(result.lastrowid)


def _persist_metrics(engine, model_run_id: int,
                     per_fold: list[dict[str, float]],
                     aggregate: dict[str, float]) -> None:
    rows = []
    for fold_idx, fold_metrics in enumerate(per_fold, start=1):
        for name, value in fold_metrics.items():
            if math.isnan(value):
                continue
            rows.append({
                "model_run_id": model_run_id,
                "fold": fold_idx,
                "metric_name": name,
                "metric_value": round(value, 6),
            })
    for name, value in aggregate.items():
        if math.isnan(value):
            continue
        rows.append({
            "model_run_id": model_run_id,
            "fold": None,
            "metric_name": name,
            "metric_value": round(value, 6),
        })
    pd.DataFrame(rows).to_sql("model_training_results", engine,
                              if_exists="append", index=False)


def _persist_predictions(engine, model_run_id: int,
                         patient_ids: np.ndarray, fold_ids: np.ndarray,
                         pred_class: np.ndarray, pred_proba: np.ndarray,
                         y_true: np.ndarray) -> None:
    df = pd.DataFrame({
        "model_run_id": model_run_id,
        "patient_id": patient_ids.astype(int),
        "fold": fold_ids.astype(int),
        "predicted_class": pred_class.astype(int),
        "predicted_probability": pred_proba.round(6),
        "true_label": y_true.astype(int),
    })
    df.to_sql("model_predictions", engine, if_exists="append", index=False)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one(model_type: str, estimator: BaseEstimator,
              X: pd.DataFrame, y: pd.Series, patient_ids: np.ndarray,
              git_sha: str | None) -> TrainOutput:
    pipeline = Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", estimator),
    ])

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    n = len(y)
    oof_proba = np.zeros(n)
    oof_pred = np.zeros(n, dtype=int)
    oof_fold = np.zeros(n, dtype=int)

    per_fold: list[dict[str, float]] = []
    y_arr = y.to_numpy()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_arr), start=1):
        pipe = clone(pipeline)
        pipe.fit(X.iloc[train_idx], y_arr[train_idx])
        proba = pipe.predict_proba(X.iloc[val_idx])[:, 1]
        preds = (proba >= THRESHOLD).astype(int)

        oof_proba[val_idx] = proba
        oof_pred[val_idx] = preds
        oof_fold[val_idx] = fold_idx

        per_fold.append(_compute_metrics(y_arr[val_idx], preds, proba))

    aggregate = _compute_metrics(y_arr, oof_pred, oof_proba)

    # Persist
    engine = get_engine()
    model_run_id = _persist_run(engine, model_type, estimator, git_sha,
                                notes=f"{N_SPLITS}-fold stratified CV; threshold={THRESHOLD}")
    _persist_metrics(engine, model_run_id, per_fold, aggregate)
    _persist_predictions(engine, model_run_id,
                         patient_ids, oof_fold, oof_pred, oof_proba, y_arr)

    return TrainOutput(model_run_id, aggregate, per_fold)


def train_all() -> list[tuple[str, TrainOutput]]:
    X, y, patient_ids = _load_data()
    git_sha = _git_sha()
    print(f"Loaded {len(y)} patients (positives: {int(y.sum())}, negatives: {int((1 - y).sum())})")
    print(f"git_sha: {git_sha}")
    print()

    results: list[tuple[str, TrainOutput]] = []
    for name, est in _build_models().items():
        print(f"==> training {name} ...", flush=True)
        out = train_one(name, est, X, y, patient_ids, git_sha)
        results.append((name, out))
        agg = out.aggregate_metrics
        print(f"    model_run_id={out.model_run_id}  "
              f"acc={agg['accuracy']:.3f}  "
              f"roc_auc={agg['roc_auc']:.3f}  "
              f"pr_auc={agg['pr_auc']:.3f}  "
              f"f1={agg['f1']:.3f}  "
              f"brier={agg['brier_score']:.3f}")
    return results


if __name__ == "__main__":
    train_all()
