"""Step 10 — SHAP explainability for the best linear and tree models.

Persists three layers of SHAP output (per the project's policy):

  1. Global feature importance — mean(|SHAP|) per (model_run, original_feature)
     into shap_global_importance.
  2. Patient-level signed contributions — one row per
     (model_run, patient_id, feature_name) into shap_patient_level, with the
     source feature_value next to it so Power BI can drill through without an
     extra join.
  3. Visual outputs — beeswarm summary, global-importance bar, and two
     example waterfall plots (correct call + missed call) into
     notebooks/figures/.

Encoding handling: the sklearn pipeline one-hot-encodes 5 categorical columns
into ~15 dummy features. We compute SHAP on the post-transform feature space,
then **sum SHAP across all dummies of the same original column** to get a
13-feature attribution that matches the data dictionary. This is the standard
practice for SHAP + one-hot.

Evaluation strategy: out-of-fold SHAP. Each fold's training rows fit the
preprocessor + model; the held-out rows are then explained with that fold's
model. This is consistent with how out-of-fold predictions were generated
and avoids leakage of validation rows into the SHAP background.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sqlalchemy import text

from src.db import PROJECT_ROOT, get_engine
from src.features import CATEGORICAL_COLS, NUMERIC_COLS, make_preprocessor
from src.train import N_SPLITS, SEED, _build_models

warnings.filterwarnings("ignore", category=UserWarning, module="shap")

FIG_DIR = PROJECT_ROOT / "notebooks" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EXPLAIN_MODELS = ["logistic_regression", "random_forest"]
ORIGINAL_FEATURES = NUMERIC_COLS + CATEGORICAL_COLS


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _load_data():
    engine = get_engine()
    cols = ", ".join(["patient_id", "target_binary"] + NUMERIC_COLS + CATEGORICAL_COLS)
    df = pd.read_sql(
        f"SELECT {cols} FROM cleaned_patient_records ORDER BY patient_id",
        engine,
    )
    return df


def _latest_run_id(engine, model_type: str) -> int:
    res = pd.read_sql(
        text("SELECT MAX(model_run_id) AS mid FROM model_runs WHERE model_type = :t"),
        engine, params={"t": model_type},
    )
    return int(res["mid"].iloc[0])


# ---------------------------------------------------------------------------
# Onehot → original feature mapping
# ---------------------------------------------------------------------------
def _onehot_groups(feature_names: Sequence[str]) -> dict[str, list[int]]:
    """Map original-feature name → indices of its post-transform columns."""
    groups: dict[str, list[int]] = {f: [] for f in ORIGINAL_FEATURES}
    for idx, name in enumerate(feature_names):
        if name.startswith("num__"):
            original = name[len("num__"):]
        elif name.startswith("cat__"):
            remaining = name[len("cat__"):]
            original = next((c for c in CATEGORICAL_COLS
                             if remaining == c or remaining.startswith(c + "_")), remaining)
        else:
            original = name
        groups.setdefault(original, []).append(idx)
    # Drop empties (shouldn't happen but safe)
    return {k: v for k, v in groups.items() if v}


# ---------------------------------------------------------------------------
# SHAP per model_type, OOF
# ---------------------------------------------------------------------------
@dataclass
class ShapResult:
    """SHAP values aggregated to the original 13 features, in patient order."""
    patient_ids: np.ndarray
    feature_order: list[str]
    shap_values: np.ndarray   # shape (n_patients, n_original_features)
    feature_values: pd.DataFrame   # original input dataframe in patient order


def _shap_for_fold(model_type: str, pipeline: Pipeline,
                   X_train_t: np.ndarray, X_val_t: np.ndarray) -> np.ndarray:
    """Return SHAP values for the positive class on X_val_t, shape (n_val, n_features_transformed)."""
    model = pipeline.named_steps["model"]
    if model_type in ("logistic_regression", "logistic_regression_l1"):
        explainer = shap.LinearExplainer(model, X_train_t)
        sv = explainer.shap_values(X_val_t)
        # LinearExplainer for binary LR returns (n, n_features) of contributions to log-odds
        return np.asarray(sv)
    elif model_type in ("random_forest", "hist_gradient_boosting"):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_val_t)
        # Newer shap returns ndarray (n, n_features, n_classes) for binary RF
        sv = np.asarray(sv)
        if sv.ndim == 3:
            # take class-1 contribution
            return sv[:, :, 1]
        if isinstance(sv, list):
            return sv[1]
        return sv
    else:
        raise NotImplementedError(model_type)


def compute_oof_shap(model_type: str, X: pd.DataFrame, y: pd.Series,
                     patient_ids: np.ndarray) -> ShapResult:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    y_arr = y.to_numpy()

    # Reference pipeline (for feature names after fit)
    reference = Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", _build_models()[model_type]),
    ]).fit(X, y_arr)
    feature_names_post = list(reference.named_steps["preprocess"].get_feature_names_out())
    groups = _onehot_groups(feature_names_post)
    # Stable feature order matching ORIGINAL_FEATURES
    feature_order = [f for f in ORIGINAL_FEATURES if f in groups]

    n = len(X)
    aggregated = np.zeros((n, len(feature_order)), dtype=float)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_arr), 1):
        pipe = clone(reference)
        pipe.fit(X.iloc[train_idx], y_arr[train_idx])
        preproc = pipe.named_steps["preprocess"]
        X_train_t = preproc.transform(X.iloc[train_idx])
        X_val_t = preproc.transform(X.iloc[val_idx])

        sv_post = _shap_for_fold(model_type, pipe, X_train_t, X_val_t)

        # Aggregate one-hot dummies back to original features (sum)
        for f_idx, feat in enumerate(feature_order):
            cols = groups[feat]
            aggregated[val_idx, f_idx] = sv_post[:, cols].sum(axis=1)

    return ShapResult(
        patient_ids=patient_ids,
        feature_order=feature_order,
        shap_values=aggregated,
        feature_values=X.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _format_feature_value(col: str, val) -> str:
    if pd.isna(val):
        return ""
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        # Keep 1-decimal for oldpeak, integer otherwise
        if col == "oldpeak":
            return f"{val:.1f}"
        return f"{val:.0f}"
    return str(val)


def _purge_old_shap_rows(engine, model_run_id: int) -> None:
    """Delete prior SHAP rows for this run (idempotent re-runs)."""
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM shap_patient_level WHERE model_run_id = :id"),
                     {"id": model_run_id})
        conn.execute(text("DELETE FROM shap_global_importance WHERE model_run_id = :id"),
                     {"id": model_run_id})


def persist_global(engine, model_run_id: int, result: ShapResult) -> None:
    mean_abs = np.abs(result.shap_values).mean(axis=0)
    df = pd.DataFrame({
        "model_run_id": model_run_id,
        "feature_name": result.feature_order,
        "mean_abs_shap": np.round(mean_abs, 6),
    })
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank_position"] = df.index + 1
    df.to_sql("shap_global_importance", engine, if_exists="append", index=False)


def persist_patient_level(engine, model_run_id: int, result: ShapResult) -> None:
    rows = []
    for i, pid in enumerate(result.patient_ids):
        for f_idx, feat in enumerate(result.feature_order):
            rows.append({
                "model_run_id": model_run_id,
                "patient_id": int(pid),
                "feature_name": feat,
                "feature_value": _format_feature_value(feat, result.feature_values.loc[i, feat]),
                "shap_value": float(round(result.shap_values[i, f_idx], 6)),
            })
    pd.DataFrame(rows).to_sql("shap_patient_level", engine,
                              if_exists="append", index=False)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _figure_dir(model_type: str) -> str:
    return model_type.upper().replace("_", "")


def fig_global_bar(model_type: str, result: ShapResult) -> Path:
    mean_abs = np.abs(result.shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    feats = [result.feature_order[i] for i in order]
    vals = mean_abs[order]
    ax.barh(feats[::-1], vals[::-1], color="#3B82C4", edgecolor="white")
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title(f"Global feature importance — {model_type}\n"
                 "(average magnitude of contribution across 303 patients)")
    fig.tight_layout()
    path = FIG_DIR / f"10a_shap_global_{_figure_dir(model_type)}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_summary_beeswarm(model_type: str, result: ShapResult) -> Path:
    # shap.summary_plot needs numeric features for color; for categoricals
    # we factorize so the colour gradient at least groups them.
    X_for_plot = result.feature_values[result.feature_order].copy()
    for col in CATEGORICAL_COLS:
        if col in X_for_plot.columns:
            X_for_plot[col] = pd.factorize(X_for_plot[col])[0]
    fig = plt.figure(figsize=(8, 5.5))
    shap.summary_plot(
        result.shap_values,
        X_for_plot,
        feature_names=result.feature_order,
        show=False,
        plot_size=None,
    )
    plt.title(f"SHAP summary (beeswarm) — {model_type}", fontsize=11)
    path = FIG_DIR / f"10b_shap_summary_{_figure_dir(model_type)}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _example_patient_ids(engine, model_run_id: int) -> tuple[int, int]:
    """Pick one true-positive (correctly classified disease) and one
    false-negative (missed disease) patient under this model's OOF predictions."""
    df = pd.read_sql(text("""
        SELECT patient_id, predicted_class, true_label, predicted_probability
        FROM model_predictions WHERE model_run_id = :id
    """), engine, params={"id": model_run_id})
    tp = df[(df["true_label"] == 1) & (df["predicted_class"] == 1)].sort_values(
        "predicted_probability", ascending=False)
    fn = df[(df["true_label"] == 1) & (df["predicted_class"] == 0)].sort_values(
        "predicted_probability", ascending=False)
    return int(tp.iloc[0]["patient_id"]), int(fn.iloc[0]["patient_id"])


def fig_waterfall(model_type: str, result: ShapResult,
                  patient_id: int, label: str, base_value: float) -> Path:
    idx = int(np.where(result.patient_ids == patient_id)[0][0])
    sv = result.shap_values[idx]
    fv = result.feature_values.loc[idx, result.feature_order].tolist()

    expl = shap.Explanation(
        values=sv,
        base_values=base_value,
        data=fv,
        feature_names=result.feature_order,
    )
    fig = plt.figure(figsize=(8.5, 5.5))
    shap.plots.waterfall(expl, show=False, max_display=10)
    plt.title(f"Patient {patient_id} — {label}  ({model_type})", fontsize=10)
    safe = label.replace(" ", "").lower()
    path = FIG_DIR / f"10c_shap_waterfall_{_figure_dir(model_type)}_{safe}_p{patient_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_for_model(engine, df: pd.DataFrame, model_type: str) -> dict:
    model_run_id = _latest_run_id(engine, model_type)
    X = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    y = df["target_binary"].astype(int)
    patient_ids = df["patient_id"].to_numpy()

    print(f"\n==> SHAP for {model_type} (model_run_id={model_run_id}) ...")
    result = compute_oof_shap(model_type, X, y, patient_ids)
    print(f"   shap matrix shape: {result.shap_values.shape}")

    _purge_old_shap_rows(engine, model_run_id)
    persist_global(engine, model_run_id, result)
    persist_patient_level(engine, model_run_id, result)
    print(f"   persisted to shap_global_importance + shap_patient_level")

    paths = [
        fig_global_bar(model_type, result),
        fig_summary_beeswarm(model_type, result),
    ]
    # Waterfall examples — base_value approximation = prevalence on logit/prob scale.
    # For both LR (log-odds) and RF (probability) we plot using the mean-prediction
    # baseline derived from the OOF probability.
    base_value = float(df["target_binary"].mean())
    tp_id, fn_id = _example_patient_ids(engine, model_run_id)
    paths.append(fig_waterfall(model_type, result, tp_id,
                               "correctly identified disease (TP)", base_value))
    paths.append(fig_waterfall(model_type, result, fn_id,
                               "missed disease case (FN)", base_value))

    for p in paths:
        print(f"   figure: {p.relative_to(PROJECT_ROOT)}")
    return {"model_run_id": model_run_id, "tp_id": tp_id, "fn_id": fn_id}


def main() -> None:
    engine = get_engine()
    df = _load_data()
    print(f"Loaded {len(df)} patients (positives: {int(df['target_binary'].sum())})")
    for model_type in EXPLAIN_MODELS:
        run_for_model(engine, df, model_type)


if __name__ == "__main__":
    main()
