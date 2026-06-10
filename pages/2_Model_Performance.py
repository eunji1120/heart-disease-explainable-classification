"""Model performance — metrics, ROC/PR, calibration, threshold, subgroup.

Heavy ops (calibration curves, subgroup metrics, threshold sweeps) are pulled
from cached helpers in `streamlit_utils.data` so the page does not recompute
on every slider move. The threshold operating-point block is wrapped in
@st.fragment so the slider only re-renders its own chart.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_utils.data import (
    best_model_name, calibration_data, load_cohort, load_curves,
    load_metrics_aggregate, load_metrics_per_fold, load_predictions,
    model_leaderboard, subgroup_metrics, threshold_sweep,
)
from streamlit_utils.styles import (
    ACCENT_AMBER, DISCLAIMER_HTML, DISEASE, ICONS, MODEL_PALETTE, NO_DISEASE,
    PAGE_CSS, SEQUENTIAL_NAVY, apply_plotly_layout, insight_card,
)

st.set_page_config(page_title="Model Performance", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)
st.title(f"{ICONS['performance']} Model performance")
st.markdown(
    "<p style='color:#6B7273;'>Discrimination (ROC, PR) <b>and</b> calibration "
    "(reliability curve, Brier score) — both matter for a health-risk classifier.</p>",
    unsafe_allow_html=True,
)

agg = load_metrics_aggregate()
per_fold = load_metrics_per_fold()
preds = load_predictions()
curves = load_curves()
best = best_model_name()
best_row = agg[agg["model_type"] == best].set_index("metric_name")["metric_value"]

# ---------------------------------------------------------------------------
# Top-line insight cards
# ---------------------------------------------------------------------------
i1, i2, i3, i4 = st.columns(4)
with i1:
    st.markdown(insight_card("teal", "Selected", best.replace("_", " ").title(),
                "Highest OOF ROC AUC; lowest Brier."), unsafe_allow_html=True)
with i2:
    st.markdown(insight_card("blue", "ROC AUC", f"{best_row['roc_auc']:.3f}",
                "Discrimination — separating disease from no-disease."), unsafe_allow_html=True)
with i3:
    st.markdown(insight_card("amber", "Recall @ 0.5", f"{best_row['recall']:.3f}",
                "Fraction of disease cases the model correctly flags."), unsafe_allow_html=True)
with i4:
    st.markdown(insight_card("rose", "Brier", f"{best_row['brier_score']:.3f}",
                "Probability quality (lower = better calibrated)."), unsafe_allow_html=True)

st.markdown("")

tabs = st.tabs([
    "Metrics comparison", "ROC & PR curves", "Calibration",
    "Per-fold CV stability", "Threshold operating point",
    "Subgroup performance",
])

# ---------------------------------------------------------------------------
# Tab 1: leaderboard + heatmap
# ---------------------------------------------------------------------------
with tabs[0]:
    lb = model_leaderboard()
    lb_show = lb.copy()
    lb_show["model_type"] = lb_show["model_type"].str.replace("_", " ").str.title()
    st.dataframe(
        lb_show, use_container_width=True, hide_index=True,
        column_config={
            "model_type": "Model",
            "roc_auc":     st.column_config.ProgressColumn("ROC AUC", min_value=0.5, max_value=1.0, format="%.3f"),
            "pr_auc":      st.column_config.ProgressColumn("PR AUC",  min_value=0.4, max_value=1.0, format="%.3f"),
            "accuracy":    st.column_config.ProgressColumn("Accuracy",min_value=0.5, max_value=1.0, format="%.3f"),
            "precision":   st.column_config.ProgressColumn("Precision",min_value=0.0,max_value=1.0, format="%.3f"),
            "recall":      st.column_config.ProgressColumn("Recall",  min_value=0.0, max_value=1.0, format="%.3f"),
            "f1":          st.column_config.ProgressColumn("F1",      min_value=0.0, max_value=1.0, format="%.3f"),
            "brier_score": st.column_config.NumberColumn("Brier", format="%.3f"),
        },
    )
    st.caption(
        f"**Selected model:** `{best}` — highest OOF ROC AUC and lowest Brier. "
        "Tied with `random_forest` and `logistic_regression_l1` within fold-level "
        "variance; the L2 LR is preferred because it is the simpler model on n = 303."
    )

# ---------------------------------------------------------------------------
# Tab 2: ROC + PR
# ---------------------------------------------------------------------------
with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        roc = curves[curves["curve"] == "ROC"]
        fig = px.line(
            roc, x="x_value", y="y_value", color="model_type",
            color_discrete_map=MODEL_PALETTE,
            labels={"x_value": "False positive rate", "y_value": "True positive rate"},
            title="ROC curves",
        )
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(color="#B0B0B0", dash="dash", width=1))
        st.plotly_chart(apply_plotly_layout(fig, height=480), use_container_width=True)
    with c2:
        pr = curves[curves["curve"] == "PR"]
        fig = px.line(
            pr, x="x_value", y="y_value", color="model_type",
            color_discrete_map=MODEL_PALETTE,
            labels={"x_value": "Recall", "y_value": "Precision"},
            title="Precision–recall curves",
        )
        prevalence = preds["true_label"].mean()
        fig.add_hline(y=prevalence, line_dash="dash", line_color="#B0B0B0",
                      annotation_text=f"prevalence ≈ {prevalence:.2f}",
                      annotation_position="top right")
        st.plotly_chart(apply_plotly_layout(fig, height=480), use_container_width=True)
    st.caption(
        "PR curves are more informative under class imbalance than ROC; here "
        "the dataset is roughly balanced (54% / 46%), so both tell a similar "
        "story. All real models cluster around AUC ≈ 0.91 — the LR's edge is "
        "small and within CV variance."
    )

# ---------------------------------------------------------------------------
# Tab 3: calibration (uses cached calibration_data)
# ---------------------------------------------------------------------------
with tabs[2]:
    cal = calibration_data()
    fig = go.Figure()
    for model_type, sub in cal.groupby("model_type"):
        brier = sub["brier"].iloc[0]
        fig.add_trace(go.Scatter(
            x=sub["prob_pred"], y=sub["prob_true"], mode="lines+markers",
            name=f"{model_type} (Brier {brier:.3f})",
            line=dict(color=MODEL_PALETTE.get(model_type, "#666"), width=2),
            marker=dict(size=8),
            hovertemplate=f"%{{x:.3f}} → %{{y:.3f}}<br>{model_type}<extra></extra>",
        ))
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color="#B0B0B0", dash="dash", width=1))
    fig.update_layout(
        title="Reliability diagram — does the model's probability match reality?",
        xaxis_title="Mean predicted probability (per bin)",
        yaxis_title="Empirical positive rate (per bin)",
    )
    st.plotly_chart(apply_plotly_layout(fig, height=520), use_container_width=True)

    brier_df = (cal.drop_duplicates("model_type")[["model_type", "brier"]]
                .sort_values("brier").reset_index(drop=True))
    fig = px.bar(
        brier_df, x="brier", y="model_type", orientation="h",
        color="model_type", color_discrete_map=MODEL_PALETTE,
        labels={"brier": "Brier score (lower is better)", "model_type": ""},
        title="Brier score — overall calibration quality",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(apply_plotly_layout(fig, height=300), use_container_width=True)
    st.caption(
        "The linear models track the diagonal closely; the gradient booster "
        "deviates at high probabilities. With n = 303, calibration estimates "
        "should be read as indicative, not definitive."
    )

# ---------------------------------------------------------------------------
# Tab 4: per-fold stability
# ---------------------------------------------------------------------------
with tabs[3]:
    metric_pick = st.selectbox(
        "Metric", ["roc_auc", "pr_auc", "f1", "accuracy", "precision", "recall", "brier_score"],
        index=0,
    )
    sub = per_fold[per_fold["metric_name"] == metric_pick]
    fig = px.box(
        sub, x="model_type", y="metric_value", color="model_type",
        color_discrete_map=MODEL_PALETTE, points="all",
        labels={"model_type": "", "metric_value": metric_pick},
        title=f"Per-fold {metric_pick} (10 folds per model)",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(apply_plotly_layout(fig, height=480), use_container_width=True)
    st.caption(
        "Overlapping IQRs across the top three models imply the model ranking "
        "should not be over-interpreted at this sample size."
    )

# ---------------------------------------------------------------------------
# Tab 5: threshold (isolated via @st.fragment — slider doesn't re-render whole page)
# ---------------------------------------------------------------------------
@st.fragment
def threshold_panel():
    model_for_thresh = st.selectbox(
        "Model", [m for m in preds["model_type"].unique() if m != "dummy"],
        index=0 if best == "logistic_regression"
              else [m for m in preds["model_type"].unique() if m != "dummy"].index(best),
        key="thresh_model",
    )
    threshold = st.slider(
        "Decision threshold on P(disease)",
        min_value=0.05, max_value=0.95, value=0.50, step=0.05,
        help="Higher threshold → fewer false alarms but more missed cases.",
        key="thresh_value",
    )

    grid = threshold_sweep(model_for_thresh)
    # Round to align with slider step
    row = grid.iloc[(grid["threshold"] - threshold).abs().argmin()]
    tp, fp, tn, fn = int(row["tp"]), int(row["fp"]), int(row["tn"]), int(row["fn"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{ICONS['disease']} Recall", f"{row['recall']:.3f}",
              help="Of records with disease, fraction correctly flagged.")
    m2.metric("Precision", f"{row['precision']:.3f}",
              help="Of records flagged as disease, fraction actually positive.")
    m3.metric("F1", f"{row['f1']:.3f}", help="Harmonic mean of precision and recall.")
    m4.metric("Accuracy", f"{row['accuracy']:.3f}",
              help="Overall fraction of correct classifications.")

    cm_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["actual no disease", "actual disease"],
        columns=["predicted no disease", "predicted disease"],
    )
    cf1, cf2 = st.columns([1, 1])
    with cf1:
        fig = px.imshow(
            cm_df.values, x=cm_df.columns, y=cm_df.index, text_auto=True,
            color_continuous_scale=SEQUENTIAL_NAVY, aspect="auto",
            title=f"Confusion matrix at threshold = {threshold:.2f}",
        )
        st.plotly_chart(apply_plotly_layout(fig, height=400), use_container_width=True)
        st.markdown(
            f"- {ICONS['tp']} **True positives** (correctly flagged disease): **{tp}**\n"
            f"- {ICONS['fn']} **False negatives** (missed disease): **{fn}** — clinically costliest\n"
            f"- {ICONS['fp']} **False positives** (over-flagged): **{fp}**\n"
            f"- {ICONS['tn']} **True negatives**: **{tn}**"
        )
    with cf2:
        long = grid.melt(id_vars=["threshold"],
                         value_vars=["precision", "recall", "f1"],
                         var_name="metric", value_name="value")
        fig = px.line(
            long, x="threshold", y="value", color="metric",
            color_discrete_map={"precision": NO_DISEASE, "recall": DISEASE, "f1": ACCENT_AMBER},
            labels={"threshold": "Threshold", "value": "Rate", "metric": ""},
            title="Operating-point sweep — recall trades against precision",
        )
        fig.add_vline(x=threshold, line_dash="dash", line_color="black",
                      annotation_text=f"current = {threshold:.2f}",
                      annotation_position="top")
        st.plotly_chart(apply_plotly_layout(fig, height=400), use_container_width=True)


with tabs[4]:
    threshold_panel()

# ---------------------------------------------------------------------------
# Tab 6: subgroup (cached)
# ---------------------------------------------------------------------------
@st.fragment
def subgroup_panel():
    sub_model = st.selectbox(
        "Model for subgroup analysis",
        [m for m in preds["model_type"].unique() if m != "dummy"],
        index=0, key="sub_model",
    )
    sub_df = subgroup_metrics(sub_model)
    if sub_df.empty:
        st.info("Not enough records in any subgroup to compute meaningful AUC.")
        return

    fig = px.bar(
        sub_df, x="roc_auc", y="subgroup", color="subgroup_type",
        orientation="h", text="roc_auc",
        labels={"roc_auc": "ROC AUC", "subgroup": "", "subgroup_type": "Group"},
        color_discrete_map={"sex": NO_DISEASE, "age_band": ACCENT_AMBER, "cp": "#3E8E7E"},
        title=f"Subgroup ROC AUC — {sub_model}",
    )
    fig.add_vline(x=0.5, line_dash="dot", line_color="#888",
                  annotation_text="no-skill", annotation_position="bottom left")
    st.plotly_chart(apply_plotly_layout(fig, height=520), use_container_width=True)

    st.markdown("##### Subgroup metrics at threshold = 0.50")
    st.dataframe(sub_df, use_container_width=True, hide_index=True)
    st.caption(
        "Small subgroups (n < 30) produce wide AUC swings — read with caution. "
        "This view is intended to *raise* fairness questions, not answer them. "
        "Race / ethnicity is not available in the dataset."
    )


with tabs[5]:
    st.markdown(
        "How does the selected model perform across **demographic and clinical "
        "subgroups**? Wide AUC variation would signal that the model works "
        "well for some subgroups but not others — a critical fairness check."
    )
    subgroup_panel()

st.markdown(DISCLAIMER_HTML, unsafe_allow_html=True)
