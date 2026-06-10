"""Heart Disease Risk Classification — interactive analytics dashboard.

Home page: KPI snapshot + top risk drivers + key insight cards + model
leaderboard + narrative. Performance optimized with cached derived data.
"""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from streamlit_utils.data import (
    best_model_name, key_insights, load_cohort, load_metrics_aggregate,
    load_shap_global, model_leaderboard,
)
from streamlit_utils.glossary import FEATURE_LABEL
from streamlit_utils.styles import (
    DISCLAIMER_HTML, ICONS, MODEL_PALETTE, PAGE_CSS, PRIMARY_DEEP,
    SEQUENTIAL_NAVY, apply_plotly_layout, insight_card,
)

st.set_page_config(
    page_title="Heart Disease Risk — Explainable Dashboard",
    page_icon="·",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(PAGE_CSS, unsafe_allow_html=True)

st.title(f"{ICONS['risk']} Explainable Heart Disease Risk Classification")
st.markdown(
    "<p style='color:#6B7273; font-size:1.02rem; margin-top:-8px;'>"
    "An end-to-end health-analytics workflow on the UCI Cleveland cohort — "
    "MySQL pipeline → six cross-validated models → SHAP explanations → "
    "this interactive dashboard.</p>",
    unsafe_allow_html=True,
)
st.markdown(DISCLAIMER_HTML, unsafe_allow_html=True)
st.markdown("")

# ---------------------------------------------------------------------------
# Headline KPI row
# ---------------------------------------------------------------------------
cohort = load_cohort()
metrics = load_metrics_aggregate()
best = best_model_name()
best_row = metrics[metrics["model_type"] == best].set_index("metric_name")["metric_value"]

n_total = len(cohort)
n_disease = int(cohort["target_binary"].sum())
pct_disease = n_disease / n_total

st.markdown(f"##### {ICONS['cohort']} Cohort at a glance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Cohort size", f"{n_total:,}")
c2.metric(f"{ICONS['disease']} Disease prevalence", f"{pct_disease:.1%}",
          help=f"{n_disease:,} of {n_total:,} records classified as heart disease present (num ≥ 1).")
c3.metric("Features", "13", help="8 numeric + 5 categorical (after cleaning).")
c4.metric("Models compared", "6",
          help="dummy / LR L2 / LR L1 / RF / HistGBM / calibrated HistGBM.")

st.markdown(f"##### {ICONS['performance']} Selected model — out-of-fold performance, 10-fold stratified CV")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Best model", best.replace("_", " ").title())
m2.metric("ROC AUC", f"{best_row['roc_auc']:.3f}",
          help="Discrimination — how well the model separates disease from no-disease. 0.5 = random, 1.0 = perfect.")
m3.metric("PR AUC", f"{best_row['pr_auc']:.3f}",
          help="Precision-recall AUC — more informative under class imbalance.")
m4.metric("Recall", f"{best_row['recall']:.3f}",
          help="Sensitivity at default threshold 0.5 — fraction of disease cases correctly identified.")
m5.metric("Brier score", f"{best_row['brier_score']:.3f}",
          help="Lower is better. Measures whether predicted probabilities match observed outcomes.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Key insights — soft pastel cards (the "what should I notice?" section)
# ---------------------------------------------------------------------------
ins = key_insights()
st.markdown(f"##### {ICONS['info']} Key insights from this analysis")
i1, i2, i3, i4 = st.columns(4)
with i1:
    feature_label_top = FEATURE_LABEL.get(ins["top3_features"][0], ins["top3_features"][0])
    st.markdown(
        insight_card(
            "teal",
            "Strongest signal",
            feature_label_top,
            "leads both linear and tree models' feature rankings — "
            "the model's most informative variable.",
        ),
        unsafe_allow_html=True,
    )
with i2:
    st.markdown(
        insight_card(
            "rose",
            "Sex disparity",
            f"{ins['male_prev']:.0%} vs {ins['female_prev']:.0%}",
            "disease prevalence in males vs females in this referral cohort — "
            "wide gap that reflects 1980s referral patterns.",
        ),
        unsafe_allow_html=True,
    )
with i3:
    st.markdown(
        insight_card(
            "amber",
            "Highest-risk age band",
            ins["highest_risk_band"],
            f"records in this band carry a {ins['highest_risk_rate']:.0%} disease rate — "
            "the model leans on age, but only modestly.",
        ),
        unsafe_allow_html=True,
    )
with i4:
    st.markdown(
        insight_card(
            "blue",
            "Data quality",
            f"{ins['n_imputed']} of {n_total}",
            "records had `ca` or `thal` missing in the source; both imputed and "
            "marked with audit flags for traceability.",
        ),
        unsafe_allow_html=True,
    )

st.markdown("")

# ---------------------------------------------------------------------------
# Two columns: Top drivers chart + Model leaderboard
# ---------------------------------------------------------------------------
left, right = st.columns([3, 2])

with left:
    st.markdown(f"##### {ICONS['risk']} Top risk drivers — what the model leans on")
    shap_global = load_shap_global()
    top_lr = (shap_global[shap_global["model_type"] == "logistic_regression"]
              .sort_values("mean_abs_shap", ascending=True).tail(8).copy())
    top_lr["display_name"] = top_lr["feature_name"].map(FEATURE_LABEL).fillna(top_lr["feature_name"])
    fig = px.bar(
        top_lr, x="mean_abs_shap", y="display_name", orientation="h",
        labels={"mean_abs_shap": "Average impact on prediction (mean |SHAP|)",
                "display_name": ""},
        color_discrete_sequence=[PRIMARY_DEEP],
        custom_data=["feature_name"],
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>raw code: <i>%{customdata[0]}</i><br>"
                      "Average impact: %{x:.3f}<extra></extra>",
        marker_line_color="white", marker_line_width=1,
    )
    st.plotly_chart(apply_plotly_layout(fig, height=380), use_container_width=True)
    st.caption(
        f"The top three — **major vessels with reduced flow**, **chest pain type**, "
        "and the **thallium stress test result** — match the cardiology literature "
        "on the most informative non-invasive cues for coronary-artery disease."
    )

with right:
    st.markdown(f"##### {ICONS['performance']} Model leaderboard")
    lb = model_leaderboard()
    lb["model_type"] = lb["model_type"].str.replace("_", " ").str.title()
    # Use only headline metrics — keep card compact
    show = lb[["model_type", "roc_auc", "pr_auc", "brier_score"]].rename(columns={
        "model_type": "Model", "roc_auc": "ROC AUC", "pr_auc": "PR AUC",
        "brier_score": "Brier",
    })
    st.dataframe(
        show, use_container_width=True, hide_index=True,
        column_config={
            "ROC AUC": st.column_config.ProgressColumn(
                "ROC AUC", min_value=0.5, max_value=1.0, format="%.3f"),
            "PR AUC": st.column_config.ProgressColumn(
                "PR AUC", min_value=0.4, max_value=1.0, format="%.3f"),
            "Brier": st.column_config.NumberColumn("Brier", format="%.3f"),
        },
    )
    st.caption(
        "Ordered by OOF ROC AUC. The top three models — L2 LR, L1 LR, and "
        "Random Forest — are within fold-level variance of each other."
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Navigation + narrative
# ---------------------------------------------------------------------------
left, right = st.columns([3, 2])

with left:
    st.markdown("### How to use this dashboard")
    st.markdown(
        f"""
        - {ICONS['cohort']} **Cohort Profile** — interactive demographic and clinical
          filters to inspect how disease prevalence varies across age, sex,
          chest-pain type, and other subgroups.
        - {ICONS['performance']} **Model Performance** — discrimination (ROC, PR),
          probability calibration (reliability diagram, Brier), per-fold
          cross-validation stability, an interactive **threshold slider** for
          choosing the recall-vs-precision operating point, and a
          **subgroup-fairness** view.
        - {ICONS['risk']} **Top Risk Factors** — feature-level importance for both
          the linear and tree models, with a cross-model comparison surfacing
          agreement.
        - {ICONS['record']} **Per-Record Explanation** — pick one of the 303 records
          and see the per-feature signed SHAP contribution behind its
          prediction, with **plain-language clinical interpretation** for
          non-expert readers.
        """
    )

with right:
    st.markdown("### Why both discrimination *and* calibration")
    st.markdown(
        """
        A health-risk classifier is consumed for its **predicted probability**,
        not just the class label — clinicians and care-pathway designers act
        on graded risk, not on a coin flip. Every page reports both:

        - **Discrimination** — can the model separate disease from no-disease
          records? (ROC, PR)
        - **Calibration** — when the model says 70%, do 70% actually have the
          condition? (reliability diagram, Brier score)

        Accuracy alone would be misleading here.
        """
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Glossary expander (HCI: help text on-demand)
# ---------------------------------------------------------------------------
with st.expander(f"{ICONS['info']} Glossary — clinical variables in plain language", expanded=False):
    from streamlit_utils.glossary import FEATURE_HELP, FEATURE_LABEL
    glossary_rows = []
    for code, label in FEATURE_LABEL.items():
        glossary_rows.append({
            "Code": code, "Variable": label, "What it means": FEATURE_HELP[code],
        })
    import pandas as pd
    st.dataframe(pd.DataFrame(glossary_rows), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    f"Source: Detrano et al., UCI Heart Disease (Cleveland subset). "
    f"Selected model: `{best}` — chosen by OOF ROC AUC and Brier on the held-out 10-fold CV."
)
