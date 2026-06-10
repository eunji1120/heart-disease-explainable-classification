"""Per-record explanation — pick a record, see signed SHAP drivers in plain language.

Wrapped in @st.fragment so the patient selector only re-renders its own block,
not the whole page (Streamlit 1.50+).
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_utils.data import load_cohort, load_shap_patient
from streamlit_utils.glossary import FEATURE_HELP, FEATURE_LABEL, value_label
from streamlit_utils.styles import (
    DISCLAIMER_HTML, DIRECTION_PALETTE, DISEASE, ICONS, NEUTRAL_DARK,
    NEUTRAL_LIGHT, NO_DISEASE, PAGE_CSS, PRIMARY_DEEP, apply_plotly_layout,
    insight_card,
)

st.set_page_config(page_title="Per-record explanation", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)

st.title(f"{ICONS['record']} Per-record explanation")
st.markdown(
    "<p style='color:#6B7273;'>Pick any record to see how the model arrived at "
    "its prediction. Each feature is shown alongside a plain-language "
    "interpretation, an icon for the direction of risk, and the magnitude of "
    "its contribution.</p>",
    unsafe_allow_html=True,
)

cohort = load_cohort()
shap_patient = load_shap_patient()
available_models = sorted(shap_patient["model_type"].unique())


def _gauge(proba: float, threshold: float = 0.5) -> go.Figure:
    """Probability gauge using soft warm/cool palette."""
    color = DISEASE if proba >= threshold else NO_DISEASE
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={"suffix": "%", "font": {"size": 28, "color": PRIMARY_DEEP}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": NEUTRAL_LIGHT},
            "bar": {"color": color, "thickness": 0.7},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 30], "color": "#E7F0EF"},
                {"range": [30, 50], "color": "#F4EFEA"},
                {"range": [50, 70], "color": "#FAEDDF"},
                {"range": [70, 100], "color": "#F7E4E1"},
            ],
            "threshold": {
                "line": {"color": "#333", "width": 2},
                "thickness": 0.85,
                "value": threshold * 100,
            },
        },
    ))
    fig.update_layout(
        height=190,
        margin=dict(l=12, r=12, t=12, b=12),
        paper_bgcolor="white",
    )
    return fig


@st.fragment
def record_panel():
    # ---- Controls ----
    ctl1, ctl2, ctl3 = st.columns([2, 3, 2])
    with ctl1:
        model_pick = st.selectbox(
            "Model", available_models, index=0,
            format_func=lambda x: x.replace("_", " ").title(),
            key="rec_model",
        )
    with ctl2:
        outcome_filter = st.radio(
            "Show only:",
            ["any record",
             f"{ICONS['tp']} true positives — correctly identified disease",
             f"{ICONS['fn']} false negatives — missed disease",
             f"{ICONS['fp']} false positives — over-flagged"],
            index=0, key="rec_filter",
        )
    with ctl3:
        sub_model = (shap_patient[shap_patient["model_type"] == model_pick]
                     .drop_duplicates("patient_id"))
        if "true positives" in outcome_filter:
            sub_model = sub_model[(sub_model["true_label"] == 1) & (sub_model["predicted_class"] == 1)]
        elif "false negatives" in outcome_filter:
            sub_model = sub_model[(sub_model["true_label"] == 1) & (sub_model["predicted_class"] == 0)]
        elif "false positives" in outcome_filter:
            sub_model = sub_model[(sub_model["true_label"] == 0) & (sub_model["predicted_class"] == 1)]
        patient_ids = sub_model["patient_id"].sort_values().tolist()
        if not patient_ids:
            st.error("No patients match this filter.")
            return
        patient_id = st.selectbox(
            "Patient record ID", patient_ids, index=0,
            help=f"{len(patient_ids)} records match.",
            key="rec_pid",
        )

    patient_row = cohort[cohort["patient_id"] == patient_id].iloc[0]
    shap_rows = (shap_patient[
        (shap_patient["model_type"] == model_pick)
        & (shap_patient["patient_id"] == patient_id)
    ].copy())

    pred_proba = float(shap_rows["predicted_probability"].iloc[0])
    pred_class = int(shap_rows["predicted_class"].iloc[0])
    true_label = int(shap_rows["true_label"].iloc[0])
    outcome = (
        f"{ICONS['tp']} true positive" if (true_label == 1 and pred_class == 1)
        else f"{ICONS['fn']} false negative" if (true_label == 1 and pred_class == 0)
        else f"{ICONS['fp']} false positive" if (true_label == 0 and pred_class == 1)
        else f"{ICONS['tn']} true negative"
    )

    # ---- Headline: gauge + outcome cards ----
    st.markdown("---")
    g_col, k_col1, k_col2, k_col3 = st.columns([2, 1, 1, 1])
    with g_col:
        st.markdown("##### Predicted probability of disease")
        st.plotly_chart(_gauge(pred_proba), use_container_width=True)
    with k_col1:
        st.markdown(insight_card(
            "rose" if pred_class == 1 else "teal",
            "Model says",
            f"{ICONS['disease']} disease" if pred_class == 1 else f"{ICONS['no_disease']} no disease",
            f"Probability ≥ 0.5 → disease, else no disease.",
        ), unsafe_allow_html=True)
    with k_col2:
        st.markdown(insight_card(
            "rose" if true_label == 1 else "teal",
            "Actual label",
            f"{ICONS['disease']} disease" if true_label == 1 else f"{ICONS['no_disease']} no disease",
            "Ground truth — what the angiography eventually showed.",
        ), unsafe_allow_html=True)
    with k_col3:
        tone = "teal" if outcome.startswith(ICONS["tp"]) or outcome.startswith(ICONS["tn"]) else "amber"
        if outcome.startswith(ICONS["fn"]):
            tone = "rose"  # missed disease is worst
        st.markdown(insight_card(
            tone, "Outcome", outcome.split(" ", 1)[1].title(),
            "Whether the prediction matched the actual label.",
        ), unsafe_allow_html=True)

    st.markdown(
        f"<div class='callout' style='margin-top:18px;'>"
        f"{ICONS['info']} The <b>{model_pick.replace('_',' ')}</b> model classified "
        f"this record as <b>{'disease present' if pred_class == 1 else 'no disease'}</b> "
        f"with a predicted probability of <b>{pred_proba:.0%}</b>. The drivers below "
        f"explain <i>why the model leaned that way</i> — they are a model-derived "
        f"explanation, not a clinical assessment.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ---- Clinical record + SHAP bars ----
    left, right = st.columns([2, 3])
    with left:
        st.markdown("##### Clinical record (plain language)")
        rows = []
        for feature in ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                        "thalach", "exang", "oldpeak", "slope", "ca", "thal"]:
            rows.append({
                "Variable": FEATURE_LABEL[feature],
                "Value": value_label(feature, patient_row[feature]),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            f"{ICONS['info']} Hover any bar on the right to see what each "
            "medical term means."
        )

    with right:
        st.markdown("##### SHAP contributions — ranked by impact")
        shap_rows["|shap|"] = shap_rows["shap_value"].abs()
        shap_rows = shap_rows.sort_values("|shap|", ascending=True).copy()

        def _row_label(r):
            arrow = ICONS["increase"] if r["shap_value"] > 0 else ICONS["decrease"]
            return f"{arrow}  {FEATURE_LABEL[r['feature_name']]}: {value_label(r['feature_name'], r['feature_value'])}"

        shap_rows["row_label"] = shap_rows.apply(_row_label, axis=1)
        shap_rows["help_text"] = shap_rows["feature_name"].map(FEATURE_HELP)

        fig = px.bar(
            shap_rows, x="shap_value", y="row_label", orientation="h",
            color="direction", color_discrete_map=DIRECTION_PALETTE,
            labels={"shap_value": "Impact on prediction (signed SHAP)",
                    "y": "", "direction": ""},
            custom_data=["help_text", "feature_name"],
        )
        fig.update_traces(
            marker_line_color="white", marker_line_width=1,
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "Impact: %{x:+.3f}<br><br>"
                "%{customdata[0]}<extra></extra>"
            ),
        )
        fig.add_vline(x=0, line_color="#444", line_width=1)
        st.plotly_chart(apply_plotly_layout(fig, height=540), use_container_width=True)
        st.caption(
            f"{ICONS['increase']} red bars push toward disease · "
            f"{ICONS['decrease']} blue bars push away from disease. "
            "Together with the model's baseline, they sum to the predicted "
            "log-odds, which produces the probability gauge above."
        )

    # ---- Plain-language summary ----
    st.markdown("---")
    st.markdown("##### What this record's prediction means — in plain terms")
    top_pos = (shap_rows[shap_rows["shap_value"] > 0]
               .sort_values("shap_value", ascending=False).head(3))
    top_neg = (shap_rows[shap_rows["shap_value"] < 0]
               .sort_values("shap_value", ascending=True).head(3))

    sm1, sm2 = st.columns(2)
    with sm1:
        st.markdown(
            f"<h6 style='color:{DISEASE};'>{ICONS['increase']} Pushing prediction toward disease</h6>",
            unsafe_allow_html=True,
        )
        if len(top_pos):
            for _, r in top_pos.iterrows():
                label = value_label(r["feature_name"], r["feature_value"])
                st.markdown(
                    f"- **{FEATURE_LABEL[r['feature_name']]}**: {label}  "
                    f"<span style='color:{NEUTRAL_LIGHT};'>(impact +{r['shap_value']:.2f})</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("*No features pushed toward disease.*")

    with sm2:
        st.markdown(
            f"<h6 style='color:{NO_DISEASE};'>{ICONS['decrease']} Pushing prediction away from disease</h6>",
            unsafe_allow_html=True,
        )
        if len(top_neg):
            for _, r in top_neg.iterrows():
                label = value_label(r["feature_name"], r["feature_value"])
                st.markdown(
                    f"- **{FEATURE_LABEL[r['feature_name']]}**: {label}  "
                    f"<span style='color:{NEUTRAL_LIGHT};'>(impact {r['shap_value']:.2f})</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("*No features pushed away from disease.*")

    st.markdown(
        f"<div class='callout' style='margin-top:18px;'>"
        f"<b>How to read this</b>: each variable nudged the model's prediction "
        f"up ({ICONS['increase']}) or down ({ICONS['decrease']}). The bigger the "
        f"impact, the more weight the model placed on that variable for this "
        f"specific record. Hover a bar above to see the medical meaning.</div>",
        unsafe_allow_html=True,
    )


record_panel()
st.markdown(DISCLAIMER_HTML, unsafe_allow_html=True)
