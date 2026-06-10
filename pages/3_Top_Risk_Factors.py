"""Top risk factors — SHAP global importance with plain-language framing."""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_utils.data import load_shap_global
from streamlit_utils.glossary import FEATURE_HELP, FEATURE_LABEL
from streamlit_utils.styles import (
    DISCLAIMER_HTML, ICONS, MODEL_PALETTE, NEUTRAL_LIGHT, PAGE_CSS,
    PRIMARY_DEEP, apply_plotly_layout,
)

st.set_page_config(page_title="Top risk factors", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)
st.title(f"{ICONS['risk']} Top risk factors")
st.markdown(
    "<p style='color:#6B7280;'>Which clinical variables drive the model's "
    "predictions across the whole cohort, measured by SHAP. The view separates "
    "<i>model performance</i> (the previous page) from <i>model explanation</i> — "
    "a model can perform well statistically and still need interpretability before "
    "being used in a health-analytics context.</p>",
    unsafe_allow_html=True,
)

shap_global = load_shap_global()
models_available = sorted(shap_global["model_type"].unique())

tabs = st.tabs(["Single model", "Cross-model comparison"])

# ---------------------------------------------------------------------------
# Single model
# ---------------------------------------------------------------------------
with tabs[0]:
    model_pick = st.selectbox("Model", models_available, index=0)
    sub = (shap_global[shap_global["model_type"] == model_pick]
           .sort_values("mean_abs_shap", ascending=True).copy())
    sub["display_name"] = sub["feature_name"].map(FEATURE_LABEL).fillna(sub["feature_name"])
    sub["help_text"] = sub["feature_name"].map(FEATURE_HELP)
    fig = px.bar(
        sub, x="mean_abs_shap", y="display_name", orientation="h",
        labels={"mean_abs_shap": "Average impact on prediction (mean |SHAP|)",
                "display_name": ""},
        title=f"Feature impact ranking — {model_pick.replace('_', ' ')}",
        color_discrete_sequence=[MODEL_PALETTE.get(model_pick, PRIMARY_DEEP)],
        custom_data=["feature_name", "help_text"],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>raw code: <i>%{customdata[0]}</i><br>"
            "mean |SHAP| = %{x:.3f}<br><br>%{customdata[1]}<extra></extra>"
        ),
        marker_line_color="white", marker_line_width=1,
    )
    st.plotly_chart(apply_plotly_layout(fig, height=540), use_container_width=True)

    top3 = sub.tail(3)
    top3_names = top3["display_name"].tolist()[::-1]
    st.markdown(
        f"**Top three drivers** for `{model_pick.replace('_', ' ')}`: "
        f"**{top3_names[0]}** → **{top3_names[1]}** → **{top3_names[2]}**."
    )
    st.caption(
        "Higher bar = the feature changes the model's predicted probability more "
        "on average across the cohort. This is a 'how much the model leans on it' "
        "ranking, not a 'how often the feature is abnormal' ranking."
    )

# ---------------------------------------------------------------------------
# Cross-model
# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown(
        "How does each feature's **rank** change between the linear and tree "
        "models? Lines that stay flat near the top show feature-importance "
        "stability across model families — the strongest signal."
    )
    fig = go.Figure()
    features = sorted(shap_global["feature_name"].unique())
    for feat in features:
        label = FEATURE_LABEL.get(feat, feat)
        sub = shap_global[shap_global["feature_name"] == feat].sort_values("model_type")
        fig.add_trace(go.Scatter(
            x=sub["model_type"], y=sub["rank_position"],
            mode="lines+markers+text", name=label,
            text=[label] * len(sub),
            textposition="middle right", textfont=dict(size=10),
            hovertemplate=f"%{{x}}<br>rank %{{y}}<br><b>{label}</b><extra></extra>",
            line=dict(width=2),
            marker=dict(size=10, line=dict(color="white", width=1.5)),
        ))
    fig.update_yaxes(autorange="reversed", title_text="Rank (1 = most important)")
    fig.update_xaxes(title_text="")
    fig.update_layout(
        title="Feature-importance rank — linear vs tree models",
        showlegend=False,
    )
    st.plotly_chart(apply_plotly_layout(fig, height=540), use_container_width=True)
    st.caption(
        "Top three drivers — `ca`, `cp`, `thal` — agree across model families, "
        "matching the cardiology literature on coronary-artery-disease risk. "
        "`chol` and `fbs` sit near the bottom in both models — consistent with "
        "modern CAD research that downgrades isolated cholesterol as an "
        "independent risk signal."
    )

st.markdown(DISCLAIMER_HTML, unsafe_allow_html=True)
