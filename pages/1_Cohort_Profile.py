"""Cohort profile — filterable EDA across the 13 features."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from streamlit_utils.data import load_cohort
from streamlit_utils.glossary import FEATURE_LABEL
from streamlit_utils.styles import (
    DISCLAIMER_HTML, DIVERGING_RDBL, ICONS, PAGE_CSS, TARGET_LABEL_PALETTE,
    apply_plotly_layout,
)

st.set_page_config(page_title="Cohort Profile", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)
st.title(f"{ICONS['cohort']} Cohort profile")
st.markdown(
    "<p style='color:#6B7280;'>Demographic and clinical distributions across "
    "the 303-record cohort, stratified by heart-disease status. Use the "
    "filters on the left to focus on a sub-cohort.</p>",
    unsafe_allow_html=True,
)

df = load_cohort()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    sex_pick = st.multiselect("Sex", ["male", "female"], default=["male", "female"])
    cp_pick = st.multiselect(
        "Chest pain type",
        ["typical_angina", "atypical_angina", "non_anginal_pain", "asymptomatic"],
        default=["typical_angina", "atypical_angina", "non_anginal_pain", "asymptomatic"],
    )
    age_min, age_max = int(df["age"].min()), int(df["age"].max())
    age_range = st.slider("Age range", age_min, age_max, (age_min, age_max))

mask = (
    df["sex"].isin(sex_pick)
    & df["cp"].isin(cp_pick)
    & df["age"].between(*age_range)
)
sub = df[mask]

# ---------------------------------------------------------------------------
# Headline numbers
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Records in filter", f"{len(sub):,}", f"of {len(df):,}",
          delta_color="off")
k2.metric(f"{ICONS['disease']} Disease prevalence",
          f"{sub['target_binary'].mean():.1%}" if len(sub) else "—")
k3.metric("Median age", f"{int(sub['age'].median())}" if len(sub) else "—")
k4.metric("Median cholesterol", f"{int(sub['chol'].median())}" if len(sub) else "—")

if len(sub) == 0:
    st.warning("No records match the current filter — relax one of the sidebar criteria.")
    st.stop()

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_age, tab_cat, tab_num, tab_corr = st.tabs(
    ["Age & sex", "Categorical features", "Numeric features", "Correlations"]
)

with tab_age:
    a1, a2 = st.columns(2)
    with a1:
        order = ["<40", "40-49", "50-59", "60-69", "70+"]
        fig = px.histogram(
            sub, x="age_band", color="target_label", barmode="stack",
            category_orders={"age_band": order, "target_label": ["no disease", "disease"]},
            color_discrete_map=TARGET_LABEL_PALETTE,
            labels={"age_band": "Age band", "target_label": "Status", "count": "Records"},
            title="Age band × heart-disease status",
        )
        st.plotly_chart(apply_plotly_layout(fig, height=400), use_container_width=True)
    with a2:
        sex_target = sub.groupby(["sex", "target_label"]).size().reset_index(name="n")
        fig = px.bar(
            sex_target, x="sex", y="n", color="target_label", barmode="stack",
            color_discrete_map=TARGET_LABEL_PALETTE,
            category_orders={"target_label": ["no disease", "disease"]},
            labels={"sex": "Sex", "n": "Records", "target_label": "Status"},
            title="Sex × heart-disease status",
        )
        st.plotly_chart(apply_plotly_layout(fig, height=400), use_container_width=True)
    st.caption(
        "Disease prevalence in this cohort skews male and concentrates in the "
        "50–69 age bands — a pattern consistent with referral-based cardiology "
        "samples of this era."
    )

with tab_cat:
    pairs = [
        ("cp", "Chest pain type"),
        ("thal", "Thalassemia status"),
        ("slope", "ST-segment slope"),
        ("restecg", "Resting ECG"),
        ("exang", "Exercise-induced angina"),
        ("fbs", "Fasting blood sugar > 120 mg/dl"),
    ]
    rows = [pairs[i:i+2] for i in range(0, len(pairs), 2)]
    for row in rows:
        cols = st.columns(2)
        for (col, label), c in zip(row, cols):
            with c:
                tab = sub.groupby([col, "target_label"]).size().reset_index(name="n")
                fig = px.bar(
                    tab, x=col, y="n", color="target_label", barmode="group",
                    color_discrete_map=TARGET_LABEL_PALETTE,
                    category_orders={"target_label": ["no disease", "disease"]},
                    labels={col: label, "n": "Records", "target_label": "Status"},
                    title=f"{label} × status",
                )
                st.plotly_chart(apply_plotly_layout(fig, height=340), use_container_width=True)

with tab_num:
    feature = st.selectbox(
        "Numeric feature",
        ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"],
        index=2,
        format_func=lambda x: FEATURE_LABEL[x],
    )
    nb1, nb2 = st.columns(2)
    with nb1:
        fig = px.histogram(
            sub, x=feature, color="target_label", marginal="box",
            nbins=30, barmode="overlay", opacity=0.55,
            color_discrete_map=TARGET_LABEL_PALETTE,
            category_orders={"target_label": ["no disease", "disease"]},
            title=f"{feature} — distribution by status",
        )
        st.plotly_chart(apply_plotly_layout(fig, height=440), use_container_width=True)
    with nb2:
        fig = px.scatter(
            sub, x="age", y=feature, color="target_label",
            color_discrete_map=TARGET_LABEL_PALETTE,
            opacity=0.7, hover_data=["patient_id", "sex", "cp"],
            category_orders={"target_label": ["no disease", "disease"]},
            title=f"{feature} vs age",
        )
        fig.update_traces(marker=dict(size=8))
        st.plotly_chart(apply_plotly_layout(fig, height=440), use_container_width=True)

with tab_corr:
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "target_binary"]
    corr = sub[numeric_cols].corr().round(3)
    fig = px.imshow(
        corr, text_auto=True, color_continuous_scale=DIVERGING_RDBL,
        zmin=-1, zmax=1, aspect="auto",
        title="Pearson correlation — numeric features + target",
    )
    st.plotly_chart(apply_plotly_layout(fig, height=500), use_container_width=True)
    st.caption(
        "`ca`, `oldpeak`, and (negatively) `thalach` are the strongest numeric "
        "correlates with the binary target. `chol` is much weaker than the "
        "lay narrative would suggest — a finding the SHAP analysis confirms."
    )

st.markdown(DISCLAIMER_HTML, unsafe_allow_html=True)
