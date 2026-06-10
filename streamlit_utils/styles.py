"""Visual style — "陶碗酒痕" Chinese-aesthetic palette + icon vocabulary.

Sophisticated, earthy, low-saturation tones picked to feel like a curated
clinical-report rather than a generic ML dashboard:
  - 瓷釉青 (porcelain glaze teal) — calm, neutral, "no disease"
  - 酒痕红 (wine-stain red)       — warm, present, "disease"
  - 杏皮褐 (apricot brown)        — accent / current selection
  - 釉面蓝 (glaze blue)           — soft callout backgrounds
"""
from __future__ import annotations

# Primary semantic tones (陶碗酒痕)
DISEASE = "#D48982"            # 酒痕红 — disease / increases risk
NO_DISEASE = "#99B6B4"         # 瓷釉青 — no disease / decreases risk
ACCENT_AMBER = "#DFB199"       # 杏皮褐 — accent, current selection
PRIMARY_DEEP = "#5A6B70"       # darker teal-gray — titles, anchors
NEUTRAL_DARK = "#6B7273"       # warm dark gray — body text
NEUTRAL_LIGHT = "#BAC4C3"      # warm light gray — secondary text, gridlines
HIGHLIGHT_BG = "#BACFCE"       # 釉面蓝 — soft background panels
WARM_CREAM = "#F4EFEA"         # warm off-white — page tint

TARGET_PALETTE = {0: NO_DISEASE, 1: DISEASE}
TARGET_LABEL_PALETTE = {"no disease": NO_DISEASE, "disease": DISEASE}
DIRECTION_PALETTE = {
    "increases risk": DISEASE,
    "decreases risk": NO_DISEASE,
    "neutral": NEUTRAL_LIGHT,
}

# Categorical model palette — pulled from the same warm family
MODEL_PALETTE = {
    "dummy": "#BAC4C3",
    "logistic_regression": "#5A6B70",
    "logistic_regression_l1": "#99B6B4",
    "random_forest": "#7B9E89",
    "hist_gradient_boosting": "#D48982",
    "hist_gradient_boosting_calibrated": "#B79E8A",
}

# Sequential heatmap — single-hue teal-gray
SEQUENTIAL_NAVY = [
    [0.0, "#F4EFEA"], [0.25, "#D6E0DF"], [0.5, "#BACFCE"],
    [0.75, "#99B6B4"], [1.0, "#5A6B70"],
]

# Diverging scale for correlations / SHAP
DIVERGING_RDBL = [
    [0.0, "#5A6B70"], [0.25, "#99B6B4"], [0.5, "#F4EFEA"],
    [0.75, "#DFB199"], [1.0, "#D48982"],
]

# ---------------------------------------------------------------------------
# Icon vocabulary — used everywhere so meaning stays consistent
# ---------------------------------------------------------------------------
ICONS = {
    # Status / target
    "disease": "🫀",            # heart — has disease
    "no_disease": "✓",          # checkmark — healthy / no disease
    # Risk direction (used in SHAP)
    "increase": "▲",           # up triangle — increases risk
    "decrease": "▼",           # down triangle — decreases risk
    # Outcome (confusion matrix)
    "tp": "🎯",                 # bullseye — correctly identified disease
    "fn": "⚠️",                 # warning — missed disease
    "fp": "❗",                 # exclamation — over-flagged
    "tn": "·",                  # neutral dot
    # Page section
    "cohort": "👥",
    "performance": "📊",
    "risk": "⚕",                # medical caduceus
    "record": "📋",
    # Meta
    "info": "ℹ️",
    "warn": "⚠️",
}


# ---------------------------------------------------------------------------
# Disclaimer banner
# ---------------------------------------------------------------------------
DISCLAIMER_HTML = (
    f"<div style='background:{WARM_CREAM}; padding:12px 16px; border-left:4px solid "
    f"{ACCENT_AMBER}; border-radius:4px; font-size:0.88em; color:#3F3F3F;'>"
    f"<b>{ICONS['warn']} Educational prototype.</b> Not for clinical decision-making. "
    "Built on the public UCI Cleveland Heart Disease dataset (n = 303, "
    "1981–84 referral cohort). Predicted probabilities reflect this historical "
    "sample and should not be applied to current patients."
    "</div>"
)


def apply_plotly_layout(fig, height: int | None = None,
                        legend_orientation: str = "h"):
    fig.update_layout(
        template="simple_white",
        font=dict(family="Inter, -apple-system, Helvetica Neue, Arial, sans-serif",
                  size=12, color=NEUTRAL_DARK),
        title=dict(font=dict(size=14, color=PRIMARY_DEEP)),
        margin=dict(l=40, r=20, t=55, b=40),
        legend=dict(
            orientation=legend_orientation, yanchor="bottom",
            y=1.02, xanchor="right", x=1, title_text="",
            font=dict(size=11),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#ECECEC", linecolor=NEUTRAL_LIGHT, title_font_size=11)
    fig.update_yaxes(gridcolor="#ECECEC", linecolor=NEUTRAL_LIGHT, title_font_size=11)
    if height:
        fig.update_layout(height=height)
    return fig


PAGE_CSS = """
<style>
  .block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1280px; }

  h1 { color: #5A6B70; font-weight: 600; letter-spacing: -0.5px; }
  h2 { color: #5A6B70; font-weight: 600; letter-spacing: -0.3px; }
  h3 { color: #5A6B70; font-weight: 600; }
  h5 { color: #6B7273; font-weight: 600; text-transform: uppercase;
       font-size: 0.78rem; letter-spacing: 1px; }

  [data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E5E2DD;
    border-radius: 8px;
    padding: 14px 18px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  }
  [data-testid="stMetricLabel"] {
    color: #6B7273 !important;
    text-transform: uppercase;
    font-size: 0.7rem;
    letter-spacing: 0.8px;
  }
  [data-testid="stMetricValue"] {
    color: #5A6B70 !important;
    font-weight: 600;
  }

  [data-testid="stSidebar"] { background-color: #F4EFEA; }
  [data-testid="stSidebar"] h2 { color: #5A6B70; }

  .caption, [data-testid="stCaptionContainer"] { color: #6B7273; }

  /* Soft callout box helper class */
  .callout {
    background: #F4EFEA;
    border-left: 4px solid #DFB199;
    border-radius: 4px;
    padding: 10px 14px;
    font-size: 0.92em;
    color: #3F3F3F;
    margin: 6px 0;
  }

  /* Insight cards — pastel-tinted soft cards (intelly-style) */
  .insight-card {
    background: #F4EFEA;
    border-radius: 12px;
    padding: 16px 18px;
    height: 100%;
    border: 1px solid rgba(0,0,0,0.04);
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  .insight-card.teal   { background: #E7F0EF; border-left: 3px solid #99B6B4; }
  .insight-card.amber  { background: #FAEDDF; border-left: 3px solid #DFB199; }
  .insight-card.rose   { background: #F7E4E1; border-left: 3px solid #D48982; }
  .insight-card.blue   { background: #E8EEED; border-left: 3px solid #BACFCE; }
  .insight-card h6 {
    color: #5A6B70;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin: 0 0 8px 0;
    font-weight: 600;
  }
  .insight-card .stat {
    font-size: 1.6rem;
    color: #5A6B70;
    font-weight: 600;
    margin: 2px 0 4px 0;
    line-height: 1.1;
  }
  .insight-card .label {
    color: #6B7273;
    font-size: 0.85rem;
    line-height: 1.35;
  }

  /* Tab pills look slightly cleaner */
  [data-baseweb="tab-list"] { gap: 4px; }
  [data-baseweb="tab"] {
    border-radius: 6px 6px 0 0;
    padding: 6px 12px !important;
  }

  /* Section dividers */
  hr { border-color: #E5E2DD !important; }
</style>
"""


def insight_card(tone: str, header: str, stat: str, label: str) -> str:
    """Render a soft pastel insight card (HTML helper)."""
    return (
        f"<div class='insight-card {tone}'>"
        f"<h6>{header}</h6>"
        f"<div class='stat'>{stat}</div>"
        f"<div class='label'>{label}</div>"
        f"</div>"
    )
