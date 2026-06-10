"""Exploratory data analysis figures for cleaned_patient_records.

Reads from MySQL (cleaned_patient_records) and writes six PNG figures to
notebooks/figures/ that are intended for the README, the model card, and the
Power BI dashboard's "exploration" page.

Style choices: seaborn whitegrid, target encoded as red (disease) vs blue
(no disease) consistently across all figures. Figures are 150 DPI, sized to be
embeddable in a Power BI tile or a Markdown document without resizing.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.db import PROJECT_ROOT, get_engine

FIG_DIR = PROJECT_ROOT / "notebooks" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

PALETTE = {0: "#3B82C4", 1: "#C43B3B"}   # blue = no disease, red = disease
TARGET_LABELS = {0: "no disease", 1: "disease"}


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"


def _load() -> pd.DataFrame:
    sql = """
        SELECT patient_id, age, sex, cp, trestbps, chol, fbs, restecg, thalach,
               exang, oldpeak, slope, ca, thal, num, target_binary
        FROM cleaned_patient_records
    """
    df = pd.read_sql(sql, get_engine())
    df["target_label"] = df["target_binary"].map(TARGET_LABELS)
    return df


def fig01_target_distribution(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    counts = df["target_binary"].value_counts().sort_index()
    bars = ax.bar(
        [TARGET_LABELS[i] for i in counts.index],
        counts.values,
        color=[PALETTE[i] for i in counts.index],
        edgecolor="white",
    )
    for bar, n in zip(bars, counts.values):
        pct = n / len(df) * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{n} ({pct:.1f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("patients")
    ax.set_title(f"Target distribution (n = {len(df)})")
    ax.set_ylim(0, counts.max() * 1.15)
    path = FIG_DIR / "01_target_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig02_numeric_kde(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, col in zip(axes.flat, NUMERIC_COLS):
        for t in (0, 1):
            sns.kdeplot(
                df.loc[df["target_binary"] == t, col],
                ax=ax, fill=True, alpha=0.35,
                color=PALETTE[t], label=TARGET_LABELS[t],
                bw_adjust=0.9, linewidth=1.2,
            )
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.legend(loc="best", frameon=False)
    fig.suptitle("Numeric feature distributions, stratified by target", y=1.02)
    fig.tight_layout()
    path = FIG_DIR / "02_numeric_kde_by_target.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig03_numeric_box(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, col in zip(axes.flat, NUMERIC_COLS):
        sns.boxplot(
            data=df, x="target_label", y=col, ax=ax,
            hue="target_label",
            palette={TARGET_LABELS[0]: PALETTE[0], TARGET_LABELS[1]: PALETTE[1]},
            order=[TARGET_LABELS[0], TARGET_LABELS[1]],
            width=0.5, linewidth=1.1, fliersize=3, legend=False,
        )
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Numeric features by target — boxplots", y=1.02)
    fig.tight_layout()
    path = FIG_DIR / "03_numeric_box_by_target.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig04_categorical(df: pd.DataFrame) -> Path:
    n = len(CATEGORICAL_COLS)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.2 * nrows))
    axes_flat = axes.flat
    for ax, col in zip(axes_flat, CATEGORICAL_COLS):
        tab = (df.groupby([col, "target_binary"]).size()
                 .unstack(fill_value=0)
                 .rename(columns=TARGET_LABELS))
        tab.plot(kind="bar", stacked=False, ax=ax,
                 color=[PALETTE[0], PALETTE[1]],
                 width=0.75, edgecolor="white")
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("patients")
        ax.tick_params(axis="x", labelrotation=20)
        ax.legend(title=None, loc="best", frameon=False, fontsize=8)
    # hide unused panels
    for ax in list(axes_flat)[n:]:
        ax.set_visible(False)
    fig.suptitle("Categorical features by target", y=1.0)
    fig.tight_layout()
    path = FIG_DIR / "04_categorical_by_target.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig05_correlation(df: pd.DataFrame) -> Path:
    # Numeric correlation including the binary target for quick screening
    numeric_with_target = NUMERIC_COLS + ["target_binary"]
    corr = df[numeric_with_target].corr()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, square=True, linewidths=0.4,
        cbar_kws={"shrink": 0.7}, ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title("Numeric correlations (Pearson)")
    path = FIG_DIR / "05_correlation_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig06_age_sex_target(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, sex_val in zip(axes, ["female", "male"]):
        sub = df[df["sex"] == sex_val]
        for t in (0, 1):
            sns.kdeplot(
                sub.loc[sub["target_binary"] == t, "age"],
                ax=ax, fill=True, alpha=0.35,
                color=PALETTE[t], label=TARGET_LABELS[t],
                bw_adjust=0.9, linewidth=1.2,
            )
        ax.set_title(f"{sex_val}  (n = {len(sub)})")
        ax.set_xlabel("age")
        ax.set_ylabel("")
        ax.legend(loc="best", frameon=False)
    fig.suptitle("Age distribution by sex and target", y=1.02)
    fig.tight_layout()
    path = FIG_DIR / "06_age_sex_target.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def run() -> list[Path]:
    _setup_style()
    df = _load()
    paths = [
        fig01_target_distribution(df),
        fig02_numeric_kde(df),
        fig03_numeric_box(df),
        fig04_categorical(df),
        fig05_correlation(df),
        fig06_age_sex_target(df),
    ]
    return paths


if __name__ == "__main__":
    paths = run()
    print("==> EDA figures saved:")
    for p in paths:
        print(f"  {p.relative_to(PROJECT_ROOT)}")
