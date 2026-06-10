# Tableau Public Dashboard — Build Guide

This guide walks through building the project's 5-page dashboard in Tableau
Public. All data lives in `data/outputs/tableau/` as 8 CSV files — Tableau
Public can connect to them directly.

> Heads-up: Tableau Public saves your workbooks to the Tableau Public cloud,
> not locally. This is fine for a portfolio piece (you get a shareable URL),
> but you cannot keep a workbook private here. If you want private drafts,
> save the `.twb` file locally via *File → Save As* before publishing.

---

## 0. Install + first-time setup (10 min)

1. Go to **https://public.tableau.com/** → **Create your Tableau Public profile**.
   Use any email; no organizational requirement.
2. Download **Tableau Desktop Public Edition** for macOS (free).
3. Open it. The home screen offers *Connect* → *To a File* → *Text File*.

---

## 1. Connect the data (5 min)

You will use **all 8 CSV files** as separate data sources (Tableau Public
treats each CSV as its own source — switch via the *Data* pane at left).

1. *Connect* → *Text File* → navigate to
   `data/outputs/tableau/cohort_cleaned.csv`. Click *Open*.
2. Drag the file into the canvas if not already there. Tableau will infer
   types. Verify:
   - `patient_id` → number (whole)
   - `age`, `chol`, `thalach`, `trestbps` → number (whole)
   - `oldpeak` → number (decimal)
   - everything else categorical → string
3. Repeat *Data → New Data Source → Text File* for the other 7 CSVs:
   - `model_metrics_aggregate.csv`
   - `model_metrics_per_fold.csv`
   - `predictions.csv`
   - `model_curves.csv`
   - `threshold_grid.csv`
   - `shap_global.csv`
   - `shap_patient.csv`

You should now see 8 data sources listed in the left **Data** pane.

---

## 2. Recommended calculated fields (10 min)

Before building visuals, create these calculated fields on each relevant
source — they make every page cleaner.

### On `cohort_cleaned`

| Calc field name | Formula | Notes |
| --- | --- | --- |
| `Disease (count)` | `IF [target_binary] = 1 THEN 1 END` | Used for prevalence cards |
| `Cohort size` | `COUNT([patient_id])` | Card |
| `Disease prevalence %` | `SUM([Disease (count)]) / COUNT([patient_id])` | Format as percentage |

### On `model_metrics_aggregate`

| Calc field | Formula |
| --- | --- |
| `Metric value (rounded)` | `ROUND([Metric Value], 3)` |
| `Is real model` | `[Model Type] != "dummy"` |

### On `predictions`

| Calc field | Formula |
| --- | --- |
| `Confusion cell` | `[True Label Str] + " / " + [Predicted Label]` |

### On `shap_patient`

| Calc field | Formula |
| --- | --- |
| `\|SHAP value\|` | `ABS([Shap Value])` |
| `Direction color` | `IF [Shap Value] > 0 THEN "increases risk" ELSE "decreases risk" END` |

---

## 3. Page 1 — Project Overview

**Data source:** `cohort_cleaned` + `model_metrics_aggregate`.

Goal: a stakeholder-friendly snapshot — what the project is and what it does.

| Visual | Type | Setup |
| --- | --- | --- |
| **Title** text block | Text | "Explainable Heart Disease Risk Classification" |
| **Cohort size** card | Big number | `cohort_cleaned` → `Cohort size` calc → BAN |
| **Disease prevalence %** card | Big number | `cohort_cleaned` → `Disease prevalence %` |
| **No disease %** card | Big number | `1 - [Disease prevalence %]` |
| **Features used** card | Constant: 13 | Text only |
| **Best model** card | Text: "Logistic Regression" | Text only |
| **Best ROC AUC** card | BAN | `model_metrics_aggregate` filter `Metric Name = roc_auc` and `Model Type = logistic_regression`, drag `Metric Value` to text |
| **Best Recall** card | BAN | Same source, filter `Metric Name = recall` |
| **Privacy note** | Text block | "Public de-identified UCI dataset. Not for clinical use." |
| **Limitations note** | Text block | "n=303, 1980s Cleveland cohort, no external validation." |

Arrange in a tile-grid dashboard (Tableau *Dashboard → New Dashboard* → drag
sheets into a 2-row layout: 4 cards top, longer text bottom).

---

## 4. Page 2 — Cohort / Data Profile

**Data source:** `cohort_cleaned`.

Use *cohort* (not "patients") in titles — health-data convention.

| Sheet | Visual | Setup |
| --- | --- | --- |
| **Age band by target** | Stacked bar | Columns: `age_band` (sorted); Rows: `Count of cohort_cleaned`; Color: `target_label` (red = disease, blue = no disease) |
| **Disease % by sex** | Clustered bar | Columns: `sex`; Rows: `Disease prevalence %`; Color: `sex` |
| **Chest pain type by target** | Stacked bar | Columns: `cp` (sorted by counts); Rows: count; Color: `target_label` |
| **Cholesterol distribution** | Histogram | Use `chol`, bin width = 25, color split by `target_label` |
| **Thalach vs Age** | Scatter | x = `age`, y = `thalach`, color = `target_label`, marker = circle |

Add a single **filter shelf** on the dashboard with `sex` and `cp` for
interactive cohort slicing.

---

## 5. Page 3 — Model Performance

**Data sources:** `model_metrics_aggregate`, `model_metrics_per_fold`,
`predictions`, `model_curves`, `threshold_grid`.

| Sheet | Visual | Setup |
| --- | --- | --- |
| **Metric comparison** | Table / heatmap | `model_metrics_aggregate`: Rows = `Model Type`; Columns = `Metric Name`; Color & text = `Metric Value` |
| **Per-fold ROC AUC boxplot** | Box plot | `model_metrics_per_fold` filtered `Metric Name = roc_auc`; Columns = `Model Type`; Rows = `Metric Value`; Show *Box Plot* from Analytics tab |
| **Confusion matrix** | Heatmap | `predictions` filtered to one model (slicer below); Rows = `True Label Str`; Columns = `Predicted Label`; Color and text = count |
| **ROC curve** | Line | `model_curves` filtered `Curve = ROC`; Columns = `X Value` (FPR); Rows = `Y Value` (TPR); Color = `Model Type`. Add a 45° reference line. |
| **PR curve** | Line | `model_curves` filtered `Curve = PR`; Columns = `X Value` (Recall); Rows = `Y Value` (Precision); Color = `Model Type` |
| **Threshold sweep** | Multi-line | `threshold_grid`: Columns = `threshold`; Rows = `recall`, `precision`, `false_positive`, `false_negative` (drag the 4 measures on top of each other to dual-axis stack) |

**Threshold slicer (the most "health analytics" piece)**: on the dashboard,
add a **Parameter** named `Threshold`, range 0.05 – 0.95 step 0.05. Create a
calculated field `Predicted at param threshold = IF [Predicted Probability]
>= [Threshold] THEN 1 ELSE 0 END`. Use this in a secondary confusion-matrix
sheet so moving the slider live-updates the matrix. This makes the page feel
genuinely interactive in 10 seconds of demo.

---

## 6. Page 4 — SHAP Global Explainability

**Data source:** `shap_global`.

> Suggested intro text on the page:
> *"The dashboard separates model performance from model explanation. A model
>  can perform well statistically but still require interpretability before
>  being used in a health analytics context."*

| Sheet | Visual | Setup |
| --- | --- | --- |
| **Top features (LR)** | Horizontal bar | Filter `Model Type = logistic_regression`; Columns = `Mean Abs Shap`; Rows = `Feature Name` sorted by importance |
| **Top features (RF)** | Horizontal bar | Same with `Model Type = random_forest` |
| **Side-by-side rank** | Slope chart | Columns = `Model Type`; Rows = `Rank Position` (reverse axis); Color = `Feature Name`; Path = `Feature Name`. Shows feature stability across the two model families. |

The slope chart is the storytelling visual: it shows which features both
models agree are top drivers (lines that stay flat near the top) versus
features one model weighs more than the other.

---

## 7. Page 5 — Individual Record Explanation

This is the **dashboard's highlight page** — patient-level SHAP drill-through.

**Data source:** `shap_patient`.

| Visual | Setup |
| --- | --- |
| **Record selector** | Parameter `Patient ID` (integer list of all 303 ids) shown as a dropdown |
| **Header cards** | Filter by `Patient ID`. Cards show `Predicted Probability`, `Predicted Label`, `True Label Str`. **Wording tip**: title the card "This record was classified as…" *not* "This patient has…" — the safer phrasing for a health prototype. |
| **Top positive SHAP drivers** | Horizontal bar filtered to `Direction = increases risk` and one model (default logistic_regression). Columns = `Shap Value`; Rows = `Feature Name` + `Feature Value`; Color = red |
| **Top negative SHAP drivers** | Same as above with `Direction = decreases risk`, Color = blue |
| **All features waterfall-style** | Bar chart sorted by `|SHAP value|`, color by `Direction`. Visually mimics the SHAP waterfall from the model card. |

Optionally add a small text block:

> "This record was classified as **{Predicted Label}** by the model.
>  The features pushing the prediction toward 'disease' are shown in red;
>  the features pushing it away are shown in blue. This is a model-derived
>  explanation, not a clinical assessment."

---

## 8. Publish (5 min)

1. *Server → Tableau Public → Save to Tableau Public As…*
2. Sign in with your Tableau Public account.
3. Workbook name: `Explainable Heart Disease Risk Classification`.
4. After upload, Tableau opens the workbook in the browser. Copy that URL.
5. Update [README.md](../README.md) and your résumé with that link.

> **One-time note**: Tableau Public requires workbooks to be public. There is
> no private option in the free tier. Since your data is the public UCI
> Cleveland dataset, this is fine — but never use Tableau Public for real PHI.

---

## 9. Style / polish checklist before publishing

- Consistent palette across all pages: **red = disease / increases risk**,
  **blue = no disease / decreases risk**.
- Use *Format → Workbook* to set a body font (Arial / Avenir / Tableau Light).
- On every page, add a footer text block:
  *"Educational prototype. Not for clinical decision-making.
   Source: UCI Cleveland Heart Disease (1988). Model: logistic regression,
   10-fold CV. See [model_card.md](model_card.md) and
   [data_quality_report.md](data_quality_report.md)."*
- Hide unused axis titles and gridlines (right-click axes → uncheck *Show Header*).
- Tooltip on every visual: include feature name, count, and the
  health-context phrase (e.g. "1 of 303 records").

---

## 10. Re-running the export

If you retrain models or add more, just re-run:

```bash
.venv/bin/python -m src.export_for_tableau
```

This overwrites the 8 CSVs in `data/outputs/tableau/`. Reopen the workbook
and Tableau will pick up the new data on next refresh
(*Data → Refresh All Extracts*).
