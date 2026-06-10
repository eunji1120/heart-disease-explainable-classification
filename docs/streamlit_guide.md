# Streamlit Dashboard — Run Locally + Deploy to Streamlit Cloud

Five-page interactive dashboard (Home / Cohort / Performance / Top Risk Factors
/ Per-Record). Reads from the eight CSVs in `data/outputs/dashboard/` — no live
database connection required, so it deploys cleanly to Streamlit Cloud.

---

## 1. Run locally (3 commands)

```bash
# from project root, with the venv already created:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL (usually <http://localhost:8501>) — open it
in any browser. The sidebar exposes the five pages.

If you need to refresh the data after a re-train, rerun the export script:

```bash
python -m src.export_for_dashboard
```

`@st.cache_data` will pick up the new CSVs on the next browser refresh.

---

## 2. Deploy to Streamlit Cloud (free public URL)

Streamlit Cloud serves any GitHub repo's Streamlit app for free, gives it a
public URL of the form `https://<username>-<repo>.streamlit.app`. Workflow:

### One-time setup

1. Go to **<https://share.streamlit.io>** and sign in with your GitHub account.
2. Click **New app**.
3. Fill in:
   - **Repository:** `eunji1120/heart-disease-explainable-classification`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** customise the slug (e.g. `heart-disease-explainable`)
4. Click **Deploy**.

Streamlit Cloud will install `requirements.txt`, run `streamlit run app.py`,
and expose the public URL. First build takes ~3 minutes (installing pandas,
sklearn, shap, plotly). Subsequent commits to `main` auto-redeploy in
~30 seconds.

### Post-deploy

- Copy the public URL.
- Update [`README.md`](../README.md) — replace the placeholder *"will be added
  after Streamlit Cloud is deployed"* with the URL.
- Add it to your résumé / LinkedIn: hiring managers can click it and interact
  with the dashboard immediately.

---

## 3. Architecture notes (for portfolio/interview talking points)

- **`app.py`** is the Home page (KPI cards, top risk drivers, insight cards,
  model leaderboard, glossary). Streamlit auto-discovers further pages from
  the `pages/` directory.
- **`streamlit_utils/data.py`** holds every data load + every derived
  computation, all wrapped in `@st.cache_data`. Heavy work
  (calibration curves, subgroup ROC AUC, threshold sweeps) runs once per
  session, not per interaction.
- **`@st.fragment`** isolates the threshold slider and the patient-record
  selector, so those widgets only re-render their own block — meaningful
  perceived latency improvement on a 303-row dataset.
- **`streamlit_utils/glossary.py`** maps each raw clinical code (`ca`,
  `thal`, `oldpeak`, etc.) to a plain-language label and a tooltip
  explanation, so a non-cardiologist reading the dashboard understands what
  each variable means. Implements the HCI principle of *explainable
  variables*.
- **`streamlit_utils/styles.py`** centralises the palette
  (陶碗酒痕 — porcelain teal + wine-stain red + warm amber) and a small set
  of CSS classes for soft pastel "insight cards" — closer to a clinical-
  report look than a generic ML demo.

---

## 4. Customising the dashboard

| Want to change | Edit |
| --- | --- |
| Palette / typography | `streamlit_utils/styles.py` |
| Plain-language variable labels | `streamlit_utils/glossary.py` |
| KPI cards on the Home page | `app.py` |
| Sub-tabs in Model Performance | `pages/2_Model_Performance.py` |
| Which models get SHAP comparison | `pages/3_Top_Risk_Factors.py` + `src/explain.py` |
| Adding a new page | drop a `pages/N_Title.py` file — Streamlit auto-discovers |

---

## 5. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ModuleNotFoundError: streamlit` | `pip install -r requirements.txt` inside the venv |
| Streamlit Cloud build fails on `shap` | Check Python version. Streamlit Cloud picks 3.10/3.11/3.12; `shap` works fine on these. Locally we use 3.9 because of system Python. |
| The CSVs in `data/outputs/dashboard/` look stale | Rerun `python -m src.export_for_dashboard` from project root |
| Page is slow after editing | Add `@st.cache_data` to any new derived computation; wrap interactive blocks in `@st.fragment` |
