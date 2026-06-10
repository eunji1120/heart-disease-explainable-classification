"""Plain-language translations for the 13 clinical features and their values.

Used to convert opaque medical codes (e.g. `ca = 3`, `thal = reversable_defect`)
into descriptions a non-clinician can read. Implements an HCI principle of
*explainable variables* — without these, the SHAP page is unreadable for any
audience outside cardiology.
"""
from __future__ import annotations

# Display name + short clinical meaning for each of the 13 features
FEATURE_LABEL: dict[str, str] = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG result",
    "thalach": "Max heart rate during exercise",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (exercise vs rest)",
    "slope": "ST recovery slope",
    "ca": "Major vessels with reduced flow",
    "thal": "Heart muscle (thallium) test",
}

# Long-form, plain-English explanation — shown in tooltips and per-record page
FEATURE_HELP: dict[str, str] = {
    "age": "Patient's age in years.",
    "sex": "Biological sex.",
    "cp": (
        "Type of chest pain at presentation. Typical angina is the classic "
        "exertion-induced pressure; asymptomatic means no chest pain — which "
        "in this dataset is paradoxically associated with higher disease risk, "
        "because silent disease tends to be more advanced when discovered."
    ),
    "trestbps": "Resting blood pressure measured at hospital admission (mm Hg).",
    "chol": "Total serum cholesterol (mg/dl) — a much weaker independent signal than the popular narrative suggests.",
    "fbs": "Whether the patient's fasting blood sugar was above 120 mg/dl (1 = yes).",
    "restecg": "Resting electrocardiogram result. Categorical: normal, ST-T abnormality, or left-ventricle thickening.",
    "thalach": "Maximum heart rate achieved during an exercise stress test. Lower max heart rate often signals reduced cardiac reserve.",
    "exang": "Whether the patient developed chest pain during the exercise stress test (1 = yes).",
    "oldpeak": (
        "How much the ST segment of the ECG depressed during exercise versus rest. "
        "A larger drop indicates greater stress-induced ischemia."
    ),
    "slope": "Shape of the ST segment during peak exercise. A flat or downward slope is more concerning than an upward slope.",
    "ca": "Number of major coronary vessels (0–3) visualized as narrowed by fluoroscopy — a direct anatomical evidence.",
    "thal": (
        "Result of a thallium nuclear stress test. 'Normal' means even uptake; "
        "'fixed defect' = a permanent scar (old infarct); 'reversable defect' = "
        "stress-induced reduced perfusion that recovers with rest."
    ),
}

# Plain-language label for each categorical value
VALUE_LABEL: dict[tuple[str, str], str] = {
    # sex
    ("sex", "male"):   "Male",
    ("sex", "female"): "Female",
    # cp
    ("cp", "typical_angina"):    "Typical angina (classic exertion chest pain)",
    ("cp", "atypical_angina"):   "Atypical angina (non-classic pattern)",
    ("cp", "non_anginal_pain"):  "Non-cardiac chest pain",
    ("cp", "asymptomatic"):      "Asymptomatic (no chest pain)",
    # restecg
    ("restecg", "normal"):              "Normal ECG",
    ("restecg", "st_t_abnormality"):    "ST-T wave abnormality",
    ("restecg", "lv_hypertrophy"):      "Probable left-ventricle thickening",
    # slope
    ("slope", "upsloping"):    "Upsloping ST recovery (least concerning)",
    ("slope", "flat"):         "Flat ST recovery",
    ("slope", "downsloping"):  "Downsloping ST recovery (most concerning)",
    # thal
    ("thal", "normal"):              "Normal — even tracer uptake",
    ("thal", "fixed_defect"):        "Fixed defect — permanent scar / old infarct",
    ("thal", "reversable_defect"):   "Reversible defect — stress-only reduced flow",
    # binary 0/1 for fbs and exang
    ("fbs", "1"):   "Yes — fasting blood sugar > 120 mg/dl",
    ("fbs", "0"):   "No — fasting blood sugar ≤ 120 mg/dl",
    ("exang", "1"): "Yes — angina developed during exercise",
    ("exang", "0"): "No exercise-induced angina",
}


def value_label(feature: str, value) -> str:
    """Return a plain-language label for (feature, value), falling back to the raw value."""
    key = (feature, str(value))
    if key in VALUE_LABEL:
        return VALUE_LABEL[key]
    # numeric values: append unit hint where useful
    if feature == "age":
        return f"{value} years old"
    if feature == "trestbps":
        return f"{value} mm Hg"
    if feature == "chol":
        return f"{value} mg/dl"
    if feature == "thalach":
        return f"{value} bpm peak"
    if feature == "oldpeak":
        return f"{value} mm ST depression"
    if feature == "ca":
        n = int(value) if str(value).isdigit() else value
        return f"{n} of 3 major vessels narrowed"
    return str(value)
