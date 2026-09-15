"""
Insomnia Prediction App — Lagos, Nigeria
------------------------------------------------------
A Streamlit application that predicts whether a person is likely to have insomnia,
based on a short clinical sleep-screening questionnaire and a few demographic details.

Loads the trained artifacts produced by insomnia_logistic_regression.ipynb:
  - model.pkl             (trained LogisticRegression model)
  - scaler.pkl            (fitted StandardScaler for the 'age' feature)
  - feature_columns.pkl   (exact column order the model was trained on)

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Insomnia Risk Screener — Lagos State",
    page_icon="🌙",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns


try:
    model, scaler, feature_columns = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError:
    artifacts_loaded = False

# ---------------------------------------------------------------------------
# Header / description
# ---------------------------------------------------------------------------
st.title("🌙 Insomnia Risk Screener")
st.markdown(
    """
This tool screens for the **likelihood of insomnia** using a short questionnaire based on
common clinical sleep-screening questions. It was built and calibrated on a synthetic
dataset reflecting the demographic and lifestyle context of **Lagos State, Nigeria**
(within the wider Lagos corridor of South-West Nigeria).

⚠️ **Disclaimer:** This is a screening aid for educational/demonstration purposes only.
It is **not** a medical diagnosis. Please consult a qualified healthcare professional for
any concerns about your sleep health.
"""
)

if not artifacts_loaded:
    st.error(
        "Could not find `model.pkl`, `scaler.pkl`, or `feature_columns.pkl` in the app "
        "directory. Please run `insomnia_logistic_regression.ipynb` first to generate "
        "these files, then place them alongside `app.py`."
    )
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# Input widgets
# ---------------------------------------------------------------------------
st.subheader("👤 Demographics")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", min_value=18, max_value=70, value=30, step=1)
with col2:
    gender = st.selectbox("Gender", ["Female", "Male"])

occupation = st.selectbox(
    "Occupation",
    [
        "Student", "Trader", "Civil Servant", "Tech Professional",
        "Healthcare Worker", "Artisan", "Banker", "Unemployed",
        "Driver", "Business Owner",
    ],
)

st.divider()
st.subheader("😴 Sleep Screening Questionnaire")
st.caption("Answer the following 10 questions as honestly as you can.")

def yn_radio(label, key):
    return st.radio(label, ["No", "Yes"], key=key, horizontal=True)

q1 = yn_radio("1. Do you have trouble falling asleep?", "q1")
q2 = yn_radio("2. Do you wake up during the night and have trouble getting back to sleep?", "q2")
q3 = yn_radio("3. Do you wake up earlier than intended and can't fall asleep again?", "q3")
q4 = yn_radio("4. Does this happen often?", "q4")
q5 = yn_radio("5. Has this been happening for over 6 months?", "q5")
q6 = yn_radio("6. Do you get enough sleep during the night?", "q6")
q7 = st.radio(
    "7. When you wake up, do you feel rested or tired?",
    ["Rested", "Tired"], key="q7", horizontal=True,
)
q8 = yn_radio(
    "8. Does poor sleep affect you during the day (concentration, communication, study, work)?",
    "q8",
)
q9 = yn_radio("9. Do you take caffeine daily?", "q9")
q10 = yn_radio("10. Are you on any drug medication?", "q10")

st.divider()

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def build_input_row():
    """Builds a single-row DataFrame matching the exact preprocessing used in training."""
    raw = {
        "age": age,
        "gender": gender,
        "occupation": occupation,
        "trouble_falling_asleep": q1,
        "trouble_returning_to_sleep": q2,
        "early_waking": q3,
        "happens_often": q4,
        "over_6_months": q5,
        "enough_sleep": q6,
        "feel_on_waking": q7,
        "daytime_impact": q8,
        "daily_caffeine": q9,
        "on_medication": q10,
    }
    row = pd.DataFrame([raw])

    binary_cols = [
        "trouble_falling_asleep", "trouble_returning_to_sleep", "early_waking",
        "happens_often", "over_6_months", "enough_sleep", "daytime_impact",
        "daily_caffeine", "on_medication",
    ]
    for col in binary_cols:
        row[col] = row[col].map({"Yes": 1, "No": 0})
    row["feel_on_waking"] = row["feel_on_waking"].map({"Tired": 1, "Rested": 0})

    row_encoded = pd.get_dummies(row, columns=["gender", "occupation"], drop_first=True)

    # Align columns exactly with what the model was trained on.
    # Any dummy column not produced for this single row (e.g. an occupation/gender
    # category that was the reference level, or simply not selected) is filled with 0.
    row_aligned = row_encoded.reindex(columns=feature_columns, fill_value=0)

    # Scale the 'age' column with the SAME scaler fitted during training.
    row_aligned["age"] = scaler.transform(row_aligned[["age"]])

    return row_aligned


if st.button("🔍 Predict", type="primary", use_container_width=True):
    input_row = build_input_row()

    prediction = model.predict(input_row)[0]
    probability = model.predict_proba(input_row)[0][1]  # probability of class 1 (insomnia)

    st.divider()
    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ Insomnia Detected  —  Predicted probability: **{probability:.1%}**")
        st.markdown(
            "Your responses suggest patterns consistent with insomnia. Consider speaking "
            "with a healthcare professional about your sleep."
        )
    else:
        st.success(f"✅ No Insomnia Detected  —  Predicted probability: **{probability:.1%}**")
        st.markdown(
            "Your responses do not show strong signs of insomnia. Keep maintaining "
            "healthy sleep habits."
        )

    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(
        "Probability represents the model's estimated likelihood that this response "
        "pattern matches someone with insomnia, based on the training data."
    )

st.divider()
st.caption(
    "Model: Logistic Regression | Trained on a synthetic dataset representative of "
    "Lagos State, Nigeria | For demonstration purposes only. Thank YOU"
)
