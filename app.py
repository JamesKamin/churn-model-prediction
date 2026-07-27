
import streamlit as st
import numpy as np
import pandas as pd
import pickle

@st.cache_resource
def load_artifacts():
    with open("/tmp/churn_rf_healthy_meals_2023.pkl", "rb") as f:
        model = pickle.load(f)
    with open("/tmp/churn_encoder_healthy_meals_2023.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = load_artifacts()


st.title("Customer Renewal Probability Predictor")
st.write("Enter customer attributes to predict the likelihood of subscription renewal.")

total_sessions    = st.number_input("Total Sessions (2022)", min_value=0, max_value=5000, value=50)
gross_total_session_length = st.number_input("Gross Total Session Length (2022)", min_value=0, max_value=100000, value=1500)
num_active_days   = st.number_input("Number of Active Days (2022)", min_value=0, max_value=365, value=10)
num_active_quarters = st.number_input("Number of Active Quarters (2022)", min_value=0, max_value=4, value=2)
avg_sessions_per_active_quarter = st.number_input("Average Sessions Per Active Quarter", min_value=0, max_value=5000, value=25)
age               = st.number_input("Age", min_value=18, max_value=100, value=35)
tech_comfort_score = st.number_input("Tech Comfort Score", min_value=1, max_value=10, value=5)
income_level      = st.radio("Income Level",  ["Low", "Medium", "High", "Very High"])
education         = st.radio("Education",     ["Graduate", "High School", "Other", "Post-Graduate"])
device_type       = st.radio("Device Type",   ["Desktop-only", "Mobile-only", "Multi-device"])

if st.button("Predict"):

    # Build categorical DataFrame — column names must match encoder exactly
    raw = pd.DataFrame([{
        'INCOME_LEVEL': income_level,
        'EDUCATION':    education,
        'DEVICE_TYPE':  device_type,
    }])

    # Apply the saved encoder (transform only — never fit_transform)
    encoded = encoder.transform(raw)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

    # Numeric features first, then encoded dummies — must match training column order
    numeric_df = pd.DataFrame([{
        'TOTAL_SESSIONS':                  total_sessions,
        'GROSS_TOTAL_SESSION_LENGTH':      gross_total_session_length,
        'NUM_ACTIVE_DAYS':                 num_active_days,
        'NUM_ACTIVE_QUARTERS':             num_active_quarters,
        'AVG_SESSIONS_PER_ACTIVE_QUARTER': avg_sessions_per_active_quarter,
        'AGE':                             age,
        'TECH_COMFORT_SCORE':              tech_comfort_score,
    }])

    input_df = pd.concat([numeric_df, encoded_df], axis=1)

    # Column 1 = P(renewed), column 0 = P(churned)
    probability = model.predict_proba(input_df)[0][1]
    risk = "Low" if probability >= 0.6 else "Medium" if probability >= 0.4 else "High"

    st.metric("Renewal Probability", f"{probability:.2f}")
    if risk == "High":
        st.error(f"Churn Risk: {risk}")
    elif risk == "Medium":
        st.warning(f"Churn Risk: {risk}")
    else:
        st.success(f"Churn Risk: {risk}")
