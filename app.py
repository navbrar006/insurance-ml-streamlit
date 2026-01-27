

import streamlit as st
import joblib
import pandas as pd

# ==============================
# Load trained ML pipeline
# ==============================
model = joblib.load("insurance_model.pkl")

st.set_page_config(page_title="Insurance Charges Predictor", layout="centered")

st.title("💰 Insurance Charges Predictor")
st.write("Predict health insurance charges using an advanced ML model")

# ==============================
# User Inputs
# ==============================
age = st.number_input("Age", min_value=1, max_value=100, value=25)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)

sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox(
    "Region",
    ["northwest", "southeast", "southwest", "northeast"]
)

# ==============================
# Prediction
# ==============================
if st.button("Predict Insurance Charges"):

    input_data = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex": sex,
        "smoker": smoker,
        "region": region
    }])

    prediction = model.predict(input_data)

    st.success(f"Estimated Insurance Charges: ₹ {prediction[0]:,.2f}")
