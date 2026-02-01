

# import streamlit as st
# import joblib
# import pandas as pd

# # ==============================
# # Load trained ML pipeline
# # ==============================
# model = joblib.load("insurance_model.pkl")

# st.set_page_config(page_title="Insurance Charges Predictor", layout="centered")

# st.title("💰 Insurance Charges Predictor")
# st.write("Predict health insurance charges using an advanced ML model")

# # ==============================
# # User Inputs
# # ==============================
# age = st.number_input("Age", min_value=1, max_value=100, value=25)
# bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
# children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)

# sex = st.selectbox("Sex", ["male", "female"])
# smoker = st.selectbox("Smoker", ["yes", "no"])
# region = st.selectbox(
#     "Region",
#     ["northwest", "southeast", "southwest", "northeast"]
# )

# # ==============================
# # Prediction
# # ==============================
# if st.button("Predict Insurance Charges"):

#     input_data = pd.DataFrame([{
#         "age": age,
#         "bmi": bmi,
#         "children": children,
#         "sex": sex,
#         "smoker": smoker,
#         "region": region
#     }])

#     prediction = model.predict(input_data)

#     st.success(f"Estimated Insurance Charges: ₹ {prediction[0]:,.2f}")

import streamlit as st
import pandas as pd
import joblib

# ==============================
# Load Model
# ==============================
model = joblib.load("insurance_model.pkl")

st.set_page_config(page_title="Insurance Charges Predictor", layout="centered")

st.title("💰 Insurance Charges Predictor")
st.caption("Advanced ML model with lifestyle & risk-based feature engineering")

# ==============================
# User Inputs
# ==============================
age = st.number_input("Age", 18, 100, 30)
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
children = st.number_input("Number of Children", 0, 5, 0)

sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox(
    "Region",
    ["northwest", "southeast", "southwest", "northeast"]
)

# ==============================
# Feature Engineering (SAME AS TRAINING)
# ==============================
def feature_engineering(df):

    df = df.copy()

    df['BMI_Category'] = pd.cut(
        df['bmi'],
        bins=[0, 18.5, 25, 30, 100],
        labels=['Underweight', 'Normal', 'Overweight', 'Obese']
    )

    df['Age_Group'] = pd.cut(
        df['age'],
        bins=[0, 30, 50, 100],
        labels=['Young', 'Middle', 'Senior']
    )

    df['Smoker_Risk_Index'] = (df['smoker'] == 'yes').astype(int) * df['bmi']
    df['Family_Load'] = df['children'] * df['age']

    df['Lifestyle_Risk_Score'] = (
        0.4 * (df['bmi'] / 50) +
        0.4 * (df['smoker'] == 'yes').astype(int) +
        0.2 * (df['age'] / 100)
    )

    return df

# ==============================
# Prediction
# ==============================
if st.button("Predict Insurance Charges"):

    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    input_df = feature_engineering(input_df)

    prediction = model.predict(input_df)[0]

    st.success(f"💰 Estimated Insurance Charges: ₹ {prediction:,.2f}")

    # ==============================
    # Explainability Output
    # ==============================
    st.subheader("🧠 Risk Insights")

    st.write(f"**Lifestyle Risk Score:** {input_df['Lifestyle_Risk_Score'].values[0]:.2f}")
    st.progress(min(input_df['Lifestyle_Risk_Score'].values[0], 1.0))

    if smoker == "yes":
        st.warning("Smoking significantly increases insurance cost.")

    if bmi >= 30:
        st.warning("BMI falls in Obese category – higher medical risk.")

    if age > 50:
        st.info("Senior age group increases healthcare expenses.")

