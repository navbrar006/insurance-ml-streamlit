import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import json
import os

from model.features import feature_engineering

# ==============================
# Streamlit Config
# ==============================
st.set_page_config(page_title="Insurance Charges Predictor", layout="centered")
st.title("💰 Insurance Charges Predictor")
st.caption("Advanced ML model with lifestyle & risk-based feature engineering + Explainable AI")

# ==============================
# Load model + metrics (cached)
# ==============================
@st.cache_resource
def load_model():
    return joblib.load("insurance_model.pkl")

@st.cache_data
def load_metrics():
    if os.path.exists("metrics.json"):
        with open("metrics.json", "r") as f:
            return json.load(f)
    return None

model = load_model()
metrics = load_metrics()

# Sidebar: model performance
with st.sidebar:
    st.header("📌 Model Info")
    if metrics:
        st.write(f"**R²:** {metrics.get('r2', 'NA'):.4f}")
        st.write(f"**MAE:** {metrics.get('mae', 'NA'):.2f}")
        st.write(f"**RMSE:** {metrics.get('rmse', 'NA'):.2f}")
        st.caption("Best Params:")
        st.json(metrics.get("best_params", {}))
    else:
        st.info("metrics.json not found. Train model and save metrics for display.")

# ==============================
# Helper: get pipeline, model & preprocessor safely
# Works for Pipeline OR TransformedTargetRegressor
# ==============================
def unwrap_model(m):
    """
    Returns: (pipeline, rf_model, preprocessor)
    """
    # Case 1: TransformedTargetRegressor -> has .regressor_
    if hasattr(m, "regressor_"):
        pipe = m.regressor_
    # Case 2: saved fitted TransformedTargetRegressor might store .regressor
    elif hasattr(m, "regressor"):
        pipe = m.regressor
    else:
        pipe = m  # assume pipeline

    # Extract steps safely
    pre = pipe.named_steps.get("preprocessor", None)
    rf = pipe.named_steps.get("model", None)
    return pipe, rf, pre

pipeline, rf_model, preprocessor = unwrap_model(model)

if rf_model is None or preprocessor is None:
    st.error("Model structure not recognized. Make sure your saved model includes 'preprocessor' and 'model' steps.")
    st.stop()

# ==============================
# User Inputs
# ==============================
age = st.number_input("Age", 18, 100, 30)

weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0)
height_feet = st.number_input("Height (feet)", min_value=3, max_value=8, value=5)
height_inches = st.number_input("Height (inches)", min_value=0, max_value=11, value=7)

# Convert height to meters
height_m = (height_feet * 12 + height_inches) * 0.0254
if height_m <= 0:
    st.error("Invalid height. Please enter a valid height.")
    st.stop()

bmi = weight / (height_m ** 2)

st.write(f"Calculated BMI: **{bmi:.2f}**")

children = st.number_input("Number of Children", 0, 5, 0)
sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northwest", "southeast", "southwest", "northeast"])

# ==============================
# Friendly feature names
# ==============================
def simplify_feature_name(feature: str) -> str:
    if "Smoker_Risk_Index" in feature:
        return "Smoking combined with BMI"
    if "Lifestyle_Risk_Score" in feature:
        return "Overall lifestyle risk score"
    if "Family_Load" in feature:
        return "Family responsibility (age × children)"
    if "BMI_Category_Obese" in feature:
        return "Obese BMI category"
    if "BMI_Category_Overweight" in feature:
        return "Overweight BMI category"
    if "Age_Group_Senior" in feature:
        return "Senior age group"
    if "Age_Group_Middle" in feature:
        return "Middle age group"
    if "smoker_yes" in feature:
        return "Smoking habit"
    if feature.endswith("bmi") or "bmi" in feature:
        return "Body Mass Index (BMI)"
    if feature.endswith("age") or "age" in feature:
        return "Age"
    return feature.replace("num__", "").replace("cat__", "")

# ==============================
# Build input dataframe
# ==============================
def make_input_df(age, sex, bmi, children, smoker, region):
    df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])
    return feature_engineering(df)

# ==============================
# What-if analysis helper
# ==============================
def predict_for(df):
    return float(model.predict(df)[0])

# ==============================
# Predict Button
# ==============================
if st.button("Predict Insurance Charges"):
    input_df = make_input_df(age, sex, bmi, children, smoker, region)

    # Prediction
    prediction = predict_for(input_df)
    st.success(f"💰 Estimated Insurance Charges: ₹ {prediction:,.2f}")

    # Risk insights
    st.subheader("🧠 Risk Insights")
    st.write(f"**Lifestyle Risk Score:** {input_df['Lifestyle_Risk_Score'].values[0]:.2f}")

    # -----------------------------
    # What-if Analysis (High impact)
    # -----------------------------
    st.subheader("🔁 What-if Analysis (How to reduce cost?)")

    col1, col2 = st.columns(2)

    # What if smoker = no
    with col1:
        df_non_smoker = make_input_df(age, sex, bmi, children, "no", region)
        pred_non_smoker = predict_for(df_non_smoker)
        st.write("If you **stop smoking**:")
        st.metric(
            label="Estimated Charges",
            value=f"₹ {pred_non_smoker:,.2f}",
            delta=f"₹ {pred_non_smoker - prediction:,.2f}"
        )

    # What if BMI decreases
    with col2:
        bmi_reduced = max(12.0, bmi - 2.0)
        df_bmi_reduce = make_input_df(age, sex, bmi_reduced, children, smoker, region)
        pred_bmi_reduce = predict_for(df_bmi_reduce)
        st.write("If your **BMI reduces by 2**:")
        st.metric(
            label="Estimated Charges",
            value=f"₹ {pred_bmi_reduce:,.2f}",
            delta=f"₹ {pred_bmi_reduce - prediction:,.2f}"
        )

    # -----------------------------
    # Explainable AI (SHAP)
    # -----------------------------
    st.subheader("🔍 Explainable AI: Why this prediction?")

    X_transformed = preprocessor.transform(input_df)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_transformed)

    feature_names = preprocessor.get_feature_names_out()

    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_series = shap_df.iloc[0]

    abs_shap = shap_series.abs()
    shap_percent = (abs_shap / abs_shap.sum()) * 100
    top_features = shap_percent.sort_values(ascending=False).head(5)

    readable_table = pd.DataFrame({
        "Factor": [simplify_feature_name(f) for f in top_features.index],
        "Contribution (%)": top_features.values.round(2)
    })

    st.write("### 🔑 Main Factors Influencing Charges (%)")
    st.dataframe(readable_table, use_container_width=True)

    main_feature = top_features.index[0]
    st.success(
        f"📌 **Main reason:** {simplify_feature_name(main_feature)} "
        f"(**{top_features.iloc[0]:.2f}% contribution**)"
    )

    st.write("### 📊 Feature Contribution (Percentage View)")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(
        [simplify_feature_name(f) for f in top_features.index][::-1],
        top_features.values[::-1]
    )
    ax.set_xlabel("Contribution (%)")
    ax.set_title("Explainable AI – Percentage Contribution")

    for i, v in enumerate(top_features.values[::-1]):
        ax.text(v + 0.5, i, f"{v:.1f}%", va='center')

    st.pyplot(fig)
    plt.clf()

# import streamlit as st
# import pandas as pd
# import joblib
# import shap
# import matplotlib.pyplot as plt

# # ==============================
# # Load Model
# # ==============================
# model = joblib.load("insurance_model.pkl")

# st.set_page_config(page_title="Insurance Charges Predictor", layout="centered")

# st.title("💰 Insurance Charges Predictor")
# st.caption("Advanced ML model with lifestyle & risk-based feature engineering")

# # Extract trained Random Forest model
# rf_model = model.named_steps['model']

# # Extract preprocessor
# preprocessor = model.named_steps['preprocessor']

# # ==============================
# # User Inputs
# # ==============================
# age = st.number_input("Age", 18, 100, 30)
# # Weight input
# weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0)

# # Height input
# height_feet = st.number_input("Height (feet)", min_value=3, max_value=8, value=5)
# height_inches = st.number_input("Height (inches)", min_value=0, max_value=11, value=7)

# # Convert height to meters
# height_m = (height_feet * 12 + height_inches) * 0.0254

# # Calculate BMI
# bmi = weight / (height_m ** 2)

# st.write(f"Calculated BMI: {bmi:.2f}")
# if bmi < 18.5:
#     category = "Underweight"
# elif bmi < 25:
#     category = "Normal"
# elif bmi < 30:
#     category = "Overweight"
# else:
#     category = "Obese"

# st.write(f"BMI Category: {category}")


# children = st.number_input("Number of Children", 0, 5, 0)

# sex = st.selectbox("Sex", ["male", "female"])
# smoker = st.selectbox("Smoker", ["yes", "no"])
# region = st.selectbox(
#     "Region",
#     ["northwest", "southeast", "southwest", "northeast"]
# )

# # ==============================
# # Feature Engineering (SAME AS TRAINING)
# # ==============================
# def feature_engineering(df):

#     df = df.copy()

#     df['BMI_Category'] = pd.cut(
#         df['bmi'],
#         bins=[0, 18.5, 25, 30, 100],
#         labels=['Underweight', 'Normal', 'Overweight', 'Obese']
#     )

#     df['Age_Group'] = pd.cut(
#         df['age'],
#         bins=[0, 30, 50, 100],
#         labels=['Young', 'Middle', 'Senior']
#     )

#     df['Smoker_Risk_Index'] = (df['smoker'] == 'yes').astype(int) * df['bmi']
#     df['Family_Load'] = df['children'] * df['age']

#     df['Lifestyle_Risk_Score'] = (
#         0.4 * (df['bmi'] / 50) +
#         0.4 * (df['smoker'] == 'yes').astype(int) +
#         0.2 * (df['age'] / 100)
#     )

#     return df
# def simplify_feature_name(feature):
#     if "Smoker_Risk_Index" in feature:
#         return "Smoking combined with BMI"
#     elif "Lifestyle_Risk_Score" in feature:
#         return "Overall lifestyle risk score"
#     elif "Family_Load" in feature:
#         return "Family responsibility (age × children)"
#     elif "BMI_Category_Obese" in feature:
#         return "Obese BMI category"
#     elif "BMI_Category_Overweight" in feature:
#         return "Overweight BMI category"
#     elif "Age_Group_Senior" in feature:
#         return "Senior age group"
#     elif "Age_Group_Middle" in feature:
#         return "Middle age group"
#     elif "smoker_yes" in feature:
#         return "Smoking habit"
#     elif "bmi" in feature:
#         return "Body Mass Index (BMI)"
#     elif "age" in feature:
#         return "Age"
#     else:
#         return feature.replace("num__", "").replace("cat__", "")

# if st.button("Predict Insurance Charges"):

#     # -----------------------------
#     # Create input dataframe
#     # -----------------------------
#     input_df = pd.DataFrame([{
#         "age": age,
#         "sex": sex,
#         "bmi": bmi,
#         "children": children,
#         "smoker": smoker,
#         "region": region
#     }])

#     # Apply feature engineering
#     input_df = feature_engineering(input_df)

#     # -----------------------------
#     # Prediction
#     # -----------------------------
#     prediction = model.predict(input_df)[0]
#     st.success(f"💰 Estimated Insurance Charges: ₹ {prediction:,.2f}")

#     # -----------------------------
#     # Risk insights
#     # -----------------------------
#     st.subheader("🧠 Risk Insights")
#     st.write(
#         f"Lifestyle Risk Score: {input_df['Lifestyle_Risk_Score'].values[0]:.2f}"
#     )

#     # =====================================================
#     # 🔍 EXPLAINABLE AI (SHAP) — CORRECT PLACEMENT
#     # =====================================================
#     st.subheader("🔍 Explainable AI: Why this prediction?")

#     # Transform input using trained preprocessor
#     X_transformed = preprocessor.transform(input_df)

#     # Create SHAP explainer
#     explainer = shap.TreeExplainer(rf_model)

#     # Compute SHAP values
#     shap_values = explainer.shap_values(X_transformed)

#     # Get feature names AFTER encoding
#     feature_names = preprocessor.get_feature_names_out()

#     # Create SHAP DataFrame (THIS WAS MISSING)
#     shap_df = pd.DataFrame(
#         shap_values,
#         columns=feature_names
#     )

#     # -----------------------------
#     # Percentage-based Explainability
#     # -----------------------------
#     shap_series = shap_df.iloc[0]

#     # Absolute SHAP values
#     abs_shap = shap_series.abs()

#     # Convert to percentage contribution
#     shap_percent = (abs_shap / abs_shap.sum()) * 100

#     # Top 5 features
#     top_features = shap_percent.sort_values(ascending=False).head(5)

#     # User-friendly table
#     readable_table = pd.DataFrame({
#       "Factor": [simplify_feature_name(f) for f in top_features.index],
#     "Contribution (%)": top_features.values.round(2)
#     })

#     st.write("### 🔑 Main Factors Influencing Insurance Charges (%)")
#     st.dataframe(readable_table, use_container_width=True)

#     # Main cause
#     main_feature = top_features.index[0]
#     main_reason = simplify_feature_name(main_feature)
#     main_percent = top_features.iloc[0]

#     st.success(
#       f"📌 **Main reason for this insurance charge:** {main_reason} "
#       f"(**{main_percent:.2f}% contribution**)"
#     )

#     # -----------------------------
#     # Visualization (Percentage)
#     # -----------------------------
#     st.write("### 📊 Feature Contribution (Percentage View)")

#     fig, ax = plt.subplots(figsize=(8, 4))

#     ax.barh(
#      [simplify_feature_name(f) for f in top_features.index][::-1],
#      top_features.values[::-1]
#     )

#     ax.set_xlabel("Contribution (%)")
#     ax.set_title("Explainable AI – Percentage Contribution")

#     for i, v in enumerate(top_features.values[::-1]):
#       ax.text(v + 0.5, i, f"{v:.1f}%", va='center')

#     st.pyplot(fig)
#     plt.clf()






