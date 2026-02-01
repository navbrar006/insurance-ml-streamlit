

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ==============================
# Load Model
# ==============================
model = joblib.load("insurance_model.pkl")

st.set_page_config(page_title="Insurance Charges Predictor", layout="centered")

st.title("💰 Insurance Charges Predictor")
st.caption("Advanced ML model with lifestyle & risk-based feature engineering")

# Extract trained Random Forest model
rf_model = model.named_steps['model']

# Extract preprocessor
preprocessor = model.named_steps['preprocessor']

# ==============================
# User Inputs
# ==============================
age = st.number_input("Age", 18, 100, 30)
# Weight input
weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0)

# Height input
height_feet = st.number_input("Height (feet)", min_value=3, max_value=8, value=5)
height_inches = st.number_input("Height (inches)", min_value=0, max_value=11, value=7)

# Convert height to meters
height_m = (height_feet * 12 + height_inches) * 0.0254

# Calculate BMI
bmi = weight / (height_m ** 2)

st.write(f"Calculated BMI: {bmi:.2f}")
if bmi < 18.5:
    category = "Underweight"
elif bmi < 25:
    category = "Normal"
elif bmi < 30:
    category = "Overweight"
else:
    category = "Obese"

st.write(f"BMI Category: {category}")


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
def simplify_feature_name(feature):
    if "Smoker_Risk_Index" in feature:
        return "Smoking combined with BMI"
    elif "Lifestyle_Risk_Score" in feature:
        return "Overall lifestyle risk score"
    elif "Family_Load" in feature:
        return "Family responsibility (age × children)"
    elif "BMI_Category_Obese" in feature:
        return "Obese BMI category"
    elif "BMI_Category_Overweight" in feature:
        return "Overweight BMI category"
    elif "Age_Group_Senior" in feature:
        return "Senior age group"
    elif "Age_Group_Middle" in feature:
        return "Middle age group"
    elif "smoker_yes" in feature:
        return "Smoking habit"
    elif "bmi" in feature:
        return "Body Mass Index (BMI)"
    elif "age" in feature:
        return "Age"
    else:
        return feature.replace("num__", "").replace("cat__", "")

if st.button("Predict Insurance Charges"):

    # -----------------------------
    # Create input dataframe
    # -----------------------------
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    # Apply feature engineering
    input_df = feature_engineering(input_df)

    # -----------------------------
    # Prediction
    # -----------------------------
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Insurance Charges: ₹ {prediction:,.2f}")

    # -----------------------------
    # Risk insights
    # -----------------------------
    st.subheader("🧠 Risk Insights")
    st.write(
        f"Lifestyle Risk Score: {input_df['Lifestyle_Risk_Score'].values[0]:.2f}"
    )

    # =====================================================
    # 🔍 EXPLAINABLE AI (SHAP) — CORRECT PLACEMENT
    # =====================================================
    st.subheader("🔍 Explainable AI: Why this prediction?")

    # Transform input using trained preprocessor
    X_transformed = preprocessor.transform(input_df)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(rf_model)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_transformed)

    # Get feature names AFTER encoding
    feature_names = preprocessor.get_feature_names_out()

    # Create SHAP DataFrame (THIS WAS MISSING)
    shap_df = pd.DataFrame(
        shap_values,
        columns=feature_names
    )

    # -----------------------------
    # Percentage-based Explainability
    # -----------------------------
    shap_series = shap_df.iloc[0]

    # Absolute SHAP values
    abs_shap = shap_series.abs()

    # Convert to percentage contribution
    shap_percent = (abs_shap / abs_shap.sum()) * 100

    # Top 5 features
    top_features = shap_percent.sort_values(ascending=False).head(5)

    # User-friendly table
    readable_table = pd.DataFrame({
      "Factor": [simplify_feature_name(f) for f in top_features.index],
    "Contribution (%)": top_features.values.round(2)
    })

    st.write("### 🔑 Main Factors Influencing Insurance Charges (%)")
    st.dataframe(readable_table, use_container_width=True)

    # Main cause
    main_feature = top_features.index[0]
    main_reason = simplify_feature_name(main_feature)
    main_percent = top_features.iloc[0]

    st.success(
      f"📌 **Main reason for this insurance charge:** {main_reason} "
      f"(**{main_percent:.2f}% contribution**)"
    )

    # -----------------------------
    # Visualization (Percentage)
    # -----------------------------
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






