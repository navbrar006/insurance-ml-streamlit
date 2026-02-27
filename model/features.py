# model/features.py
import pandas as pd

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["BMI_Category"] = pd.cut(
        df["bmi"], bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"]
    )

    df["Age_Group"] = pd.cut(
        df["age"], bins=[0, 30, 50, 100],
        labels=["Young", "Middle", "Senior"]
    )

    df["Smoker_Risk_Index"] = (df["smoker"] == "yes").astype(int) * df["bmi"]
    df["Family_Load"] = df["children"] * df["age"]

    df["Lifestyle_Risk_Score"] = (
        0.4 * (df["bmi"] / 50) +
        0.4 * (df["smoker"] == "yes").astype(int) +
        0.2 * (df["age"] / 100)
    )
    return df