# --------- Imports ----------
import pandas as pd
import glob
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ==============================
# Load and merge CSV files
# ==============================
files = glob.glob("dataset/*.csv")
df_list = [pd.read_csv(file) for file in files]
data = pd.concat(df_list, ignore_index=True)

print("Total records:", data.shape[0])

# ==============================
# Select required columns
# ==============================
columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
data = data[columns]

# ==============================
# Data Cleaning
# ==============================
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

print("Data after cleaning:", data.shape)

# ==============================
# Split features & target
# ==============================
X = data.drop('charges', axis=1)
y = data['charges']

# ==============================
# Identify column types
# ==============================
categorical_features = ['sex', 'smoker', 'region']
numerical_features = ['age', 'bmi', 'children']

# ==============================
# Preprocessing pipeline
# ==============================
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'),
         categorical_features),
        ('num', 'passthrough', numerical_features)
    ]
)

# ==============================
# Model: Random Forest
# ==============================
rf = RandomForestRegressor(random_state=42)

# ==============================
# Full Pipeline
# ==============================
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rf)
])

# ==============================
# Hyperparameter Tuning
# ==============================
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=15,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=42
)

# ==============================
# Train-test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# Train model
# ==============================
print("Training Random Forest model...")
search.fit(X_train, y_train)

best_model = search.best_estimator_

print("Best Parameters:")
print(search.best_params_)

# ==============================
# Model Evaluation
# ==============================
y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Performance:")
print("R² Score:", r2)
print("MAE:", mae)
print("RMSE:", rmse)

# ==============================
# Save the trained pipeline
# ==============================
base_dir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(base_dir, "insurance_model.pkl")

joblib.dump(best_model, model_path)

print("\nModel saved at:")
print(model_path)

