import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

df = pd.read_csv("insurance_dataset.csv")
X = df.drop(columns=["charges"])
y = df["charges"]

numerical_features = ["age", "bmi", "children"]
categorical_features = ["gender", "smoker", "region", "medical_history", 
                        "family_medical_history", "exercise_frequency", 
                        "occupation", "coverage_level"]

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(random_state=42))
    ]
)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

with open("trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

if __name__ == "__main__":
    print("Model trained and saved as 'trained_model.pkl'")
