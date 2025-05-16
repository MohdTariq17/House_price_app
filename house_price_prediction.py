import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Title
st.title("House Price Prediction App")
st.write("Predict California housing prices using input features.")

# Load Dataset
data = pd.read_csv("https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv")
data.dropna(inplace=True)

# Feature Engineering
data['rooms_per_household'] = data['total_rooms'] / data['households']
data['bedrooms_per_room'] = data['total_bedrooms'] / data['total_rooms']
data['population_per_household'] = data['population'] / data['households']

features = [
    "median_income",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "latitude",
    "longitude",
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household"
]

X = data[features]
y = data["median_house_value"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Sidebar Inputs
st.sidebar.header("Input Features")

input_data = {}
for col in features:
    min_val = float(X[col].min())
    max_val = float(X[col].max())
    mean_val = float(X[col].mean())
    input_data[col] = st.sidebar.slider(col.replace("_", " ").title(), min_val, max_val, mean_val)

# Prediction
input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)[0]
st.subheader("Predicted House Price:")
st.write(f"${prediction:,.2f}")

# Evaluation
st.subheader("Model Evaluation on Test Set")
y_pred = model.predict(X_test)
st.write(f"R2 Score: {r2_score(y_test, y_pred):.2f}")
st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

