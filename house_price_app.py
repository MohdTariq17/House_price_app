
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

# Load California Housing dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit UI
st.title("🏠 California House Price Predictor")

st.markdown("Adjust the sliders to set the features of a house, and get the predicted price.")

user_input = []
for feature in data.feature_names:
    min_val = float(X[feature].min())
    max_val = float(X[feature].max())
    mean_val = float(X[feature].mean())
    value = st.slider(f"{feature}", min_val, max_val, mean_val)
    user_input.append(value)

# Predict
input_array = np.array(user_input).reshape(1, -1)
prediction = model.predict(input_array)[0]

st.subheader(f"Predicted Median House Price:")
st.success(f"${prediction * 100000:.2f}")
