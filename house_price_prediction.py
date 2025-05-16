import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Streamlit App
st.title("California House Price Prediction")

st.markdown("""
This app predicts the **median house price** in California (in $100,000s) using features from the [California Housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html).
""")

# Create sliders dynamically
st.sidebar.header("Input Features")
input_data = {}
for feature in X.columns:
    min_val = float(X[feature].min())
    max_val = float(X[feature].max())
    mean_val = float(X[feature].mean())
    input_data[feature] = st.sidebar.slider(
        feature, min_val, max_val, mean_val
    )

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Make prediction
prediction = model.predict(input_df)[0]

# Show results
st.subheader("Predicted House Price")
st.write(f"💰 **${prediction * 100000:,.2f}**")

# Optional: Show model performance on test set
if st.checkbox("Show model performance"):
    y_pred = model.predict(X_test)
    st.write("**R² Score:**", round(r2_score(y_test, y_pred), 2))
    st.write("**RMSE:**", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))
    st.write("**MAE:**", round(mean_absolute_error(y_test, y_pred), 2))

