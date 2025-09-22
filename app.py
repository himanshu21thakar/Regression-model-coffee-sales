import streamlit as st
import pandas as pd
import joblib
import numpy as np
import pandas as pd

df = pd.read_csv("Coffe_sales.csv")  # add this at the top


# Load the trained model
model = joblib.load("best_coffee_sales_model.pkl")

# Set page configuration
st.set_page_config(page_title="Coffee Sales Predictor", page_icon="☕", layout="centered")

# Custom CSS for dynamic styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 15px;
    }
    .stButton>button {
        background-color: #ff6600;
        color: white;
        height: 3em;
        width: 100%;
        border-radius: 10px;
        font-size: 18px;
    }
    .stTextInput>div>input {
        height: 2.5em;
        border-radius: 10px;
        padding-left: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("☕ Coffee Sales Predictor")
st.markdown("Predict daily coffee sales based on various features.")

# --- Dynamic Input Fields ---
st.header("Enter the details:")

# Example of extracting model feature names
# If you saved the ColumnTransformer, you can dynamically fetch features; otherwise, define manually:
categorical_features = ['cash_type', 'coffee_name', 'Time_of_Day', 'Weekday', 'Month_name']  # replace with your dataset categorical columns
numeric_features = ['hour_of_day', 'Weekdaysort', 'Monthsort']  # replace with your numeric columns

inputs = {}

# Categorical inputs
for col in categorical_features:
    options = model.named_steps['preprocessor'].named_transformers_['cat'].categories_[categorical_features.index(col)]
    inputs[col] = st.selectbox(f"{col.replace('_', ' ').title()}:", options)

# Numeric inputs
for col in numeric_features:
    min_val = int(model.named_steps['preprocessor'].transformers[0][1].get_feature_names_out().shape[0]) if col not in df else int(df[col].min())
    max_val = int(df[col].max())
    mean_val = int(df[col].mean())
    inputs[col] = st.number_input(f"{col.replace('_', ' ').title()}:", min_value=min_val, max_value=max_val, value=mean_val)

# Predict button
if st.button("Predict Sales"):
    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Coffee Sales: {prediction:.2f}")
