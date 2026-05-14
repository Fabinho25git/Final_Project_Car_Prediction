import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚘",
    layout="centered"
)

# --- HEADER ---
st.title("🚘 Used Car Price Predictor")
st.markdown("Enter the vehicle's specifications below to get a data-driven market price estimate powered by XGBoost.")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Make sure 'test_final.pkl' is in the same folder as app.py
    return joblib.load('test_final.pkl')

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    st.error("⚠️ Model file 'test_final.pkl' not found. Please ensure it is in the same directory.")
    model_loaded = False

# --- USER INPUT FORM ---
if model_loaded:
    with st.form("prediction_form"):
        st.subheader("Vehicle Identity")
        col1, col2 = st.columns(2)
        with col1:
            brand = st.text_input("Brand", value="BMW", help="e.g., BMW, Toyota, Porsche")
            car_class = st.selectbox("Car Class", ['1. Luxury', '2. Premium', '3. Mainstream', '4. Entry Level'])
            model_year = st.number_input("Model Year", min_value=1990, max_value=2026, value=2021, step=1)
        with col2:
            milage = st.number_input("Milage (miles)", min_value=0, value=30000, step=1000)
            accident = st.selectbox("Accident History", ["None Reported", "Accident/Damage Reported"])
        
        st.markdown("---")
        
        st.subheader("Technical Specifications")
        col3, col4 = st.columns(2)
        with col3:
            horsepower = st.number_input("Horsepower (HP)", min_value=0.0, value=335.0, step=10.0)
            engine_liter = st.number_input("Engine Capacity (L)", min_value=0.0, value=3.0, step=0.1)
            cylinders = st.number_input("Cylinders", min_value=0.0, value=6.0, step=1.0)
        with col4:
            fuel_type = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Hybrid", "Electric", "E85 Flex Fuel", "Other"])
            transmission = st.selectbox("Transmission", ["Automatic", "Manual", "Dual Shift", "CVT", "Other"])
            
        st.markdown("---")
        
        st.subheader("Cosmetics")
        col5, col6 = st.columns(2)
        with col5:
            ext_col_group = st.selectbox("Exterior Color", ["Neutral", "Exotic"])
        with col6:
            int_col_group = st.selectbox("Interior Color", ["Neutral", "Exotic"])

        # Submit button
        submit_button = st.form_submit_button(label="Predict Market Price", type="primary")

# --- PREDICTION LOGIC ---
if model_loaded and submit_button:
    # 1. Map the text selection back to the binary integer your model expects
    accident_val = 0 if accident == "None Reported" else 1

    # 2. Package inputs into a DataFrame mapping exactly to your training features
    input_data = pd.DataFrame({
        'model_year': [model_year],
        'milage': [milage],
        'horsepower': [horsepower],
        'engine_liter': [engine_liter],
        'cylinders': [cylinders],
        'brand': [brand],
        'fuel_type': [fuel_type],
        'transmission': [transmission],
        'car_class': [car_class],
        'ext_col_group': [ext_col_group],
        'int_col_group': [int_col_group],
        'accident': [accident_val]
    })

    # 3. Execute Prediction
    try:
        # Predict returns the log_price
        log_price_pred = model.predict(input_data)
        
        # Convert log_price back to actual dollar amount
        actual_price = np.expm1(log_price_pred)[0]

        # 4. Display Result
        st.success("### 🎯 Estimated Market Value")
        st.metric(label=f"{brand} ({car_class})", value=f"${actual_price:,.2f}")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# --- FOOTER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Developed by Fabian RM</p>", unsafe_allow_html=True)