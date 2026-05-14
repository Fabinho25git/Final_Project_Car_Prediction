import streamlit as st
import pandas as pd
import numpy as np
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
    return joblib.load('test_final.pkl')

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    st.error("⚠️ Model file 'test_final.pkl' not found. Please ensure it is in the same directory.")
    model_loaded = False

# --- LOGIC: OTOMATISASI CAR CLASS ---
luxury_brands = ['Porsche', 'Lamborghini', 'Bentley', 'Aston Martin', 'Ferrari', 'McLaren', 'Rolls-Royce', 'Lotus', 'Bugatti']
premium_brands = ['BMW', 'Mercedes-Benz', 'Audi', 'Lexus', 'Cadillac', 'Jaguar', 'Genesis', 'Lincoln', 'Land Rover', 'Alfa Romeo', 'Maserati']
entry_level_brands = ['Kia', 'Hyundai', 'Mitsubishi', 'Nissan', 'Suzuki', 'FIAT', 'smart', 'Scion', 'Saturn', 'Pontiac', 'Mercury']

def get_car_class(selected_brand):
    if selected_brand in luxury_brands: return '1. Luxury'
    elif selected_brand in premium_brands: return '2. Premium'
    elif selected_brand in entry_level_brands: return '4. Entry Level'
    else: return '3. Mainstream'

# --- LOGIC: DATABASE DYNAMIC DROPDOWN ---
# Catatan: Karena kita tidak me-load file CSV di cloud agar aplikasi tetap ringan,
# kita membuat kamus data (dictionary) untuk memetakan Brand ke Modelnya.
brand_model_dict = {
    "Aston Martin": ["Vantage", "DB11", "DBS", "Valhalla"],
    "BMW": ["330i Sport", "M3", "M4", "X5", "7 Series"],
    "Dodge": ["Charger SRT", "Challenger", "Durango", "Viper"],
    "Hyundai": ["Elantra Standard", "Palisade", "Santa Fe", "Ioniq 5"],
    "Porsche": ["911 Turbo", "Cayman", "Panamera", "Macan", "Taycan"],
    "Toyota": ["Camry XLE", "Corolla", "Land Cruiser", "Supra", "GR Yaris"],
    "Honda": ["Civic Type R", "Accord", "CR-V", "HR-V"],
    "Ford": ["Mustang", "F-150", "Bronco", "Explorer"]
}

# --- USER INPUT FORM ---
if model_loaded:
    with st.form("prediction_form"):
        st.subheader("Vehicle Identity")
        
        # Baris 1: Brand dan Model (Dinamis)
        col1, col2 = st.columns(2)
        with col1:
            # Dropdown Brand mengambil dari kunci (keys) di dictionary
            brand = st.selectbox("Brand", list(brand_model_dict.keys()))
        with col2:
            # Dropdown Model otomatis berubah isi list-nya mengikuti Brand yang dipilih
            model_name = st.selectbox("Model", brand_model_dict[brand])

        # Baris 2: Tahun dan Milage
        col3, col4 = st.columns(2)
        with col3:
            model_year = st.selectbox("Model Year", list(range(2026, 1989, -1)), index=5) # Default di tahun 2021
        with col4:
            milage = st.number_input("Milage (miles)", min_value=0, value=30000, step=1000)
            
        st.markdown("---")
        
        st.subheader("Technical Specifications")
        
        # Tampilkan Car Class secara otomatis tanpa perlu diinput user
        car_class = get_car_class(brand)
        st.info(f"💡 System Auto-Detected Car Segment: **{car_class}**")
        
        col5, col6 = st.columns(2)
        with col5:
            horsepower = st.number_input("Horsepower (HP)", min_value=0.0, value=335.0, step=10.0)
            engine_liter = st.number_input("Engine Capacity (L)", min_value=0.0, value=3.0, step=0.1)
            cylinders = st.number_input("Cylinders", min_value=0.0, value=6.0, step=1.0)
        with col6:
            fuel_type = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Hybrid", "Electric", "E85 Flex Fuel", "Other"])
            transmission = st.selectbox("Transmission", ["Automatic", "Manual", "Dual Shift", "CVT", "Other"])
            accident = st.selectbox("Accident History", ["None Reported", "Accident/Damage Reported"])
            
        st.markdown("---")
        
        st.subheader("Cosmetics")
        col7, col8 = st.columns(2)
        with col7:
            ext_col_group = st.selectbox("Exterior Color", ["Neutral", "Exotic"])
        with col8:
            int_col_group = st.selectbox("Interior Color", ["Neutral", "Exotic"])

        # Submit button
        submit_button = st.form_submit_button(label="Predict Market Price", type="primary")

# --- PREDICTION LOGIC ---
if model_loaded and submit_button:
    accident_val = 0 if accident == "None Reported" else 1

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

    try:
        log_price_pred = model.predict(input_data)
        actual_price = np.expm1(log_price_pred)[0]

        st.success("### 🎯 Estimated Market Value")
        st.metric(label=f"{brand} {model_name} ({model_year})", value=f"${actual_price:,.2f}")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# --- FOOTER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Developed by Fabian RM</p>", unsafe_allow_html=True)
