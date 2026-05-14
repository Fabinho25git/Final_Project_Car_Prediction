import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- CONFIG PAGE ---
st.set_page_config(page_title="Car Price Predictor", page_icon="🚘", layout="centered")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Pastikan file pkl ada di direktori yang sama
    return joblib.load('test_final.pkl')

try:
    model = load_model()
    model_loaded = True
except:
    st.error("⚠️ Model 'test_final.pkl' tidak ditemukan!")
    model_loaded = False

# --- DATABASE MOBIL LENGKAP ---
# Struktur: { Brand: { Model: [List Tahun] } }
car_database = {
    "Porsche": {
        "911 Turbo": [2024, 2023, 2022, 2021, 2020],
        "Taycan": [2024, 2023, 2022, 2021],
        "Cayenne": [2024, 2023, 2022, 2021, 2020, 2019, 2018],
        "Macan": [2024, 2023, 2022, 2021, 2017]
    },
    "BMW": {
        "330i Sport": [2022, 2021, 2020, 2019],
        "M3": [2024, 2023, 2022, 2021],
        "X5": [2024, 2023, 2022, 2021, 2020, 2019, 2018],
        "7 Series": [2024, 2023, 2022, 2021]
    },
    "Toyota": {
        "Camry XLE": [2024, 2023, 2022, 2021, 2020, 2019, 2018],
        "GR Supra": [2024, 2023, 2022, 2021, 2020],
        "Land Cruiser": [2024, 2023, 2022, 2021, 2020, 2015, 2010],
        "Corolla Cross": [2024, 2023, 2022, 2021]
    },
    "Honda": {
        "Civic Type R": [2024, 2023, 2022, 2021],
        "Accord": [2024, 2023, 2022, 2021, 2020, 2019],
        "CR-V": [2024, 2023, 2022, 2021, 2020, 2019, 2018]
    },
    "Hyundai": {
        "Ioniq 5": [2024, 2023, 2022, 2021],
        "Palisade": [2024, 2023, 2022, 2021, 2020, 2019],
        "Elantra": [2024, 2023, 2022, 2021, 2020]
    },
    "Mercedes-Benz": {
        "C-Class": [2024, 2023, 2022, 2021, 2020, 2019],
        "E-Class": [2024, 2023, 2022, 2021, 2020, 2019],
        "G-Wagon": [2024, 2023, 2022, 2021, 2018]
    },
    "Aston Martin": {
        "Vantage": [2024, 2023, 2022, 2021, 2020],
        "DBS": [2023, 2022, 2021, 2020],
        "DBX": [2024, 2023, 2022, 2021]
    }
}

# --- LOGIC CAR CLASS ---
luxury_brands = ['Porsche', 'Lamborghini', 'Bentley', 'Aston Martin', 'Ferrari', 'McLaren', 'Rolls-Royce', 'Lotus', 'Bugatti']
premium_brands = ['BMW', 'Mercedes-Benz', 'Audi', 'Lexus', 'Cadillac', 'Jaguar', 'Genesis', 'Lincoln', 'Land Rover', 'Alfa Romeo', 'Maserati']
entry_level_brands = ['Kia', 'Hyundai', 'Mitsubishi', 'Nissan', 'Suzuki', 'FIAT', 'smart', 'Scion', 'Saturn', 'Pontiac', 'Mercury']

def get_car_class(brand):
    if brand in luxury_brands: return '1. Luxury'
    elif brand in premium_brands: return '2. Premium'
    elif brand in entry_level_brands: return '4. Entry Level'
    else: return '3. Mainstream'

# --- UI START ---
st.title("🚘 Car Price Predictor")
st.markdown("Masukkan identitas mobil di bawah ini (Pilihan akan update otomatis).")

# BAGIAN IDENTITAS (DI LUAR FORM UNTUK FIX BUG)
col1, col2, col3 = st.columns(3)
with col1:
    selected_brand = st.selectbox("Jenama (Brand)", sorted(list(car_database.keys())))
with col2:
    selected_model = st.selectbox("Model", sorted(list(car_database[selected_brand].keys())))
with col3:
    selected_year = st.selectbox("Tahun", car_database[selected_brand][selected_model])

st.markdown("---")

# BAGIAN SPESIFIKASI (DI DALAM FORM)
if model_loaded:
    with st.form("spec_form"):
        st.subheader("Detail Spesifikasi & Estetika")
        
        car_class = get_car_class(selected_brand)
        st.info(f"📋 Segmen Terdeteksi: **{car_class}**")
        
        c1, c2 = st.columns(2)
        with c1:
            milage = st.number_input("Jarak Tempuh (Miles)", min_value=0, value=30000)
            horsepower = st.number_input("Horsepower (HP)", min_value=0.0, value=250.0)
            engine_liter = st.number_input("Kapasiti Enjin (L)", min_value=0.0, value=2.0, step=0.1)
            cylinders = st.number_input("Silinder", min_value=0.0, value=4.0, step=1.0)
        
        with c2:
            fuel_type = st.selectbox("Bahan Bakar", ["Gasoline", "Diesel", "Hybrid", "Electric"])
            transmission = st.selectbox("Transmisi", ["Automatic", "Manual", "CVT", "Dual Shift"])
            accident = st.selectbox("Rekod Kemalangan", ["None Reported", "Accident Reported"])
            ext_col = st.selectbox("Warna Luar", ["Neutral", "Exotic"])
            int_col = st.selectbox("Warna Dalam", ["Neutral", "Exotic"])

        predict_btn = st.form_submit_button("Ramal Harga Pasaran", type="primary")

    if predict_btn:
        # Menyesuaikan input sesuai format training
        acc_val = 0 if accident == "None Reported" else 1
        
        input_df = pd.DataFrame({
            'model_year': [selected_year],
            'milage': [milage],
            'horsepower': [horsepower],
            'engine_liter': [engine_liter],
            'cylinders': [cylinders],
            'brand': [selected_brand],
            'fuel_type': [fuel_type],
            'transmission': [transmission],
            'car_class': [car_class],
            'ext_col_group': [ext_col],
            'int_col_group': [int_col],
            'accident': [acc_val]
        })

        try:
            pred_log = model.predict(input_df)
            final_price = np.expm1(pred_log)[0]

            st.success(f"### 🎯 Anggaran Harga: ${final_price:,.2f}")
            st.balloons()
        except Exception as e:
            st.error(f"Terjadi ralat: {e}")

st.markdown("<br><hr><p style='text-align: center; color: grey;'>Developed by Fabian RM</p>", unsafe_allow_html=True)
