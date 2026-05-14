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
st.markdown("Masukkan spesifikasi kenderaan untuk mendapatkan anggaran harga pasaran berasaskan data.")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return joblib.load('test_final.pkl')

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    st.error("⚠️ Fail model 'test_final.pkl' tidak dijumpai.")
    model_loaded = False

# --- LOGIC: DATABASE KENDERAAN REALISTIK ---
# Kita gunakan struktur: { "Brand": { "Model": [Senarai Tahun] } }
car_database = {
    "Porsche": {
        "911 Turbo": [2024, 2023, 2022, 2021, 2020],
        "Taycan": [2024, 2023, 2022, 2021, 2020],
        "Cayenne": [2024, 2023, 2022, 2021, 2020, 2019, 2018]
    },
    "BMW": {
        "330i Sport": [2022, 2021, 2020, 2019],
        "M4": [2024, 2023, 2022, 2021],
        "X5": [2024, 2023, 2022, 2021, 2020, 2019]
    },
    "Toyota": {
        "Camry XLE": [2024, 2023, 2022, 2021, 2020, 2019, 2018],
        "GR Supra": [2024, 2023, 2022, 2021, 2020],
        "Land Cruiser": [2024, 2023, 2022, 2021, 2020, 2015, 2010]
    },
    "Hyundai": {
        "Ioniq 5": [2024, 2023, 2022],
        "Elantra": [2024, 2023, 2022, 2021, 2020],
        "Palisade": [2024, 2023, 2022, 2021, 2020]
    }
}

# --- LOGIC: PENENTUAN CAR CLASS OTOMATIS ---
luxury_brands = ['Porsche', 'Lamborghini', 'Bentley', 'Aston Martin', 'Ferrari']
premium_brands = ['BMW', 'Mercedes-Benz', 'Audi', 'Lexus', 'Land Rover']
entry_brands = ['Kia', 'Hyundai', 'Mitsubishi', 'Nissan']

def get_car_class(brand):
    if brand in luxury_brands: return '1. Luxury'
    elif brand in premium_brands: return '2. Premium'
    elif brand in entry_brands: return '4. Entry Level'
    else: return '3. Mainstream'

# --- USER INPUT FORM ---
if model_loaded:
    with st.form("prediction_form"):
        st.subheader("Identiti Kenderaan")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            # 1. Pilih Brand
            brand_list = sorted(list(car_database.keys()))
            selected_brand = st.selectbox("Jenama", brand_list)
            
        with col2:
            # 2. Pilih Model berdasarkan Brand yang dipilih
            model_list = sorted(list(car_database[selected_brand].keys()))
            selected_model = st.selectbox("Model", model_list)
            
        with col3:
            # 3. Pilih Tahun berdasarkan Model yang dipilih
            year_list = car_database[selected_brand][selected_model]
            selected_year = st.selectbox("Tahun", year_list)

        st.markdown("---")
        
        st.subheader("Spesifikasi Teknikal")
        
        # Car Class dikesan secara automatik
        car_class = get_car_class(selected_brand)
        st.info(f"📋 Segmen Dikesan: **{car_class}**")
        
        col4, col5 = st.columns(2)
        with col4:
            milage = st.number_input("Batu Nautika (Milage)", min_value=0, value=25000, step=1000)
            horsepower = st.number_input("Horsepower (HP)", min_value=0.0, value=300.0)
            engine_liter = st.number_input("Kapasiti Enjin (L)", min_value=0.0, value=2.0, step=0.1)
        with col5:
            fuel_type = st.selectbox("Jenis Bahan Api", ["Gasoline", "Diesel", "Hybrid", "Electric"])
            transmission = st.selectbox("Transmisi", ["Automatic", "Manual", "CVT", "Dual Shift"])
            accident = st.selectbox("Rekod Kemalangan", ["None Reported", "Accident Reported"])

        st.markdown("---")
        
        st.subheader("Estetika")
        col6, col7 = st.columns(2)
        with col6:
            ext_col = st.selectbox("Warna Luaran", ["Neutral", "Exotic"])
            cylinders = st.number_input("Bilangan Silinder", min_value=0.0, value=4.0, step=1.0)
        with col7:
            int_col = st.selectbox("Warna Dalaman", ["Neutral", "Exotic"])

        submit_button = st.form_submit_button(label="Ramal Harga", type="primary")

# --- PREDICTION LOGIC ---
if model_loaded and submit_button:
    accident_val = 0 if accident == "None Reported" else 1

    # Format data untuk input model (XGBoost)
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
        'accident': [accident_val]
    })

    try:
        pred_log = model.predict(input_df)
        final_price = np.expm1(pred_log)[0]

        st.success(f"### 🎯 Anggaran Harga: ${final_price:,.2f}")
        st.caption(f"Ramalan untuk {selected_brand} {selected_model} ({selected_year})")
        
    except Exception as e:
        st.error(f"Ralat semasa ramalan: {e}")

# --- FOOTER ---
st.markdown("<br><hr><p style='text-align: center; color: grey;'>Developed by Fabian RM</p>", unsafe_allow_html=True)
