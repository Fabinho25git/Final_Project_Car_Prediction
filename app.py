import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- CONFIG PAGE ---
st.set_page_config(page_title="Car Price Predictor", page_icon="🚘", layout="centered")

# --- LOAD DATA & MODEL ---
@st.cache_data
def load_data():
    df = pd.read_csv('test.csv') 
    return df

@st.cache_resource
def load_model():
    return joblib.load('test_final.pkl')

try:
    df_raw = load_data()
    model = load_model()
    data_ready = True
except Exception as e:
    st.error(f"⚠️ Gagal memuat data atau model: {e}")
    data_ready = False

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
st.markdown("Prediksi harga mobil bekas berdasarkan dataset real-time.")

if data_ready:
    # --- DYNAMIC FILTERING LOGIC ---
    brand_list = sorted(df_raw['brand'].unique())
    selected_brand = st.selectbox("Pilih Jenama (Brand)", brand_list)

    filtered_models = df_raw[df_raw['brand'] == selected_brand]
    model_list = sorted(filtered_models['model'].unique())
    selected_model = st.selectbox("Pilih Model", model_list)

    filtered_years = filtered_models[filtered_models['model'] == selected_model]
    year_list = sorted(filtered_years['model_year'].unique(), reverse=True)
    selected_year = st.selectbox("Pilih Tahun", year_list)

    # 4. Ambil spesifikasi EXACT (Pasti) untuk mobil di tahun tersebut
    exact_car = filtered_years[filtered_years['model_year'] == selected_year]

    st.markdown("---")

    # --- FORM SPESIFIKASI ---
    with st.form("spec_form"):
        st.subheader("Detail Spesifikasi")
        
        car_class = get_car_class(selected_brand)
        st.info(f"📋 Segmen Terdeteksi: **{car_class}**")
        
        col1, col2 = st.columns(2)
        with col1:
            # Mengambil median milage untuk tahun tersebut sebagai default
            milage = st.number_input("Jarak Tempuh (Miles)", min_value=0, value=int(exact_car['milage'].median()))
            
            # Value otomatis mengikuti dataset untuk mobil tersebut
            horsepower = st.number_input("Horsepower (HP)", min_value=0.0, value=float(exact_car['horsepower'].median()))
            engine_liter = st.number_input("Kapasiti Enjin (L)", min_value=0.0, value=float(exact_car['engine_liter'].median()), step=0.1
