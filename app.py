import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- CONFIG PAGE ---
st.set_page_config(page_title="Car Price Predictor", page_icon="🚘", layout="centered")

# --- LOAD DATA & MODEL ---
@st.cache_data
def load_data():
    # Membaca file CSV yang ada di repository
    df = pd.read_csv('test.csv') 
    return df

@st.cache_resource
def load_model():
    return joblib.load('test_final.pkl')

# Menjalankan fungsi load
try:
    df_raw = load_data()
    model = load_model()
    data_ready = True
except Exception as e:
    st.error(f"⚠️ Gagal memuat data atau model: {e}")
    data_ready = False

# --- LOGIC CAR CLASS (Sesuai Aturan Project Maestro) ---
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
    
    # 1. Pilih Brand (Diambil dari kolom 'brand' di CSV)
    brand_list = sorted(df_raw['brand'].unique())
    selected_brand = st.selectbox("Pilih Jenama (Brand)", brand_list)

    # 2. Filter Model berdasarkan Brand yang dipilih
    filtered_models = df_raw[df_raw['brand'] == selected_brand]
    model_list = sorted(filtered_models['model'].unique())
    selected_model = st.selectbox("Pilih Model", model_list)

    # 3. Filter Tahun berdasarkan Model yang dipilih
    filtered_years = filtered_models[filtered_models['model'] == selected_model]
    year_list = sorted(filtered_years['model_year'].unique(), reverse=True)
    selected_year = st.selectbox("Pilih Tahun", year_list)

    st.markdown("---")

    # --- FORM SPESIFIKASI ---
    with st.form("spec_form"):
        st.subheader("Detail Spesifikasi")
        
        car_class = get_car_class(selected_brand)
        st.info(f"📋 Segmen Terdeteksi: **{car_class}**")
        
        col1, col2 = st.columns(2)
        with col1:
            milage = st.number_input("Jarak Tempuh (Miles)", min_value=0, value=int(filtered_years['milage'].median()))
            horsepower = st.number_input("Horsepower (HP)", min_value=0.0, value=float(filtered_years['horsepower'].median()))
            engine_liter = st.number_input("Kapasiti Enjin (L)", min_value=0.0, value=float(filtered_years['engine_liter'].median()), step=0.1)
            cylinders = st.number_input("Silinder", min_value=0.0, value=float(filtered_years['cylinders'].median()), step=1.0)
        
        with col2:
            fuel_type = st.selectbox("Bahan Bakar", sorted(df_raw['fuel_type'].unique()))
            transmission = st.selectbox("Transmisi", sorted(df_raw['transmission'].unique()))
            accident = st.selectbox("Rekod Kemalangan", ["None Reported", "Accident Reported"])
            ext_col = st.selectbox("Warna Luar", ["Neutral", "Exotic"])
            int_col = st.selectbox("Warna Dalam", ["Neutral", "Exotic"])

        predict_btn = st.form_submit_button("Ramal Harga Pasaran", type="primary")

    if predict_btn:
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
            st.caption(f"Hasil prediksi untuk {selected_brand} {selected_model} ({selected_year})")
            
        except Exception as e:
            st.error(f"Terjadi ralat saat prediksi: {e}")
