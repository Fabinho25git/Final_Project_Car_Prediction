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
    st.error(f"⚠️ Failed to load data or model: {e}")
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
st.markdown("Used car price prediction based on real-time dataset specifications.")

if data_ready:
    # --- DYNAMIC FILTERING LOGIC ---
    brand_list = sorted(df_raw['brand'].unique())
    selected_brand = st.selectbox("Brand", brand_list)

    filtered_models = df_raw[df_raw['brand'] == selected_brand]
    model_list = sorted(filtered_models['model'].unique())
    selected_model = st.selectbox("Model", model_list)

    filtered_years = filtered_models[filtered_models['model'] == selected_model]
    year_list = sorted(filtered_years['model_year'].unique(), reverse=True)
    selected_year = st.selectbox("Model Year", year_list)

    # Filter data spesifik untuk kombinasi Brand, Model, dan Tahun
    exact_car = filtered_years[filtered_years['model_year'] == selected_year]

    st.markdown("---")

    # --- FORM SPESIFIKASI ---
    with st.form("spec_form"):
        st.subheader("Technical Specifications")
        
        car_class = get_car_class(selected_brand)
        st.info(f"📋 Detected Car Class: **{car_class}**")
        
        col1, col2 = st.columns(2)
        with col1:
            # Milage tetap input angka karena variasi per mobil sangat tinggi
            milage = st.number_input("Milage", min_value=0, value=int(exact_car['milage'].median()))
            
            # HP, Engine Liter, dan Cylinders sekarang dinamis berdasarkan varian di dataset
            hp_options = sorted(exact_car['horsepower'].unique().tolist())
            el_options = sorted(exact_car['engine_liter'].unique().tolist())
            cyl_options = sorted(exact_car['cylinders'].unique().tolist())
            
            horsepower = st.selectbox("Horsepower", hp_options)
            engine_liter = st.selectbox("Engine Liter", el_options)
            cylinders = st.selectbox("Cylinders", cyl_options)
        
        with col2:
            fuel_options = exact_car['fuel_type'].unique().tolist()
            trans_options = exact_car['transmission'].unique().tolist()
            
            fuel_type = st.selectbox("Fuel Type", fuel_options)
            transmission = st.selectbox("Transmission", trans_options)
            accident = st.selectbox("Accident", ["None Reported", "Accident Reported"])
            ext_col = st.selectbox("Exterior Color", ["Neutral", "Exotic"])
            int_col = st.selectbox("Interior Color", ["Neutral", "Exotic"])

        predict_btn = st.form_submit_button("Predict Market Price", type="primary")

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

            st.success(f"### 🎯 Estimated Price: ${final_price:,.2f}")
            st.caption(f"Prediction for {selected_brand} {selected_model} ({selected_year})")
            
        except Exception as e:
            st.error(f"Prediction error occurred: {e}")

st.markdown("<br><hr><p style='text-align: center; color: grey;'>Developed by Fabian RM</p>", unsafe_allow_html=True)
