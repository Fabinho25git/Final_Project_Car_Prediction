import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- CONFIG PAGE ---
st.set_page_config(page_title="Car Price Predictor", page_icon="🚘", layout="centered")

# --- CUSTOM CSS UI ---
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fb 0%, #eef2f7 100%);
    }

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 980px;
    }

    /* Hero card */
    .hero-card {
        background: linear-gradient(135deg, #101828 0%, #1d2939 100%);
        padding: 2.2rem 2.4rem;
        border-radius: 24px;
        box-shadow: 0 18px 45px rgba(16, 24, 40, 0.18);
        margin-bottom: 1.8rem;
        color: white;
        text-align: left;
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 1.2rem;
        letter-spacing: -0.04em;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #d0d5dd;
        margin-top: 1.2rem;
        margin-bottom: 0;
    }

    .team-list {
        margin-top: 0.8rem;
        margin-bottom: 1rem;
        color: #f2f4f7;
        font-size: 0.98rem;
        line-height: 1.7;
        font-weight: 500;
        padding-left: 1.2rem;
    }

    .team-list li {
        margin: 0.18rem 0;
        padding-left: 0.25rem;
    }

    /* Section card */
    .section-card {
        background: white;
        padding: 1.6rem;
        border-radius: 22px;
        box-shadow: 0 12px 30px rgba(16, 24, 40, 0.08);
        border: 1px solid rgba(208, 213, 221, 0.7);
        margin-bottom: 1.4rem;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 750;
        color: #101828;
        margin-bottom: 0.4rem;
    }

    .section-desc {
        color: #667085;
        font-size: 0.92rem;
        margin-bottom: 1.2rem;
    }

    /* Streamlit widgets */
    div[data-baseweb="select"] > div {
        border-radius: 14px;
        border-color: #d0d5dd;
        min-height: 44px;
    }

    div[data-baseweb="input"] > div {
        border-radius: 14px;
        border-color: #d0d5dd;
        min-height: 44px;
    }

    .stNumberInput input {
        border-radius: 14px;
    }

    label {
        font-weight: 650 !important;
        color: #344054 !important;
        font-size: 0.92rem !important;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        height: 3.3rem;
        border-radius: 16px;
        border: none;
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        font-weight: 800;
        font-size: 1rem;
        box-shadow: 0 12px 25px rgba(220, 38, 38, 0.25);
        transition: all 0.2s ease-in-out;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 35px rgba(220, 38, 38, 0.35);
        color: white;
    }

    /* Info box */
    div[data-testid="stAlert"] {
        border-radius: 16px;
        border: none;
        box-shadow: 0 6px 18px rgba(16, 24, 40, 0.06);
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #ecfdf3 0%, #d1fadf 100%);
        border: 1px solid #abefc6;
        padding: 1.8rem;
        border-radius: 22px;
        text-align: center;
        box-shadow: 0 14px 35px rgba(22, 163, 74, 0.16);
        margin-top: 1.3rem;
    }

    .result-label {
        color: #067647;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .result-price {
        color: #054f31;
        font-size: 2.4rem;
        font-weight: 900;
        margin-bottom: 0.4rem;
        letter-spacing: -0.04em;
    }

    .result-caption {
        color: #067647;
        font-size: 0.95rem;
    }

    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: #eaecf0;
        margin: 1.5rem 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #98a2b3;
        font-size: 0.9rem;
        margin-top: 2rem;
    }

    /* Hide Streamlit default decoration */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

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
st.markdown("""
<div class="hero-card">
    <div class="hero-title">🚘 Final Project Data Science Group 7 : Team Outliers</div>
    <ol class="team-list">
        <li>Artorius Weelyn Jawra</li>
        <li>Fabian Rashed Majduddin</li>
        <li>Kurniati</li>
        <li>Gunaryono Ary</li>
        <li>Hashfi Hawali</li>
    </ol>
    <p class="hero-subtitle">Used car price prediction based on real-time dataset specifications.</p>
</div>
""", unsafe_allow_html=True)

if data_ready:
    # --- DYNAMIC FILTERING LOGIC ---
    st.markdown("""
    <div class="section-card">
        <div class="section-title">Vehicle Selection</div>
        <div class="section-desc">Choose the car identity before entering the technical specifications.</div>
    """, unsafe_allow_html=True)

    brand_list = sorted(df_raw['brand'].unique())
    selected_brand = st.selectbox("Brand", brand_list)

    filtered_models = df_raw[df_raw['brand'] == selected_brand]
    model_list = sorted(filtered_models['model'].unique())
    selected_model = st.selectbox("Model", model_list)

    filtered_years = filtered_models[filtered_models['model'] == selected_model]
    year_list = sorted(filtered_years['model_year'].unique(), reverse=True)
    selected_year = st.selectbox("Model Year", year_list)

    st.markdown("</div>", unsafe_allow_html=True)

    # Filter data spesifik untuk kombinasi Brand, Model, dan Tahun
    exact_car = filtered_years[filtered_years['model_year'] == selected_year]

    # --- FORM SPESIFIKASI ---
    st.markdown("""
    <div class="section-card">
        <div class="section-title">Technical Specifications</div>
        <div class="section-desc">Complete the specification details to generate the estimated market price.</div>
    """, unsafe_allow_html=True)

    with st.form("spec_form"):
        car_class = get_car_class(selected_brand)

        # Versi display hanya untuk tampilan UI
        car_class_display = car_class.split(". ", 1)[1]

        st.info(f"📋 Detected Car Class: **{car_class_display}**")
        
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

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.form_submit_button("Predict Market Price", type="primary")

    st.markdown("</div>", unsafe_allow_html=True)

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

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">🎯 Estimated Price</div>
                <div class="result-price">${final_price:,.2f}</div>
                <div class="result-caption">Prediction for {selected_brand} {selected_model} ({selected_year})</div>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Prediction error occurred: {e}")

st.markdown("<div class='footer'>Developed by Fabian RM</div>", unsafe_allow_html=True)
