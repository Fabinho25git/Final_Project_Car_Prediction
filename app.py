import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests

# --- CONFIG PAGE ---
st.set_page_config(page_title="Car Price Predictor", page_icon="🚘", layout="wide")

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

# --- LOGIKA PENCARIAN GAMBAR WIKIPEDIA (DINAMIS & AKURAT) ---
@st.cache_data(show_spinner=False)
def get_car_image(brand, model_name):
    # Kata kunci pencarian ditambah kata 'car' agar Wikipedia tidak bingung
    search_query = f"{brand} {model_name} car automobile"
    search_url = "https://en.wikipedia.org/w/api.php"
    
    # Langkah 1: Cari judul halaman Wikipedia yang paling relevan
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": search_query,
        "format": "json"
    }
    
    try:
        search_res = requests.get(search_url, params=search_params).json()
        if search_res['query']['search']:
            # Ambil judul artikel teratas
            title = search_res['query']['search'][0]['title']
            
            # Langkah 2: Ambil gambar thumbnail utama dari artikel tersebut
            img_params = {
                "action": "query",
                "titles": title,
                "prop": "pageimages",
                "format": "json",
                "pithumbsize": 800 # Ukuran gambar 800px
            }
            img_res = requests.get(search_url, params=img_params).json()
            pages = img_res['query']['pages']
            
            for page_id in pages:
                if 'thumbnail' in pages[page_id]:
                    return pages[page_id]['thumbnail']['source']
    except Exception:
        pass
    
    # Gambar siluet default jika Wikipedia tidak memiliki foto mobil tersebut
    return "https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?auto=format&fit=crop&w=800&q=80"

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
    # --- DYNAMIC FILTERING & IMAGE RENDER ---
    col_img, col_filter = st.columns([1, 1.2]) 

    with col_filter:
        st.subheader("Vehicle Identity")
        brand_list = sorted(df_raw['brand'].unique())
        selected_brand = st.selectbox("Brand", brand_list)

        filtered_models = df_raw[df_raw['brand'] == selected_brand]
        model_list = sorted(filtered_models['model'].unique())
        selected_model = st.selectbox("Model", model_list)

        filtered_years = filtered_models[filtered_models['model'] == selected_model]
        year_list = sorted(filtered_years['model_year'].unique(), reverse=True)
        selected_year = st.selectbox("Model Year", year_list)

    with col_img:
        # Menjalankan fungsi penarik gambar otomatis dari Wikipedia
        with st.spinner("Mencari gambar mobil..."):
            img_url = get_car_image(selected_brand, selected_model)
        st.image(img_url, caption=f"{selected_brand} {selected_model}", use_container_width=True)

    exact_car = filtered_years[filtered_years['model_year'] == selected_year]

    st.markdown("---")

    # --- FORM SPESIFIKASI ---
    with st.form("spec_form"):
        st.subheader("Technical Specifications")
        
        car_class = get_car_class(selected_brand)
        st.info(f"📋 Detected Car Class: **{car_class}**")
        
        col1, col2 = st.columns(2)
        with col1:
            milage = st.number_input("Milage", min_value=0, value=int(exact_car['milage'].median()))
            
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
