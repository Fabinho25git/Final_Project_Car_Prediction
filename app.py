import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- CONFIG PAGE ---
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚘",
    layout="wide"
)

# --- CUSTOM CSS UI ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fb 0%, #eef2f7 100%);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101828 0%, #1d2939 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    section[data-testid="stSidebar"] label {
        color: #f2f4f7 !important;
        font-weight: 700 !important;
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: white;
    }

    .sidebar-title {
        color: white;
        font-size: 1.45rem;
        font-weight: 850;
        line-height: 1.2;
        margin-bottom: 0.35rem;
    }

    .sidebar-desc {
        color: #d0d5dd;
        font-size: 0.9rem;
        line-height: 1.5;
        margin-bottom: 1.2rem;
    }

    .sidebar-divider {
        height: 1px;
        background: rgba(255, 255, 255, 0.14);
        margin: 1.2rem 0;
    }

    .hero-card {
        background: linear-gradient(135deg, #101828 0%, #1d2939 100%);
        padding: 2rem 2.2rem;
        border-radius: 28px;
        box-shadow: 0 18px 45px rgba(16, 24, 40, 0.18);
        margin-bottom: 1.5rem;
        color: white;
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 900;
        margin-bottom: 1rem;
        letter-spacing: -0.04em;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #d0d5dd;
        margin-top: 1rem;
        margin-bottom: 0;
    }

    .team-list {
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
        color: #f2f4f7;
        font-size: 0.98rem;
        line-height: 1.75;
        font-weight: 500;
        padding-left: 1.2rem;
    }

    .team-list li {
        margin: 0.12rem 0;
        padding-left: 0.25rem;
    }

    .dashboard-card {
        background: white;
        padding: 1.4rem 1.5rem;
        border-radius: 24px;
        box-shadow: 0 12px 30px rgba(16, 24, 40, 0.08);
        border: 1px solid rgba(208, 213, 221, 0.8);
        margin-bottom: 1.2rem;
    }

    .dashboard-title {
        font-size: 1.2rem;
        font-weight: 850;
        color: #101828;
        margin-bottom: 0.35rem;
    }

    .dashboard-desc {
        color: #667085;
        font-size: 0.92rem;
        margin-bottom: 1.15rem;
    }

    .summary-card {
        background: white;
        border-radius: 22px;
        padding: 1.25rem;
        border: 1px solid #eaecf0;
        box-shadow: 0 8px 22px rgba(16, 24, 40, 0.06);
        min-height: 108px;
    }

    .summary-label {
        color: #667085;
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.35rem;
    }

    .summary-value {
        color: #101828;
        font-size: 1.28rem;
        font-weight: 900;
        line-height: 1.25;
    }

    .summary-small {
        color: #475467;
        font-size: 0.92rem;
        font-weight: 650;
        margin-top: 0.2rem;
    }

    .mini-card {
        background: #f9fafb;
        border: 1px solid #eaecf0;
        border-radius: 18px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }

    .mini-label {
        color: #667085;
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }

    .mini-value {
        color: #101828;
        font-size: 1.05rem;
        font-weight: 850;
    }

    .feature-pill {
        display: inline-block;
        background: #f2f4f7;
        color: #344054;
        padding: 0.38rem 0.65rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
        margin: 0.2rem 0.15rem;
        border: 1px solid #eaecf0;
    }

    .step-box {
        background: #f9fafb;
        border: 1px solid #eaecf0;
        border-radius: 16px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
        color: #344054;
        font-size: 0.9rem;
        font-weight: 650;
    }

    div[data-baseweb="select"] > div {
        border-radius: 14px;
        border-color: #d0d5dd;
        min-height: 44px;
        box-shadow: none;
    }

    div[data-baseweb="input"] > div {
        border-radius: 14px;
        border-color: #d0d5dd;
        min-height: 44px;
        box-shadow: none;
    }

    .stNumberInput input {
        border-radius: 14px;
    }

    label {
        font-weight: 700 !important;
        color: #344054 !important;
        font-size: 0.92rem !important;
    }

    .stButton > button {
        width: 100%;
        height: 3.4rem;
        border-radius: 16px;
        border: none;
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        font-weight: 850;
        font-size: 1rem;
        box-shadow: 0 12px 25px rgba(220, 38, 38, 0.25);
        transition: all 0.2s ease-in-out;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 35px rgba(220, 38, 38, 0.35);
        color: white;
    }

    div[data-testid="stAlert"] {
        border-radius: 16px;
        border: none;
        box-shadow: 0 6px 18px rgba(16, 24, 40, 0.06);
    }

    .result-card {
        background: linear-gradient(135deg, #ecfdf3 0%, #d1fadf 100%);
        border: 1px solid #abefc6;
        padding: 2rem;
        border-radius: 26px;
        box-shadow: 0 16px 38px rgba(22, 163, 74, 0.16);
        margin-bottom: 1.2rem;
    }

    .result-label {
        color: #067647;
        font-size: 1rem;
        font-weight: 850;
        margin-bottom: 0.5rem;
    }

    .result-price {
        color: #054f31;
        font-size: 2.65rem;
        font-weight: 950;
        margin-bottom: 0.5rem;
        letter-spacing: -0.05em;
        line-height: 1.05;
    }

    .result-caption {
        color: #067647;
        font-size: 0.95rem;
        font-weight: 600;
    }

    .empty-result-card {
        background: white;
        border: 1px dashed #d0d5dd;
        padding: 2rem;
        border-radius: 26px;
        text-align: center;
        color: #667085;
        margin-bottom: 1.2rem;
    }

    .empty-result-title {
        color: #344054;
        font-size: 1.15rem;
        font-weight: 850;
        margin-bottom: 0.35rem;
    }

    .footer {
        text-align: center;
        color: #98a2b3;
        font-size: 0.9rem;
        margin-top: 2rem;
    }

    @media (max-width: 768px) {
        .hero-title {
            font-size: 1.55rem;
        }

        .result-price {
            font-size: 2.3rem;
        }

        .hero-card {
            padding: 1.5rem;
        }
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
if data_ready:
    # --- SIDEBAR VEHICLE SELECTION ---
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-title">🚘 Vehicle Selection</div>
        <div class="sidebar-desc">
            Select the brand, model, and year before entering the technical specifications.
        </div>
        <div class="sidebar-divider"></div>
        """, unsafe_allow_html=True)

        brand_list = sorted(df_raw['brand'].unique())
        selected_brand = st.selectbox("Brand", brand_list)

        filtered_models = df_raw[df_raw['brand'] == selected_brand]
        model_list = sorted(filtered_models['model'].unique())
        selected_model = st.selectbox("Model", model_list)

        filtered_years = filtered_models[filtered_models['model'] == selected_model]
        year_list = sorted(filtered_years['model_year'].unique(), reverse=True)
        selected_year = st.selectbox("Model Year", year_list)

        st.markdown("""
        <div class="sidebar-divider"></div>
        <div class="sidebar-desc">
            This app predicts used car market price based on the selected vehicle specification.
        </div>
        """, unsafe_allow_html=True)

    # Filter data spesifik untuk kombinasi Brand, Model, dan Tahun
    exact_car = filtered_years[filtered_years['model_year'] == selected_year]

    # --- CAR CLASS ---
    car_class = get_car_class(selected_brand)
    car_class_display = car_class.split(". ", 1)[1]

    # --- DATASET INSIGHTS ---
    similar_records = len(exact_car)
    median_milage = int(exact_car['milage'].median())
    median_hp = round(exact_car['horsepower'].median(), 1)
    median_engine = round(exact_car['engine_liter'].median(), 1)

    min_milage = int(exact_car['milage'].min())
    max_milage = int(exact_car['milage'].max())
    avg_milage = int(exact_car['milage'].mean())

    # --- HERO SECTION ---
    st.markdown("""
    <div class="hero-card">
        <div class="hero-title">🚘 Final Project Data Science Group 7 : Team Outliers</div>
        <ol class="team-list">
            <li>Artorius Weelyn Jawra (Ketua)</li>
            <li>Fabian Rashed Majduddin</li>
            <li>Kurniati</li>
            <li>Gunaryono Ary</li>
            <li>Hashfi Hawali</li>
        </ol>
        <p class="hero-subtitle">Used car price prediction based on real-time dataset specifications.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- SELECTED CAR SUMMARY ---
    st.markdown("""
    <div class="dashboard-card">
        <div class="dashboard-title">Selected Vehicle Overview</div>
        <div class="dashboard-desc">Current selected vehicle based on sidebar input.</div>
    </div>
    """, unsafe_allow_html=True)

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

    with summary_col1:
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-label">Brand</div>
            <div class="summary-value">{selected_brand}</div>
        </div>
        """, unsafe_allow_html=True)

    with summary_col2:
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-label">Model</div>
            <div class="summary-value">{selected_model}</div>
        </div>
        """, unsafe_allow_html=True)

    with summary_col3:
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-label">Model Year</div>
            <div class="summary-value">{selected_year}</div>
        </div>
        """, unsafe_allow_html=True)

    with summary_col4:
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-label">Car Class</div>
            <div class="summary-value">{car_class_display}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- DATASET INSIGHT SUMMARY ---
    st.markdown("""
    <div class="dashboard-card">
        <div class="dashboard-title">Dataset Insight</div>
        <div class="dashboard-desc">Quick benchmark from similar records in the dataset.</div>
    </div>
    """, unsafe_allow_html=True)

    insight_col1, insight_col2, insight_col3, insight_col4 = st.columns(4)

    with insight_col1:
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-label">Similar Records</div>
            <div class="summary-value">{similar_records}</div>
            <div class="summary-small">cars found</div>
        </div>
        """, unsafe_allow_html=True)

    with insight_col2:
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-label">Median Milage</div>
            <div class="summary-value">{median_milage:,}</div>
            <div class="summary-small">miles</div>
        </div>
        """, unsafe_allow_html=True)

    with insight_col3:
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-label">Median Horsepower</div>
            <div class="summary-value">{median_hp}</div>
            <div class="summary-small">HP</div>
        </div>
        """, unsafe_allow_html=True)

    with insight_col4:
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-label">Median Engine</div>
            <div class="summary-value">{median_engine}L</div>
            <div class="summary-small">engine capacity</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- MAIN DASHBOARD LAYOUT ---
    left_col, right_col = st.columns([1.25, 0.75], gap="large")

    with left_col:
        st.markdown("""
        <div class="dashboard-card">
            <div class="dashboard-title">Technical Specifications</div>
            <div class="dashboard-desc">Complete the specification details to generate the estimated market price.</div>
        """, unsafe_allow_html=True)

        with st.form("spec_form"):
            st.info(f"📋 Detected Car Class: **{car_class_display}**")

            input_col1, input_col2 = st.columns(2)

            with input_col1:
                # Milage tetap input angka karena variasi per mobil sangat tinggi
                milage = st.number_input("Milage", min_value=0, value=int(exact_car['milage'].median()))
                
                # HP, Engine Liter, dan Cylinders sekarang dinamis berdasarkan varian di dataset
                hp_options = sorted(exact_car['horsepower'].unique().tolist())
                el_options = sorted(exact_car['engine_liter'].unique().tolist())
                cyl_options = sorted(exact_car['cylinders'].unique().tolist())
                
                horsepower = st.selectbox("Horsepower", hp_options)
                engine_liter = st.selectbox("Engine Liter", el_options)
                cylinders = st.selectbox("Cylinders", cyl_options)
            
            with input_col2:
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

    # --- PREDICTION LOGIC ---
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

            st.session_state.final_price = final_price
            st.session_state.prediction_caption = f"Prediction for {selected_brand} {selected_model} ({selected_year})"

        except Exception as e:
            st.error(f"Prediction error occurred: {e}")

    with right_col:
        st.markdown("""
        <div class="dashboard-card">
            <div class="dashboard-title">Prediction Result</div>
            <div class="dashboard-desc">Estimated market price will appear after prediction.</div>
        """, unsafe_allow_html=True)

        if "final_price" in st.session_state:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">🎯 Estimated Market Price</div>
                <div class="result-price">${st.session_state.final_price:,.2f}</div>
                <div class="result-caption">{st.session_state.prediction_caption}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-result-card">
                <div class="empty-result-title">No Prediction Yet</div>
                Complete the technical specifications and click <b>Predict Market Price</b>.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="dashboard-card">
            <div class="dashboard-title">Selected Car Benchmark</div>
            <div class="dashboard-desc">Milage distribution from similar selected vehicles.</div>

            <div class="mini-card">
                <div class="mini-label">Lowest Milage</div>
                <div class="mini-value">{min_milage:,} miles</div>
            </div>

            <div class="mini-card">
                <div class="mini-label">Average Milage</div>
                <div class="mini-value">{avg_milage:,} miles</div>
            </div>

            <div class="mini-card">
                <div class="mini-label">Highest Milage</div>
                <div class="mini-value">{max_milage:,} miles</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="dashboard-card">
            <div class="dashboard-title">How Prediction Works</div>
            <div class="dashboard-desc">Short workflow of the prediction system.</div>

            <div class="step-box">1. Select brand, model, and model year from the sidebar.</div>
            <div class="step-box">2. Technical options adjust based on available dataset records.</div>
            <div class="step-box">3. The regression model estimates the used car market price.</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="dashboard-card">
            <div class="dashboard-title">Model Features</div>
            <div class="dashboard-desc">Features used by the prediction model.</div>

            <span class="feature-pill">Model Year</span>
            <span class="feature-pill">Milage</span>
            <span class="feature-pill">Horsepower</span>
            <span class="feature-pill">Engine Liter</span>
            <span class="feature-pill">Cylinders</span>
            <span class="feature-pill">Brand</span>
            <span class="feature-pill">Fuel Type</span>
            <span class="feature-pill">Transmission</span>
            <span class="feature-pill">Car Class</span>
            <span class="feature-pill">Exterior Color</span>
            <span class="feature-pill">Interior Color</span>
            <span class="feature-pill">Accident</span>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="hero-card">
        <div class="hero-title">🚘 Final Project Data Science Group 7 : Team Outliers</div>
        <p class="hero-subtitle">Used car price prediction based on real-time dataset specifications.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='footer'>Developed by Fabian RM</div>", unsafe_allow_html=True)
