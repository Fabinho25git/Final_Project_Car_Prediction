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
        margin
