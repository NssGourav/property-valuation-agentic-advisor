import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

PAGE_TITLE = "Property Valuation Advisor"
PAGE_ICON = "🏠"
MODEL_PATH = Path("models/house_model.pkl")

NUMERICAL_FEATURES = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
BINARY_FEATURES = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning']
ALL_FEATURES = NUMERICAL_FEATURES + BINARY_FEATURES

class ValuationApp:
    def __init__(self):
        self.model = self._load_model()
        
    def _load_model(self):
        if not MODEL_PATH.exists():
            st.error(f"⚠️ Model file not found at {MODEL_PATH}. Please run `train_model.py` first.")
            return None
        
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return None

    def render_header(self):
        st.title(f"{PAGE_ICON} {PAGE_TITLE}")
        st.markdown("""
        **Welcome!** This tool uses a machine learning model to estimate the market value of a property.
        
        Please fill in the details below to get a prediction.
        """)
        st.markdown("---")

    def render_form(self):
        if self.model is None:
            return

        st.info("Input form implementation coming soon...")

def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)
    
    app = ValuationApp()
    app.render_header()
    app.render_form()

if __name__ == "__main__":
    main()
