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

        with st.form("valuation_form"):
            st.subheader("Property Details")
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                input_area = st.number_input("Area (sq ft)", min_value=500, max_value=20000, value=5000, step=100)
                input_bedrooms = st.slider("Bedrooms", 1, 6, 3)
                input_bathrooms = st.slider("Bathrooms", 1, 4, 1)
                input_stories = st.selectbox("Stories", [1, 2, 3, 4])
                
            with col_right:
                input_parking = st.slider("Parking Spots", 0, 4, 1)
                input_mainroad = st.checkbox("Main Road Access", value=True)
                input_guestroom = st.checkbox("Has Guestroom")
                input_basement = st.checkbox("Has Basement")
                
            st.subheader("Amenities")
            col_amenities_1, col_amenities_2 = st.columns(2)
            with col_amenities_1:
                input_hotwater = st.checkbox("Hot Water Heating")
            with col_amenities_2:
                input_ac = st.checkbox("Air Conditioning", value=True)

            submit_btn = st.form_submit_button("💰 Estimate Price", use_container_width=True)

        if submit_btn:
            self.handle_prediction(
                input_area, input_bedrooms, input_bathrooms, input_stories, input_parking,
                input_mainroad, input_guestroom, input_basement, input_hotwater, input_ac
            )

    def handle_prediction(self, area, bedrooms, bathrooms, stories, parking, 
                          mainroad, guestroom, basement, hotwater, ac):
        input_data = {
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'stories': stories,
            'parking': parking,
            'mainroad': 1 if mainroad else 0,
            'guestroom': 1 if guestroom else 0,
            'basement': 1 if basement else 0,
            'hotwaterheating': 1 if hotwater else 0,
            'airconditioning': 1 if ac else 0
        }
        
        df_inputs = pd.DataFrame([input_data])
        
        try:
            df_inputs = df_inputs[ALL_FEATURES] 
            
            predicted_price = self.model.predict(df_inputs)[0]
            
            st.success(f"### Estimated Value: ${predicted_price:,.2f}")
            
            with st.expander("See details used for prediction"):
                st.json(input_data)
                
        except Exception as e:
            st.error(f"Oops! Something went wrong during prediction: {str(e)}")

def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)
    
    app = ValuationApp()
    app.render_header()
    app.render_form()

if __name__ == "__main__":
    main()
