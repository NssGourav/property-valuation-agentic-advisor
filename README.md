# Intelligent Property Price Prediction 🏠

A professional machine learning project focused on real-estate price prediction using traditional regression algorithms. This project aims to provide accurate property valuation based on historical datasets.

## 🔗 Live Demo
**Access the live application here:** [Live Demo](https://property-valuation-agentic-advisor-xvfy6pzq5caq72fmxlzrak.streamlit.app/)

## 🎯 Project Overview
The core objective is to develop a robust system that predicts property values using historical housing data. The solution employs traditional Machine Learning methods to ensure interpretability and reliability.

### Key Features
- **Price Prediction**: Interactive form to input property details and get instant value estimates.
- **Model Insights**: Comparative analysis and performance metrics (R², MAE, RMSE).
- **Feature Importance**: Visual breakdown of which factors most influence property pricing.
- **Traditional ML**: Entirely built using standard regression techniques (Random Forest & Linear Regression) without any Generative AI dependencies.

## 🛠️ Technology Stack
- **Languages**: Python 3.13+
- **Machine Learning**: `scikit-learn` (Random Forest, Linear Regression)
- **Data Processing**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn` (for metadata generation), `streamlit` (native charts)
- **Deployment**: `Streamlit Community Cloud`

## 📊 Model Performance
The system was trained on the [Kaggle Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset).

| Metric | Random Forest | Linear Regression |
| :--- | :--- | :--- |
| **R-squared (R²)** | 0.582 | 0.627 |
| **MAE** | ₹1,080,958 | ₹999,836 |

## 🚀 Local Setup & Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/NssGourav/property-valuation-agentic-advisor.git
   cd property-valuation-agentic-advisor
   ```

2. **Environment Setup**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Train the Model (Optional)**
   ```bash
   python3 train_model.py
   ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure
- `app.py`: Streamlit frontend and application logic.
- `train_model.py`: Data preprocessing and model training pipeline.
- `models/`: Contains the serialized `house_model.pkl`.
- `assets/`: Contains `model_metadata.json` and static assets.
- `data/`: Local storage for the dataset (excluded from Git).

## 👥 Team
- **Nss Gourav**
- **Subham Sangwan**

---
*Developed for Project 9: Intelligent Property Price Prediction.*
