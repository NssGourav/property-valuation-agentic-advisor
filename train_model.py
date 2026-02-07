import logging
import sys
import kagglehub
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ASSETS_DIR = Path("assets")
MODELS_DIR = Path("models")
DATA_DIR = Path("data")

for folder in [ASSETS_DIR, MODELS_DIR, DATA_DIR]:
    folder.mkdir(exist_ok=True)

DATA_FILE = DATA_DIR / "Housing.csv"
MODEL_FILE = MODELS_DIR / "house_model.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.2

BINARY_COLUMNS = ["mainroad","guestroom","basement","hotwaterheating","airconditioning"]

FEATURES = ["area","bedrooms","bathrooms","stories","mainroad","guestroom","basement","hotwaterheating","airconditioning","parking"]

TARGET = "price"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def load_data(file_path: Path) -> pd.DataFrame:
    """Download dataset from Kaggle if not present and load it."""
    if not file_path.exists():
        logging.info("Downloading dataset from Kaggle...")
        try:
            download_path = kagglehub.dataset_download("yasserh/housing-prices-dataset")
            downloaded_file = Path(download_path) / "Housing.csv"
            if downloaded_file.exists():
                import shutil
                shutil.copy(downloaded_file, file_path)
                logging.info(f"Dataset saved to {file_path}")
            else:
                logging.error("Could not find Housing.csv in the downloaded package.")
                sys.exit(1)
        except Exception as e:
            logging.error(f"Failed to download dataset: {e}")
            sys.exit(1)
    
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    for col in BINARY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map({"yes": 1, "no": 0})
    return df

def train_models(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    rf_model.fit(X_train, y_train)
    logging.info("--- Random Forest Evaluation ---")
    evaluate_model(rf_model, X_test, y_test)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    logging.info("--- Linear Regression Evaluation ---")
    evaluate_model(lr_model, X_test, y_test)
    return rf_model, lr_model

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    logging.info("Model Evaluation Results:")
    logging.info(f"  R-squared (R2):       {r2:.4f}")
    logging.info(f"  Mean Absolute Error:  {mae:.4f}")
    logging.info(f"  RMSE:                 {rmse:.4f}")

def save_model(model, output_path: Path):
    joblib.dump(model, output_path)
    logging.info(f"Model successfully saved to {output_path}")

def main():
    logging.info("Starting model training pipeline...")
    data = load_data(DATA_FILE)
    processed_data = preprocess_data(data)
    if not all(feature in processed_data.columns for feature in FEATURES):
        logging.error(f"Missing required features. Expected: {FEATURES}")
        sys.exit(1)
    X = processed_data[FEATURES]
    y = processed_data[TARGET]
    rf_model, lr_model = train_models(X, y)
    save_model(rf_model, MODEL_FILE)

if __name__ == "__main__":
    main()
