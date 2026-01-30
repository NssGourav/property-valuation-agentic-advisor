import logging
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

DATA_FILE = Path("Housing.csv")
MODEL_FILE = Path("house_model.pkl")
RANDOM_STATE = 42
TEST_SIZE = 0.2

BINARY_COLUMNS = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
]

FEATURES = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "parking",
]

TARGET = "price"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        logging.error(f"Dataset not found at {file_path}")
        sys.exit(1)
    return pd.read_csv(file_path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    
    for col in BINARY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map({"yes": 1, "no": 0})
            
    return df


def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    
    evaluate_model(model, X_test, y_test)
    return model


def evaluate_model(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series):
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    
    logging.info("Model Evaluation Results:")
    logging.info(f"  R-squared (R2):       {r2:.4f}")
    logging.info(f"  Mean Absolute Error:  {mae:.4f}")


def save_model(model: RandomForestRegressor, output_path: Path):
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
    
    model = train_model(X, y)
    save_model(model, MODEL_FILE)

main()
