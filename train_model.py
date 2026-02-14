import logging
import sys
import kagglehub
import pandas as pd
import numpy as np
import joblib
import shutil
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ASSETS_DIR = Path("assets")
MODELS_DIR = Path("models")
DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "Housing.csv"
MODEL_FILE = MODELS_DIR / "house_model.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.2

BINARY_COLUMNS = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning"]

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

def setup_directories():
    """Ensure required directories exist."""
    for folder in [ASSETS_DIR, MODELS_DIR, DATA_DIR]:
        folder.mkdir(exist_ok=True)

def load_data(file_path: Path) -> pd.DataFrame:
    """Download dataset from Kaggle if not present and load it."""
    if not file_path.exists():
        logging.info("Downloading dataset from Kaggle...")
        try:
            download_path = kagglehub.dataset_download("yasserh/housing-prices-dataset")
            downloaded_file = Path(download_path) / "Housing.csv"
            
            if downloaded_file.exists():
                shutil.copy(downloaded_file, file_path)
                logging.info(f"Dataset saved to {file_path}")
            else:
                logging.error("Could not find Housing.csv in the downloaded package after downloading from Kaggle.")
                raise RuntimeError("Downloaded Kaggle dataset does not contain 'Housing.csv'.")
        except Exception as e:
            logging.error(
                "Failed to download dataset from Kaggle. Ensure you have network access and Kaggle "
                "credentials configured. Place your 'kaggle.json' file in the '~/.kaggle/' directory "
                "with appropriate permissions, or set the 'KAGGLE_USERNAME' and 'KAGGLE_KEY' "
                "environment variables. Original error: %s",
                e,
            )
            raise RuntimeError("Failed to download dataset from Kaggle.") from e
    
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and encode binary features."""
    df = df.dropna()
    for col in BINARY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map({"yes": 1, "no": 0})
    return df

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """Evaluate a regression model using standard metrics."""
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    logging.info("Model Evaluation Results:")
    logging.info(f"  R-squared (R2):       {r2:.4f}")
    logging.info(f"  Mean Absolute Error:  {mae:.4f}")
    logging.info(f"  RMSE:                 {rmse:.4f}")

def train_models(X: pd.DataFrame, y: pd.Series):
    """Train and evaluate multiple regression models."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    rf_model.fit(X_train, y_train)
    logging.info("--- Random Forest Evaluation ---")
    evaluate_model(rf_model, X_test, y_test)
    
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    logging.info("--- Linear Regression Evaluation ---")
    evaluate_model(lr_model, X_test, y_test)
    
    return rf_model, lr_model

def save_model(model, output_path: Path):
    """Serialize the trained model to disk."""
    joblib.dump(model, output_path)
    logging.info(f"Model successfully saved to {output_path}")

def main():
    """Main execution pipeline."""
    setup_directories()
    logging.info("Starting model training pipeline...")
    
    try:
        data = load_data(DATA_FILE)
    except RuntimeError as e:
        logging.error(str(e))
        sys.exit(1)
        
    processed_data = preprocess_data(data)
    
    if not all(feature in processed_data.columns for feature in FEATURES):
        logging.error(f"Missing required features. Expected: {FEATURES}")
        sys.exit(1)
        
    X = processed_data[FEATURES]
    y = processed_data[TARGET]
    
    rf_model, _ = train_models(X, y)
    
    # Save model
    save_model(rf_model, MODEL_FILE)
    importances = rf_model.feature_importances_
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    predictions = rf_model.predict(X_test)
    
    metadata = {
        "metrics": {
            "r2": r2_score(y_test, predictions),
            "mae": mean_absolute_error(y_test, predictions),
            "rmse": np.sqrt(mean_squared_error(y_test, predictions))
        },
        "feature_importance": dict(zip(FEATURES, importances.tolist()))
    }
    
    metadata_path = ASSETS_DIR / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"Model metadata saved to {metadata_path}")

if __name__ == "__main__":
    main()
