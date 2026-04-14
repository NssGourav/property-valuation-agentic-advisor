import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import kagglehub
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")

ASSETS_DIR = Path("assets")
MODELS_DIR = Path("models")
DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "Housing.csv"
MODEL_FILE = MODELS_DIR / "house_model.pkl"
METADATA_FILE = ASSETS_DIR / "model_metadata.json"
FEATURE_IMPORTANCE_PLOT = ASSETS_DIR / "feature_importance.png"
PREDICTED_VS_ACTUAL_PLOT = ASSETS_DIR / "predicted_vs_actual.png"

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
sns.set_theme(style="whitegrid")


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
                logging.info("Dataset saved to %s", file_path)
            else:
                logging.error("Could not find Housing.csv in the downloaded package after downloading from Kaggle.")
                raise RuntimeError("Downloaded Kaggle dataset does not contain 'Housing.csv'.")
        except Exception as exc:
            logging.error(
                "Failed to download dataset from Kaggle. Ensure you have network access and Kaggle "
                "credentials configured. Place your 'kaggle.json' file in the '~/.kaggle/' directory "
                "with appropriate permissions, or set the 'KAGGLE_USERNAME' and 'KAGGLE_KEY' "
                "environment variables. Original error: %s",
                exc,
            )
            raise RuntimeError("Failed to download dataset from Kaggle.") from exc

    return pd.read_csv(file_path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and encode binary features."""
    processed = df.dropna().copy()
    for col in BINARY_COLUMNS:
        if col in processed.columns:
            processed[col] = processed[col].str.lower().map({"yes": 1, "no": 0})
    return processed


def split_data(X: pd.DataFrame, y: pd.Series):
    """Create a reproducible train/test split."""
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def evaluate_model(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate a regression model and return the results."""
    predictions = model.predict(X_test)
    metrics = {
        "r2": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
    }

    logging.info("--- %s Evaluation ---", name)
    logging.info("  R-squared (R2):       %.4f", metrics["r2"])
    logging.info("  Mean Absolute Error:  %.4f", metrics["mae"])
    logging.info("  RMSE:                 %.4f", metrics["rmse"])

    return {
        "metrics": metrics,
        "predictions": predictions,
    }


def train_models(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, dict]:
    """Train the candidate models and collect their evaluation outputs."""
    random_forest = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    linear_regression = LinearRegression()

    random_forest.fit(X_train, y_train)
    linear_regression.fit(X_train, y_train)

    results = {
        "random_forest": {
            "label": "Random Forest",
            "model": random_forest,
            "evaluation": evaluate_model("Random Forest", random_forest, X_test, y_test),
        },
        "linear_regression": {
            "label": "Linear Regression",
            "model": linear_regression,
            "evaluation": evaluate_model("Linear Regression", linear_regression, X_test, y_test),
        },
    }

    selected_key = "random_forest"
    return results, results[selected_key]


def save_model(model, output_path: Path):
    """Serialize the trained model to disk."""
    joblib.dump(model, output_path)
    logging.info("Model successfully saved to %s", output_path)


def save_feature_importance_plot(feature_importance: dict[str, float], output_path: Path) -> None:
    """Create a horizontal feature importance chart for the selected model."""
    importance_df = (
        pd.DataFrame({"Feature": list(feature_importance.keys()), "Importance": list(feature_importance.values())})
        .sort_values(by="Importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df["Feature"], importance_df["Importance"], color="#1f77b4")
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved feature importance plot to %s", output_path)


def save_predicted_vs_actual_plot(y_true: pd.Series, y_pred: np.ndarray, output_path: Path) -> None:
    """Create a predicted-vs-actual scatter plot with a perfect-fit reference line."""
    lower_bound = min(float(np.min(y_true)), float(np.min(y_pred)))
    upper_bound = max(float(np.max(y_true)), float(np.max(y_pred)))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.75, color="#2ca02c", edgecolors="white", linewidth=0.5)
    ax.plot([lower_bound, upper_bound], [lower_bound, upper_bound], linestyle="--", color="#d62728", linewidth=2)
    ax.set_title("Predicted vs Actual Prices")
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved predicted-vs-actual plot to %s", output_path)


def build_metadata(
    selected_result: dict,
    all_results: dict,
    raw_rows: int,
    processed_rows: int,
) -> dict:
    """Build the metadata artifact consumed by the app and the README story."""
    selected_model = selected_result["model"]
    selected_metrics = selected_result["evaluation"]["metrics"]
    feature_importance = dict(zip(FEATURES, selected_model.feature_importances_.tolist()))

    benchmarks = {}
    for key, result in all_results.items():
        benchmarks[key] = {
            "label": result["label"],
            "metrics": result["evaluation"]["metrics"],
        }

    return {
        "metrics": selected_metrics,
        "feature_importance": feature_importance,
        "benchmark_summary": {
            "selected_model": "RandomForestRegressor",
            "selection_reason": (
                "Random Forest remains the production model because it captures non-linear feature interactions "
                "and exposes feature importance for the app insights tab."
            ),
            "candidates": benchmarks,
        },
        "training": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_rows": {
                "raw": raw_rows,
                "processed": processed_rows,
            },
            "split": {
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
            },
        },
        "schema": {
            "features": FEATURES,
            "target": TARGET,
            "binary_columns": BINARY_COLUMNS,
        },
        "artifacts": {
            "predicted_vs_actual": str(PREDICTED_VS_ACTUAL_PLOT),
            "feature_importance": str(FEATURE_IMPORTANCE_PLOT),
        },
    }


def save_metadata(metadata: dict, output_path: Path) -> None:
    """Write the metadata JSON artifact."""
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)
    logging.info("Model metadata saved to %s", output_path)


def main():
    """Main execution pipeline."""
    setup_directories()
    logging.info("Starting model training pipeline...")

    try:
        raw_data = load_data(DATA_FILE)
    except RuntimeError as exc:
        logging.error(str(exc))
        sys.exit(1)

    processed_data = preprocess_data(raw_data)

    if not all(feature in processed_data.columns for feature in FEATURES):
        logging.error("Missing required features. Expected: %s", FEATURES)
        sys.exit(1)

    X = processed_data[FEATURES]
    y = processed_data[TARGET]
    X_train, X_test, y_train, y_test = split_data(X, y)

    results, selected_result = train_models(X_train, y_train, X_test, y_test)

    save_model(selected_result["model"], MODEL_FILE)

    save_feature_importance_plot(
        dict(zip(FEATURES, selected_result["model"].feature_importances_.tolist())),
        FEATURE_IMPORTANCE_PLOT,
    )
    save_predicted_vs_actual_plot(
        y_test,
        selected_result["evaluation"]["predictions"],
        PREDICTED_VS_ACTUAL_PLOT,
    )

    metadata = build_metadata(
        selected_result=selected_result,
        all_results=results,
        raw_rows=len(raw_data),
        processed_rows=len(processed_data),
    )
    save_metadata(metadata, METADATA_FILE)


if __name__ == "__main__":
    main()
