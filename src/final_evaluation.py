import pandas as pd
import numpy as np

from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# FILE PATHS
# ============================================================

TEST_FILE = "data/processed/test_v2.csv"

MODEL_FILE = "models/transitai_xgboost_v2.json"

FEATURE_FILE = "models/feature_columns_v2.csv"

OUTPUT_DIR = Path("data/processed")


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    print("\nLoading V2 test dataset...")

    test = pd.read_csv(TEST_FILE)

    print(f"Test rows: {len(test):,}")

    return test


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(test, feature_columns):

    print("\nPreparing test features...")

    # One-hot encode route
    encoded = pd.get_dummies(
        test,
        columns=["bus_route"],
        dtype=int
    )

    # Make sure every model feature exists
    for column in feature_columns:

        if column not in encoded.columns:
            encoded[column] = 0

    # Keep exact same feature order as training
    X_test = encoded[feature_columns].copy()

    y_test = test["total_ridership"].copy()

    return X_test, y_test


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, predictions):

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions
        )
    )

    # sMAPE
    denominator = (
        np.abs(y_true) +
        np.abs(predictions)
    )

    mask = denominator != 0

    smape = np.mean(
        2 *
        np.abs(
            predictions[mask] -
            y_true[mask]
        ) /
        denominator[mask]
    ) * 100

    return mae, rmse, smape


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TRANSITAI - FINAL MODEL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    test = load_test_data()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading final V2 XGBoost model...")

    model = XGBRegressor()

    model.load_model(
        MODEL_FILE
    )

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Load feature list
    # --------------------------------------------------------

    print("\nLoading feature list...")

    feature_columns = pd.read_csv(
        FEATURE_FILE,
        header=None
    )[0].tolist()

    print(
        f"Features required: {len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Prepare test data
    # --------------------------------------------------------

    X_test, y_test = prepare_features(
        test,
        feature_columns
    )

    print(
        f"Final test feature shape: {X_test.shape}"
    )

    # --------------------------------------------------------
    # Check missing values
    # --------------------------------------------------------

    missing_values = X_test.isna().sum().sum()

    print(
        f"Missing feature values: {missing_values}"
    )

    if missing_values > 0:

        print(
            "\nERROR: Missing values detected."
        )

        return

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    print("\nGenerating final predictions...")

    predictions = model.predict(
        X_test
    )

    # Demand cannot be negative
    predictions = np.maximum(
        predictions,
        0
    )

    print("Predictions generated.")

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    mae, rmse, smape = calculate_metrics(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL V2 MODEL RESULTS")
    print("=" * 70)

    print(
        f"\nTest MAE:   {mae:.2f} riders"
    )

    print(
        f"Test RMSE:  {rmse:.2f} riders"
    )

    print(
        f"sMAPE:      {smape:.2f}%"
    )

    # --------------------------------------------------------
    # V1 comparison
    # --------------------------------------------------------

    v1_mae = 13.14
    v1_rmse = 29.13

    mae_improvement = (
        (v1_mae - mae) /
        v1_mae
    ) * 100

    rmse_improvement = (
        (v1_rmse - rmse) /
        v1_rmse
    ) * 100

    print("\n" + "=" * 70)
    print("V1 VS FINAL V2")
    print("=" * 70)

    print(
        f"\nV1 MAE:       {v1_mae:.2f}"
    )

    print(
        f"V2 MAE:       {mae:.2f}"
    )

    print(
        f"MAE improvement:  {mae_improvement:.2f}%"
    )

    print(
        f"\nV1 RMSE:      {v1_rmse:.2f}"
    )

    print(
        f"V2 RMSE:      {rmse:.2f}"
    )

    print(
        f"RMSE improvement: {rmse_improvement:.2f}%"
    )

    # --------------------------------------------------------
    # Prediction statistics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREDICTION STATISTICS")
    print("=" * 70)

    print(
        f"\nActual mean demand:      {y_test.mean():.2f}"
    )

    print(
        f"Predicted mean demand:   {predictions.mean():.2f}"
    )

    print(
        f"Actual maximum demand:   {y_test.max():.0f}"
    )

    print(
        f"Predicted maximum demand:{predictions.max():.0f}"
    )

    print(
        f"Actual minimum demand:   {y_test.min():.0f}"
    )

    print(
        f"Predicted minimum demand:{predictions.min():.0f}"
    )

    # --------------------------------------------------------
    # Save final predictions
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = test[
        [
            "transit_timestamp",
            "bus_route",
            "total_ridership"
        ]
    ].copy()

    results["predicted_ridership"] = predictions

    results["absolute_error"] = np.abs(
        results["total_ridership"] -
        results["predicted_ridership"]
    )

    output_file = (
        OUTPUT_DIR /
        "final_v2_predictions.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nFinal predictions saved to: {output_file}"
    )

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL MODEL DECISION")
    print("=" * 70)

    if mae < v1_mae:

        print(
            "\n🏆 V2 SELECTED AS FINAL PRODUCTION MODEL"
        )

        print(
            f"V2 achieves {mae:.2f} MAE "
            f"vs V1 {v1_mae:.2f} MAE."
        )

    else:

        print(
            "\nV1 remains the better model."
        )

    print("\n" + "=" * 70)
    print("FINAL EVALUATION COMPLETE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()