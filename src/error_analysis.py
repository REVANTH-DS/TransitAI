import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIGURATION
# ============================================================

TEST_FILE = Path("data/processed/test.csv")
MODEL_FILE = Path("models/transitai_xgboost.json")
FEATURE_FILE = Path("models/feature_columns.csv")

OUTPUT_DIR = Path("data/processed")


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\nLoading XGBoost model...")

    model = xgb.XGBRegressor()

    model.load_model(
        MODEL_FILE
    )

    print("Model loaded.")

    return model


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features():

    features = pd.read_csv(
        FEATURE_FILE,
        header=None
    )[0].tolist()

    return features


# ============================================================
# PREPARE TEST DATA
# ============================================================

def prepare_test_data(test, features):

    print("\nPreparing test data...")

    # Convert route into one-hot encoding
    test_encoded = pd.get_dummies(
        test,
        columns=["bus_route"],
        dtype=int
    )

    # Make sure every model feature exists
    for feature in features:

        if feature not in test_encoded.columns:

            test_encoded[feature] = 0

    X_test = test_encoded[
        features
    ]

    y_test = test_encoded[
        "total_ridership"
    ]

    return X_test, y_test, test_encoded


# ============================================================
# OVERALL ERROR
# ============================================================

def overall_error_analysis(
    y_true,
    predictions
):

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

    errors = (
        predictions - y_true
    )

    absolute_errors = np.abs(
        errors
    )

    print("\n" + "=" * 70)
    print("OVERALL ERROR ANALYSIS")
    print("=" * 70)

    print(
        f"\nMAE:  {mae:.2f} riders"
    )

    print(
        f"RMSE: {rmse:.2f} riders"
    )

    print(
        f"Mean absolute error: "
        f"{absolute_errors.mean():.2f}"
    )

    print(
        f"Maximum error: "
        f"{absolute_errors.max():.2f}"
    )

    return errors, absolute_errors


# ============================================================
# ROUTE LEVEL ERROR
# ============================================================

def route_error_analysis(
    test,
    absolute_errors
):

    print("\n" + "=" * 70)
    print("ROUTE-LEVEL ERROR ANALYSIS")
    print("=" * 70)

    analysis = pd.DataFrame({

        "bus_route":
            test["bus_route"].values,

        "absolute_error":
            absolute_errors

    })

    route_stats = (
        analysis
        .groupby("bus_route")
        .agg(
            mean_error=(
                "absolute_error",
                "mean"
            ),

            median_error=(
                "absolute_error",
                "median"
            ),

            max_error=(
                "absolute_error",
                "max"
            ),

            records=(
                "absolute_error",
                "count"
            )
        )
        .sort_values(
            "mean_error",
            ascending=False
        )
    )

    print(
        "\nTop 20 routes with highest MAE:"
    )

    print(
        route_stats
        .head(20)
        .to_string()
    )

    return route_stats


# ============================================================
# HOURLY ERROR
# ============================================================

def hourly_error_analysis(
    test,
    absolute_errors
):

    print("\n" + "=" * 70)
    print("HOURLY ERROR ANALYSIS")
    print("=" * 70)

    analysis = pd.DataFrame({

        "hour":
            test["hour"].values,

        "absolute_error":
            absolute_errors

    })

    hourly_stats = (
        analysis
        .groupby("hour")
        .agg(
            mean_error=(
                "absolute_error",
                "mean"
            ),

            median_error=(
                "absolute_error",
                "median"
            ),

            records=(
                "absolute_error",
                "count"
            )
        )
        .sort_values(
            "mean_error",
            ascending=False
        )
    )

    print(
        "\nHours with highest prediction error:"
    )

    print(
        hourly_stats
        .to_string()
    )

    return hourly_stats


# ============================================================
# WEEKDAY / WEEKEND ERROR
# ============================================================

def weekday_error_analysis(
    test,
    absolute_errors
):

    print("\n" + "=" * 70)
    print("WEEKDAY VS WEEKEND ERROR")
    print("=" * 70)

    analysis = pd.DataFrame({

        "is_weekend":
            test["is_weekend"].values,

        "absolute_error":
            absolute_errors

    })

    stats = (
        analysis
        .groupby("is_weekend")
        .agg(
            mean_error=(
                "absolute_error",
                "mean"
            ),

            median_error=(
                "absolute_error",
                "median"
            ),

            records=(
                "absolute_error",
                "count"
            )
        )
    )

    stats.index = [
        "Weekday"
        if value == 0
        else "Weekend"
        for value in stats.index
    ]

    print(
        stats.to_string()
    )

    return stats


# ============================================================
# PEAK VS NON-PEAK ERROR
# ============================================================

def peak_error_analysis(
    test,
    absolute_errors
):

    print("\n" + "=" * 70)
    print("PEAK VS NON-PEAK ERROR")
    print("=" * 70)

    peak_hours = (
        test["hour"].between(
            6,
            10
        )
        |
        test["hour"].between(
            16,
            20
        )
    )

    analysis = pd.DataFrame({

        "period": np.where(
            peak_hours,
            "Peak",
            "Non-Peak"
        ),

        "absolute_error":
            absolute_errors

    })

    stats = (
        analysis
        .groupby("period")
        .agg(
            mean_error=(
                "absolute_error",
                "mean"
            ),

            median_error=(
                "absolute_error",
                "median"
            ),

            records=(
                "absolute_error",
                "count"
            )
        )
    )

    print(
        stats.to_string()
    )

    return stats


# ============================================================
# UNDER / OVER PREDICTION
# ============================================================

def prediction_bias_analysis(
    y_true,
    predictions
):

    print("\n" + "=" * 70)
    print("UNDERPREDICTION VS OVERPREDICTION")
    print("=" * 70)

    errors = (
        predictions - y_true
    )

    underpredictions = (
        errors < 0
    ).sum()

    overpredictions = (
        errors > 0
    ).sum()

    accurate = (
        errors == 0
    ).sum()

    total = len(errors)

    print(
        f"\nUnderpredictions: "
        f"{underpredictions:,} "
        f"({underpredictions / total * 100:.2f}%)"
    )

    print(
        f"Overpredictions:  "
        f"{overpredictions:,} "
        f"({overpredictions / total * 100:.2f}%)"
    )

    print(
        f"Exact predictions: "
        f"{accurate:,} "
        f"({accurate / total * 100:.2f}%)"
    )


# ============================================================
# LARGEST ERRORS
# ============================================================

def largest_errors_analysis(
    test,
    y_true,
    predictions,
    absolute_errors
):

    print("\n" + "=" * 70)
    print("LARGEST PREDICTION ERRORS")
    print("=" * 70)

    result = pd.DataFrame({

        "timestamp":
            test["transit_timestamp"].values,

        "bus_route":
            test["bus_route"].values,

        "actual":
            y_true.values,

        "predicted":
            predictions,

        "error":
            predictions - y_true.values,

        "absolute_error":
            absolute_errors

    })

    result = result.sort_values(
        "absolute_error",
        ascending=False
    )

    print(
        "\nTop 20 largest errors:"
    )

    print(
        result
        .head(20)
        .to_string(
            index=False
        )
    )

    return result


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    route_stats,
    hourly_stats,
    weekday_stats,
    peak_stats,
    largest_errors
):

    print(
        "\nSaving error analysis results..."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    route_stats.to_csv(
        OUTPUT_DIR /
        "error_by_route.csv"
    )

    hourly_stats.to_csv(
        OUTPUT_DIR /
        "error_by_hour.csv"
    )

    weekday_stats.to_csv(
        OUTPUT_DIR /
        "error_weekday_weekend.csv"
    )

    peak_stats.to_csv(
        OUTPUT_DIR /
        "error_peak_vs_nonpeak.csv"
    )

    largest_errors.to_csv(
        OUTPUT_DIR /
        "largest_prediction_errors.csv",
        index=False
    )

    print(
        "Error analysis files saved."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "TRANSITAI - MODEL ERROR ANALYSIS"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print(
        "\nLoading test dataset..."
    )

    test = pd.read_csv(
        TEST_FILE
    )

    print(
        f"Test rows: {len(test):,}"
    )

    features = load_features()

    model = load_model()

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    X_test, y_test, test_encoded = (
        prepare_test_data(
            test,
            features
        )
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print(
        "\nGenerating test predictions..."
    )

    predictions = model.predict(
        X_test
    )

    predictions = np.maximum(
        predictions,
        0
    )

    print(
        "Predictions generated."
    )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    errors, absolute_errors = (
        overall_error_analysis(
            y_test,
            predictions
        )
    )

    # --------------------------------------------------------
    # Route
    # --------------------------------------------------------

    route_stats = (
        route_error_analysis(
            test,
            absolute_errors
        )
    )

    # --------------------------------------------------------
    # Hour
    # --------------------------------------------------------

    hourly_stats = (
        hourly_error_analysis(
            test,
            absolute_errors
        )
    )

    # --------------------------------------------------------
    # Weekday / Weekend
    # --------------------------------------------------------

    weekday_stats = (
        weekday_error_analysis(
            test,
            absolute_errors
        )
    )

    # --------------------------------------------------------
    # Peak / Non-Peak
    # --------------------------------------------------------

    peak_stats = (
        peak_error_analysis(
            test,
            absolute_errors
        )
    )

    # --------------------------------------------------------
    # Bias
    # --------------------------------------------------------

    prediction_bias_analysis(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # Largest errors
    # --------------------------------------------------------

    largest_errors = (
        largest_errors_analysis(
            test,
            y_test,
            predictions,
            absolute_errors
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        route_stats,
        hourly_stats,
        weekday_stats,
        peak_stats,
        largest_errors
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "ERROR ANALYSIS COMPLETE"
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()