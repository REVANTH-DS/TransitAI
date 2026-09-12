import pandas as pd
import numpy as np

from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = "data/processed/train_v2.csv"
VALID_FILE = "data/processed/validation_v2.csv"
TEST_FILE = "data/processed/test_v2.csv"

MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "transitai_xgboost_v2.json"
FEATURE_FILE = MODEL_DIR / "feature_columns_v2.csv"


# ============================================================
# V2 FEATURES
# ============================================================

BASE_FEATURES = [
    # Time
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "is_weekend",

    # Cyclical
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",

    # Historical demand
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "lag_48",

    # Original rolling features
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_mean_24",

    # ========================================================
    # V2 IMPROVEMENT FEATURES
    # ========================================================

    # Peak periods
    "is_morning_peak",
    "is_evening_peak",
    "is_night",
    "is_daytime",

    # Calendar
    "is_month_start",
    "is_month_end",
    "is_week_start",
    "is_week_end",

    # Demand-period categories
    "demand_period_morning_peak",
    "demand_period_midday",
    "demand_period_evening_peak",
    "demand_period_evening",
    "demand_period_night",

    # Additional rolling statistics
    "rolling_median_6",
    "rolling_std_6",
    "rolling_std_24",

    # Demand momentum
    "demand_change_1h",
    "demand_change_3h",
    "demand_ratio_24h",
]


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(train, validation, test):

    print("\nEncoding route information...")

    # Combine datasets so every route receives
    # the same one-hot encoded column.
    combined = pd.concat(
        [train, validation, test],
        axis=0,
        ignore_index=True
    )

    combined = pd.get_dummies(
        combined,
        columns=["bus_route"],
        dtype=int
    )

    # Split back into original datasets
    train_end = len(train)

    validation_end = (
        train_end + len(validation)
    )

    train_encoded = combined.iloc[
        :train_end
    ].copy()

    validation_encoded = combined.iloc[
        train_end:validation_end
    ].copy()

    test_encoded = combined.iloc[
        validation_end:
    ].copy()

    # Find route columns
    route_columns = [
        column
        for column in combined.columns
        if column.startswith("bus_route_")
    ]

    feature_columns = (
        BASE_FEATURES +
        route_columns
    )

    # Check that all V2 features exist
    missing_features = [
        feature
        for feature in BASE_FEATURES
        if feature not in combined.columns
    ]

    if missing_features:

        print("\nERROR: Missing V2 features:")

        for feature in missing_features:
            print(f"  - {feature}")

        raise ValueError(
            "Required V2 features are missing."
        )

    # Build X
    X_train = train_encoded[
        feature_columns
    ]

    X_valid = validation_encoded[
        feature_columns
    ]

    X_test = test_encoded[
        feature_columns
    ]

    # Build y
    y_train = train_encoded[
        "total_ridership"
    ]

    y_valid = validation_encoded[
        "total_ridership"
    ]

    y_test = test_encoded[
        "total_ridership"
    ]

    return (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        feature_columns
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(y_true, predictions):

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

    return mae, rmse


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TRANSITAI - XGBOOST V2 FORECASTING MODEL")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading V2 datasets...")

    train = pd.read_csv(
        TRAIN_FILE
    )

    validation = pd.read_csv(
        VALID_FILE
    )

    test = pd.read_csv(
        TEST_FILE
    )

    print(
        f"Train rows:      {len(train):,}"
    )

    print(
        f"Validation rows: {len(validation):,}"
    )

    print(
        f"Test rows:       {len(test):,}"
    )

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        feature_columns
    ) = prepare_data(
        train,
        validation,
        test
    )

    print(
        f"\nTraining features: {X_train.shape[1]}"
    )

    print(
        f"V2 engineered features: {len(BASE_FEATURES)}"
    )

    print(
        f"Route features: "
        f"{X_train.shape[1] - len(BASE_FEATURES)}"
    )

    # --------------------------------------------------------
    # TRAIN XGBOOST V2
    # --------------------------------------------------------

    print("\nTraining XGBoost V2...")

    model = XGBRegressor(

        n_estimators=600,

        learning_rate=0.05,

        max_depth=8,

        min_child_weight=5,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="reg:squarederror",

        eval_metric="mae",

        tree_method="hist",

        random_state=42,

        n_jobs=-1
    )

    model.fit(

        X_train,

        y_train,

        eval_set=[
            (X_valid, y_valid)
        ],

        verbose=False
    )

    print(
        "Training complete."
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print("\nEvaluating validation set...")

    valid_predictions = model.predict(
        X_valid
    )

    valid_predictions = np.maximum(
        valid_predictions,
        0
    )

    valid_mae, valid_rmse = evaluate(
        y_valid,
        valid_predictions
    )

    print(
        f"Validation MAE:  {valid_mae:.2f}"
    )

    print(
        f"Validation RMSE: {valid_rmse:.2f}"
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print("\nEvaluating test set...")

    test_predictions = model.predict(
        X_test
    )

    test_predictions = np.maximum(
        test_predictions,
        0
    )

    test_mae, test_rmse = evaluate(
        y_test,
        test_predictions
    )

    print(
        f"Test MAE:  {test_mae:.2f}"
    )

    print(
        f"Test RMSE: {test_rmse:.2f}"
    )

    # --------------------------------------------------------
    # V1 VS V2
    # --------------------------------------------------------

    V1_TEST_MAE = 13.14

    improvement = (
        (V1_TEST_MAE - test_mae)
        / V1_TEST_MAE
    ) * 100

    print("\n" + "=" * 70)
    print("XGBOOST V1 VS V2")
    print("=" * 70)

    print(
        f"\nV1 Test MAE: {V1_TEST_MAE:.2f}"
    )

    print(
        f"V2 Test MAE: {test_mae:.2f}"
    )

    if improvement > 0:

        print(
            f"V2 Improvement: {improvement:.2f}%"
        )

        print(
            "\n🏆 V2 OUTPERFORMS V1"
        )

    elif improvement < 0:

        print(
            f"V2 Change: {improvement:.2f}%"
        )

        print(
            "\nV1 currently performs better."
        )

    else:

        print(
            "\nV1 and V2 have identical MAE."
        )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance = pd.DataFrame({

        "feature": feature_columns,

        "importance":
            model.feature_importances_

    }).sort_values(

        "importance",

        ascending=False
    )

    print("\nTop 25 V2 Features:")

    print(
        importance.head(25).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model.save_model(
        MODEL_FILE
    )

    print(
        f"\nModel saved to: {MODEL_FILE}"
    )

    # --------------------------------------------------------
    # SAVE FEATURE LIST
    # --------------------------------------------------------

    pd.Series(
        feature_columns
    ).to_csv(
        FEATURE_FILE,
        index=False,
        header=False
    )

    print(
        f"Feature list saved to: {FEATURE_FILE}"
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("XGBOOST V2 TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"\nV1 MAE: {V1_TEST_MAE:.2f}"
    )

    print(
        f"V2 MAE: {test_mae:.2f}"
    )

    print(
        f"V2 RMSE: {test_rmse:.2f}"
    )

    print(
        f"V2 improvement: {improvement:.2f}%"
    )

    print("\nNext step:")
    print("Compare V1 and V2 and select the final model.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()