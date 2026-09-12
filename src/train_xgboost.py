import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path


TRAIN_FILE = "data/processed/train.csv"
VALID_FILE = "data/processed/validation.csv"
TEST_FILE = "data/processed/test.csv"

MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "transitai_xgboost.json"


FEATURES = [
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "is_weekend",

    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",

    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "lag_48",

    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_mean_24",
]


def prepare_data(train, validation, test):

    # Combine route categories so every dataset
    # uses the same encoding.

    combined = pd.concat(
        [train, validation, test],
        axis=0
    )

    combined = pd.get_dummies(
        combined,
        columns=["bus_route"],
        dtype=int
    )

    train_end = len(train)

    validation_end = train_end + len(validation)

    train_encoded = combined.iloc[
        :train_end
    ].copy()

    validation_encoded = combined.iloc[
        train_end:validation_end
    ].copy()

    test_encoded = combined.iloc[
        validation_end:
    ].copy()

    # Route columns
    route_columns = [
        col for col in combined.columns
        if col.startswith("bus_route_")
    ]

    feature_columns = FEATURES + route_columns

    X_train = train_encoded[feature_columns]
    X_valid = validation_encoded[feature_columns]
    X_test = test_encoded[feature_columns]

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


def main():

    print("=" * 70)
    print("TRANSITAI - XGBOOST FORECASTING MODEL")
    print("=" * 70)

    # --------------------------------------------------
    # Load
    # --------------------------------------------------

    print("\nLoading datasets...")

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

    # --------------------------------------------------
    # Prepare
    # --------------------------------------------------

    print("\nEncoding route information...")

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
        f"Training features: {X_train.shape[1]}"
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    print("\nTraining XGBoost...")

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

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Test
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Compare with baseline
    # --------------------------------------------------

    baseline_mae = 24.76

    improvement = (
        (baseline_mae - test_mae)
        / baseline_mae
    ) * 100

    print("\n" + "=" * 70)
    print("BASELINE VS XGBOOST")
    print("=" * 70)

    print(
        f"\nLag-1 baseline MAE: {baseline_mae:.2f}"
    )

    print(
        f"XGBoost MAE:        {test_mae:.2f}"
    )

    print(
        f"Improvement:        {improvement:.2f}%"
    )

    # --------------------------------------------------
    # Feature importance
    # --------------------------------------------------

    importance = pd.DataFrame({

        "feature": feature_columns,

        "importance": model.feature_importances_

    }).sort_values(
        "importance",
        ascending=False
    )

    print("\nTop 20 Features:")

    print(
        importance.head(20).to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

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

    # Save feature names
    pd.Series(
        feature_columns
    ).to_csv(
        MODEL_DIR / "feature_columns.csv",
        index=False,
        header=False
    )

    print(
        "Feature list saved."
    )

    print("\n" + "=" * 70)
    print("XGBOOST TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()