import pandas as pd
import numpy as np


TRAIN_FILE = "data/processed/train.csv"
VALID_FILE = "data/processed/validation.csv"
TEST_FILE = "data/processed/test.csv"


def calculate_metrics(actual, predicted):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mae = np.mean(
        np.abs(actual - predicted)
    )

    rmse = np.sqrt(
        np.mean(
            (actual - predicted) ** 2
        )
    )

    # sMAPE is safer than MAPE when demand can be zero
    denominator = (
        np.abs(actual) +
        np.abs(predicted)
    )

    mask = denominator != 0

    smape = np.mean(
        2 * np.abs(
            actual[mask] - predicted[mask]
        ) / denominator[mask]
    ) * 100

    return mae, rmse, smape


def evaluate_dataset(df, name):

    actual = df["total_ridership"]

    # Baseline 1:
    # Previous hour
    pred_lag1 = df["lag_1"]

    # Baseline 2:
    # Previous day
    pred_lag24 = df["lag_24"]

    mae1, rmse1, smape1 = calculate_metrics(
        actual,
        pred_lag1
    )

    mae24, rmse24, smape24 = calculate_metrics(
        actual,
        pred_lag24
    )

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("\nLag-1 Baseline")
    print(
        f"MAE:   {mae1:.2f}"
    )
    print(
        f"RMSE:  {rmse1:.2f}"
    )
    print(
        f"sMAPE: {smape1:.2f}%"
    )

    print("\nLag-24 Baseline")
    print(
        f"MAE:   {mae24:.2f}"
    )
    print(
        f"RMSE:  {rmse24:.2f}"
    )
    print(
        f"sMAPE: {smape24:.2f}%"
    )

    return {
        "lag1_mae": mae1,
        "lag1_rmse": rmse1,
        "lag1_smape": smape1,
        "lag24_mae": mae24,
        "lag24_rmse": rmse24,
        "lag24_smape": smape24
    }


def main():

    print("=" * 70)
    print("TRANSITAI - BASELINE FORECASTING")
    print("=" * 70)

    print("\nLoading datasets...")

    train = pd.read_csv(TRAIN_FILE)
    validation = pd.read_csv(VALID_FILE)
    test = pd.read_csv(TEST_FILE)

    # --------------------------------------------------
    # Evaluate validation
    # --------------------------------------------------

    validation_results = evaluate_dataset(
        validation,
        "VALIDATION RESULTS"
    )

    # --------------------------------------------------
    # Evaluate test
    # --------------------------------------------------

    test_results = evaluate_dataset(
        test,
        "TEST RESULTS"
    )

    # --------------------------------------------------
    # Final comparison
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)

    print(
        f"\nValidation Lag-1 MAE: "
        f"{validation_results['lag1_mae']:.2f}"
    )

    print(
        f"Validation Lag-24 MAE: "
        f"{validation_results['lag24_mae']:.2f}"
    )

    print(
        f"\nTest Lag-1 MAE: "
        f"{test_results['lag1_mae']:.2f}"
    )

    print(
        f"Test Lag-24 MAE: "
        f"{test_results['lag24_mae']:.2f}"
    )

    print("\n" + "=" * 70)
    print("BASELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()