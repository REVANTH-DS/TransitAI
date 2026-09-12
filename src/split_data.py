import pandas as pd
from pathlib import Path


INPUT_FILE = "data/processed/model_data.csv"

TRAIN_FILE = "data/processed/train.csv"
VALID_FILE = "data/processed/validation.csv"
TEST_FILE = "data/processed/test.csv"


def main():

    print("=" * 70)
    print("TRANSITAI - TIME SERIES DATA SPLIT")
    print("=" * 70)

    # --------------------------------------------------
    # Load
    # --------------------------------------------------

    print("\nLoading model data...")

    df = pd.read_csv(INPUT_FILE)

    df["transit_timestamp"] = pd.to_datetime(
        df["transit_timestamp"]
    )

    df = df.sort_values(
        "transit_timestamp"
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Determine date range
    # --------------------------------------------------

    start_date = df["transit_timestamp"].min()
    end_date = df["transit_timestamp"].max()

    print("\nDataset range:")
    print(start_date)
    print("→")
    print(end_date)

    # --------------------------------------------------
    # Use chronological split
    # --------------------------------------------------

    # Last 20% = test
    # Previous 10% = validation
    # First 70% = train

    unique_dates = sorted(
        df["transit_timestamp"].dt.date.unique()
    )

    total_days = len(unique_dates)

    train_end_index = int(
        total_days * 0.70
    )

    valid_end_index = int(
        total_days * 0.90
    )

    train_end_date = unique_dates[
        train_end_index
    ]

    valid_end_date = unique_dates[
        valid_end_index
    ]

    # --------------------------------------------------
    # Split
    # --------------------------------------------------

    train = df[
        df["transit_timestamp"].dt.date
        < train_end_date
    ]

    validation = df[
        (df["transit_timestamp"].dt.date >= train_end_date)
        &
        (df["transit_timestamp"].dt.date < valid_end_date)
    ]

    test = df[
        df["transit_timestamp"].dt.date
        >= valid_end_date
    ]

    # --------------------------------------------------
    # Print split information
    # --------------------------------------------------

    print("\n" + "-" * 70)
    print("TRAIN")
    print("-" * 70)

    print("Rows:", f"{len(train):,}")
    print("Start:", train["transit_timestamp"].min())
    print("End:", train["transit_timestamp"].max())

    print("\n" + "-" * 70)
    print("VALIDATION")
    print("-" * 70)

    print("Rows:", f"{len(validation):,}")
    print("Start:", validation["transit_timestamp"].min())
    print("End:", validation["transit_timestamp"].max())

    print("\n" + "-" * 70)
    print("TEST")
    print("-" * 70)

    print("Rows:", f"{len(test):,}")
    print("Start:", test["transit_timestamp"].min())
    print("End:", test["transit_timestamp"].max())

    # --------------------------------------------------
    # Check leakage
    # --------------------------------------------------

    print("\n" + "-" * 70)
    print("LEAKAGE CHECK")
    print("-" * 70)

    if (
        train["transit_timestamp"].max()
        < validation["transit_timestamp"].min()
        and
        validation["transit_timestamp"].max()
        < test["transit_timestamp"].min()
    ):

        print("✓ No temporal leakage detected")

    else:

        print("✗ WARNING: Temporal overlap detected")

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    Path(
        "data/processed"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    train.to_csv(
        TRAIN_FILE,
        index=False
    )

    validation.to_csv(
        VALID_FILE,
        index=False
    )

    test.to_csv(
        TEST_FILE,
        index=False
    )

    print("\nSaved files:")

    print(TRAIN_FILE)
    print(VALID_FILE)
    print(TEST_FILE)

    print("\n" + "=" * 70)
    print("DATA SPLIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()