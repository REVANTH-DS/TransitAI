import pandas as pd
from pathlib import Path


INPUT_FILE = "data/processed/model_data_v2.csv"

TRAIN_FILE = "data/processed/train_v2.csv"
VALIDATION_FILE = "data/processed/validation_v2.csv"
TEST_FILE = "data/processed/test_v2.csv"


def main():

    print("=" * 70)
    print("TRANSITAI - V2 TIME SERIES DATA SPLIT")
    print("=" * 70)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    print("\nLoading V2 model data...")

    df = pd.read_csv(INPUT_FILE)

    df["transit_timestamp"] = pd.to_datetime(
        df["transit_timestamp"]
    )

    df = df.sort_values(
        "transit_timestamp"
    ).reset_index(drop=True)

    print(f"\nTotal rows: {len(df):,}")

    print(
        f"Date range:\n"
        f"{df['transit_timestamp'].min()}\n"
        f"→\n"
        f"{df['transit_timestamp'].max()}"
    )

    # --------------------------------------------------
    # Get UNIQUE timestamps
    # --------------------------------------------------

    timestamps = sorted(
        df["transit_timestamp"].unique()
    )

    total_timestamps = len(timestamps)

    print(f"\nUnique timestamps: {total_timestamps:,}")

    # --------------------------------------------------
    # Calculate timestamp boundaries
    # --------------------------------------------------

    train_timestamp_end = int(
        total_timestamps * 0.70
    )

    validation_timestamp_end = int(
        total_timestamps * 0.90
    )

    train_end_time = timestamps[
        train_timestamp_end - 1
    ]

    validation_end_time = timestamps[
        validation_timestamp_end - 1
    ]

    # --------------------------------------------------
    # Create splits
    # --------------------------------------------------

    train = df[
        df["transit_timestamp"] <= train_end_time
    ].copy()

    validation = df[
        (df["transit_timestamp"] > train_end_time)
        &
        (df["transit_timestamp"] <= validation_end_time)
    ].copy()

    test = df[
        df["transit_timestamp"] > validation_end_time
    ].copy()

    # --------------------------------------------------
    # Display split information
    # --------------------------------------------------

    print("\n" + "-" * 70)
    print("TRAIN")
    print("-" * 70)

    print(f"Rows: {len(train):,}")
    print(f"Start: {train['transit_timestamp'].min()}")
    print(f"End:   {train['transit_timestamp'].max()}")

    print("\n" + "-" * 70)
    print("VALIDATION")
    print("-" * 70)

    print(f"Rows: {len(validation):,}")
    print(f"Start: {validation['transit_timestamp'].min()}")
    print(f"End:   {validation['transit_timestamp'].max()}")

    print("\n" + "-" * 70)
    print("TEST")
    print("-" * 70)

    print(f"Rows: {len(test):,}")
    print(f"Start: {test['transit_timestamp'].min()}")
    print(f"End:   {test['transit_timestamp'].max()}")

    # --------------------------------------------------
    # Strong leakage checks
    # --------------------------------------------------

    print("\n" + "-" * 70)
    print("LEAKAGE CHECK")
    print("-" * 70)

    train_max = train["transit_timestamp"].max()
    validation_min = validation["transit_timestamp"].min()

    validation_max = validation["transit_timestamp"].max()
    test_min = test["transit_timestamp"].min()

    # Check timestamp boundaries
    timestamp_boundary_ok = (
        train_max < validation_min
        and
        validation_max < test_min
    )

    # Check overlapping timestamps
    train_times = set(
        train["transit_timestamp"].unique()
    )

    validation_times = set(
        validation["transit_timestamp"].unique()
    )

    test_times = set(
        test["transit_timestamp"].unique()
    )

    overlap_train_validation = (
        train_times & validation_times
    )

    overlap_validation_test = (
        validation_times & test_times
    )

    if (
        timestamp_boundary_ok
        and
        len(overlap_train_validation) == 0
        and
        len(overlap_validation_test) == 0
    ):

        print("✓ No temporal leakage detected")
        print("✓ No train/validation timestamp overlap")
        print("✓ No validation/test timestamp overlap")

    else:

        print("✗ TEMPORAL LEAKAGE DETECTED")

        print(
            f"Train/Validation overlap: "
            f"{len(overlap_train_validation)} timestamps"
        )

        print(
            f"Validation/Test overlap: "
            f"{len(overlap_validation_test)} timestamps"
        )

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
        VALIDATION_FILE,
        index=False
    )

    test.to_csv(
        TEST_FILE,
        index=False
    )

    # --------------------------------------------------
    # Final report
    # --------------------------------------------------

    print("\nSaved files:")

    print(f"  {TRAIN_FILE}")
    print(f"  {VALIDATION_FILE}")
    print(f"  {TEST_FILE}")

    print("\n" + "=" * 70)
    print("V2 DATA SPLIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()