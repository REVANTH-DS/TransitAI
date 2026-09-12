import pandas as pd
import numpy as np
from pathlib import Path


INPUT_FILE = "data/processed/model_data.csv"
OUTPUT_FILE = "data/processed/model_data_v2.csv"


def main():

    print("=" * 70)
    print("TRANSITAI - FEATURE IMPROVEMENT V2")
    print("=" * 70)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    print("\nLoading model data...")

    df = pd.read_csv(INPUT_FILE)

    df["transit_timestamp"] = pd.to_datetime(
        df["transit_timestamp"]
    )

    print(f"Original rows: {len(df):,}")

    # --------------------------------------------------
    # Sort data
    # --------------------------------------------------

    df = df.sort_values(
        ["bus_route", "transit_timestamp"]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Peak-period features
    # --------------------------------------------------

    print("\nCreating peak-period features...")

    # Morning commute
    df["is_morning_peak"] = (
        df["hour"].between(6, 9)
    ).astype(int)

    # Evening commute
    df["is_evening_peak"] = (
        df["hour"].between(16, 19)
    ).astype(int)

    # Night
    df["is_night"] = (
        (df["hour"] >= 22) |
        (df["hour"] <= 5)
    ).astype(int)

    # General daytime
    df["is_daytime"] = (
        df["hour"].between(6, 21)
    ).astype(int)

    # --------------------------------------------------
    # Calendar features
    # --------------------------------------------------

    print("Creating calendar features...")

    df["is_month_start"] = (
        df["day_of_month"] <= 3
    ).astype(int)

    df["is_month_end"] = (
        df["day_of_month"] >= 28
    ).astype(int)

    df["is_week_start"] = (
        df["day_of_week"] == 0
    ).astype(int)

    df["is_week_end"] = (
        df["day_of_week"] == 6
    ).astype(int)

    # --------------------------------------------------
    # Rush-hour category
    # --------------------------------------------------

    print("Creating demand-period category...")

    def get_period(hour):

        if 6 <= hour <= 9:
            return "morning_peak"

        elif 10 <= hour <= 15:
            return "midday"

        elif 16 <= hour <= 19:
            return "evening_peak"

        elif 20 <= hour <= 21:
            return "evening"

        else:
            return "night"

    df["demand_period"] = df["hour"].apply(
        get_period
    )

    # One-hot encode demand period
    df = pd.get_dummies(
        df,
        columns=["demand_period"],
        dtype=int
    )

    # --------------------------------------------------
    # Additional rolling features
    # --------------------------------------------------

    print("Creating additional rolling features...")

    # Important:
    # shift(1) prevents the current target from
    # leaking into the rolling calculation.

    grouped = df.groupby(
        "bus_route",
        group_keys=False
    )

    df["rolling_median_6"] = grouped[
        "total_ridership"
    ].transform(
        lambda x: x.shift(1).rolling(
            window=6,
            min_periods=1
        ).median()
    )

    df["rolling_std_6"] = grouped[
        "total_ridership"
    ].transform(
        lambda x: x.shift(1).rolling(
            window=6,
            min_periods=1
        ).std()
    )

    df["rolling_std_24"] = grouped[
        "total_ridership"
    ].transform(
        lambda x: x.shift(1).rolling(
            window=24,
            min_periods=1
        ).std()
    )

    # --------------------------------------------------
    # Demand momentum
    # --------------------------------------------------

    print("Creating demand momentum features...")

    # Difference between recent demand and
    # previous-hour demand.

    df["demand_change_1h"] = (
        df["lag_1"] - df["lag_2"]
    )

    df["demand_change_3h"] = (
        df["lag_1"] - df["lag_3"]
    )

    # Ratio between recent demand and 24-hour average.
    # Small epsilon prevents division by zero.

    df["demand_ratio_24h"] = (
        df["lag_1"] /
        (df["rolling_mean_24"] + 1)
    )

    # --------------------------------------------------
    # Clean numeric values
    # --------------------------------------------------

    print("Cleaning generated features...")

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # The new rolling features can create NaN values
    # at the beginning of individual routes.

    df = df.dropna(
        subset=[
            "rolling_median_6",
            "rolling_std_6",
            "rolling_std_24"
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    Path(
        "data/processed"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE IMPROVEMENT COMPLETE")
    print("=" * 70)

    print(
        f"\nOriginal rows: {len(pd.read_csv(INPUT_FILE)):,}"
    )

    print(
        f"Final rows:    {len(df):,}"
    )

    print(
        f"Rows removed:  "
        f"{len(pd.read_csv(INPUT_FILE)) - len(df):,}"
    )

    print(
        f"\nOriginal columns: "
        f"{len(pd.read_csv(INPUT_FILE).columns)}"
    )

    print(
        f"Final columns:    "
        f"{len(df.columns)}"
    )

    print("\nNew features:")

    new_features = [
        "is_morning_peak",
        "is_evening_peak",
        "is_night",
        "is_daytime",
        "is_month_start",
        "is_month_end",
        "is_week_start",
        "is_week_end",
        "demand_period_morning_peak",
        "demand_period_midday",
        "demand_period_evening_peak",
        "demand_period_evening",
        "demand_period_night",
        "rolling_median_6",
        "rolling_std_6",
        "rolling_std_24",
        "demand_change_1h",
        "demand_change_3h",
        "demand_ratio_24h"
    ]

    for feature in new_features:

        if feature in df.columns:
            print(f"  ✓ {feature}")

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()