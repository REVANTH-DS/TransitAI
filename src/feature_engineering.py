import pandas as pd
import numpy as np
from pathlib import Path


INPUT_FILE = "data/processed/mta_hourly_demand.csv"
OUTPUT_FILE = "data/processed/mta_features.csv"


def load_data():

    df = pd.read_csv(INPUT_FILE)

    df["transit_timestamp"] = pd.to_datetime(
        df["transit_timestamp"]
    )

    df["total_ridership"] = pd.to_numeric(
        df["total_ridership"],
        errors="coerce"
    )

    return df


def create_time_features(df):

    df["hour"] = df["transit_timestamp"].dt.hour

    df["day_of_week"] = (
        df["transit_timestamp"].dt.dayofweek
    )

    df["day_of_month"] = (
        df["transit_timestamp"].dt.day
    )

    df["month"] = (
        df["transit_timestamp"].dt.month
    )

    df["week_of_year"] = (
        df["transit_timestamp"].dt.isocalendar().week.astype(int)
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    return df


def create_cyclical_features(df):

    # Hour of day
    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    # Day of week
    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    return df


def create_lag_features(df):

    df = df.sort_values(
        ["bus_route", "transit_timestamp"]
    )

    grouped = df.groupby("bus_route")[
        "total_ridership"
    ]

    # Previous hours
    df["lag_1"] = grouped.shift(1)
    df["lag_2"] = grouped.shift(2)
    df["lag_3"] = grouped.shift(3)

    # Previous day
    df["lag_24"] = grouped.shift(24)

    # Previous two days
    df["lag_48"] = grouped.shift(48)

    # Previous week
    df["lag_168"] = grouped.shift(168)

    return df


def create_rolling_features(df):

    df = df.sort_values(
        ["bus_route", "transit_timestamp"]
    )

    grouped = df.groupby("bus_route")[
        "total_ridership"
    ]

    # Shift first to prevent data leakage
    shifted = grouped.shift(1)

    df["rolling_mean_3"] = (
        shifted
        .groupby(df["bus_route"])
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["rolling_mean_6"] = (
        shifted
        .groupby(df["bus_route"])
        .rolling(6)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["rolling_mean_24"] = (
        shifted
        .groupby(df["bus_route"])
        .rolling(24)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["rolling_mean_168"] = (
        shifted
        .groupby(df["bus_route"])
        .rolling(168)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df


def main():

    print("=" * 70)
    print("TRANSITAI - FEATURE ENGINEERING")
    print("=" * 70)

    print("\nLoading data...")

    df = load_data()

    print(
        f"Original rows: {len(df):,}"
    )

    print("\nCreating time features...")

    df = create_time_features(df)

    print("Creating cyclical features...")

    df = create_cyclical_features(df)

    print("Creating lag features...")

    df = create_lag_features(df)

    print("Creating rolling features...")

    df = create_rolling_features(df)

    # Remove rows without sufficient historical information
    feature_columns = [
        "lag_168",
        "rolling_mean_168"
    ]

    df = df.dropna(
        subset=feature_columns
    )

    # Save
    Path(OUTPUT_FILE).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nFeature engineering complete.")

    print(
        f"Final rows: {len(df):,}"
    )

    print(
        f"Final columns: {len(df.columns)}"
    )

    print("\nColumns:")

    print(
        df.columns.tolist()
    )

    print("\nSample:")

    print(
        df.head().to_string(index=False)
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()