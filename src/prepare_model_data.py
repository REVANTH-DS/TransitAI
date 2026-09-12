import pandas as pd
from pathlib import Path


INPUT_FILE = "data/processed/mta_features.csv"
OUTPUT_FILE = "data/processed/model_data.csv"


def main():

    print("=" * 70)
    print("TRANSITAI - MODEL DATA PREPARATION")
    print("=" * 70)

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    print("\nLoading feature dataset...")

    df = pd.read_csv(INPUT_FILE)

    df["transit_timestamp"] = pd.to_datetime(
        df["transit_timestamp"]
    )

    df["total_ridership"] = pd.to_numeric(
        df["total_ridership"],
        errors="coerce"
    )

    # --------------------------------------------------
    # Sort
    # --------------------------------------------------

    df = df.sort_values(
        ["bus_route", "transit_timestamp"]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Check existing feature availability
    # --------------------------------------------------

    required_features = [
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
        "rolling_mean_24"
    ]

    print("\nChecking required features...")

    missing_features = [
        col for col in required_features
        if col not in df.columns
    ]

    if missing_features:

        print(
            "Missing features:",
            missing_features
        )

        return

    # --------------------------------------------------
    # Remove rows with missing required features
    # --------------------------------------------------

    print("\nRemoving incomplete feature rows...")

    before = len(df)

    df = df.dropna(
        subset=required_features
    )

    after = len(df)

    print(
        f"Rows before: {before:,}"
    )

    print(
        f"Rows after:  {after:,}"
    )

    print(
        f"Rows removed: {before - after:,}"
    )

    # --------------------------------------------------
    # Keep only useful columns
    # --------------------------------------------------

    model_columns = [
        "transit_timestamp",
        "bus_route",
        "total_ridership",

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
        "rolling_mean_24"
    ]

    df = df[model_columns]

    # --------------------------------------------------
    # Final checks
    # --------------------------------------------------

    print("\nFinal missing values:")

    print(
        df.isnull().sum()
        .sum()
    )

    print("\nFinal shape:")

    print(
        df.shape
    )

    print("\nRoutes:")

    print(
        df["bus_route"].nunique()
    )

    print("\nDate range:")

    print(
        df["transit_timestamp"].min()
    )

    print(
        "→",
        df["transit_timestamp"].max()
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    Path(OUTPUT_FILE).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )

    print("\nSample:")

    print(
        df.head(10).to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("MODEL DATA PREPARATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()