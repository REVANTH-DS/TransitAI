import pandas as pd


FILE_PATH = "data/processed/mta_demand_sample.csv"


def load_data():
    """Load the extracted demand dataset."""
    df = pd.read_csv(FILE_PATH)

    df["transit_timestamp"] = pd.to_datetime(
        df["transit_timestamp"]
    )

    df["total_ridership"] = pd.to_numeric(
        df["total_ridership"],
        errors="coerce"
    )

    return df


def run_quality_checks(df):
    """Run basic data quality checks."""

    print("=" * 60)
    print("TRANSITAI - DATA QUALITY REPORT")
    print("=" * 60)

    # 1. Dataset shape
    print("\n1. Dataset Shape")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # 2. Columns
    print("\n2. Columns")
    print(df.columns.tolist())

    # 3. Missing values
    print("\n3. Missing Values")
    print(df.isnull().sum())

    # 4. Duplicate records
    print("\n4. Duplicate Rows")
    print(df.duplicated().sum())

    # 5. Duplicate route-hour combinations
    print("\n5. Duplicate Route-Hour Records")

    duplicate_route_hours = df.duplicated(
        subset=["transit_timestamp", "bus_route"]
    ).sum()

    print(duplicate_route_hours)

    # 6. Negative ridership
    print("\n6. Negative Ridership")

    negative_values = (
        df["total_ridership"] < 0
    ).sum()

    print(negative_values)

    # 7. Zero ridership
    print("\n7. Zero Ridership")

    zero_values = (
        df["total_ridership"] == 0
    ).sum()

    print(zero_values)

    # 8. Number of routes
    print("\n8. Unique Routes")

    print(
        df["bus_route"].nunique()
    )

    # 9. Timestamp range
    print("\n9. Timestamp Range")

    print(
        "Start:",
        df["transit_timestamp"].min()
    )

    print(
        "End:",
        df["transit_timestamp"].max()
    )

    # 10. Demand statistics
    print("\n10. Ridership Statistics")

    print(
        df["total_ridership"].describe()
    )

    print("\n" + "=" * 60)
    print("QUALITY CHECK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":

    df = load_data()

    run_quality_checks(df)