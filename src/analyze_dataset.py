import pandas as pd

FILE_PATH = "data/processed/mta_hourly_demand.csv"


def main():

    print("=" * 70)
    print("TRANSITAI - DATASET ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(FILE_PATH)

    df["transit_timestamp"] = pd.to_datetime(
        df["transit_timestamp"]
    )

    df["total_ridership"] = pd.to_numeric(
        df["total_ridership"],
        errors="coerce"
    )

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nDate range:")
    print(df["transit_timestamp"].min())
    print("→")
    print(df["transit_timestamp"].max())

    print("\nUnique routes:")
    print(df["bus_route"].nunique())

    print("\nTop 20 routes by number of records:")
    print(
        df["bus_route"]
        .value_counts()
        .head(20)
    )

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate route-hour records:")
    print(
        df.duplicated(
            subset=[
                "transit_timestamp",
                "bus_route"
            ]
        ).sum()
    )

    print("\nRidership statistics:")
    print(
        df["total_ridership"].describe()
    )

    print("\nZero-demand records:")
    print(
        (df["total_ridership"] == 0).sum()
    )

    print("\nSample:")
    print(
        df.head(10).to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()