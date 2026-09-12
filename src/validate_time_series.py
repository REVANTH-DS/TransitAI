import pandas as pd


FILE_PATH = "data/processed/mta_features.csv"


def main():

    print("=" * 70)
    print("TRANSITAI - TIME SERIES VALIDATION")
    print("=" * 70)

    df = pd.read_csv(FILE_PATH)

    df["transit_timestamp"] = pd.to_datetime(
        df["transit_timestamp"]
    )

    # --------------------------------------------------
    # 1. Overall time range
    # --------------------------------------------------

    print("\n1. Overall Time Range")

    print("Start:", df["transit_timestamp"].min())
    print("End:", df["transit_timestamp"].max())

    # --------------------------------------------------
    # 2. Number of routes
    # --------------------------------------------------

    print("\n2. Number of Routes")

    print(
        df["bus_route"].nunique()
    )

    # --------------------------------------------------
    # 3. Records per route
    # --------------------------------------------------

    print("\n3. Records Per Route")

    route_counts = (
        df.groupby("bus_route")
        .size()
    )

    print(
        route_counts.describe()
    )

    print("\nRoutes with lowest record counts:")

    print(
        route_counts
        .sort_values()
        .head(10)
    )

    # --------------------------------------------------
    # 4. Timestamp gaps
    # --------------------------------------------------

    print("\n4. Timestamp Gap Analysis")

    df = df.sort_values(
        ["bus_route", "transit_timestamp"]
    )

    df["time_gap"] = (
        df.groupby("bus_route")[
            "transit_timestamp"
        ]
        .diff()
    )

    print("\nMost common time gaps:")

    print(
        df["time_gap"]
        .value_counts()
        .head(10)
    )

    # --------------------------------------------------
    # 5. Gaps greater than one hour
    # --------------------------------------------------

    print("\n5. Gaps Greater Than 1 Hour")

    large_gaps = df[
        df["time_gap"] > pd.Timedelta(hours=1)
    ]

    print(
        f"Total large gaps: {len(large_gaps):,}"
    )

    # --------------------------------------------------
    # 6. Gaps by route
    # --------------------------------------------------

    print("\n6. Routes With Most Large Gaps")

    gaps_by_route = (
        large_gaps
        .groupby("bus_route")
        .size()
        .sort_values(
            ascending=False
        )
    )

    print(
        gaps_by_route.head(20)
    )

    # --------------------------------------------------
    # 7. Exact hourly continuity
    # --------------------------------------------------

    print("\n7. One-Hour Continuity")

    hourly_gaps = (
        df["time_gap"]
        == pd.Timedelta(hours=1)
    ).mean()

    print(
        f"Hourly continuity: "
        f"{hourly_gaps * 100:.2f}%"
    )

    # --------------------------------------------------
    # 8. Feature missingness
    # --------------------------------------------------

    print("\n8. Feature Missing Values")

    print(
        df.isnull().sum()
        .sort_values(
            ascending=False
        )
        .head(10)
    )

    print("\n" + "=" * 70)
    print("TIME SERIES VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()