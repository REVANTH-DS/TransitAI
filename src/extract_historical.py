import requests
import pandas as pd
import time
from pathlib import Path


API_URL = "https://data.ny.gov/resource/gxb3-akrn.json"

OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "mta_hourly_demand.csv"


def extract_month(start_date, end_date):
    """
    Extract hourly ridership aggregated by
    timestamp and bus route for a date range.
    """

    params = {
        "$select": (
            "transit_timestamp,"
            "bus_route,"
            "sum(ridership) as total_ridership"
        ),
        "$where": (
            f"transit_timestamp >= '{start_date}' "
            f"AND transit_timestamp < '{end_date}'"
        ),
        "$group": "transit_timestamp, bus_route",
        "$order": "transit_timestamp, bus_route",
        "$limit": 50000
    }

    print(f"Requesting: {start_date} → {end_date}")

    response = requests.get(
        API_URL,
        params=params,
        timeout=180
    )

    response.raise_for_status()

    data = response.json()

    return pd.DataFrame(data)


def clean_data(df):
    """Clean extracted data."""

    if df.empty:
        return df

    df["transit_timestamp"] = pd.to_datetime(
        df["transit_timestamp"]
    )

    df["total_ridership"] = pd.to_numeric(
        df["total_ridership"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "transit_timestamp",
            "bus_route",
            "total_ridership"
        ]
    )

    return df


def main():

    print("=" * 70)
    print("TRANSITAI - HISTORICAL DATA EXTRACTION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_data = []

    # 2025-01 through 2026-08
   months = pd.date_range(
    start="2025-01-01",
    end="2025-01-01",
    freq="MS"
)
    

    for start in months:

        end = start + pd.offsets.MonthBegin(1)

        start_date = start.strftime(
            "%Y-%m-%dT%H:%M:%S"
        )

        end_date = end.strftime(
            "%Y-%m-%dT%H:%M:%S"
        )

        try:

            df = extract_month(
                start_date,
                end_date
            )

            df = clean_data(df)

            print(
                f"Retrieved {len(df):,} rows"
            )

            all_data.append(df)

            time.sleep(1)

        except Exception as e:

            print(
                f"ERROR for {start_date}: {e}"
            )

    if not all_data:

        print("No data extracted.")
        return

    final_df = pd.concat(
        all_data,
        ignore_index=True
    )

    # Remove accidental duplicates
    final_df = final_df.drop_duplicates(
        subset=[
            "transit_timestamp",
            "bus_route"
        ]
    )

    final_df = final_df.sort_values(
        [
            "transit_timestamp",
            "bus_route"
        ]
    )

    final_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"\nTotal rows: {len(final_df):,}"
    )

    print(
        f"Routes: {final_df['bus_route'].nunique():,}"
    )

    print(
        f"Start: {final_df['transit_timestamp'].min()}"
    )

    print(
        f"End: {final_df['transit_timestamp'].max()}"
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()