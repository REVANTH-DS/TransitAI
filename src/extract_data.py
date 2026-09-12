import requests
import pandas as pd
from pathlib import Path


API_URL = "https://data.ny.gov/resource/gxb3-akrn.json"

START_DATE = "2025-01-01T00:00:00"
END_DATE = "2025-01-07T23:00:00"


def extract_weekly_demand():
    """
    Extract hourly passenger demand aggregated by
    timestamp and bus route.
    """

    params = {
        "$select": (
            "transit_timestamp,"
            "bus_route,"
            "sum(ridership) as total_ridership"
        ),
        "$where": (
            f"transit_timestamp between "
            f"'{START_DATE}' and '{END_DATE}'"
        ),
        "$group": "transit_timestamp, bus_route",
        "$order": "transit_timestamp, bus_route",
        "$limit": 50000
    }

    print("Requesting aggregated MTA data...")

    response = requests.get(
        API_URL,
        params=params,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return pd.DataFrame(data)


def main():

    print("=" * 60)
    print("TRANSITAI - DATA EXTRACTION TEST")
    print("=" * 60)

    df = extract_weekly_demand()

    print("\nExtraction successful!")

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 10 rows:")
    print(df.head(10))

    print("\nData types:")
    print(df.dtypes)

    # Convert columns
    df["transit_timestamp"] = pd.to_datetime(
        df["transit_timestamp"]
    )

    df["total_ridership"] = pd.to_numeric(
        df["total_ridership"],
        errors="coerce"
    )

    # Create processed directory
    output_path = Path("data/processed")
    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save sample
    file_path = output_path / "mta_demand_sample.csv"

    df.to_csv(
        file_path,
        index=False
    )

    print("\nSaved file:")
    print(file_path)

    print("\nFinal preview:")
    print(df.head())

    print("\n" + "=" * 60)
    print("EXTRACTION TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()