import requests
import pandas as pd

API_URL = "https://data.ny.gov/resource/gxb3-akrn.json"


def api_request(params):
    """Send a request to the MTA API."""
    
    response = requests.get(
        API_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    return response.json()


def get_sample_data(limit=100):
    """Get a small sample of the dataset."""

    params = {
        "$limit": limit
    }

    return pd.DataFrame(api_request(params))


def get_earliest_record():
    """Get the earliest available timestamp."""

    params = {
        "$select": "transit_timestamp",
        "$order": "transit_timestamp ASC",
        "$limit": 1
    }

    return api_request(params)


def get_latest_record():
    """Get the latest available timestamp."""

    params = {
        "$select": "transit_timestamp",
        "$order": "transit_timestamp DESC",
        "$limit": 1
    }

    return api_request(params)


if __name__ == "__main__":

    print("=" * 60)
    print("TRANSITAI - MTA DATASET PROFILE")
    print("=" * 60)

    # Sample
    print("\n1. Getting sample data...")

    df = get_sample_data()

    print("\nSample Data:")
    print(df.head())

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nSample Shape:")
    print(df.shape)

    # Earliest record
    print("\n2. Finding earliest timestamp...")

    earliest = get_earliest_record()

    print("Earliest record:")
    print(earliest)

    # Latest record
    print("\n3. Finding latest timestamp...")

    latest = get_latest_record()

    print("Latest record:")
    print(latest)

    print("\n" + "=" * 60)
    print("DATASET PROFILE COMPLETE")
    print("=" * 60)