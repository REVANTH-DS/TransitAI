import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = Path("models/transitai_xgboost.json")
FEATURE_FILE = Path("models/feature_columns.csv")
DATA_FILE = Path("data/processed/model_data.csv")


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model = xgb.XGBRegressor()

    model.load_model(MODEL_FILE)

    return model


# ============================================================
# LOAD FEATURE LIST
# ============================================================

def load_features():

    features = pd.read_csv(
        FEATURE_FILE,
        header=None
    )[0].tolist()

    return features


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

def load_data():

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["transit_timestamp"]
    )

    df = df.sort_values(
        ["bus_route", "transit_timestamp"]
    ).reset_index(drop=True)

    return df


# ============================================================
# CREATE BASE FEATURES
# ============================================================

def create_base_features(timestamp, route_df):

    hour = timestamp.hour
    day_of_week = timestamp.dayofweek
    day_of_month = timestamp.day
    month = timestamp.month
    week_of_year = timestamp.isocalendar().week

    is_weekend = int(day_of_week >= 5)

    hour_sin = np.sin(
        2 * np.pi * hour / 24
    )

    hour_cos = np.cos(
        2 * np.pi * hour / 24
    )

    dow_sin = np.sin(
        2 * np.pi * day_of_week / 7
    )

    dow_cos = np.cos(
        2 * np.pi * day_of_week / 7
    )

    # --------------------------------------------------------
    # Historical demand
    # --------------------------------------------------------

    historical = route_df[
        route_df["transit_timestamp"] < timestamp
    ].sort_values(
        "transit_timestamp"
    )

    demand = historical["total_ridership"]

    if len(demand) == 0:
        last_values = [0] * 48

    else:
        last_values = demand.tail(168).tolist()

    def get_lag(n):

        if len(last_values) >= n:
            return last_values[-n]

        return 0

    lag_1 = get_lag(1)
    lag_2 = get_lag(2)
    lag_3 = get_lag(3)

    lag_24 = get_lag(24)
    lag_48 = get_lag(48)

    # --------------------------------------------------------
    # Rolling features
    # --------------------------------------------------------

    recent_3 = demand.tail(3)
    recent_6 = demand.tail(6)
    recent_24 = demand.tail(24)

    rolling_mean_3 = (
        recent_3.mean()
        if len(recent_3) > 0
        else 0
    )

    rolling_mean_6 = (
        recent_6.mean()
        if len(recent_6) > 0
        else 0
    )

    rolling_mean_24 = (
        recent_24.mean()
        if len(recent_24) > 0
        else 0
    )

    return {

        "hour": hour,

        "day_of_week": day_of_week,

        "day_of_month": day_of_month,

        "month": month,

        "week_of_year": int(week_of_year),

        "is_weekend": is_weekend,

        "hour_sin": hour_sin,

        "hour_cos": hour_cos,

        "dow_sin": dow_sin,

        "dow_cos": dow_cos,

        "lag_1": lag_1,

        "lag_2": lag_2,

        "lag_3": lag_3,

        "lag_24": lag_24,

        "lag_48": lag_48,

        "rolling_mean_3": rolling_mean_3,

        "rolling_mean_6": rolling_mean_6,

        "rolling_mean_24": rolling_mean_24
    }


# ============================================================
# DEMAND LEVEL
# ============================================================

def demand_level(value):

    if value >= 750:
        return "VERY HIGH"

    elif value >= 300:
        return "HIGH"

    elif value >= 100:
        return "MODERATE"

    else:
        return "LOW"


# ============================================================
# FORECAST
# ============================================================

def generate_forecast(
    model,
    features,
    df,
    route,
    hours=24
):

    route_df = df[
        df["bus_route"] == route
    ].copy()

    if route_df.empty:

        raise ValueError(
            f"Route '{route}' not found in dataset."
        )

    route_df = route_df.sort_values(
        "transit_timestamp"
    )

    last_timestamp = route_df[
        "transit_timestamp"
    ].max()

    predictions = []

    print(
        "\nGenerating 24-hour forecast..."
    )

    for i in range(1, hours + 1):

        forecast_time = (
            last_timestamp
            + pd.Timedelta(hours=i)
        )

        feature_dict = create_base_features(
            forecast_time,
            route_df
        )

        feature_df = pd.DataFrame(
            [feature_dict]
        )

        # ----------------------------------------------------
        # Add route one-hot encoding
        # ----------------------------------------------------

        for feature in features:

            if feature.startswith(
                "bus_route_"
            ):

                route_name = feature.replace(
                    "bus_route_",
                    ""
                )

                feature_df[feature] = int(
                    route_name == route
                )

        # ----------------------------------------------------
        # Ensure every model feature exists
        # ----------------------------------------------------

        for feature in features:

            if feature not in feature_df.columns:

                feature_df[feature] = 0

        feature_df = feature_df[
            features
        ]

        prediction = model.predict(
            feature_df
        )[0]

        prediction = max(
            0,
            prediction
        )

        prediction = int(
            round(prediction)
        )

        predictions.append({

            "timestamp": forecast_time,

            "predicted_ridership": prediction,

            "demand_level": demand_level(
                prediction
            )
        })

        # ----------------------------------------------------
        # Add prediction to history
        # ----------------------------------------------------

        new_row = pd.DataFrame({

            "transit_timestamp": [
                forecast_time
            ],

            "bus_route": [
                route
            ],

            "total_ridership": [
                prediction
            ]
        })

        route_df = pd.concat(
            [
                route_df,
                new_row
            ],
            ignore_index=True
        )

    return pd.DataFrame(
        predictions
    )


# ============================================================
# PEAK DEMAND INTELLIGENCE
# ============================================================

def analyze_forecast(forecast):

    peak_row = forecast.loc[
        forecast[
            "predicted_ridership"
        ].idxmax()
    ]

    lowest_row = forecast.loc[
        forecast[
            "predicted_ridership"
        ].idxmin()
    ]

    average_demand = forecast[
        "predicted_ridership"
    ].mean()

    peak_demand = peak_row[
        "predicted_ridership"
    ]

    lowest_demand = lowest_row[
        "predicted_ridership"
    ]

    peak_ratio = (
        peak_demand / average_demand
        if average_demand > 0
        else 0
    )

    high_demand = forecast[
        forecast[
            "predicted_ridership"
        ] >= 300
    ]

    very_high_demand = forecast[
        forecast[
            "predicted_ridership"
        ] >= 750
    ]

    morning = forecast[
        forecast["timestamp"].dt.hour.between(
            6,
            10
        )
    ]

    evening = forecast[
        forecast["timestamp"].dt.hour.between(
            16,
            20
        )
    ]

    morning_peak = (
        morning["predicted_ridership"].max()
        if not morning.empty
        else 0
    )

    evening_peak = (
        evening["predicted_ridership"].max()
        if not evening.empty
        else 0
    )

    return {

        "peak_demand": peak_demand,

        "peak_time": peak_row["timestamp"],

        "lowest_demand": lowest_demand,

        "lowest_time": lowest_row["timestamp"],

        "average_demand": average_demand,

        "peak_ratio": peak_ratio,

        "high_demand_hours": len(
            high_demand
        ),

        "very_high_demand_hours": len(
            very_high_demand
        ),

        "morning_peak": morning_peak,

        "evening_peak": evening_peak
    }


# ============================================================
# PRINT INTELLIGENCE
# ============================================================

def print_analysis(
    analysis
):

    print("\n" + "=" * 70)

    print(
        "TRANSITAI - PEAK DEMAND INTELLIGENCE"
    )

    print("=" * 70)

    print(
        f"\nOverall peak demand: "
        f"{analysis['peak_demand']:.0f} riders"
    )

    print(
        f"Peak time: "
        f"{analysis['peak_time']}"
    )

    print(
        f"\nLowest demand: "
        f"{analysis['lowest_demand']:.0f} riders"
    )

    print(
        f"Lowest time: "
        f"{analysis['lowest_time']}"
    )

    print(
        f"\nAverage predicted demand: "
        f"{analysis['average_demand']:.0f} riders"
    )

    print(
        f"\nPeak / Average ratio: "
        f"{analysis['peak_ratio']:.2f}x"
    )

    print(
        f"\nMorning peak demand: "
        f"{analysis['morning_peak']:.0f} riders"
    )

    print(
        f"Evening peak demand: "
        f"{analysis['evening_peak']:.0f} riders"
    )

    print(
        f"\nHigh-demand hours: "
        f"{analysis['high_demand_hours']}"
    )

    print(
        f"Very-high-demand hours: "
        f"{analysis['very_high_demand_hours']}"
    )

    # --------------------------------------------------------
    # Operational alert
    # --------------------------------------------------------

    if analysis["peak_ratio"] >= 2:

        print(
            "\n🚨 OPERATIONAL ALERT:"
        )

        print(
            "Demand is significantly above "
            "the daily average during peak periods."
        )

        print(
            "Consider deploying additional "
            "capacity during peak hours."
        )

    elif analysis["peak_ratio"] >= 1.5:

        print(
            "\n⚠️ OPERATIONAL NOTICE:"
        )

        print(
            "Moderate-to-high demand concentration "
            "is expected."
        )

    else:

        print(
            "\n✅ DEMAND STATUS:"
        )

        print(
            "Demand is relatively balanced "
            "throughout the forecast period."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "TRANSITAI - PUBLIC TRANSPORT DEMAND FORECAST"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print(
        "\nLoading model..."
    )

    model = load_model()

    print(
        "Model loaded."
    )

    print(
        "\nLoading feature list..."
    )

    features = load_features()

    print(
        f"Features loaded: {len(features)}"
    )

    print(
        "\nLoading historical data..."
    )

    df = load_data()

    print(
        f"Historical rows: {len(df):,}"
    )

    # --------------------------------------------------------
    # Route selection
    # --------------------------------------------------------

    print("\nAvailable route example: B1")

    route = input(
        "\nEnter bus route: "
    ).strip().upper()

    # --------------------------------------------------------
    # Forecast
    # --------------------------------------------------------

    forecast = generate_forecast(

        model=model,

        features=features,

        df=df,

        route=route,

        hours=24
    )

    # --------------------------------------------------------
    # Display forecast
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        f"24-HOUR DEMAND FORECAST — {route}"
    )

    print("=" * 70)

    display_forecast = forecast.copy()

    display_forecast[
        "timestamp"
    ] = display_forecast[
        "timestamp"
    ].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    print(
        display_forecast.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    peak_idx = forecast[
        "predicted_ridership"
    ].idxmax()

    low_idx = forecast[
        "predicted_ridership"
    ].idxmin()

    peak_demand = forecast.loc[
        peak_idx,
        "predicted_ridership"
    ]

    peak_time = forecast.loc[
        peak_idx,
        "timestamp"
    ]

    lowest_demand = forecast.loc[
        low_idx,
        "predicted_ridership"
    ]

    lowest_time = forecast.loc[
        low_idx,
        "timestamp"
    ]

    average_demand = forecast[
        "predicted_ridership"
    ].mean()

    print("\n" + "=" * 70)

    print(
        "FORECAST SUMMARY"
    )

    print("=" * 70)

    print(
        f"\nPeak demand: "
        f"{peak_demand} riders"
    )

    print(
        f"Peak time: "
        f"{peak_time}"
    )

    print(
        f"\nLowest demand: "
        f"{lowest_demand} riders"
    )

    print(
        f"Lowest time: "
        f"{lowest_time}"
    )

    print(
        f"\nAverage predicted demand: "
        f"{average_demand:.0f} riders"
    )

    # --------------------------------------------------------
    # Peak intelligence
    # --------------------------------------------------------

    analysis = analyze_forecast(
        forecast
    )

    print_analysis(
        analysis
    )

    print("\n" + "=" * 70)

    print(
        "24-HOUR FORECAST COMPLETE"
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()