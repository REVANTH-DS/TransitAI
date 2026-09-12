import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor


# ============================================================
# TRANSITAI - V2 PRODUCTION FORECAST ENGINE
# ============================================================

MODEL_FILE = Path("models/transitai_xgboost_v2.json")
FEATURE_FILE = Path("models/feature_columns_v2.csv")
DATA_FILE = Path("data/processed/model_data_v2.csv")

TIMESTAMP_COLUMN = "transit_timestamp"
ROUTE_COLUMN = "bus_route"
TARGET_COLUMN = "total_ridership"


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\nLoading final V2 model...")

    model = XGBRegressor()

    model.load_model(MODEL_FILE)

    print("V2 model loaded")

    return model


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features():

    features = pd.read_csv(
        FEATURE_FILE,
        header=None
    )[0].tolist()

    print(f"Loaded {len(features)} model features")

    return features


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\nLoading historical data...")

    df = pd.read_csv(DATA_FILE)

    required = [
        TIMESTAMP_COLUMN,
        ROUTE_COLUMN,
        TARGET_COLUMN
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df[TIMESTAMP_COLUMN] = pd.to_datetime(
        df[TIMESTAMP_COLUMN]
    )

    df = df.sort_values(
        [ROUTE_COLUMN, TIMESTAMP_COLUMN]
    ).reset_index(drop=True)

    print(f"Loaded {len(df):,} rows")

    return df


# ============================================================
# ROUTE ENCODING
# ============================================================

def encode_routes(df, feature_columns):

    route_columns = [
        col
        for col in feature_columns
        if col.startswith("bus_route_")
    ]

    route_values = df[ROUTE_COLUMN].astype(str)

    route_data = {}

    for column in route_columns:

        route_name = column.replace(
            "bus_route_",
            "",
            1
        )

        route_data[column] = (
            route_values == route_name
        ).astype(int)

    route_df = pd.DataFrame(
        route_data,
        index=df.index
    )

    return route_df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df, feature_columns):

    feature_df = pd.DataFrame(
        index=df.index
    )

    # --------------------------------------------------------
    # Existing numerical features
    # --------------------------------------------------------

    for feature in feature_columns:

        if feature.startswith("bus_route_"):
            continue

        if feature in df.columns:

            feature_df[feature] = pd.to_numeric(
                df[feature],
                errors="coerce"
            )

        else:

            feature_df[feature] = 0

    # --------------------------------------------------------
    # Route encoding
    # --------------------------------------------------------

    route_df = encode_routes(
        df,
        feature_columns
    )

    feature_df = pd.concat(
        [
            feature_df,
            route_df
        ],
        axis=1
    )

    # --------------------------------------------------------
    # Exact feature order
    # --------------------------------------------------------

    feature_df = feature_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # --------------------------------------------------------
    # Clean values
    # --------------------------------------------------------

    feature_df = feature_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    feature_df = feature_df.fillna(0)

    return feature_df


# ============================================================
# DEMAND LEVEL
# ============================================================

def demand_level(value):

    if value < 100:
        return "LOW"

    elif value < 300:
        return "MODERATE"

    elif value < 600:
        return "HIGH"

    else:
        return "VERY HIGH"


# ============================================================
# FORECAST
# ============================================================

def generate_forecast(
    model,
    df,
    route,
    feature_columns,
    hours=24
):

    route_df = df[
        df[ROUTE_COLUMN].astype(str).str.upper()
        == route.upper()
    ].copy()

    if route_df.empty:

        raise ValueError(
            f"Route '{route}' not found."
        )

    route_df = route_df.sort_values(
        TIMESTAMP_COLUMN
    )

    # --------------------------------------------------------
    # Last known observation
    # --------------------------------------------------------

    last_time = route_df[
        TIMESTAMP_COLUMN
    ].max()

    print(
        f"\nRoute selected: {route.upper()}"
    )

    print(
        f"Last historical timestamp: "
        f"{last_time}"
    )

    # --------------------------------------------------------
    # Generate future timestamps
    # --------------------------------------------------------

    future_times = pd.date_range(
        start=last_time + pd.Timedelta(hours=1),
        periods=hours,
        freq="h"
    )

    # --------------------------------------------------------
    # Use latest historical rows as feature templates
    # --------------------------------------------------------

    history = route_df.tail(48).copy()

    predictions = []

    # --------------------------------------------------------
    # Recursive forecasting
    # --------------------------------------------------------

    for timestamp in future_times:

        hour = timestamp.hour

        day_of_week = timestamp.dayofweek

        day_of_month = timestamp.day

        month = timestamp.month

        week_of_year = timestamp.isocalendar().week

        is_weekend = int(
            day_of_week >= 5
        )

        # ----------------------------------------------
        # Lag values
        # ----------------------------------------------

        values = history[
            TARGET_COLUMN
        ].tolist()

        lag_1 = values[-1] if len(values) >= 1 else 0

        lag_2 = values[-2] if len(values) >= 2 else 0

        lag_3 = values[-3] if len(values) >= 3 else 0

        lag_24 = (
            values[-24]
            if len(values) >= 24
            else lag_1
        )

        lag_48 = (
            values[-48]
            if len(values) >= 48
            else lag_24
        )

        # ----------------------------------------------
        # Rolling features
        # ----------------------------------------------

        rolling_mean_3 = np.mean(
            values[-3:]
        )

        rolling_mean_6 = np.mean(
            values[-6:]
        )

        rolling_mean_24 = np.mean(
            values[-24:]
        )

        rolling_median_6 = np.median(
            values[-6:]
        )

        rolling_std_6 = (
            np.std(values[-6:])
            if len(values) >= 2
            else 0
        )

        rolling_std_24 = (
            np.std(values[-24:])
            if len(values) >= 2
            else 0
        )

        # ----------------------------------------------
        # Calendar features
        # ----------------------------------------------

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

        is_morning_peak = int(
            6 <= hour <= 9
        )

        is_evening_peak = int(
            15 <= hour <= 18
        )

        is_night = int(
            hour >= 22 or hour <= 5
        )

        is_daytime = int(
            6 <= hour <= 21
        )

        is_month_start = int(
            day_of_month <= 3
        )

        # approximate month end
        next_month = timestamp + pd.offsets.MonthBegin(1)

        month_end = (
            next_month - pd.Timedelta(days=1)
        ).day

        is_month_end = int(
            day_of_month >= month_end - 2
        )

        is_week_start = int(
            day_of_week == 0
        )

        is_week_end = int(
            day_of_week == 6
        )

        # ----------------------------------------------
        # Demand period
        # ----------------------------------------------

        demand_period_morning_peak = int(
            6 <= hour <= 9
        )

        demand_period_midday = int(
            10 <= hour <= 14
        )

        demand_period_evening_peak = int(
            15 <= hour <= 18
        )

        demand_period_evening = int(
            19 <= hour <= 21
        )

        demand_period_night = int(
            hour >= 22 or hour <= 5
        )

        # ----------------------------------------------
        # Momentum
        # ----------------------------------------------

        demand_change_1h = (
            lag_1 - lag_2
        )

        demand_change_3h = (
            lag_1 - lag_3
        )

        if lag_24 > 0:

            demand_ratio_24h = (
                lag_1 / lag_24
            )

        else:

            demand_ratio_24h = 0

        # ----------------------------------------------
        # Build row
        # ----------------------------------------------

        row = {

            "hour": hour,

            "day_of_week": day_of_week,

            "day_of_month": day_of_month,

            "month": month,

            "week_of_year": week_of_year,

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

            "rolling_mean_24": rolling_mean_24,

            "is_morning_peak":
                is_morning_peak,

            "is_evening_peak":
                is_evening_peak,

            "is_night":
                is_night,

            "is_daytime":
                is_daytime,

            "is_month_start":
                is_month_start,

            "is_month_end":
                is_month_end,

            "is_week_start":
                is_week_start,

            "is_week_end":
                is_week_end,

            "demand_period_morning_peak":
                demand_period_morning_peak,

            "demand_period_midday":
                demand_period_midday,

            "demand_period_evening_peak":
                demand_period_evening_peak,

            "demand_period_evening":
                demand_period_evening,

            "demand_period_night":
                demand_period_night,

            "rolling_median_6":
                rolling_median_6,

            "rolling_std_6":
                rolling_std_6,

            "rolling_std_24":
                rolling_std_24,

            "demand_change_1h":
                demand_change_1h,

            "demand_change_3h":
                demand_change_3h,

            "demand_ratio_24h":
                demand_ratio_24h
        }

        feature_row = pd.DataFrame(
            [row]
        )

        # ----------------------------------------------
        # Route features
        # ----------------------------------------------

        for feature in feature_columns:

            if feature.startswith(
                "bus_route_"
            ):

                route_name = feature.replace(
                    "bus_route_",
                    "",
                    1
                )

                feature_row[feature] = int(
                    route_name == route
                )

        # ----------------------------------------------
        # Ensure exact model features
        # ----------------------------------------------

        feature_row = feature_row.reindex(
            columns=feature_columns,
            fill_value=0
        )

        feature_row = feature_row.replace(
            [np.inf, -np.inf],
            np.nan
        ).fillna(0)

        # ----------------------------------------------
        # Predict
        # ----------------------------------------------

        prediction = model.predict(
            feature_row
        )[0]

        prediction = max(
            0,
            round(float(prediction))
        )

        predictions.append(
            prediction
        )

        # ----------------------------------------------
        # Add prediction to history
        # ----------------------------------------------

        new_row = {
            TIMESTAMP_COLUMN:
                timestamp,

            ROUTE_COLUMN:
                route,

            TARGET_COLUMN:
                prediction
        }

        history = pd.concat(
            [
                history,
                pd.DataFrame([new_row])
            ],
            ignore_index=True
        )

        if len(history) > 48:

            history = history.tail(48)

    # --------------------------------------------------------
    # Result dataframe
    # --------------------------------------------------------

    result = pd.DataFrame({

        "timestamp":
            future_times,

        "predicted_ridership":
            predictions
    })

    result["demand_level"] = (
        result["predicted_ridership"]
        .apply(demand_level)
    )

    return result


# ============================================================
# SUMMARY
# ============================================================

def print_summary(result):

    peak_row = result.loc[
        result["predicted_ridership"].idxmax()
    ]

    low_row = result.loc[
        result["predicted_ridership"].idxmin()
    ]

    average_demand = (
        result["predicted_ridership"]
        .mean()
    )

    morning = result[
        result["timestamp"].dt.hour.between(
            6,
            10
        )
    ]

    evening = result[
        result["timestamp"].dt.hour.between(
            15,
            18
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

    high_hours = (
        result["predicted_ridership"] >= 300
    ).sum()

    very_high_hours = (
        result["predicted_ridership"] >= 600
    ).sum()

    print("\n" + "=" * 70)

    print("FORECAST SUMMARY")

    print("=" * 70)

    print(
        f"\nPeak demand: "
        f"{int(peak_row['predicted_ridership'])} riders"
    )

    print(
        f"Peak time: "
        f"{peak_row['timestamp']}"
    )

    print(
        f"\nLowest demand: "
        f"{int(low_row['predicted_ridership'])} riders"
    )

    print(
        f"Lowest time: "
        f"{low_row['timestamp']}"
    )

    print(
        f"\nAverage predicted demand: "
        f"{average_demand:.0f} riders"
    )

    print(
        f"\nMorning peak demand: "
        f"{int(morning_peak)} riders"
    )

    print(
        f"Evening peak demand: "
        f"{int(evening_peak)} riders"
    )

    print(
        f"\nHigh-demand hours: "
        f"{high_hours}"
    )

    print(
        f"Very-high-demand hours: "
        f"{very_high_hours}"
    )

    if peak_row["predicted_ridership"] > (
        average_demand * 1.5
    ):

        print(
            "\nOPERATIONAL ALERT:"
        )

        print(
            "Demand is significantly above "
            "the daily average during peak periods."
        )

        print(
            "Consider deploying additional "
            "capacity during peak hours."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "TRANSITAI - V2 PRODUCTION FORECAST ENGINE"
    )

    print("=" * 70)

    route = input(
        "\nEnter bus route (example: B1): "
    ).strip().upper()

    try:

        model = load_model()

        feature_columns = load_features()

        df = load_data()

        available_routes = (
            df[ROUTE_COLUMN]
            .astype(str)
            .str.upper()
            .unique()
        )

        if route not in available_routes:

            raise ValueError(
                f"Route '{route}' not found."
            )

        print(
            f"\nGenerating 24-hour forecast..."
        )

        result = generate_forecast(
            model,
            df,
            route,
            feature_columns,
            hours=24
        )

        print("\n" + "=" * 70)

        print(
            f"24-HOUR DEMAND FORECAST — {route}"
        )

        print("=" * 70)

        print(
            result.to_string(
                index=False
            )
        )

        print_summary(result)

        output_file = Path(
            f"data/processed/forecast_{route}_v2.csv"
        )

        result.to_csv(
            output_file,
            index=False
        )

        print(
            f"\nForecast saved to: {output_file}"
        )

        print("\n" + "=" * 70)

        print(
            "V2 FORECAST COMPLETE"
        )

        print("=" * 70)

    except Exception as e:

        print(
            f"\nERROR: {e}"
        )


if __name__ == "__main__":
    main()