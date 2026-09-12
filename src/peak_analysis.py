import pandas as pd


def analyze_forecast(forecast):

    # ---------------------------------------------
    # Peak and lowest demand
    # ---------------------------------------------

    peak = forecast.loc[
        forecast["predicted_ridership"].idxmax()
    ]

    lowest = forecast.loc[
        forecast["predicted_ridership"].idxmin()
    ]

    average = forecast[
        "predicted_ridership"
    ].mean()

    # ---------------------------------------------
    # High-demand hours
    # ---------------------------------------------

    high_demand = forecast[
        forecast["predicted_ridership"] >= 300
    ].copy()

    # ---------------------------------------------
    # Morning peak
    # ---------------------------------------------

    morning = forecast[
        forecast["timestamp"].dt.hour.between(6, 10)
    ]

    morning_peak = morning.loc[
        morning["predicted_ridership"].idxmax()
    ]

    # ---------------------------------------------
    # Evening peak
    # ---------------------------------------------

    evening = forecast[
        forecast["timestamp"].dt.hour.between(15, 19)
    ]

    evening_peak = evening.loc[
        evening["predicted_ridership"].idxmax()
    ]

    # ---------------------------------------------
    # Demand ratio
    # ---------------------------------------------

    peak_ratio = (
        peak["predicted_ridership"]
        / average
    )

    return {
        "peak_demand": int(
            peak["predicted_ridership"]
        ),

        "peak_time": peak["timestamp"],

        "lowest_demand": int(
            lowest["predicted_ridership"]
        ),

        "lowest_time": lowest["timestamp"],

        "average_demand": round(
            average
        ),

        "morning_peak_demand": int(
            morning_peak["predicted_ridership"]
        ),

        "morning_peak_time": morning_peak[
            "timestamp"
        ],

        "evening_peak_demand": int(
            evening_peak["predicted_ridership"]
        ),

        "evening_peak_time": evening_peak[
            "timestamp"
        ],

        "high_demand_hours": len(
            high_demand
        ),

        "peak_ratio": round(
            peak_ratio,
            2
        )
    }


def print_analysis(analysis):

    print("\n" + "=" * 70)
    print("TRANSITAI - PEAK DEMAND INTELLIGENCE")
    print("=" * 70)

    print(
        f"\n🔴 Overall Peak:"
        f" {analysis['peak_demand']} riders"
    )

    print(
        f"   Time:"
        f" {analysis['peak_time']}"
    )

    print(
        f"\n🟢 Lowest Demand:"
        f" {analysis['lowest_demand']} riders"
    )

    print(
        f"   Time:"
        f" {analysis['lowest_time']}"
    )

    print(
        f"\n📊 Average Demand:"
        f" {analysis['average_demand']} riders"
    )

    print(
        f"\n🌅 Morning Peak:"
        f" {analysis['morning_peak_demand']} riders"
    )

    print(
        f"   Time:"
        f" {analysis['morning_peak_time']}"
    )

    print(
        f"\n🌇 Evening Peak:"
        f" {analysis['evening_peak_demand']} riders"
    )

    print(
        f"   Time:"
        f" {analysis['evening_peak_time']}"
    )

    print(
        f"\n🚨 High-demand hours:"
        f" {analysis['high_demand_hours']}"
    )

    print(
        f"\n📈 Peak/Average ratio:"
        f" {analysis['peak_ratio']}x"
    )

    # ---------------------------------------------
    # Operational recommendation
    # ---------------------------------------------

    if analysis["peak_ratio"] >= 2:

        print(
            "\n⚠️ OPERATIONAL ALERT:"
        )

        print(
            "Demand is significantly above the "
            "daily average during the peak."
        )

        print(
            "Consider increased fleet capacity "
            "during peak periods."
        )

    elif analysis["peak_ratio"] >= 1.5:

        print(
            "\n⚠️ OPERATIONAL NOTICE:"
        )

        print(
            "Moderate peak concentration detected."
        )

    else:

        print(
            "\n✓ Demand is relatively stable."
        )

    print("\n" + "=" * 70)