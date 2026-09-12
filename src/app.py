import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TransitAI | Public Transport Intelligence",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =========================
       GLOBAL
       ========================= */

    .stApp {
        background:
            radial-gradient(
                circle at 0% 0%,
                rgba(14, 165, 233, 0.12),
                transparent 30%
            ),
            radial-gradient(
                circle at 100% 0%,
                rgba(37, 99, 235, 0.10),
                transparent 30%
            ),
            #07111f;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }


    /* =========================
       HEADINGS
       ========================= */

    h1, h2, h3 {
        color: #f8fafc !important;
    }

    p {
        color: #94a3b8;
    }


    /* =========================
       SELECTBOX
       ========================= */

    div[data-baseweb="select"] > div {
        background-color: #111c2e;
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 12px;
    }


    /* =========================
       BUTTON
       ========================= */

    .stButton > button {
        width: 100%;
        min-height: 48px;

        border: none;
        border-radius: 12px;

        background: linear-gradient(
            135deg,
            #0ea5e9,
            #2563eb
        );

        color: white;

        font-size: 1rem;
        font-weight: 700;

        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);

        box-shadow:
            0 10px 30px rgba(14, 165, 233, 0.25);
    }


    /* =========================
       METRICS
       ========================= */

    div[data-testid="stMetric"] {
        background:
            linear-gradient(
                145deg,
                #172238,
                #0f192a
            );

        border: 1px solid
            rgba(148, 163, 184, 0.15);

        border-radius: 18px;

        padding: 20px;

        box-shadow:
            0 15px 35px
            rgba(0, 0, 0, 0.22);

        min-height: 125px;
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 800 !important;
    }

    div[data-testid="stMetricDelta"] {
        color: #38bdf8 !important;
    }


    /* =========================
       DATAFRAME
       ========================= */

    div[data-testid="stDataFrame"] {
        border-radius: 15px;
        overflow: hidden;
    }


    /* =========================
       DIVIDER
       ========================= */

    hr {
        border-color:
            rgba(148, 163, 184, 0.12) !important;
    }


    /* =========================
       FOOTER
       ========================= */

    .footer-text {
        text-align: center;
        color: #64748b;
        padding-top: 25px;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns(
    [5, 1]
)

with header_left:

    st.title("🚌 TransitAI")

    st.caption(
        "Smart Public Transport Demand Intelligence"
    )

with header_right:

    st.write("")

    st.success(
        "● ENGINE ONLINE"
    )


st.divider()


# ============================================================
# ROUTE DISCOVERY
# ============================================================

def find_routes():

    routes = []

    if DATA_DIR.exists():

        files = DATA_DIR.glob(
            "forecast_*_v2.csv"
        )

        for file in files:

            route = file.stem

            route = route.replace(
                "forecast_",
                ""
            )

            route = route.replace(
                "_v2",
                ""
            )

            if route:
                routes.append(route)

    if not routes:
        routes = ["B1"]

    return sorted(
        set(routes)
    )


routes = find_routes()


# ============================================================
# ROUTE SELECTION
# ============================================================

st.subheader(
    "Choose your route"
)

st.caption(
    "Select a bus route to discover expected passenger demand for the next 24 hours."
)


route_col, button_col = st.columns(
    [5, 1],
    gap="medium"
)


with route_col:

    selected_route = st.selectbox(
        "Bus Route",
        options=routes,
        index=0,
    )


with button_col:

    st.write("")

    forecast_button = st.button(
        "🔮 Forecast",
        use_container_width=True,
    )


# ============================================================
# LOAD FORECAST
# ============================================================

def load_forecast(route):

    file_path = (
        DATA_DIR /
        f"forecast_{route}_v2.csv"
    )

    if not file_path.exists():

        return None, file_path

    try:

        data = pd.read_csv(
            file_path
        )

        return data, file_path

    except Exception as error:

        st.error(
            f"Could not read forecast file: {error}"
        )

        return None, file_path


# ============================================================
# RUN FORECAST
# ============================================================

if forecast_button:

    with st.spinner(
        "Running XGBoost V2 forecast..."
    ):

        forecast_data, forecast_path = (
            load_forecast(
                selected_route
            )
        )

    if forecast_data is None:

        st.error(
            f"""
            Forecast file not found.

            Expected:

            `{forecast_path}`
            """
        )

        st.stop()

    st.session_state[
        "forecast_data"
    ] = forecast_data

    st.session_state[
        "forecast_route"
    ] = selected_route


# ============================================================
# LANDING STATE
# ============================================================

if "forecast_data" not in st.session_state:

    st.info(
        "👆 Select a route and click **Forecast** to generate the 24-hour prediction."
    )

    st.markdown(
        """
        ### 🚀 TransitAI

        **XGBoost V2**  
        ↓  
        **Passenger Demand Prediction**  
        ↓  
        **Peak Period Detection**  
        ↓  
        **Demand-Level Intelligence**  
        ↓  
        **Transport Planning**
        """
    )

    st.markdown(
        """
        <div class="footer-text">
        🚌 TransitAI • Intelligent Public Transport Demand Forecasting
        <br><br>
        XGBoost V2 • Python • Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# FORECAST DATA
# ============================================================

df = st.session_state[
    "forecast_data"
].copy()

active_route = st.session_state[
    "forecast_route"
]


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_columns = [
    "timestamp",
    "predicted_ridership",
    "demand_level",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )

    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce",
)

df["predicted_ridership"] = pd.to_numeric(
    df["predicted_ridership"],
    errors="coerce",
)

df = df.dropna(
    subset=[
        "timestamp",
        "predicted_ridership",
    ]
)

df = df.sort_values(
    "timestamp"
)

# Keep 24 hours
df = df.head(24).copy()


if df.empty:

    st.error(
        "No valid forecast data available."
    )

    st.stop()


# ============================================================
# CALCULATIONS
# ============================================================

peak_row = df.loc[
    df["predicted_ridership"].idxmax()
]

lowest_row = df.loc[
    df["predicted_ridership"].idxmin()
]


peak_demand = int(
    round(
        peak_row[
            "predicted_ridership"
        ]
    )
)

lowest_demand = int(
    round(
        lowest_row[
            "predicted_ridership"
        ]
    )
)

average_demand = int(
    round(
        df[
            "predicted_ridership"
        ].mean()
    )
)


peak_time = peak_row[
    "timestamp"
].strftime(
    "%I:%M %p"
)

lowest_time = lowest_row[
    "timestamp"
].strftime(
    "%I:%M %p"
)


# ============================================================
# DEMAND LEVELS
# ============================================================

demand_levels = (
    df["demand_level"]
    .astype(str)
    .str.upper()
)


low_hours = int(
    (demand_levels == "LOW").sum()
)

medium_hours = int(
    (demand_levels == "MEDIUM").sum()
)

high_hours = int(
    (demand_levels == "HIGH").sum()
)

very_high_hours = int(
    (demand_levels == "VERY HIGH").sum()
)


# ============================================================
# FORECAST HEADER
# ============================================================

st.divider()

st.header(
    f"📊 Route {active_route} — 24-Hour Forecast"
)

st.caption(
    "XGBoost V2 passenger demand prediction and operational intelligence"
)


# ============================================================
# KPI CARDS
# ============================================================

k1, k2, k3, k4 = st.columns(
    4,
    gap="medium"
)


with k1:

    st.metric(
        label="Peak Demand",
        value=f"{peak_demand:,}",
        delta="riders",
    )


with k2:

    st.metric(
        label="Peak Time",
        value=peak_time,
    )


with k3:

    st.metric(
        label="Average Demand",
        value=f"{average_demand:,}",
        delta="riders / hour",
    )


with k4:

    st.metric(
        label="Lowest Demand",
        value=f"{lowest_demand:,}",
        delta=lowest_time,
    )


# ============================================================
# MAIN FORECAST CHART
# ============================================================

st.subheader(
    "📈 Demand Forecast"
)

st.caption(
    "Expected passenger demand across the next 24 hours."
)


fig = go.Figure()


fig.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["predicted_ridership"],

        mode="lines+markers",

        name="Predicted Riders",

        line=dict(
            color="#38bdf8",
            width=4,
        ),

        marker=dict(
            color="#38bdf8",
            size=8,
        ),

        fill="tozeroy",

        fillcolor="rgba(56,189,248,0.08)",

        hovertemplate=
            "<b>%{x|%d %b %I:%M %p}</b>"
            "<br>Predicted riders: %{y:,}"
            "<extra></extra>",
    )
)


# Peak marker

fig.add_trace(
    go.Scatter(
        x=[
            peak_row["timestamp"]
        ],

        y=[
            peak_row[
                "predicted_ridership"
            ]
        ],

        mode="markers+text",

        name="Peak",

        marker=dict(
            color="#f43f5e",
            size=13,
            symbol="circle",
        ),

        text=[
            f"Peak: {peak_demand:,}"
        ],

        textposition="top center",

        hovertemplate=
            "<b>Peak Demand</b>"
            "<br>%{y:,} riders"
            "<extra></extra>",
    )
)


fig.update_layout(

    height=500,

    margin=dict(
        l=20,
        r=20,
        t=30,
        b=30,
    ),

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="#0b1626",

    font=dict(
        color="#cbd5e1",
    ),

    xaxis=dict(
        title="Time",

        gridcolor=
            "rgba(148,163,184,0.10)",

        zeroline=False,
    ),

    yaxis=dict(
        title="Predicted Riders",

        gridcolor=
            "rgba(148,163,184,0.10)",

        zeroline=False,
    ),

    legend=dict(
        bgcolor="rgba(0,0,0,0)",
    ),

    hoverlabel=dict(
        bgcolor="#111827",
        font_color="#f8fafc",
    ),
)


st.plotly_chart(
    fig,
    use_container_width=True,
)


# ============================================================
# OPERATIONAL INTELLIGENCE
# ============================================================

st.subheader(
    "🧠 Operational Intelligence"
)


intel1, intel2 = st.columns(
    2,
    gap="medium"
)


with intel1:

    if very_high_hours > 0:

        st.warning(
            f"""
            **⚠️ HIGH CAPACITY ALERT**

            **{very_high_hours} hour(s)** are predicted
            to experience **VERY HIGH** demand.

            Consider increasing bus frequency or
            allocating additional fleet capacity
            during these periods.
            """
        )

    elif high_hours > 0:

        st.info(
            f"""
            **📊 ELEVATED DEMAND**

            **{high_hours} hour(s)** are predicted
            to experience HIGH demand.
            """
        )

    else:

        st.success(
            "✓ Demand remains within manageable levels during the forecast window."
        )


with intel2:

    st.info(
        f"""
        **📈 PEAK PERIOD**

        Maximum predicted demand:

        **{peak_demand:,} riders**

        Expected at:

        **{peak_time}**

        Average demand:

        **{average_demand:,} riders/hour**
        """
    )


# ============================================================
# DEMAND LEVEL SUMMARY
# ============================================================

st.subheader(
    "🎯 Demand Level Summary"
)


d1, d2, d3, d4 = st.columns(
    4,
    gap="medium"
)


with d1:

    st.metric(
        "LOW",
        low_hours,
        "hours",
    )


with d2:

    st.metric(
        "MEDIUM",
        medium_hours,
        "hours",
    )


with d3:

    st.metric(
        "HIGH",
        high_hours,
        "hours",
    )


with d4:

    st.metric(
        "VERY HIGH",
        very_high_hours,
        "hours",
    )


# ============================================================
# DEMAND LEVEL CHART
# ============================================================

level_df = pd.DataFrame(
    {
        "Demand Level": [
            "LOW",
            "MEDIUM",
            "HIGH",
            "VERY HIGH",
        ],

        "Hours": [
            low_hours,
            medium_hours,
            high_hours,
            very_high_hours,
        ],
    }
)


fig2 = go.Figure()


fig2.add_trace(
    go.Bar(
        x=level_df[
            "Demand Level"
        ],

        y=level_df[
            "Hours"
        ],

        text=level_df[
            "Hours"
        ],

        textposition="outside",

        marker=dict(
            color="#38bdf8"
        ),

        hovertemplate=
            "<b>%{x}</b>"
            "<br>Hours: %{y}"
            "<extra></extra>",
    )
)


fig2.update_layout(

    height=380,

    margin=dict(
        l=20,
        r=20,
        t=30,
        b=30,
    ),

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="#0b1626",

    font=dict(
        color="#cbd5e1",
    ),

    xaxis=dict(
        title="Demand Level",
        gridcolor=
            "rgba(148,163,184,0.08)",
    ),

    yaxis=dict(
        title="Hours",
        dtick=1,
        gridcolor=
            "rgba(148,163,184,0.10)",
    ),
)


st.plotly_chart(
    fig2,
    use_container_width=True,
)


# ============================================================
# 24-HOUR FORECAST TABLE
# ============================================================

st.subheader(
    "🕐 24-Hour Forecast"
)

table_df = df[
    [
        "timestamp",
        "predicted_ridership",
        "demand_level",
    ]
].copy()


table_df["timestamp"] = table_df[
    "timestamp"
].dt.strftime(
    "%d %b %Y • %I:%M %p"
)


table_df[
    "predicted_ridership"
] = (
    table_df[
        "predicted_ridership"
    ]
    .round()
    .astype(int)
)


table_df.columns = [
    "Timestamp",
    "Predicted Riders",
    "Demand Level",
]


st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# DOWNLOAD
# ============================================================

st.subheader(
    "📥 Export Forecast"
)


csv_data = df.to_csv(
    index=False
).encode(
    "utf-8"
)


st.download_button(
    label="⬇️ Download 24-Hour Forecast CSV",

    data=csv_data,

    file_name=
        f"TransitAI_{active_route}_forecast.csv",

    mime="text/csv",
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🚌 TransitAI • Intelligent Public Transport Demand Forecasting"
)

st.caption(
    "XGBoost V2 • Python • Streamlit"
)