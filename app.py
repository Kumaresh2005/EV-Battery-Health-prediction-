import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

LOGO_PATH = Path(__file__).resolve().parent / "logo.png"

st.set_page_config(
    page_title="EV Battery AI Dashboard",
    page_icon="🔋",
    layout="wide"
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --ink: #e8f1f5;
        --muted: #9bb0b9;
        --surface: #12232a;
        --surface-soft: #172d35;
        --line: #2a4852;
        --cyan: #27d3c2;
        --amber: #ffbf69;
    }

    .stApp {
        background:
            radial-gradient(circle at 85% 5%, rgba(39, 211, 194, 0.14), transparent 28rem),
            linear-gradient(145deg, #08151a 0%, #0d2027 55%, #111d2b 100%);
        color: var(--ink);
        font-family: 'Space Grotesk', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(8, 21, 26, 0.78);
    }

    [data-testid="stSidebar"] {
        background: #0b1b21;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] * {
        color: var(--ink);
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0;
    }

    h1 {
        font-weight: 700;
        text-shadow: 0 0 28px rgba(39, 211, 194, 0.2);
    }

    p, label, [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    [data-testid="stMetric"] {
        background: linear-gradient(135deg, var(--surface), var(--surface-soft));
        border: 1px solid var(--line);
        border-left: 4px solid var(--cyan);
        border-radius: 10px;
        padding: 1rem 1.1rem;
        box-shadow: 0 14px 34px rgba(0, 0, 0, 0.2);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    [data-testid="stMetricValue"] {
        color: var(--cyan);
    }

    .stButton > button {
        background: linear-gradient(100deg, var(--cyan), #58e6d7);
        color: #062126;
        border: 0;
        border-radius: 8px;
        font-weight: 700;
        min-height: 3rem;
        box-shadow: 0 10px 24px rgba(39, 211, 194, 0.2);
    }

    .stButton > button:hover {
        background: linear-gradient(100deg, #58e6d7, var(--amber));
        color: #062126;
        border: 0;
    }

    [data-baseweb="tab-list"] {
        gap: 0.35rem;
        border-bottom: 1px solid var(--line);
    }

    [data-baseweb="tab"] {
        color: var(--muted);
        font-weight: 600;
    }

    [aria-selected="true"] {
        color: var(--cyan) !important;
        border-bottom-color: var(--cyan) !important;
    }

    [data-testid="stAlert"] {
        background: rgba(23, 45, 53, 0.86);
        border: 1px solid var(--line);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(18, 35, 42, 0.76);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1rem 1.15rem;
        box-shadow: 0 16px 36px rgba(0, 0, 0, 0.16);
    }

    .status-card {
        background: linear-gradient(135deg, rgba(39, 211, 194, 0.14), rgba(18, 35, 42, 0.8));
        border: 1px solid rgba(39, 211, 194, 0.35);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin: 0.75rem 0 1.25rem;
    }

    .status-card strong {
        color: var(--cyan);
        font-size: 1.35rem;
    }

    [data-testid="stSlider"] [role="slider"] {
        background: var(--cyan);
    }

    [data-baseweb="select"] > div {
        background: #172d35;
        border-color: var(--line);
        border-radius: 10px;
    }

    hr {
        border-color: var(--line);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD DATA AND MODELS
# =========================================================

@st.cache_resource
def load_models():

    health_model = joblib.load(
        "best_battery_health_model.pkl"
    )

    rul_model = joblib.load(
        "best_rul_model.pkl"
    )

    return health_model, rul_model


@st.cache_data
def load_data():

    return pd.read_csv(
        "ev_battery_prediction_dataset.csv"
    )


health_model, rul_model = load_models()
df = load_data()


# =========================================================
# FEATURES
# =========================================================

numeric_features = [
    "Battery_Age_Months",
    "Charging_Cycles",
    "Average_Temperature_C",
    "Maximum_Temperature_C",
    "Average_Charging_Time_Hours",
    "Fast_Charging_Frequency_Percent",
    "Average_Discharge_Rate_Percent",
    "Depth_of_Discharge_Percent",
    "Daily_Driving_Distance_KM",
    "Average_Speed_KMH",
    "Regenerative_Braking_Usage_Percent",
    "Battery_Capacity_KWh",
    "Charging_Efficiency_Percent",
    "Discharging_Efficiency_Percent",
    "Number_of_Full_Charges",
    "Number_of_Deep_Discharges",
    "Operating_Hours"
]

categorical_features = [
    "Vehicle_Type",
    "Battery_Type",
    "Usage_Type",
    "Charging_Type"
]


# =========================================================
# HEADER
# =========================================================

header_logo, header_title = st.columns([1, 5], vertical_alignment="center")

with header_logo:

    st.image(
        str(LOGO_PATH),
        width=150
    )

with header_title:

    st.title("EV Battery AI Prediction Dashboard")

st.markdown(
    """
    **Predictive intelligence for cleaner, longer-lasting electric mobility.**
    Tune the operating profile below, then analyze battery health and remaining useful life.
    """
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Battery Intelligence")

st.sidebar.info(
    "Adjust the battery profile and click "
    "**Analyze Battery** to generate predictions."
)


# =========================================================
# INPUT PROFILE
# =========================================================

def numeric_input(column, label, step=0.1):

    return st.number_input(
        label,
        min_value=float(df[column].min()),
        max_value=float(df[column].max()),
        value=float(df[column].median()),
        step=step,
        format="%.2f"
    )


# =========================================================
# BATTERY CONDITION
# =========================================================

with st.container(border=True):

    st.subheader("🔋 Battery Condition")
    st.caption("Core indicators of battery wear and accumulated usage.")

    col1, col2, col3 = st.columns(3)

    with col1:
        battery_age = numeric_input(
            "Battery_Age_Months",
            "Battery Age (Months)"
        )

    with col2:
        charging_cycles = numeric_input(
            "Charging_Cycles",
            "Charging Cycles",
            step=1.0
        )

    with col3:
        depth_discharge = numeric_input(
            "Depth_of_Discharge_Percent",
            "Depth of Discharge (%)"
        )

    col1, col2 = st.columns(2)

    with col1:
        discharge_rate = numeric_input(
            "Average_Discharge_Rate_Percent",
            "Average Discharge Rate (%)"
        )

    with col2:
        operating_hours = numeric_input(
            "Operating_Hours",
            "Operating Hours"
        )


# =========================================================
# TEMPERATURE & DRIVING
# =========================================================

with st.container(border=True):

    st.subheader("🌡️ Temperature & Driving")
    st.caption("Thermal load and driving behavior that shape battery stress.")

    col1, col2, col3 = st.columns(3)

    with col1:
        avg_temp = numeric_input(
            "Average_Temperature_C",
            "Average Temperature (°C)"
        )

    with col2:
        max_temp = numeric_input(
            "Maximum_Temperature_C",
            "Maximum Temperature (°C)"
        )

    with col3:
        driving_distance = numeric_input(
            "Daily_Driving_Distance_KM",
            "Daily Driving Distance (KM)"
        )

    col1, col2 = st.columns(2)

    with col1:
        avg_speed = numeric_input(
            "Average_Speed_KMH",
            "Average Speed (KM/H)"
        )

    with col2:
        regenerative = numeric_input(
            "Regenerative_Braking_Usage_Percent",
            "Regenerative Braking Usage (%)"
        )


# =========================================================
# CHARGING & EFFICIENCY
# =========================================================

with st.container(border=True):

    st.subheader("⚡ Charging & Efficiency")
    st.caption("Charging habits and conversion efficiency across the battery system.")

    col1, col2, col3 = st.columns(3)

    with col1:
        avg_charge_time = numeric_input(
            "Average_Charging_Time_Hours",
            "Average Charging Time (Hours)"
        )

        fast_charge = numeric_input(
            "Fast_Charging_Frequency_Percent",
            "Fast Charging Frequency (%)"
        )

        battery_capacity = numeric_input(
            "Battery_Capacity_KWh",
            "Battery Capacity (kWh)"
        )

    with col2:
        charging_efficiency = numeric_input(
            "Charging_Efficiency_Percent",
            "Charging Efficiency (%)"
        )

        discharging_efficiency = numeric_input(
            "Discharging_Efficiency_Percent",
            "Discharging Efficiency (%)"
        )

    with col3:
        full_charges = numeric_input(
            "Number_of_Full_Charges",
            "Number of Full Charges",
            step=1.0
        )

        deep_discharges = numeric_input(
            "Number_of_Deep_Discharges",
            "Number of Deep Discharges",
            step=1.0
        )


# =========================================================
# VEHICLE PROFILE
# =========================================================

with st.container(border=True):

    st.subheader("🚗 Vehicle Profile")
    st.caption("Select the vehicle and battery configuration for this analysis.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        vehicle_type = st.selectbox(
            "Vehicle Type",
            sorted(df["Vehicle_Type"].dropna().unique())
        )

    with col2:
        battery_type = st.selectbox(
            "Battery Type",
            sorted(df["Battery_Type"].dropna().unique())
        )

    with col3:
        usage_type = st.selectbox(
            "Usage Type",
            sorted(df["Usage_Type"].dropna().unique())
        )

    with col4:
        charging_type = st.selectbox(
            "Charging Type",
            sorted(df["Charging_Type"].dropna().unique())
        )


# =========================================================
# CREATE INPUT DATA
# =========================================================

input_data = pd.DataFrame({

    "Battery_Age_Months": [battery_age],

    "Charging_Cycles": [charging_cycles],

    "Average_Temperature_C": [avg_temp],

    "Maximum_Temperature_C": [max_temp],

    "Average_Charging_Time_Hours": [avg_charge_time],

    "Fast_Charging_Frequency_Percent": [fast_charge],

    "Average_Discharge_Rate_Percent": [discharge_rate],

    "Depth_of_Discharge_Percent": [depth_discharge],

    "Daily_Driving_Distance_KM": [driving_distance],

    "Average_Speed_KMH": [avg_speed],

    "Regenerative_Braking_Usage_Percent": [regenerative],

    "Battery_Capacity_KWh": [battery_capacity],

    "Charging_Efficiency_Percent": [charging_efficiency],

    "Discharging_Efficiency_Percent": [discharging_efficiency],

    "Number_of_Full_Charges": [full_charges],

    "Number_of_Deep_Discharges": [deep_discharges],

    "Operating_Hours": [operating_hours],

    "Vehicle_Type": [vehicle_type],

    "Battery_Type": [battery_type],

    "Usage_Type": [usage_type],

    "Charging_Type": [charging_type]
})


# =========================================================
# ENCODING
# =========================================================

input_encoded = pd.get_dummies(
    input_data,
    columns=categorical_features,
    dtype=int
)


# =========================================================
# ALIGN FEATURES WITH TRAINING MODEL
# =========================================================

health_columns = health_model.feature_names_in_

rul_columns = rul_model.feature_names_in_


health_input = input_encoded.reindex(
    columns=health_columns,
    fill_value=0
)

rul_input = input_encoded.reindex(
    columns=rul_columns,
    fill_value=0
)


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.divider()

predict_button = st.button(
    "⚡ Analyze Battery",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    health_prediction = health_model.predict(
        health_input
    )[0]

    rul_prediction = rul_model.predict(
        rul_input
    )[0]


    # Keep predictions in sensible range

    health_prediction = np.clip(
        health_prediction,
        0,
        100
    )

    rul_prediction = max(
        0,
        rul_prediction
    )

    def scale_input(column, value):

        minimum = float(df[column].min())
        maximum = float(df[column].max())

        if maximum == minimum:
            return 0.0

        return np.clip((value - minimum) / (maximum - minimum), 0, 1)


    thermal_stress = np.mean([
        scale_input("Average_Temperature_C", avg_temp),
        scale_input("Maximum_Temperature_C", max_temp)
    ]) * 100

    charging_stress = np.mean([
        scale_input("Charging_Cycles", charging_cycles),
        scale_input("Fast_Charging_Frequency_Percent", fast_charge),
        scale_input("Depth_of_Discharge_Percent", depth_discharge)
    ]) * 100

    driving_load = np.mean([
        scale_input("Daily_Driving_Distance_KM", driving_distance),
        scale_input("Average_Speed_KMH", avg_speed),
        scale_input("Average_Discharge_Rate_Percent", discharge_rate)
    ]) * 100

    efficiency_score = np.mean([
        scale_input("Charging_Efficiency_Percent", charging_efficiency),
        scale_input("Discharging_Efficiency_Percent", discharging_efficiency)
    ]) * 100

    battery_risk = 100 - health_prediction


    # =====================================================
    # RESULTS
    # =====================================================

    st.success("Analysis complete. Your battery profile has been evaluated.")

    st.subheader("📊 Battery Intelligence Overview")

    result1, result2, result3, result4 = st.columns(4)

    with result1:

        st.metric(
            "🔋 Battery Health",
            f"{health_prediction:.2f}%"
        )

    with result2:

        st.metric(
            "⏳ Remaining Useful Life",
            f"{rul_prediction:.0f} cycles"
        )

    with result3:

        st.metric(
            "⚡ Efficiency Score",
            f"{efficiency_score:.0f}%"
        )

    with result4:

        st.metric(
            "🛡️ Battery Risk",
            f"{battery_risk:.0f}%"
        )


    # =====================================================
    # BATTERY STATUS
    # =====================================================

    st.divider()

    st.subheader("🔎 Battery Condition")

    if health_prediction >= 80:

        status = "🟢 Excellent"
        status_color = "#27d3c2"

        message = (
            "Battery is in good condition "
            "and appears relatively healthy."
        )

    elif health_prediction >= 60:

        status = "🟡 Moderate"
        status_color = "#ffbf69"

        message = (
            "Battery shows moderate degradation. "
            "Regular monitoring is recommended."
        )

    else:

        status = "🔴 Poor"
        status_color = "#ff7b72"

        message = (
            "Battery health is relatively low. "
            "Inspection or replacement may be required."
        )


    st.markdown(
        f"""
        <div class="status-card">
            <strong style="color: {status_color};">{status}</strong>
            <p>{message}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # HEALTH GAUGE
    # =====================================================

    st.subheader("🔋 Battery Health Gauge")

    fig_health = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=health_prediction,
            title={
                "text": "Battery Health (%)"
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#9bb0b9"
                },
                "bar": {
                    "thickness": 0.7,
                    "color": "#27d3c2"
                },
                "steps": [
                    {
                        "range": [0, 60],
                        "color": "#26373d"
                    },
                    {
                        "range": [60, 80],
                        "color": "#514833"
                    },
                    {
                        "range": [80, 100],
                        "color": "#194b48"
                    }
                ],
                "threshold": {
                    "line": {
                        "width": 4,
                        "color": "#ffbf69"
                    },
                    "value": health_prediction
                }
            }
        )
    )

    fig_health.update_layout(
        height=350,
        paper_bgcolor="rgba(0, 0, 0, 0)",
        font={"color": "#e8f1f5", "family": "Space Grotesk"},
        margin={"t": 60, "b": 20, "l": 20, "r": 20}
    )

    st.plotly_chart(
        fig_health,
        use_container_width=True
    )


    # =====================================================
    # RUL GAUGE
    # =====================================================

    st.subheader("⏳ Remaining Useful Life")

    max_rul_display = max(
        rul_prediction * 1.5,
        100
    )

    fig_rul = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=rul_prediction,
            title={
                "text": "Remaining Charging Cycles"
            },
            gauge={
                "axis": {
                    "range": [0, max_rul_display],
                    "tickcolor": "#9bb0b9"
                },
                "bar": {
                    "thickness": 0.7,
                    "color": "#65d9a6"
                },
                "steps": [
                    {
                        "range": [0, max_rul_display * 0.35],
                        "color": "#3b3032"
                    },
                    {
                        "range": [max_rul_display * 0.35, max_rul_display * 0.7],
                        "color": "#514833"
                    },
                    {
                        "range": [max_rul_display * 0.7, max_rul_display],
                        "color": "#194b48"
                    }
                ],
                "threshold": {
                    "line": {
                        "width": 4,
                        "color": "#ffbf69"
                    },
                    "value": rul_prediction
                }
            }
        )
    )

    fig_rul.update_layout(
        height=350,
        paper_bgcolor="rgba(0, 0, 0, 0)",
        font={"color": "#e8f1f5", "family": "Space Grotesk"},
        margin={"t": 60, "b": 20, "l": 20, "r": 20}
    )

    st.plotly_chart(
        fig_rul,
        use_container_width=True
    )


    # =====================================================
    # DIAGNOSTIC ANALYTICS
    # =====================================================

    st.divider()

    st.subheader("🧭 Battery Diagnostic Analytics")

    diagnostic_left, diagnostic_right = st.columns(2)

    with diagnostic_left:

        profile_labels = [
            "Battery Health",
            "Efficiency",
            "Thermal Stability",
            "Cycle Reserve",
            "Driving Balance"
        ]

        profile_values = [
            health_prediction,
            efficiency_score,
            100 - thermal_stress,
            np.clip(100 - charging_stress, 0, 100),
            np.clip(100 - driving_load, 0, 100)
        ]

        fig_radar = go.Figure(
            go.Scatterpolar(
                r=profile_values + [profile_values[0]],
                theta=profile_labels + [profile_labels[0]],
                fill="toself",
                fillcolor="rgba(39, 211, 194, 0.22)",
                line={"color": "#27d3c2", "width": 3},
                marker={"color": "#ffbf69", "size": 7},
                name="Battery profile"
            )
        )

        fig_radar.update_layout(
            title="Battery Performance Profile",
            height=390,
            paper_bgcolor="rgba(0, 0, 0, 0)",
            polar={
                "bgcolor": "rgba(18, 35, 42, 0.72)",
                "radialaxis": {
                    "range": [0, 100],
                    "gridcolor": "#2a4852",
                    "tickfont": {"color": "#9bb0b9"}
                },
                "angularaxis": {
                    "gridcolor": "#2a4852",
                    "tickfont": {"color": "#e8f1f5"}
                }
            },
            font={"color": "#e8f1f5", "family": "Space Grotesk"},
            margin={"t": 70, "b": 20, "l": 45, "r": 45},
            showlegend=False
        )

        st.plotly_chart(
            fig_radar,
            use_container_width=True
        )

    with diagnostic_right:

        stress_labels = [
            "Thermal Stress",
            "Charging Stress",
            "Driving Load",
            "Battery Risk"
        ]

        stress_values = [
            thermal_stress,
            charging_stress,
            driving_load,
            battery_risk
        ]

        fig_stress = go.Figure(
            go.Bar(
                x=stress_values,
                y=stress_labels,
                orientation="h",
                marker={
                    "color": ["#ffbf69", "#f28f6b", "#d87582", "#ff7b72"],
                    "line": {"color": "#e8f1f5", "width": 0.5}
                },
                text=[f"{value:.0f}%" for value in stress_values],
                textposition="outside",
                cliponaxis=False
            )
        )

        fig_stress.update_layout(
            title="Stress & Risk Signals",
            height=390,
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(18, 35, 42, 0.72)",
            xaxis={
                "range": [0, 110],
                "gridcolor": "#2a4852",
                "tickfont": {"color": "#9bb0b9"},
                "title": {
                    "text": "Relative intensity (%)",
                    "font": {"color": "#9bb0b9"}
                }
            },
            yaxis={
                "tickfont": {"color": "#e8f1f5"},
                "categoryorder": "array",
                "categoryarray": stress_labels
            },
            font={"color": "#e8f1f5", "family": "Space Grotesk"},
            margin={"t": 70, "b": 55, "l": 35, "r": 55},
            showlegend=False
        )

        st.plotly_chart(
            fig_stress,
            use_container_width=True
        )


    visual_left, visual_right = st.columns(2)

    with visual_left:

        fig_health_split = go.Figure(
            go.Pie(
                labels=["Remaining Health", "Degradation Risk"],
                values=[health_prediction, battery_risk],
                hole=0.68,
                marker={
                    "colors": ["#27d3c2", "#ff7b72"],
                    "line": {"color": "#12232a", "width": 3}
                },
                textinfo="label+percent",
                textfont={"color": "#e8f1f5", "size": 12},
                hovertemplate="%{label}: %{value:.1f}%<extra></extra>"
            )
        )

        fig_health_split.update_layout(
            title="Health Composition",
            height=330,
            paper_bgcolor="rgba(0, 0, 0, 0)",
            font={"color": "#e8f1f5", "family": "Space Grotesk"},
            margin={"t": 65, "b": 20, "l": 20, "r": 20},
            showlegend=False,
            annotations=[
                {
                    "text": f"{health_prediction:.0f}%<br>health",
                    "showarrow": False,
                    "font": {"size": 20, "color": "#e8f1f5"}
                }
            ]
        )

        st.plotly_chart(
            fig_health_split,
            use_container_width=True
        )

    with visual_right:

        efficiency_labels = [
            "Charging Efficiency",
            "Discharging Efficiency",
            "Regenerative Braking"
        ]

        efficiency_values = [
            charging_efficiency,
            discharging_efficiency,
            regenerative
        ]

        fig_efficiency = go.Figure(
            go.Bar(
                x=efficiency_labels,
                y=efficiency_values,
                marker={
                    "color": ["#27d3c2", "#65d9a6", "#ffbf69"],
                    "line": {"color": "#e8f1f5", "width": 0.5}
                },
                text=[f"{value:.0f}%" for value in efficiency_values],
                textposition="outside",
                cliponaxis=False
            )
        )

        fig_efficiency.update_layout(
            title="Energy Recovery & Efficiency",
            height=330,
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(18, 35, 42, 0.72)",
            yaxis={
                "range": [0, 110],
                "gridcolor": "#2a4852",
                "tickfont": {"color": "#9bb0b9"},
                "title": {
                    "text": "Performance (%)",
                    "font": {"color": "#9bb0b9"}
                }
            },
            xaxis={"tickfont": {"color": "#e8f1f5"}},
            font={"color": "#e8f1f5", "family": "Space Grotesk"},
            margin={"t": 65, "b": 75, "l": 45, "r": 25},
            showlegend=False
        )

        st.plotly_chart(
            fig_efficiency,
            use_container_width=True
        )

    driving_labels = [
        "Daily Distance",
        "Average Speed",
        "Discharge Rate",
        "Fast Charging"
    ]

    driving_values = [
        driving_distance,
        avg_speed,
        discharge_rate,
        fast_charge
    ]

    driving_columns = [
        "Daily_Driving_Distance_KM",
        "Average_Speed_KMH",
        "Average_Discharge_Rate_Percent",
        "Fast_Charging_Frequency_Percent"
    ]

    driving_percentiles = [
        scale_input(column, value) * 100
        for column, value in zip(driving_columns, driving_values)
    ]

    fig_driving = go.Figure(
        go.Scatter(
            x=driving_labels,
            y=driving_percentiles,
            mode="lines+markers+text",
            text=[f"{value:.0f}%" for value in driving_percentiles],
            textposition="top center",
            line={"color": "#27d3c2", "width": 3, "shape": "spline"},
            marker={"color": "#ffbf69", "size": 10},
            fill="tozeroy",
            fillcolor="rgba(39, 211, 194, 0.13)",
            hovertemplate="%{x}: %{y:.1f}% of dataset range<extra></extra>"
        )
    )

    fig_driving.update_layout(
        title="Operating Profile Intensity",
        height=340,
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(18, 35, 42, 0.72)",
        yaxis={
            "range": [0, 110],
            "gridcolor": "#2a4852",
            "tickfont": {"color": "#9bb0b9"},
            "title": {
                "text": "Relative position in dataset (%)",
                "font": {"color": "#9bb0b9"}
            }
        },
        xaxis={"tickfont": {"color": "#e8f1f5"}},
        font={"color": "#e8f1f5", "family": "Space Grotesk"},
        margin={"t": 65, "b": 55, "l": 55, "r": 25},
        showlegend=False
    )

    st.plotly_chart(
        fig_driving,
        use_container_width=True
    )


    # =====================================================
    # INPUT PROFILE
    # =====================================================

    st.divider()

    st.subheader("📈 Battery Input Profile")

    profile_data = pd.DataFrame({

        "Parameter": [
            "Battery Age",
            "Charging Cycles",
            "Avg Temperature",
            "Depth of Discharge",
            "Fast Charging",
            "Charging Efficiency",
            "Discharging Efficiency"
        ],

        "Value": [
            battery_age,
            charging_cycles,
            avg_temp,
            depth_discharge,
            fast_charge,
            charging_efficiency,
            discharging_efficiency
        ]
    })

    st.bar_chart(
        profile_data.set_index(
            "Parameter"
        )
    )


    # =====================================================
    # RECOMMENDATION
    # =====================================================

    st.divider()

    st.subheader("💡 AI Recommendations")

    if health_prediction >= 80:

        st.info(
            """
            **Battery condition is good.**

            Continue regular maintenance and avoid
            excessive temperature and unnecessary
            deep-discharge cycles.
            """
        )

    elif health_prediction >= 60:

        st.warning(
            """
            **Battery degradation is noticeable.**

            Monitor battery temperature, charging behavior
            and deep-discharge frequency regularly.
            """
        )

    else:

        st.error(
            """
            **Battery health is low.**

            Consider professional battery inspection and
            evaluate whether replacement may be required.
            """
        )

    if thermal_stress >= 70:

        st.warning(
            "**Thermal alert:** operating temperatures are elevated. "
            "Improve cooling and avoid repeated high-temperature sessions."
        )

    if charging_stress >= 70:

        st.warning(
            "**Charging alert:** cycle depth, fast charging, or charge count "
            "is contributing to higher battery stress."
        )

    if efficiency_score >= 75 and thermal_stress < 70:

        st.info(
            "**Efficiency insight:** the battery is converting energy well. "
            "Maintain the current charging and driving profile."
        )


    # =====================================================
    # SUMMARY
    # =====================================================

    st.divider()

    st.subheader("📋 Prediction Summary")

    summary = pd.DataFrame({

        "Metric": [
            "Battery Health",
            "Remaining Useful Life",
            "Battery Status"
        ],

        "Prediction": [
            f"{health_prediction:.2f}%",
            f"{rul_prediction:.0f} cycles",
            status
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    report = pd.DataFrame({
        "Analysis": [
            "Battery Health",
            "Remaining Useful Life",
            "Battery Status",
            "Efficiency Score",
            "Thermal Stress",
            "Charging Stress",
            "Driving Load",
            "Battery Risk"
        ],
        "Value": [
            f"{health_prediction:.2f}%",
            f"{rul_prediction:.0f} cycles",
            status,
            f"{efficiency_score:.0f}%",
            f"{thermal_stress:.0f}%",
            f"{charging_stress:.0f}%",
            f"{driving_load:.0f}%",
            f"{battery_risk:.0f}%"
        ]
    })

    st.download_button(
        "⬇️ Download Analysis Summary",
        report.to_csv(index=False),
        file_name="ev_battery_analysis_summary.csv",
        mime="text/csv",
        use_container_width=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "EV Battery Prediction System | "
    "Machine Learning Project"
)