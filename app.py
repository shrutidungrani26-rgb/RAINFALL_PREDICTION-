"""
=============================================================================
🌧️ Rainfall Prediction System - Main Streamlit Application
=============================================================================
Features:
- Unique Glassmorphic Design with Animated Background Rain & Atmospheric Canvas
- Theme Switcher ("Deep Ocean Aurora" vs "Sky Blue Glass")
- Modularized Folder Organization (Assets/, Docs/, Model/, Reports/, Web_App/)
- 10 Meteorological Weather Feature Controls
- Instant Weather Presets (⛈️ Rainy Day, ☀️ Sunny Day, 🔄 Average Weather)
- Real-Time Probability Estimation with Glowing Status Badges
- Performance Evaluation & High-Resolution EDA Visualization Tabs
=============================================================================
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Resolve base directory
BASE_DIR = Path(__file__).resolve().parent

# Ensure local modular imports work seamlessly
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from Python_Code.utils import get_paths, load_trained_pipeline, FEATURE_NAMES
from Web_App.theme import apply_custom_theme
from Web_App.components import render_hero_banner, render_prediction_badge, render_disclaimer

# Configure Streamlit page
st.set_page_config(
    page_title="Rainfall Prediction System",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

PATHS = get_paths()
pipeline = load_trained_pipeline()

# Preset weather profiles
PRESETS = {
    "default": {
        "pressure": 1013.0, "maxtemp": 26.2, "temparature": 23.8, "mintemp": 21.9,
        "dewpoint": 20.0, "humidity": 80.0, "cloud": 70.0, "sunshine": 4.5,
        "winddirection": 100.0, "windspeed": 21.5
    },
    "rainy": {
        "pressure": 1006.0, "maxtemp": 23.5, "temparature": 21.0, "mintemp": 19.5,
        "dewpoint": 19.0, "humidity": 94.0, "cloud": 92.0, "sunshine": 0.5,
        "winddirection": 60.0, "windspeed": 28.0
    },
    "sunny": {
        "pressure": 1022.0, "maxtemp": 31.0, "temparature": 28.5, "mintemp": 24.0,
        "dewpoint": 15.0, "humidity": 55.0, "cloud": 15.0, "sunshine": 10.5,
        "winddirection": 180.0, "windspeed": 14.0
    }
}

if "form_values" not in st.session_state:
    st.session_state.form_values = PRESETS["default"].copy()

# Sidebar: Controls & Theme Selection
with st.sidebar:
    st.markdown("### 🎨 Unique Style & Theme")
    theme_choice = st.radio(
        "Choose Background Aesthetic:",
        ["Deep Ocean Aurora", "Sky Blue Glass"],
        index=0,
        help="Select between Dark Aurora Glassmorphism and Fresh Sky Blue daylight theme"
    )

    st.markdown("---")
    st.markdown("### 🎛️ Weather Presets")
    st.caption("Load realistic test conditions with a single click:")
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("⛈️ Rainy Day"):
            st.session_state.form_values = PRESETS["rainy"].copy()
            st.rerun()
    with col_b2:
        if st.button("☀️ Sunny Day"):
            st.session_state.form_values = PRESETS["sunny"].copy()
            st.rerun()

    if st.button("🔄 Reset to Averages"):
        st.session_state.form_values = PRESETS["default"].copy()
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Model Architecture")
    st.markdown("""
    - **Algorithm:** Logistic Regression
    - **Pipeline:** Imputer + Scaler + Model
    - **Accuracy:** `83.78%`
    - **F1 Score:** `88.46%`
    - **Dataset:** 366 Historical Days
    """)
    st.markdown("---")
    st.markdown("### 📁 Organized Folders")
    st.caption("`Assets/` • `Docs/` • `Model/` • `Reports/` • `Web_App/`")

# Apply dynamic custom theme & background
apply_custom_theme(BASE_DIR, theme_choice)

# Render Hero Banner
render_hero_banner()

# Navigation Tabs
tab_predict, tab_metrics, tab_eda, tab_folders, tab_guide = st.tabs([
    "🔮 Predict Rainfall",
    "📊 Model Performance",
    "📈 EDA Visualizations",
    "📁 Organized Folders",
    "ℹ️ Weather Feature Guide"
])

# -----------------------------------------------------------------------------
# TAB 1: PREDICTION INTERFACE
# -----------------------------------------------------------------------------
with tab_predict:
    if pipeline is None:
        st.error("⚠️ Trained model not found in `Model/rainfall_model.joblib`. Please run `py Python_Code/project.py` first!")
    else:
        st.markdown("### 📝 Enter Weather Observations")
        st.caption("Adjust meteorological parameters below to evaluate rainfall likelihood:")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="glass-card-header">🌡️ Thermal Indicators</div>', unsafe_allow_html=True)
            maxtemp = st.number_input(
                "Maximum Temperature (°C)",
                min_value=5.0, max_value=45.0,
                value=float(st.session_state.form_values["maxtemp"]),
                step=0.1, help="Highest recorded temperature during the day"
            )
            temparature = st.number_input(
                "Average Temperature (°C)",
                min_value=0.0, max_value=40.0,
                value=float(st.session_state.form_values["temparature"]),
                step=0.1, help="Mean daily ambient temperature"
            )
            mintemp = st.number_input(
                "Minimum Temperature (°C)",
                min_value=0.0, max_value=35.0,
                value=float(st.session_state.form_values["mintemp"]),
                step=0.1, help="Lowest temperature recorded overnight"
            )
            dewpoint = st.number_input(
                "Dew Point (°C)",
                min_value=-5.0, max_value=30.0,
                value=float(st.session_state.form_values["dewpoint"]),
                step=0.1, help="Temperature at which moisture condenses"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="glass-card-header">☁️ Atmospheric Conditions</div>', unsafe_allow_html=True)
            pressure = st.number_input(
                "Atmospheric Pressure (hPa)",
                min_value=980.0, max_value=1050.0,
                value=float(st.session_state.form_values["pressure"]),
                step=0.1, help="Barometric air pressure at station level"
            )
            humidity = st.slider(
                "Relative Humidity (%)",
                min_value=10.0, max_value=100.0,
                value=float(st.session_state.form_values["humidity"]),
                step=1.0, help="Moisture saturation level in the atmosphere"
            )
            cloud = st.slider(
                "Cloud Cover (%)",
                min_value=0.0, max_value=100.0,
                value=float(st.session_state.form_values["cloud"]),
                step=1.0, help="Fraction of the sky obscured by clouds"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="glass-card-header">💨 Wind & Solar Radiation</div>', unsafe_allow_html=True)
            sunshine = st.number_input(
                "Sunshine Duration (Hours)",
                min_value=0.0, max_value=15.0,
                value=float(st.session_state.form_values["sunshine"]),
                step=0.1, help="Total daily hours of direct sunlight"
            )
            winddirection = st.slider(
                "Wind Direction (Degrees)",
                min_value=0.0, max_value=360.0,
                value=float(st.session_state.form_values["winddirection"]),
                step=5.0, help="Azimuth wind direction (0=N, 90=E, 180=S, 270=W)"
            )
            windspeed = st.number_input(
                "Wind Speed (km/h)",
                min_value=0.0, max_value=80.0,
                value=float(st.session_state.form_values["windspeed"]),
                step=0.1, help="Average wind velocity in km per hour"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("🔮 Predict Rainfall", use_container_width=True)

        if predict_clicked:
            input_dict = {
                'pressure': [pressure],
                'maxtemp': [maxtemp],
                'temparature': [temparature],
                'mintemp': [mintemp],
                'dewpoint': [dewpoint],
                'humidity': [humidity],
                'cloud': [cloud],
                'sunshine': [sunshine],
                'winddirection': [winddirection],
                'windspeed': [windspeed]
            }
            input_df = pd.DataFrame(input_dict)

            # Predict using pipeline
            prediction = pipeline.predict(input_df)[0]
            probabilities = pipeline.predict_proba(input_df)[0]
            rain_prob = probabilities[1]
            dry_prob = probabilities[0]

            render_prediction_badge(prediction, rain_prob, dry_prob)

            with st.expander("🔍 View Processed Meteorological Vector", expanded=False):
                st.dataframe(input_df.T.rename(columns={0: "Input Value"}), use_container_width=True)

        render_disclaimer()

# -----------------------------------------------------------------------------
# TAB 2: MODEL PERFORMANCE
# -----------------------------------------------------------------------------
with tab_metrics:
    st.markdown("### 📊 Model Evaluation Performance")
    st.write("Trained using an **80:20 Stratified Split** (`random_state=42`) with scikit-learn Pipeline:")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", "83.78%", "Test Data")
    m2.metric("Precision", "85.19%", "Rain Class")
    m3.metric("Recall", "92.00%", "Rain Sensitivity")
    m4.metric("F1 Score", "88.46%", "Harmonic Mean")

    st.markdown("---")
    col_cm, col_rep = st.columns([1, 1])

    with col_cm:
        st.markdown("#### 🧩 Confusion Matrix")
        cm_path = PATHS["reports_figures"] / "confusion_matrix.png"
        if not cm_path.exists():
            cm_path = PATHS["reports"] / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrix on Unseen Test Observations (74 Days)")

    with col_rep:
        st.markdown("#### 📋 Detailed Classification Report")
        rep_path = PATHS["reports_metrics"] / "model_evaluation_report.txt"
        if not rep_path.exists():
            rep_path = PATHS["reports"] / "model_evaluation_report.txt"
        if rep_path.exists():
            with open(rep_path, "r", encoding="utf-8") as f:
                st.code(f.read(), language="text")

# -----------------------------------------------------------------------------
# TAB 3: EDA VISUALIZATIONS
# -----------------------------------------------------------------------------
with tab_eda:
    st.markdown("### 📈 Exploratory Data Analysis & Visualizations")
    st.write("Visual analytical charts generated from the original 366-day meteorological dataset:")

    eda_c1, eda_c2 = st.columns(2)
    fig_dir = PATHS["reports_figures"]
    fallback_dir = PATHS["reports"]

    def find_fig(name):
        p1 = fig_dir / name
        p2 = fallback_dir / name
        return str(p1) if p1.exists() else (str(p2) if p2.exists() else None)

    with eda_c1:
        target_f = find_fig("rainfall_target_distribution.png")
        if target_f:
            st.image(target_f, caption="Rainfall Target Distribution (Yes vs No)")
        
        miss_f = find_fig("missing_values_chart.png")
        if miss_f:
            st.image(miss_f, caption="Missing Values Analysis")

    with eda_c2:
        temp_f = find_fig("temperature_distribution.png")
        if temp_f:
            st.image(temp_f, caption="Temperature Variations & Distributions")

        corr_f = find_fig("correlation_heatmap.png")
        if corr_f:
            st.image(corr_f, caption="Meteorological Feature Correlation Matrix")

    html_rep = PATHS["reports_interactive"] / "rainfall_eda_report.html"
    if not html_rep.exists():
        html_rep = PATHS["reports"] / "rainfall_eda_report.html"
    if html_rep.exists():
        st.success(f"💡 Comprehensive Interactive HTML EDA Report available at: `{html_rep.relative_to(BASE_DIR)}`")

# -----------------------------------------------------------------------------
# TAB 4: ORGANIZED FOLDERS
# -----------------------------------------------------------------------------
with tab_folders:
    st.markdown("### 📁 Organized Project Structure (Dedicated Folders)")
    st.write("All components of the project are neatly separated into dedicated folders:")

    st.markdown("""
    ```text
    Rainfall Prediction Model/
    ├── Assets/
    │   ├── css/
    │   │   └── style.css                     # Unique styling and rain animations
    │   └── img/
    │       └── weather_bg.jpg                # 3D Atmospheric weather artwork
    ├── Dataset/
    │   └── rainfall.csv                      # Cleaned original 366-day dataset
    ├── Docs/
    │   ├── User_Manual.md                    # Beginner step-by-step user manual
    │   └── Architecture.md                   # System and pipeline architecture
    ├── Model/
    │   └── rainfall_model.joblib             # Serialized scikit-learn pipeline
    ├── Python_Code/
    │   ├── project.py                        # Model training and export script
    │   ├── eda.py                            # EDA chart generation script
    │   └── utils.py                          # Reusable helper utilities
    ├── Reports/
    │   ├── Figures/                          # Dedicated folder for all PNG charts
    │   ├── Metrics/                          # Text logs and evaluation summaries
    │   └── Interactive/                      # Standalone interactive HTML report
    ├── Web_App/
    │   ├── components.py                     # Modular UI cards and badges
    │   └── theme.py                          # Dynamic background and CSS injector
    ├── app.py                                # Main Streamlit web application
    ├── requirements.txt                      # Python dependencies
    └── README.md                             # Comprehensive project documentation
    ```
    """)

# -----------------------------------------------------------------------------
# TAB 5: WEATHER FEATURE GUIDE
# -----------------------------------------------------------------------------
with tab_guide:
    st.markdown("### ℹ️ Meteorological Feature Reference Guide")
    features_info = [
        {"Feature": "pressure", "Description": "Atmospheric pressure in hectopascals (hPa). Low pressure often correlates with clouds and precipitation.", "Normal Range": "1000 - 1030 hPa"},
        {"Feature": "maxtemp", "Description": "Highest temperature reached during the day.", "Normal Range": "15°C - 35°C"},
        {"Feature": "temparature", "Description": "Average ambient air temperature throughout the day.", "Normal Range": "10°C - 32°C"},
        {"Feature": "mintemp", "Description": "Lowest temperature recorded, typically near dawn.", "Normal Range": "5°C - 28°C"},
        {"Feature": "dewpoint", "Description": "The temperature at which moisture condenses. High dew points indicate moist, humid air.", "Normal Range": "10°C - 26°C"},
        {"Feature": "humidity", "Description": "Relative percentage of moisture the air holds relative to saturation.", "Normal Range": "40% - 100%"},
        {"Feature": "cloud", "Description": "Percentage of sky covered by clouds. Cloud cover > 80% strongly correlates with rain.", "Normal Range": "0% - 100%"},
        {"Feature": "sunshine", "Description": "Total hours of direct sunlight. Rain days typically exhibit very low sunshine (0 - 3 hours).", "Normal Range": "0 - 12 hours"},
        {"Feature": "winddirection", "Description": "Compass angle from which wind blows (0°=North, 90°=East, 180°=South, 270°=West).", "Normal Range": "0° - 360°"},
        {"Feature": "windspeed", "Description": "Average wind velocity in kilometers per hour.", "Normal Range": "5 - 50 km/h"}
    ]
    st.table(pd.DataFrame(features_info))
