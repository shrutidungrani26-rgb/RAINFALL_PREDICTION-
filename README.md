# 🌧️ Rainfall Prediction System (Machine Learning & Web App)

A modern, beginner-friendly Machine Learning project in Python that predicts rainfall occurrence based on meteorological observations. Features a **unique glassmorphic design**, **animated rainfall particles**, and a **3D atmospheric background**.

---

## 🎨 Unique Aesthetic & Styling
- **3D Atmospheric Backdrop:** Custom high-resolution digital weather artwork (Assets/img/weather_bg.jpg).
- **Animated Rain Particles:** Pure GPU-accelerated CSS falling rain drops in the background canvas.
- **Dynamic Theme Switcher:**
  - 🌌 **Deep Ocean Aurora:** Midnight oceanic glassmorphism with glowing cyan neon accents.
  - 🌤️ **Sky Blue Glass:** Airy daylight glassmorphism with golden sunbeam accents.
- **Pulsating Prediction Aura:** Glowing alert cards with animated probability meters.
- **One-Click Weather Scenarios:** Instant ⛈️ Rainy Day and ☀️ Sunny Day preset buttons.

---

## 📊 Actual Model Evaluation Metrics (Unseen Test Data)
Trained using an **80:20 Stratified Split** (andom_state=42) with Logistic Regression inside a scikit-learn Pipeline:

| Metric | Score | Percentage | Notes |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **0.8378** | **83.78%** | High overall diagnostic accuracy |
| **Precision** | **0.8519** | **85.19%** | Positive Predictive Value for rain |
| **Recall** | **0.9200** | **92.00%** | Catches 92% of actual rainy days |
| **F1 Score** | **0.8846** | **88.46%** | Harmonic mean of precision & recall |

---

## 📁 Neatly Organized Folder Architecture
All components are separated into dedicated, modular folders:

`	ext
Rainfall Prediction Model/
├── Assets/
│   ├── css/
│   │   └── style.css                 # Custom glassmorphic styles and rain keyframes
│   └── img/
│       └── weather_bg.jpg            # 3D Atmospheric weather artwork
├── Dataset/
│   └── rainfall.csv                  # Original dataset (366 rows, cleaned headers)
├── Docs/
│   ├── User_Manual.md                # Beginner step-by-step user manual
│   └── Architecture.md               # System & pipeline architectural blueprint
├── Model/
│   └── rainfall_model.joblib         # Trained scikit-learn pipeline (Imputer + Scaler + Model)
├── Python_Code/
│   ├── project.py                    # Model training, evaluation & joblib exporter
│   ├── eda.py                        # Exploratory Data Analysis & visual charts script
│   └── utils.py                      # Reusable path, dataset & pipeline utilities
├── Reports/
│   ├── Figures/                      # Dedicated folder for all PNG visualizations
│   │   ├── rainfall_target_distribution.png
│   │   ├── temperature_distribution.png
│   │   ├── correlation_heatmap.png
│   │   ├── confusion_matrix.png
│   │   └── missing_values_chart.png
│   ├── Metrics/                      # Text metrics logs & classification reports
│   │   ├── missing_values_summary.txt
│   │   └── model_evaluation_report.txt
│   └── Interactive/                  # Standalone interactive HTML report
│       └── rainfall_eda_report.html
├── Web_App/
│   ├── components.py                 # Modular UI cards, badges & preset loaders
│   └── theme.py                      # Dynamic CSS injector & background animations
├── app.py                            # Streamlit web application entry point
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
`

---

## 🚀 Step-by-Step Running Instructions

Open PowerShell or Command Prompt inside the project directory:
`powershell
cd "C:\Users\DELL\Desktop\Rainfall Prediction Model"
`

### 1. Install Dependencies
`powershell
py -m pip install -r requirements.txt
`

### 2. Train the Machine Learning Model
`powershell
py Python_Code/project.py
`

### 3. Generate Visual EDA Charts & Interactive HTML Report
`powershell
py Python_Code/eda.py
`

### 4. Launch the Web Application
`powershell
py -m streamlit run app.py
`
Open http://localhost:8501 in your web browser to interact with the system!

---

## ⚠️ Weather Forecast Disclaimer
This system uses a machine learning classification model trained on historical data. Predictions reflect probabilistic machine learning inferences and do **not** claim to be official meteorological weather forecasts.
