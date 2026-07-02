# 🌫️ ML-AQI — India Air Quality Intelligence Dashboard

An interactive **Streamlit dashboard** that predicts hyper-local PM2.5 air pollution across five major Indian cities using machine learning, real CPCB monitoring data, and Open-Meteo weather data.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-app-red)
![Model](https://img.shields.io/badge/model-Random%20Forest%20R²%200.9873-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 🔍 Overview

Ground-truth air quality monitoring stations only cover a tiny fraction of any city, leaving most neighborhoods "blind" to their actual pollution exposure. **ML-AQI** fills those gaps by training a machine learning model on real Central Pollution Control Board (CPCB) station readings plus weather data, then using it to predict PM2.5 concentrations across an entire city grid — including areas with no physical sensor.

The dashboard covers **5 cities**, **50 monitoring stations**, and **105,585+ readings** from January–March 2023, and lets you explore pollution maps, trends, model performance, health risk, sensor placement recommendations, and a live what-if prediction simulator.

## ✨ Features

| Tab | What it does |
|---|---|
| 🗺️ **Pollution Maps** | ML-predicted PM2.5 heatmap across the city vs. an IDW (Inverse Distance Weighting) spatial baseline, with real CPCB station markers overlaid |
| 📈 **Trends & EDA** | Time-series trends, hourly patterns, seasonal (winter → spring) comparisons, per-station breakdowns, and a feature correlation matrix |
| 🤖 **ML Models** | Side-by-side comparison of Random Forest, XGBoost, SVR, Ridge, and Linear Regression, feature importance, and Leave-One-Station-Out (LOSO) spatial validation |
| ⚕️ **Health Risk** | Percentage of city area exceeding WHO and Indian national PM2.5 standards, a color-coded health risk map, and the top 10 pollution hotspots |
| 📡 **Sensor Placement** | Recommends the top 3 locations for new monitoring sensors, balancing pollution severity against distance from existing stations |
| 🎛️ **What-If Simulator** | Adjust temperature, humidity, wind speed, hour of day, and PM10 to get a live PM2.5 prediction, plus quick preset scenarios (Rush Hour, Strong Wind, Spring Day) |
| 🏙️ **City Comparison** | Cross-city PM2.5 comparison, summary statistics table, combined station map, and LOSO validation scores by city |

## 🏙️ Cities Covered

Delhi · Chennai · Bengaluru · Hyderabad · Jaipur

## 🤖 Model Performance

The primary prediction model is a **Random Forest Regressor**, selected after comparison against several alternatives:

| Algorithm | MAE (µg/m³) | RMSE (µg/m³) | R² |
|---|---|---|---|
| **Random Forest** ⭐ | 4.25 | 6.69 | **0.9873** |
| XGBoost | 6.97 | 10.19 | 0.9707 |
| SVR | 9.31 | 14.90 | 0.9373 |
| Ridge Regression | 14.30 | 20.65 | 0.8795 |
| Linear Regression | 14.30 | 20.65 | 0.8795 |

Model quality is further validated using **Leave-One-Station-Out (LOSO)** cross-validation, which tests how well the model generalizes to a station it has never seen — a more realistic test of spatial prediction quality than a random train/test split.

## 🗂️ Project Structure

```
ML-AQI/
├── app.py                # Main Streamlit application (all 7 dashboard tabs)
├── assets/                # Custom CSS and static assets
├── data/                  # CPCB station data, weather data, prediction grids
├── utils/
│   ├── predictions.py    # Model loading, encoding, and single-point prediction
│   ├── maps.py            # Folium map builders (heatmap, health risk, sensor placement)
│   └── charts.py          # Plotly chart builders (trends, seasonal, feature importance, etc.)
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/InanyaMadan/ML-AQI.git
cd ML-AQI

# (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** `requirements.txt` currently lists the core ML dependencies (`numpy`, `pandas`, `scikit-learn`, `xgboost`, `streamlit`, `joblib`). The app also imports `folium`, `streamlit-folium`, and `plotly` for the mapping and charting tabs — install these as well if they aren't already present:
> ```bash
> pip install folium streamlit-folium plotly
> ```

### Run the app

```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`.

## 📊 Data Sources

- **Air quality data:** Real CPCB (Central Pollution Control Board) station measurements, sourced via Kaggle
- **Weather data:** Open-Meteo API (temperature, humidity, wind speed anomalies)
- **Coverage period:** January – March 2023
- **Scale:** 5 cities, 50 stations, 105,585+ readings

## 🧠 How It Works

1. **Data collection** — Historical CPCB PM2.5/PM10 readings are merged with corresponding weather data (temperature, humidity, wind speed) by station, date, and hour.
2. **Feature engineering** — Features include PM10, hour of day, day of year, wind speed, humidity, temperature, and encoded city identity.
3. **Model training** — Several regression algorithms are trained and compared; the Random Forest model is selected as the production model based on R², MAE, and RMSE.
4. **Spatial prediction** — The trained model predicts PM2.5 across a dense grid of coordinates per city, producing a continuous pollution surface even in areas without physical sensors.
5. **Validation** — LOSO cross-validation checks that the model still performs well when an entire station's data is withheld, simulating prediction in a genuinely unmonitored area.
6. **Interpretation** — Predictions are translated into WHO/India standard exceedance ratios, AQI categories, health risk levels, and sensor placement priorities.

## 🛠️ Tech Stack

- **App framework:** [Streamlit](https://streamlit.io/)
- **Maps:** [Folium](https://python-visualization.github.io/folium/) + [streamlit-folium](https://github.com/randyzwitch/streamlit-folium)
- **Charts:** [Plotly](https://plotly.com/python/)
- **ML:** scikit-learn (Random Forest, SVR, Ridge, Linear Regression), XGBoost
- **Data processing:** pandas, numpy

## 📌 Notes

- PM2.5 thresholds used in the app: **WHO guideline** ≤ 15 µg/m³, **India national standard** ≤ 60 µg/m³.
- The IDW (Inverse Distance Weighting) map mode serves as a simple, model-free spatial interpolation baseline for comparison against the ML predictions.
- The Hyderabad Patancheruvu station shows a notably poor LOSO R² due to its geographic isolation from other stations — flagged in-app as a priority location for new sensor deployment.

## 📄 License

No license file is currently included in this repository. Consider adding one (e.g., MIT) if you intend for others to reuse this code.

## 🙋 Author

Built by [InanyaMadan](https://github.com/InanyaMadan)