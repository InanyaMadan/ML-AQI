# Prediction utility functions
# Loads trained models and generates PM2.5 predictions

import joblib
import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), "data")

def load_models():
    rf_model  = joblib.load(
        os.path.join(DATA_DIR, "random_forest_model.pkl"))
    scaler    = joblib.load(
        os.path.join(DATA_DIR, "feature_scaler.pkl"))
    le        = joblib.load(
        os.path.join(DATA_DIR, "city_encoder.pkl"))
    features  = joblib.load(
        os.path.join(DATA_DIR, "feature_names.pkl"))
    return rf_model, scaler, le, features

def get_city_encoded_map(le):
    return dict(zip(le.classes_,
                    le.transform(le.classes_)))

def pm25_to_category(pm25):
    if pm25 <= 30:    return "Good"
    elif pm25 <= 60:  return "Satisfactory"
    elif pm25 <= 90:  return "Moderate"
    elif pm25 <= 120: return "Poor"
    elif pm25 <= 250: return "Very Poor"
    else:             return "Severe"

def pm25_to_color(pm25):
    if pm25 <= 30:    return "#00e400"
    elif pm25 <= 60:  return "#92d050"
    elif pm25 <= 90:  return "#ffff00"
    elif pm25 <= 120: return "#ff7e00"
    elif pm25 <= 250: return "#ff0000"
    else:             return "#7e0023"

def predict_single(city, temperature, humidity,
                   wind_speed, hour, day_of_year,
                   pm10, rf_model, scaler, features,
                   city_encoded_map, aqi_df):

    city_data = aqi_df[aqi_df["city"] == city]
    avg_no2   = city_data["NO2"].mean()
    avg_so2   = city_data["SO2"].mean()
    avg_co    = city_data["CO"].mean()

    city_centers = {
        "Delhi":     (28.63, 77.20),
        "Chennai":   (13.05, 80.22),
        "Bengaluru": (12.97, 77.59),
        "Hyderabad": (17.38, 78.49),
        "Jaipur":    (26.91, 75.79),
    }

    lat, lon = city_centers[city]

    input_data = pd.DataFrame([{
        "city_encoded": city_encoded_map[city],
        "lat":          lat,
        "lon":          lon,
        "temperature":  temperature,
        "humidity":     humidity,
        "wind_speed":   wind_speed,
        "wind_deg":     180,
        "road_density": 8.0,
        "hour":         hour,
        "day_of_year":  day_of_year,
        "NO2":          avg_no2,
        "SO2":          avg_so2,
        "CO":           avg_co,
        "PM10":         pm10
    }])

    pm25 = rf_model.predict(input_data[features])[0]
    return pm25

def load_pred_grid(city):
    path = os.path.join(DATA_DIR, f"{city.lower()}_pred_grid.csv")
    return pd.read_csv(path)

def load_master_data():
    path = os.path.join(DATA_DIR, "master_dataset_multicity.csv")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def load_station_means():
    path = os.path.join(DATA_DIR, "station_means.csv")
    return pd.read_csv(path)

def load_loso_results():
    path = os.path.join(DATA_DIR, "loso_results.csv")
    return pd.read_csv(path)