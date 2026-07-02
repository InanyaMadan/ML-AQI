# Map generation utility functions
# Creates Folium maps for all features

import folium
from folium.plugins import HeatMap
import pandas as pd
import numpy as np
from utils.predictions import (pm25_to_category,
                                pm25_to_color)

CITY_CENTERS = {
    "Delhi":     [28.63, 77.20],
    "Chennai":   [13.05, 80.22],
    "Bengaluru": [12.97, 77.59],
    "Hyderabad": [17.38, 78.49],
    "Jaipur":    [26.91, 75.79],
}

CITY_ZOOM = {
    "Delhi": 11, "Chennai": 11,
    "Bengaluru": 11, "Hyderabad": 11,
    "Jaipur": 12
}

def make_heatmap(city, pred_df):
    # ML prediction heatmap for a city
    center = CITY_CENTERS[city]
    m = folium.Map(location=center,
                   zoom_start=CITY_ZOOM[city],
                   tiles="CartoDB positron")

    min_p = pred_df["predicted_pm25"].min()
    max_p = pred_df["predicted_pm25"].max()

    heat_data = [
        [r["lat"], r["lon"],
         (r["predicted_pm25"]-min_p)/(max_p-min_p)]
        for _, r in pred_df.iterrows()
    ]
    HeatMap(heat_data, radius=22, blur=18,
            min_opacity=0.4,
            gradient={"0.0":"green","0.4":"yellow",
                      "0.7":"orange","1.0":"red"}
            ).add_to(m)
    return m

def add_station_markers(m, city, aqi_df):
    # Add station circle markers to a map
    for station, grp in aqi_df[
            aqi_df["city"]==city].groupby("station"):
        avg = grp["pm25"].mean()
        col = pm25_to_color(avg)
        folium.CircleMarker(
            location=[grp["lat"].iloc[0],
                      grp["lon"].iloc[0]],
            radius=10, color="black",
            fill=True, fill_color=col,
            fill_opacity=0.9,
            popup=(f"<b>{station}</b><br>"
                   f"Avg PM2.5: {avg:.1f} µg/m³<br>"
                   f"Category: "
                   f"{pm25_to_category(avg)}")
        ).add_to(m)
    return m

def make_health_risk_map(city, pred_df):
    # Health risk map showing WHO exceedance
    def risk_color(pm25):
        if pm25 <= 15:    return "#00e400"
        elif pm25 <= 30:  return "#92d050"
        elif pm25 <= 60:  return "#ffff00"
        elif pm25 <= 90:  return "#ff7e00"
        elif pm25 <= 120: return "#ff0000"
        else:             return "#7e0023"

    center = CITY_CENTERS[city]
    m = folium.Map(location=center,
                   zoom_start=CITY_ZOOM[city],
                   tiles="CartoDB positron")

    for _, row in pred_df.iloc[::2].iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=7, color=None, fill=True,
            fill_color=risk_color(
                row["predicted_pm25"]),
            fill_opacity=0.7,
            popup=(f"PM2.5: "
                   f"{row['predicted_pm25']:.1f}<br>"
                   f"WHO: "
                   f"{row['predicted_pm25']/15:.1f}x<br>"
                   f"Risk: "
                   f"{pm25_to_category(row['predicted_pm25'])}")
        ).add_to(m)
    return m

def make_sensor_map(city, pred_df, aqi_df,
                    sensor_recs):
    # Sensor placement recommendation map
    from scipy.spatial.distance import cdist

    center = CITY_CENTERS[city]
    m = folium.Map(location=center,
                   zoom_start=CITY_ZOOM[city],
                   tiles="CartoDB positron")

    # Background heatmap
    min_p = pred_df["predicted_pm25"].min()
    max_p = pred_df["predicted_pm25"].max()
    heat_data = [
        [r["lat"], r["lon"],
         (r["predicted_pm25"]-min_p)/(max_p-min_p)]
        for _, r in pred_df.iterrows()
    ]
    HeatMap(heat_data, radius=18, blur=15,
            min_opacity=0.3,
            gradient={"0.0":"green","0.4":"yellow",
                      "0.7":"orange","1.0":"red"}
            ).add_to(m)

    # Existing stations
    for station, grp in aqi_df[
            aqi_df["city"]==city].groupby("station"):
        folium.CircleMarker(
            location=[grp["lat"].iloc[0],
                      grp["lon"].iloc[0]],
            radius=10, color="black",
            fill=True, fill_color="white",
            fill_opacity=0.9,
            popup=f"<b>EXISTING: {station}</b>"
        ).add_to(m)

    # Recommended sensors
    for idx, row in sensor_recs.iterrows():
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=(f"<b>RECOMMENDED #{idx+1}</b><br>"
                   f"PM2.5: "
                   f"{row['predicted_pm25']:.1f}<br>"
                   f"Score: "
                   f"{row['priority_score']:.3f}"),
            icon=folium.Icon(color="blue",
                             icon="star")
        ).add_to(m)
    return m

def make_all_cities_map(aqi_df):
    # Overview map showing all 50 stations
    m = folium.Map(location=[20.0, 78.0],
                   zoom_start=5,
                   tiles="CartoDB positron")

    city_colors = {
        "Delhi":     "red",
        "Chennai":   "orange",
        "Bengaluru": "green",
        "Hyderabad": "blue",
        "Jaipur":    "purple"
    }

    for station, grp in aqi_df.groupby("station"):
        city = grp["city"].iloc[0]
        avg  = grp["pm25"].mean()
        folium.CircleMarker(
            location=[grp["lat"].iloc[0],
                      grp["lon"].iloc[0]],
            radius=8, color="black", fill=True,
            fill_color=city_colors[city],
            fill_opacity=0.8,
            popup=(f"<b>{station}</b><br>"
                   f"City: {city}<br>"
                   f"Avg PM2.5: {avg:.1f}")
        ).add_to(m)
    return m

def compute_sensor_recommendations(city, pred_df,
                                    aqi_df):
    # Compute top 3 sensor placement recommendations
    from scipy.spatial.distance import cdist
    import numpy as np

    existing = aqi_df[aqi_df["city"]==city][
        ["lat","lon"]].drop_duplicates().values
    candidate = pred_df.copy()

    distances = cdist(
        candidate[["lat","lon"]].values,
        existing, metric="euclidean")
    candidate["min_dist"] = distances.min(axis=1)

    candidate["norm_pm25"] = (
        (candidate["predicted_pm25"] -
         candidate["predicted_pm25"].min()) /
        (candidate["predicted_pm25"].max() -
         candidate["predicted_pm25"].min()))
    candidate["norm_dist"] = (
        (candidate["min_dist"] -
         candidate["min_dist"].min()) /
        (candidate["min_dist"].max() -
         candidate["min_dist"].min()))
    candidate["priority_score"] = (
        0.6 * candidate["norm_pm25"] +
        0.4 * candidate["norm_dist"])

    sorted_c = candidate.sort_values(
        "priority_score", ascending=False
    ).reset_index(drop=True)

    selected = []
    for _, row in sorted_c.iterrows():
        if len(selected) == 3:
            break
        if len(selected) == 0:
            selected.append(row)
        else:
            sel_coords = np.array(
                [[r["lat"], r["lon"]]
                 for r in selected])
            cur = np.array([[row["lat"], row["lon"]]])
            if cdist(cur, sel_coords).min() > 0.03:
                selected.append(row)

    return pd.DataFrame(selected).reset_index(
        drop=True)