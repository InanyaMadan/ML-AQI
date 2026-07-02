# AQI Dashboard — Multi City Air Quality Prediction
# Built with Streamlit, Folium, Plotly
# Data: Real CPCB measurements + Open-Meteo weather
# Model: Random Forest R2=0.9873

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings("ignore")

from utils.predictions import (
    load_models, get_city_encoded_map,
    pm25_to_category, pm25_to_color,
    predict_single, load_pred_grid,
    load_master_data, load_station_means,
    load_loso_results
)
from utils.maps import (
    make_heatmap, add_station_markers,
    make_health_risk_map, make_sensor_map,
    make_all_cities_map,
    compute_sensor_recommendations
)
from utils.charts import (
    make_trend_chart, make_city_comparison_chart,
    make_hourly_chart, make_seasonal_chart,
    make_station_chart, make_feature_importance_chart,
    make_loso_chart, make_correlation_chart
)

# Page configuration
st.set_page_config(
    page_title="AQI India Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
css_path = os.path.join(
    os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>",
                    unsafe_allow_html=True)

# Load all data and models
@st.cache_resource
def load_all():
    rf_model, scaler, le, features = load_models()
    aqi_df        = load_master_data()
    station_means = load_station_means()
    loso_df       = load_loso_results()
    city_enc_map  = get_city_encoded_map(le)
    pred_grids    = {
        city: load_pred_grid(city)
        for city in ["Delhi", "Chennai", "Bengaluru",
                     "Hyderabad", "Jaipur"]
    }
    return (rf_model, scaler, le, features,
            aqi_df, station_means, loso_df,
            city_enc_map, pred_grids)

(rf_model, scaler, le, features, aqi_df,
 station_means, loso_df, city_enc_map,
 pred_grids) = load_all()

AQI_COLORS = {
    "Good":         "#00e400",
    "Satisfactory": "#92d050",
    "Moderate":     "#ffff00",
    "Poor":         "#ff7e00",
    "Very Poor":    "#ff0000",
    "Severe":       "#7e0023"
}

CITIES = ["Delhi", "Chennai", "Bengaluru",
          "Hyderabad", "Jaipur"]

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌫️ AQI Dashboard")
    st.markdown(
        "**Hyper-Local Air Quality**  \n"
        "Prediction for Indian Cities")
    st.divider()

    st.markdown("### 🏙️ Select City")
    selected_city = st.selectbox(
        "City", CITIES,
        label_visibility="collapsed")

    st.divider()
    st.markdown("### 📊 Data Info")
    st.markdown("**Source:** Real CPCB Data")
    st.markdown("**Period:** Jan-Mar 2023")
    st.markdown("**Cities:** 5")
    st.markdown("**Stations:** 50")
    st.markdown("**Model R²:** 0.9873")
    st.markdown("**Algorithm:** Random Forest")
    st.divider()

    st.markdown("### ⚠️ AQI Scale")
    for cat, color in AQI_COLORS.items():
        st.markdown(
            f'<div style="display:flex;'
            f'align-items:center;gap:8px;'
            f'margin:4px 0;">'
            f'<div style="width:16px;height:16px;'
            f'background:{color};'
            f'border-radius:3px;"></div>'
            f'<span style="font-size:0.85rem;">'
            f'{cat}</span></div>',
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("# 🌫️ India Air Quality Intelligence")
st.markdown(
    f"**{selected_city}** — Hyper-local PM2.5 "
    f"prediction using Machine Learning  |  "
    f"Real CPCB Data + Open-Meteo Weather")
st.divider()

# ─────────────────────────────────────────
# KEY METRICS ROW
# ─────────────────────────────────────────
city_data   = aqi_df[aqi_df["city"] == selected_city]
avg_pm25    = city_data["pm25"].mean()
max_pm25    = city_data["pm25"].max()
who_exceed  = avg_pm25 / 15
category    = pm25_to_category(avg_pm25)
winter_avg  = city_data[
    city_data["month"] <= 2]["pm25"].mean()
spring_avg  = city_data[
    city_data["month"] == 3]["pm25"].mean()
improvement = (winter_avg - spring_avg) / winter_avg * 100

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Avg PM2.5", f"{avg_pm25:.1f} µg/m³")
with col2:
    st.metric("AQI Category", category)
with col3:
    st.metric("WHO Exceedance", f"{who_exceed:.1f}x")
with col4:
    st.metric("Peak PM2.5", f"{max_pm25:.0f} µg/m³")
with col5:
    delta_str = (
        f"-{improvement:.1f}%"
        if improvement > 0
        else f"+{abs(improvement):.1f}%"
    )
    st.metric("Winter→Spring", delta_str,
              delta=delta_str)

st.divider()

# ─────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────
tabs = st.tabs([
    "🗺️ Pollution Maps",
    "📈 Trends & EDA",
    "🤖 ML Models",
    "⚕️ Health Risk",
    "📡 Sensor Placement",
    "🎛️ What-If Simulator",
    "🏙️ City Comparison"
])

# ─────────────────────────────────────────
# TAB 1 — POLLUTION MAPS
# ─────────────────────────────────────────
with tabs[0]:
    st.markdown(f"### 🗺️ {selected_city} Pollution Heatmap")
    st.markdown(
        "ML-predicted PM2.5 across the city including "
        "unmonitored areas. Markers show real CPCB "
        "monitoring stations.")

    map_type = st.radio(
        "Map type",
        ["ML Prediction", "IDW Baseline"],
        horizontal=True)

    pred_df = pred_grids[selected_city]

    if map_type == "ML Prediction":
        m = make_heatmap(selected_city, pred_df)
        m = add_station_markers(m, selected_city, aqi_df)
        caption = "Random Forest prediction — R²=0.9873"
    else:
        from utils.maps import CITY_CENTERS, CITY_ZOOM
        from folium.plugins import HeatMap as FHM

        def idw_fn(tlat, tlon, df, power=2):
            d = np.sqrt(
                (df["lat"] - tlat)**2 +
                (df["lon"] - tlon)**2
            ).replace(0, 1e-10)
            w = 1 / d**power
            return (w * df["pm25"]).sum() / w.sum()

        city_bounds = {
            "Delhi":     {"lat": (28.50, 28.78),
                          "lon": (77.00, 77.38)},
            "Chennai":   {"lat": (12.85, 13.20),
                          "lon": (80.10, 80.35)},
            "Bengaluru": {"lat": (12.75, 13.15),
                          "lon": (77.45, 77.80)},
            "Hyderabad": {"lat": (17.20, 17.60),
                          "lon": (78.25, 78.65)},
            "Jaipur":    {"lat": (26.75, 27.00),
                          "lon": (75.70, 75.90)},
        }
        bounds   = city_bounds[selected_city]
        lats     = np.linspace(
            bounds["lat"][0], bounds["lat"][1], 25)
        lons     = np.linspace(
            bounds["lon"][0], bounds["lon"][1], 25)
        sm       = station_means[
            station_means["city"] == selected_city]
        idw_data = []
        for la in lats:
            for lo in lons:
                est = idw_fn(la, lo, sm)
                idw_data.append([la, lo, est])

        idw_df = pd.DataFrame(
            idw_data, columns=["lat", "lon", "aqi"])
        min_v  = idw_df["aqi"].min()
        max_v  = idw_df["aqi"].max()

        m = folium.Map(
            location=CITY_CENTERS[selected_city],
            zoom_start=CITY_ZOOM[selected_city],
            tiles="CartoDB positron")
        heat = [
            [r["lat"], r["lon"],
             (r["aqi"] - min_v) / (max_v - min_v)]
            for _, r in idw_df.iterrows()
        ]
        FHM(heat, radius=25, blur=20,
            min_opacity=0.3,
            gradient={"0.0": "green", "0.4": "yellow",
                      "0.7": "orange", "1.0": "red"}
            ).add_to(m)
        m = add_station_markers(m, selected_city, aqi_df)
        caption = "IDW spatial interpolation baseline"

    st_folium(m, width=900, height=500)
    st.caption(caption)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        worst = pred_df.loc[
            pred_df["predicted_pm25"].idxmax()]
        st.metric(
            "Highest predicted location",
            f"{worst['predicted_pm25']:.1f} µg/m³",
            delta=pm25_to_category(
                worst["predicted_pm25"]))
    with c2:
        best = pred_df.loc[
            pred_df["predicted_pm25"].idxmin()]
        st.metric(
            "Lowest predicted location",
            f"{best['predicted_pm25']:.1f} µg/m³",
            delta=pm25_to_category(
                best["predicted_pm25"]))
    with c3:
        pct_above_india = (
            pred_df["predicted_pm25"] > 60
        ).mean() * 100
        st.metric(
            "Area exceeding India standard",
            f"{pct_above_india:.1f}%")

# ─────────────────────────────────────────
# TAB 2 — TRENDS AND EDA
# ─────────────────────────────────────────
with tabs[1]:
    st.markdown(
        f"### 📈 {selected_city} Air Quality Analysis")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(
            make_trend_chart(aqi_df, selected_city),
            use_container_width=True)
    with r1c2:
        st.plotly_chart(
            make_hourly_chart(aqi_df, selected_city),
            use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.plotly_chart(
            make_seasonal_chart(aqi_df, selected_city),
            use_container_width=True)
    with r2c2:
        st.plotly_chart(
            make_station_chart(aqi_df, selected_city),
            use_container_width=True)

    st.markdown("### 🔗 Feature Correlation")
    st.plotly_chart(
        make_correlation_chart(aqi_df, features),
        use_container_width=True)

# ─────────────────────────────────────────
# TAB 3 — ML MODELS
# ─────────────────────────────────────────
with tabs[2]:
    st.markdown("### 🤖 Machine Learning Models")

    model_results = {
        "Random Forest":    {"MAE": 4.25,  "RMSE": 6.69,  "R2": 0.9873},
        "XGBoost":          {"MAE": 6.97,  "RMSE": 10.19, "R2": 0.9707},
        "SVR":              {"MAE": 9.31,  "RMSE": 14.90, "R2": 0.9373},
        "Ridge Regression": {"MAE": 14.30, "RMSE": 20.65, "R2": 0.8795},
        "Linear Regression":{"MAE": 14.30, "RMSE": 20.65, "R2": 0.8795},
    }

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric("Best Model", "Random Forest")
    with mc2:
        st.metric("Best R²", "0.9873")
    with mc3:
        st.metric("Best MAE", "4.25 µg/m³")

    st.markdown("#### Model Comparison Table")
    results_df = pd.DataFrame(model_results).T
    results_df.index.name = "Model"
    results_df = results_df.reset_index()
    st.dataframe(
        results_df.style.highlight_max(
            subset=["R2"], color="#2ecc71"
        ).highlight_min(
            subset=["MAE", "RMSE"], color="#2ecc71"
        ),
        use_container_width=True,
        hide_index=True
    )

    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown("#### Feature Importance")
        st.plotly_chart(
            make_feature_importance_chart(
                rf_model, features),
            use_container_width=True)
    with fc2:
        st.markdown("#### LOSO Spatial Validation")
        st.plotly_chart(
            make_loso_chart(loso_df),
            use_container_width=True)
        st.markdown("**LOSO Summary:**")
        loso_city = loso_df.groupby(
            "city")["R2"].mean().reset_index()
        loso_city.columns = ["City", "Avg R²"]
        loso_city["Avg R²"] = loso_city[
            "Avg R²"].round(4)
        st.dataframe(loso_city,
                     use_container_width=True,
                     hide_index=True)

    st.markdown("#### Why These Algorithms?")
    algo_data = {
        "Algorithm": [
            "Linear Regression", "Ridge Regression",
            "SVR", "Random Forest",
            "XGBoost", "KMeans", "IDW"],
        "Family": [
            "Linear", "Linear+Reg", "Kernel",
            "Tree Ensemble", "Tree Ensemble",
            "Clustering", "Spatial"],
        "Purpose": [
            "Simplest baseline",
            "Tests regularization",
            "Non-linear kernel method",
            "Main model — best accuracy",
            "State of art boosting",
            "Zone detection",
            "Spatial baseline"],
        "R²": [
            "0.8795", "0.8795", "0.9373",
            "0.9873", "0.9707", "—", "—"]
    }
    st.dataframe(
        pd.DataFrame(algo_data),
        use_container_width=True,
        hide_index=True)

# ─────────────────────────────────────────
# TAB 4 — HEALTH RISK
# ─────────────────────────────────────────
with tabs[3]:
    st.markdown(
        f"### ⚕️ {selected_city} Health Risk Analysis")

    pred_df = pred_grids[selected_city]

    hc1, hc2, hc3 = st.columns(3)
    with hc1:
        who_pct = (
            pred_df["predicted_pm25"] > 15
        ).mean() * 100
        st.metric("Exceeds WHO limit",
                  f"{who_pct:.1f}%",
                  delta="of city area")
    with hc2:
        india_pct = (
            pred_df["predicted_pm25"] > 60
        ).mean() * 100
        st.metric("Exceeds India standard",
                  f"{india_pct:.1f}%",
                  delta="of city area")
    with hc3:
        avg_who = (pred_df["predicted_pm25"].mean()
                   / 15)
        st.metric("Avg WHO exceedance",
                  f"{avg_who:.1f}x",
                  delta="above safe limit")

    st.markdown("#### Health Risk Map")
    st.markdown(
        "Color coded by WHO exceedance level. "
        "Click any point for details.")

    m_health = make_health_risk_map(
        selected_city, pred_df)
    m_health = add_station_markers(
        m_health, selected_city, aqi_df)

    legend_html = """
    <div style="position:fixed;bottom:40px;left:40px;
         z-index:1000;background:white;padding:12px;
         border-radius:8px;border:2px solid grey;
         font-size:11px;line-height:1.8;">
      <b>Health Risk</b><br>
      <span style="color:#00e400">&#9679;</span>
      Within WHO (&le;15)<br>
      <span style="color:#92d050">&#9679;</span>
      Moderate (&le;30)<br>
      <span style="color:#ffcc00">&#9679;</span>
      India limit (&le;60)<br>
      <span style="color:#ff7e00">&#9679;</span>
      Exceeds India (&le;90)<br>
      <span style="color:#ff0000">&#9679;</span>
      Poor (&le;120)<br>
      <span style="color:#7e0023">&#9679;</span>
      Severe (&gt;120)
    </div>"""
    m_health.get_root().html.add_child(
        folium.Element(legend_html))
    st_folium(m_health, width=900, height=480)

    st.markdown("#### Top 10 Pollution Hotspots")
    hotspots = pred_df.nlargest(
        10, "predicted_pm25")[
        ["lat", "lon", "predicted_pm25"]
    ].copy()
    hotspots["category"] = hotspots[
        "predicted_pm25"].apply(pm25_to_category)
    hotspots["WHO exceedance"] = (
        hotspots["predicted_pm25"] / 15
    ).round(1).astype(str) + "x"
    hotspots["predicted_pm25"] = hotspots[
        "predicted_pm25"].round(1)
    hotspots.index = range(1, len(hotspots) + 1)
    hotspots.index.name = "Rank"
    st.dataframe(
        hotspots.rename(columns={
            "lat":            "Latitude",
            "lon":            "Longitude",
            "predicted_pm25": "PM2.5 (µg/m³)",
            "category":       "AQI Category"
        }),
        use_container_width=True)

# ─────────────────────────────────────────
# TAB 5 — SENSOR PLACEMENT
# ─────────────────────────────────────────
with tabs[4]:
    st.markdown(
        f"### 📡 {selected_city} "
        f"Sensor Placement Recommender")
    st.markdown(
        "Recommends the top 3 locations for new "
        "air quality sensors based on pollution "
        "severity and distance from existing stations.")

    pred_df    = pred_grids[selected_city]
    sensor_rec = compute_sensor_recommendations(
        selected_city, pred_df, aqi_df)

    sc1, sc2, sc3 = st.columns(3)
    for i, (_, row) in enumerate(
            sensor_rec.iterrows()):
        col = [sc1, sc2, sc3][i]
        with col:
            cat = pm25_to_category(
                row["predicted_pm25"])
            st.metric(
                f"Recommendation #{i+1}",
                f"{row['predicted_pm25']:.1f} µg/m³",
                delta=cat)
            st.markdown(
                f"📍 `{row['lat']:.4f}, "
                f"{row['lon']:.4f}`")
            st.markdown(
                f"**Priority score:** "
                f"`{row['priority_score']:.3f}`")

    st.markdown("#### Sensor Placement Map")
    st.markdown(
        "⚪ White = existing stations  |  "
        "⭐ Blue = recommended new sensors")

    m_sensor = make_sensor_map(
        selected_city, pred_df, aqi_df, sensor_rec)
    st_folium(m_sensor, width=900, height=480)

    st.markdown("#### Priority Score Formula")
    st.code(
        "Priority Score = 0.6 × Pollution Severity\n"
        "               + 0.4 × Distance from "
        "Existing Sensors",
        language=None)
    st.markdown(
        "Locations with **high pollution** AND "
        "**far from existing sensors** score highest "
        "— these are the most critical monitoring gaps.")

    if selected_city == "Hyderabad":
        st.warning(
            "⚠️ **Note:** Patancheruvu station in "
            "Hyderabad showed R²=-0.097 in LOSO "
            "validation — the model struggles there "
            "due to geographic isolation. This is a "
            "critical location for a new sensor.")

# ─────────────────────────────────────────
# TAB 6 — WHAT-IF SIMULATOR
# ─────────────────────────────────────────
with tabs[5]:
    st.markdown("### 🎛️ What-If Scenario Simulator")
    st.markdown(
        "Adjust weather and pollution conditions "
        "to see predicted PM2.5 in real time.")

    sim_col1, sim_col2 = st.columns([1, 1])

    with sim_col1:
        st.markdown("#### ⚙️ Adjust Conditions")

        sim_city = st.selectbox(
            "City", CITIES, key="sim_city")
        temperature = st.slider(
            "🌡️ Temperature (°C)", 5, 45, 15)
        humidity = st.slider(
            "💧 Humidity (%)", 20, 100, 65)
        wind_speed = st.slider(
            "💨 Wind Speed (m/s)",
            0.0, 15.0, 2.5)
        hour = st.slider(
            "🕐 Hour of Day", 0, 23, 12)
        day_of_year = st.slider(
            "📅 Day (1=Jan1, 90=Mar31)", 1, 90, 45)
        pm10_default = 200 if sim_city == "Delhi" \
                       else 100
        pm10 = st.slider(
            "🏭 PM10 (µg/m³)", 10, 500,
            pm10_default)

        is_rush    = hour in [8, 9, 18, 19]
        is_winter  = day_of_year <= 60
        rush_str   = ("🚗 Rush Hour"
                      if is_rush else "🌙 Off-Peak")
        season_str = ("❄️ Winter"
                      if is_winter else "🌸 Spring")
        st.markdown(
            f"**{season_str}** | **{rush_str}**")

    with sim_col2:
        st.markdown("#### 📊 Prediction Result")

        pm25_pred = predict_single(
            sim_city, temperature, humidity,
            wind_speed, hour, day_of_year,
            pm10, rf_model, scaler, features,
            city_enc_map, aqi_df)

        cat   = pm25_to_category(pm25_pred)
        color = pm25_to_color(pm25_pred)
        who_x = pm25_pred / 15
        india = pm25_pred > 60

        st.markdown(
            f'<div style="background:#1a2035;'
            f'border-radius:16px;padding:24px;'
            f'text-align:center;'
            f'border:2px solid {color};">'
            f'<div style="font-size:3rem;'
            f'font-weight:800;color:{color};">'
            f'{pm25_pred:.1f}</div>'
            f'<div style="font-size:1rem;'
            f'color:#8892a4;margin-top:4px;">'
            f'µg/m³ PM2.5</div>'
            f'<div style="margin-top:16px;'
            f'padding:8px 20px;'
            f'background:{color}22;'
            f'border-radius:20px;'
            f'display:inline-block;'
            f'color:{color};font-weight:700;'
            f'font-size:1.1rem;">{cat}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown("")
        wm1, wm2 = st.columns(2)
        with wm1:
            st.metric("WHO Exceedance",
                      f"{who_x:.1f}x")
        with wm2:
            st.metric(
                "India Standard",
                "❌ Exceeds" if india
                else "✅ Within")

        st.markdown("**Pollution Level:**")
        st.progress(min(pm25_pred / 300, 1.0))
        st.markdown(
            "`0` ────────────────── `300+ µg/m³`")

        st.markdown("---")
        st.markdown("#### Quick Scenarios")
        qs1, qs2, qs3 = st.columns(3)
        scenarios = [
            ("Rush Hour",   8,  2.5),
            ("Strong Wind", 12, 12.0),
            ("Spring Day",  45, 2.5),
        ]
        for col, (label, sc_day, sc_wind) in zip(
                [qs1, qs2, qs3], scenarios):
            p = predict_single(
                sim_city, temperature, humidity,
                sc_wind, hour, sc_day,
                pm10, rf_model, scaler, features,
                city_enc_map, aqi_df)
            col.metric(
                label, f"{p:.0f} µg/m³",
                delta=pm25_to_category(p))

# ─────────────────────────────────────────
# TAB 7 — CITY COMPARISON
# ─────────────────────────────────────────
with tabs[6]:
    st.markdown("### 🏙️ Multi-City Comparison")

    st.plotly_chart(
        make_city_comparison_chart(aqi_df),
        use_container_width=True)

    st.markdown("#### City Summary Table")
    summary_rows = []
    for city in CITIES:
        cd        = aqi_df[aqi_df["city"] == city]
        avg       = cd["pm25"].mean()
        who_x     = avg / 15
        india_pct = (cd["pm25"] > 60).mean() * 100
        cat       = pm25_to_category(avg)
        loso_r2   = loso_df[
            loso_df["city"] == city]["R2"].mean()
        w         = cd[cd["month"] <= 2][
            "pm25"].mean()
        s         = cd[cd["month"] == 3][
            "pm25"].mean()
        imp       = (w - s) / w * 100
        summary_rows.append({
            "City":          city,
            "Avg PM2.5":     f"{avg:.1f}",
            "WHO Exceed":    f"{who_x:.1f}x",
            "India Std %":   f"{india_pct:.0f}%",
            "AQI Category":  cat,
            "LOSO R²":       f"{loso_r2:.4f}",
            "Season Improv": f"{imp:.1f}%"
        })
    st.dataframe(
        pd.DataFrame(summary_rows),
        use_container_width=True,
        hide_index=True)

    st.markdown("#### All Cities Station Map")
    m_all = make_all_cities_map(aqi_df)
    st_folium(m_all, width=900, height=500)

    st.markdown("#### LOSO Validation by City")
    loso_city_avg = loso_df.groupby(
        "city")["R2"].agg(
        ["mean", "min", "max"]
    ).round(4).reset_index()
    loso_city_avg.columns = [
        "City", "Mean R²", "Min R²", "Max R²"]
    st.dataframe(loso_city_avg,
                 use_container_width=True,
                 hide_index=True)

    st.markdown("#### Seasonal Comparison")
    seasonal_data = []
    for city in CITIES:
        cd  = aqi_df[aqi_df["city"] == city]
        w   = cd[cd["month"] <= 2]["pm25"].mean()
        s   = cd[cd["month"] == 3]["pm25"].mean()
        seasonal_data.append({
            "City":             city,
            "Winter (Jan-Feb)": round(w, 1),
            "Spring (Mar)":     round(s, 1),
            "Improvement %":    round((w-s)/w*100, 1)
        })

    seas_df = pd.DataFrame(seasonal_data)
    fig_seas = go.Figure()
    fig_seas.add_trace(go.Bar(
        name="Winter (Jan-Feb)",
        x=seas_df["City"],
        y=seas_df["Winter (Jan-Feb)"],
        marker_color="#3498db"))
    fig_seas.add_trace(go.Bar(
        name="Spring (Mar)",
        x=seas_df["City"],
        y=seas_df["Spring (Mar)"],
        marker_color="#2ecc71"))
    fig_seas.add_hline(
        y=15, line_dash="dash",
        line_color="green",
        annotation_text="WHO limit")
    fig_seas.add_hline(
        y=60, line_dash="dash",
        line_color="orange",
        annotation_text="India standard")
    fig_seas.update_layout(
        barmode="group",
        title="Winter vs Spring PM2.5 by City",
        yaxis_title="PM2.5 (µg/m³)",
        template="plotly_dark",
        height=400)
    st.plotly_chart(
        fig_seas, use_container_width=True)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;"
    "color:#8892a4;font-size:0.8rem;'>"
    "AQI India Dashboard  |  "
    "Data: CPCB via Kaggle + Open-Meteo  |  "
    "Model: Random Forest R²=0.9873  |  "
    "5 Cities · 50 Stations · 105,585 readings"
    "</div>",
    unsafe_allow_html=True
)