# Chart generation utility functions
# Creates Plotly and Matplotlib charts

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.predictions import pm25_to_category

def make_trend_chart(aqi_df, city):
    # 90 day PM2.5 trend with rolling average
    city_data = aqi_df[aqi_df["city"]==city].copy()
    city_data["date"] = city_data[
        "timestamp"].dt.date
    daily = city_data.groupby("date")["pm25"].agg(
        ["mean","min","max"]).reset_index()
    daily["date"]    = pd.to_datetime(daily["date"])
    daily["rolling"] = daily["mean"].rolling(
        7, center=True).mean()

    fig = go.Figure()

    # Min-max band
    fig.add_trace(go.Scatter(
        x=pd.concat([daily["date"],
                     daily["date"][::-1]]),
        y=pd.concat([daily["max"],
                     daily["min"][::-1]]),
        fill="toself",
        fillcolor="rgba(99,110,250,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Daily range"
    ))

    # Rolling average
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["rolling"],
        mode="lines",
        line=dict(color="#636EFA", width=2.5),
        name="7-day avg"
    ))

    # Reference lines
    fig.add_hline(y=15, line_dash="dash",
                  line_color="green",
                  annotation_text="WHO limit (15)")
    fig.add_hline(y=60, line_dash="dash",
                  line_color="orange",
                  annotation_text="India standard (60)")

    fig.update_layout(
        title=f"{city} PM2.5 Trend — Jan-Mar 2019",
        xaxis_title="Date",
        yaxis_title="PM2.5 (µg/m³)",
        template="plotly_dark",
        height=350,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def make_city_comparison_chart(aqi_df):
    # Bar chart comparing all 5 cities
    city_avg = aqi_df.groupby("city")[
        "pm25"].mean().reset_index()
    city_avg = city_avg.sort_values(
        "pm25", ascending=False)
    city_avg["category"] = city_avg["pm25"].apply(
        pm25_to_category)

    color_map = {
        "Delhi":     "#e74c3c",
        "Jaipur":    "#9b59b6",
        "Hyderabad": "#3498db",
        "Chennai":   "#f39c12",
        "Bengaluru": "#2ecc71"
    }
    colors = [color_map.get(c, "#95a5a6")
              for c in city_avg["city"]]

    fig = go.Figure(go.Bar(
        x=city_avg["city"],
        y=city_avg["pm25"],
        marker_color=colors,
        text=city_avg["pm25"].round(1),
        textposition="outside"
    ))
    fig.add_hline(y=15, line_dash="dash",
                  line_color="green",
                  annotation_text="WHO (15)")
    fig.add_hline(y=60, line_dash="dash",
                  line_color="orange",
                  annotation_text="India (60)")
    fig.update_layout(
        title="Average PM2.5 by City",
        yaxis_title="PM2.5 (µg/m³)",
        template="plotly_dark",
        height=350,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def make_hourly_chart(aqi_df, city):
    # PM2.5 by hour showing rush hour effect
    city_data = aqi_df[aqi_df["city"]==city]
    hourly    = city_data.groupby(
        "hour")["pm25"].mean().reset_index()
    hourly["is_rush"] = hourly["hour"].isin(
        [8,9,18,19])

    colors = ["#e74c3c" if r else "#3498db"
              for r in hourly["is_rush"]]

    fig = go.Figure(go.Bar(
        x=hourly["hour"],
        y=hourly["pm25"],
        marker_color=colors,
        text=hourly["pm25"].round(1),
        textposition="outside"
    ))
    fig.add_hline(y=hourly["pm25"].mean(),
                  line_dash="dash",
                  line_color="white",
                  annotation_text="Mean")
    fig.update_layout(
        title=f"{city} — PM2.5 by Hour of Day<br>"
              f"<sup>Red = Rush hours (8-9am, 6-7pm)"
              f"</sup>",
        xaxis_title="Hour",
        yaxis_title="PM2.5 (µg/m³)",
        template="plotly_dark",
        height=350,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def make_seasonal_chart(aqi_df, city):
    # Winter vs Spring comparison
    city_data = aqi_df[aqi_df["city"]==city]
    monthly   = city_data.groupby(
        "month")["pm25"].mean()
    month_map = {1:"January",2:"February",3:"March"}

    fig = go.Figure()
    colors = ["#e74c3c","#f39c12","#2ecc71"]
    for i, (month, avg) in enumerate(monthly.items()):
        fig.add_trace(go.Bar(
            x=[month_map[month]],
            y=[avg],
            name=month_map[month],
            marker_color=colors[i],
            text=[f"{avg:.1f}"],
            textposition="outside"
        ))

    fig.add_hline(y=60, line_dash="dash",
                  line_color="orange",
                  annotation_text="India standard")
    fig.update_layout(
        title=f"{city} — Monthly PM2.5 Average",
        yaxis_title="PM2.5 (µg/m³)",
        template="plotly_dark",
        height=350,
        showlegend=False,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def make_station_chart(aqi_df, city):
    # PM2.5 by station
    city_data   = aqi_df[aqi_df["city"]==city]
    station_avg = city_data.groupby(
        "station")["pm25"].mean().sort_values(
        ascending=False).reset_index()
    city_mean   = city_data["pm25"].mean()

    colors = ["#e74c3c" if v > city_mean
              else "#3498db"
              for v in station_avg["pm25"]]

    fig = go.Figure(go.Bar(
        x=station_avg["station"],
        y=station_avg["pm25"],
        marker_color=colors,
        text=station_avg["pm25"].round(1),
        textposition="outside"
    ))
    fig.add_hline(y=city_mean, line_dash="dash",
                  line_color="white",
                  annotation_text=f"City mean: "
                  f"{city_mean:.1f}")
    fig.update_layout(
        title=f"{city} — PM2.5 by Station<br>"
              f"<sup>Red = above city mean</sup>",
        xaxis_title="Station",
        yaxis_title="PM2.5 (µg/m³)",
        template="plotly_dark",
        height=380,
        margin=dict(l=40, r=40, t=60, b=100)
    )
    fig.update_xaxes(tickangle=45)
    return fig

def make_feature_importance_chart(rf_model,
                                   features):
    # Feature importance bar chart
    imp = pd.DataFrame({
        "feature":    features,
        "importance": rf_model.feature_importances_
    }).sort_values("importance", ascending=True)

    colors = ["#e74c3c" if i >= len(imp)-3
              else "#3498db"
              for i in range(len(imp))]

    fig = go.Figure(go.Bar(
        x=imp["importance"],
        y=imp["feature"],
        orientation="h",
        marker_color=colors,
        text=imp["importance"].round(3),
        textposition="outside"
    ))
    fig.update_layout(
        title="Feature Importance — Random Forest<br>"
              "<sup>Red = Top 3 most important"
              "</sup>",
        xaxis_title="Importance Score",
        template="plotly_dark",
        height=500,
        margin=dict(l=120, r=60, t=60, b=40)
    )
    return fig

def make_loso_chart(loso_df):
    # LOSO validation results
    city_colors = {
        "Delhi":     "#e74c3c",
        "Chennai":   "#f39c12",
        "Bengaluru": "#2ecc71",
        "Hyderabad": "#3498db",
        "Jaipur":    "#9b59b6"
    }
    colors = [city_colors.get(c, "#95a5a6")
              for c in loso_df["city"]]

    fig = go.Figure(go.Bar(
        x=loso_df["station"],
        y=loso_df["R2"],
        marker_color=colors,
        text=loso_df["R2"].round(3),
        textposition="outside"
    ))
    fig.add_hline(y=loso_df["R2"].mean(),
                  line_dash="dash",
                  line_color="white",
                  annotation_text=f"Mean R²: "
                  f"{loso_df['R2'].mean():.3f}")
    fig.update_layout(
        title="Leave-One-Station-Out Validation<br>"
              "<sup>R² per station across 5 cities"
              "</sup>",
        xaxis_title="Station",
        yaxis_title="R²",
        template="plotly_dark",
        height=400,
        margin=dict(l=40, r=40, t=60, b=120)
    )
    fig.update_xaxes(tickangle=90, tickfont_size=9)
    return fig

def make_correlation_chart(aqi_df, features):
    # Correlation heatmap
    corr = aqi_df[features + ["pm25"]].corr()
    fig  = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Feature Correlation Matrix"
    )
    fig.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig