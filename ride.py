import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Ride Demand AI + Prediction", layout="wide")

st.title("🚗 Ride Demand Analyzer + Prediction Model")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    if not all(col in df.columns for col in ["latitude", "longitude", "time"]):
        st.error("Dataset must have latitude, longitude, time")
    else:
        # Time processing
        df["time"] = pd.to_datetime(df["time"])
        df["hour"] = df["time"].dt.hour
        df["day"] = df["time"].dt.dayofweek

        # Sidebar
        st.sidebar.header("⚙️ Controls")
        eps = st.sidebar.slider("Epsilon", 0.001, 0.05, 0.01)
        min_samples = st.sidebar.slider("Min Samples", 2, 15, 5)

        # DBSCAN clustering
        coords = df[["latitude", "longitude"]]
        db = DBSCAN(eps=eps, min_samples=min_samples)
        df["cluster"] = db.fit_predict(coords)

        # Aggregate demand per hour
        demand_df = df.groupby(["hour"]).size().reset_index(name="rides")

        # Train ML model
        X = demand_df[["hour"]]
        y = demand_df["rides"]

        model = RandomForestRegressor()
        model.fit(X, y)

        # Predict for full day
        future_hours = pd.DataFrame({"hour": range(24)})
        predictions = model.predict(future_hours)

        future_hours["predicted_rides"] = predictions

        # 📊 Actual vs Predicted
        st.subheader("📊 Demand Prediction")

        fig = px.line(title="Actual vs Predicted Demand")

        fig.add_scatter(x=demand_df["hour"], y=demand_df["rides"],
                        mode='lines+markers', name='Actual')

        fig.add_scatter(x=future_hours["hour"], y=future_hours["predicted_rides"],
                        mode='lines', name='Predicted')

        st.plotly_chart(fig, use_container_width=True)

        # 🔮 Peak hour detection
        peak_hour = future_hours.loc[future_hours["predicted_rides"].idxmax()]

        st.success(f"🔥 Predicted Peak Hour: {int(peak_hour['hour'])}:00 with ~{int(peak_hour['predicted_rides'])} rides")

        # 🗺️ Map
        st.subheader("🗺️ Cluster Map")

        m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=12)

        for _, row in df.iterrows():
            color = "red" if row["cluster"] == -1 else "blue"

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=4,
                color=color,
                fill=True
            ).add_to(m)

        st_folium(m, width=1000, height=500)

        # 📈 Hourly demand chart
        st.subheader("📈 Actual Demand")

        fig2 = px.bar(demand_df, x="hour", y="rides", title="Ride Requests per Hour")
        st.plotly_chart(fig2, use_container_width=True)

        # Insights
        st.subheader("📌 Insights")

        st.write(f"""
        - Model used: Random Forest Regressor  
        - Peak predicted hour: {int(peak_hour['hour'])}:00  
        
        🚀 Business Use:
        - Deploy more drivers during peak hours  
        - Reduce idle drivers during low demand  
        - Improve customer wait time  
        """)

else:
    st.info("Upload dataset to start")