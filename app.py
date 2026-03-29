"""
app.py — Streamlit UI for Mess Food Quality Predictor
======================================================
Run: streamlit run app.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ──────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mess Food Quality Predictor",
    page_icon="🍛",
    layout="centered"
)

# ──────────────────────────────────────────────────────
# Load model & encoders (cached so they load only once)
# ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

@st.cache_data
def load_data():
    return pd.read_csv("data/mess_food_data.csv")


# Check that the model has been trained
if not os.path.exists("models/best_model.pkl"):
    st.error("⚠️ Model not found. Please run `python train_model.py` first.")
    st.stop()

model, encoders = load_artifacts()
df = load_data()

# ──────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────
st.markdown("## 🍛 Mess Food Quality Predictor")
st.markdown(
    "Predict the expected food rating **(out of 10)** for your next meal "
    "based on the day, meal type, and food item."
)
st.divider()

# ──────────────────────────────────────────────────────
# Input Section
# ──────────────────────────────────────────────────────
st.subheader("📋 Enter Meal Details")

col1, col2 = st.columns(2)

with col1:
    day = st.selectbox(
        "Day of the Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday",
         "Friday", "Saturday", "Sunday"]
    )
    meal = st.selectbox(
        "Meal Type",
        ["Breakfast", "Lunch", "Dinner"]
    )

# Food items depend on meal type
meal_foods = {
    "Breakfast": ["Poha", "Upma", "Idli-Sambar", "Paratha", "Bread-Butter", "Aloo Puri"],
    "Lunch":     ["Dal Rice", "Rajma Rice", "Chole Bhature", "Veg Biryani", "Roti Sabzi", "Paneer Curry"],
    "Dinner":    ["Khichdi", "Dal Tadka", "Palak Paneel", "Aloo Matar", "Fried Rice", "Chapati Sabzi"]
}
# Fix typo for consistency with training data
meal_foods["Dinner"][2] = "Palak Paneer"

with col2:
    food_item = st.selectbox("Food Item", meal_foods[meal])

# ──────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────
if st.button("🔮 Predict Rating", type="primary", use_container_width=True):
    try:
        day_enc  = encoders["Day"].transform([day])[0]
        meal_enc = encoders["Meal"].transform([meal])[0]
        food_enc = encoders["Food_Item"].transform([food_item])[0]
        X_input  = np.array([[day_enc, meal_enc, food_enc]])
        rating   = round(float(model.predict(X_input)[0]), 1)
        rating   = max(1.0, min(10.0, rating))

        st.divider()
        st.subheader("🎯 Prediction Result")

        # Colour the metric card based on rating
        if rating >= 8.5:
            verdict, color = "🌟 Excellent!", "#1D9E75"
        elif rating >= 7.0:
            verdict, color = "😊 Good",       "#3B6D11"
        elif rating >= 5.5:
            verdict, color = "😐 Average",    "#BA7517"
        else:
            verdict, color = "😞 Below Average", "#A32D2D"

        st.markdown(f"""
        <div style="
            background: #F9F9F6;
            border-radius: 12px;
            border: 1.5px solid {color};
            padding: 1.5rem 2rem;
            text-align: center;
        ">
            <p style="font-size:1rem; color:#666; margin:0">{day} · {meal} · {food_item}</p>
            <h1 style="font-size:3rem; color:{color}; margin:0.3rem 0">{rating} / 10</h1>
            <p style="font-size:1.3rem; margin:0">{verdict}</p>
        </div>
        """, unsafe_allow_html=True)

        # Progress bar visual
        st.markdown("")
        st.progress(rating / 10)

    except ValueError as e:
        st.error(f"Prediction failed: {e}. Make sure you ran train_model.py with this food item.")

st.divider()

# ──────────────────────────────────────────────────────
# Charts Section
# ──────────────────────────────────────────────────────
st.subheader("📊 Data Insights")
tab1, tab2, tab3 = st.tabs(["By Day", "By Meal", "Top Food Items"])

with tab1:
    avg_day = df.groupby("Day")["Rating"].mean().reindex(
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    )
    fig, ax = plt.subplots(figsize=(8, 3.5))
    colors = ["#7F77DD" if d in ("Saturday","Sunday") else "#9FE1CB" for d in avg_day.index]
    ax.bar(avg_day.index, avg_day.values, color=colors, edgecolor="white")
    ax.set_ylim(0, 10)
    ax.set_ylabel("Avg Rating")
    ax.set_title("Average Rating by Day (purple = weekend)", fontsize=11)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    avg_meal = df.groupby("Meal")["Rating"].mean()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(avg_meal.index, avg_meal.values,
           color=["#FAC775","#5DCAA5","#F0997B"], edgecolor="white", width=0.5)
    ax.set_ylim(0, 10)
    ax.set_ylabel("Avg Rating")
    ax.set_title("Average Rating by Meal Type", fontsize=11)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    top_foods = (
        df.groupby("Food_Item")["Rating"]
          .mean()
          .sort_values(ascending=True)
          .tail(10)
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top_foods.index, top_foods.values, color="#7F77DD", edgecolor="white")
    ax.set_xlabel("Avg Rating")
    ax.set_title("Top 10 Food Items by Rating", fontsize=11)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

# ──────────────────────────────────────────────────────
# Raw data expander
# ──────────────────────────────────────────────────────
with st.expander("📄 View Raw Dataset"):
    st.dataframe(df, use_container_width=True)

st.markdown("---")
st.caption("Built with ❤️ using Python · scikit-learn · Streamlit")
