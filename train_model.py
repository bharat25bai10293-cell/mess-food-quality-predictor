"""
train_model.py
==============
Preprocesses data, trains a Random Forest model, evaluates it,
and saves everything needed for prediction.

Run: python train_model.py
"""

import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
DATA_PATH = "data/mess_food_data.csv"

if not os.path.exists(DATA_PATH):
    print("Dataset not found. Generating it first…")
    exec(open("data/generate_dataset.py").read())

df = pd.read_csv(DATA_PATH)
print("📂 Data loaded:", df.shape)
print(df.head())

# ─────────────────────────────────────────────
# 2. PREPROCESSING — ENCODE CATEGORICAL COLUMNS
# ─────────────────────────────────────────────
# Machine learning models only understand numbers.
# We use LabelEncoder to convert text → integer codes.
# Example: Monday→0, Tuesday→1, …, Sunday→6

encoders = {}          # We'll save these so predictions use the same mapping
df_encoded = df.copy()

for col in ["Day", "Meal", "Food_Item"]:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"  {col} classes: {list(le.classes_)}")

# Features (X) and Target (y)
X = df_encoded[["Day", "Meal", "Food_Item"]]
y = df_encoded["Rating"]

# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
# 80 % for training, 20 % for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n🔀 Train size: {len(X_train)}, Test size: {len(X_test)}")

# ─────────────────────────────────────────────
# 4. TRAIN MULTIPLE MODELS AND COMPARE
# ─────────────────────────────────────────────
models = {
    "Linear Regression":  LinearRegression(),
    "Decision Tree":      DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "model": model}
    print(f"\n📊 {name}")
    print(f"   MAE  = {mae:.3f}  (average error in rating points)")
    print(f"   RMSE = {rmse:.3f}")
    print(f"   R²   = {r2:.3f}  (1.0 = perfect)")

# ─────────────────────────────────────────────
# 5. SELECT BEST MODEL (lowest MAE)
# ─────────────────────────────────────────────
best_name = min(results, key=lambda n: results[n]["MAE"])
best_model = results[best_name]["model"]
print(f"\n🏆 Best model: {best_name}")

# ─────────────────────────────────────────────
# 6. SAVE MODEL & ENCODERS
# ─────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("models/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
print("💾 Model and encoders saved in models/")

# ─────────────────────────────────────────────
# 7. VISUALISATIONS
# ─────────────────────────────────────────────
os.makedirs("plots", exist_ok=True)

# Use a clean style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F9F9F6",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.color":       "#E0DDD5",
    "grid.linewidth":   0.6,
    "font.family":      "sans-serif",
})

# ── Plot 1: Average rating by day ──
avg_by_day = df.groupby("Day")["Rating"].mean().reindex(
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
)
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(avg_by_day.index, avg_by_day.values,
              color=["#7F77DD" if d in ("Saturday","Sunday") else "#9FE1CB"
                     for d in avg_by_day.index],
              edgecolor="white", linewidth=0.8)
ax.set_title("Average Food Rating by Day of the Week", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Day", fontsize=11)
ax.set_ylabel("Avg Rating (out of 10)", fontsize=11)
ax.set_ylim(0, 10)
ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("plots/avg_rating_by_day.png", dpi=150)
plt.close()
print("📈 Saved: plots/avg_rating_by_day.png")

# ── Plot 2: Average rating by meal type ──
avg_by_meal = df.groupby("Meal")["Rating"].mean()
fig, ax = plt.subplots(figsize=(6, 4))
colors = ["#FAC775", "#5DCAA5", "#F0997B"]
bars = ax.bar(avg_by_meal.index, avg_by_meal.values,
              color=colors, edgecolor="white", linewidth=0.8, width=0.5)
ax.set_title("Average Rating by Meal Type", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Meal", fontsize=11)
ax.set_ylabel("Avg Rating (out of 10)", fontsize=11)
ax.set_ylim(0, 10)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig("plots/avg_rating_by_meal.png", dpi=150)
plt.close()
print("📈 Saved: plots/avg_rating_by_meal.png")

# ── Plot 3: Model comparison bar chart ──
model_names = list(results.keys())
mae_vals    = [results[n]["MAE"]  for n in model_names]
r2_vals     = [results[n]["R2"]   for n in model_names]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.barh(model_names, mae_vals, color=["#B5D4F4","#5DCAA5","#7F77DD"], edgecolor="white")
ax1.set_title("MAE (lower = better)", fontsize=12, fontweight="bold")
ax1.set_xlabel("Mean Absolute Error")
for i, v in enumerate(mae_vals):
    ax1.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)

ax2.barh(model_names, r2_vals, color=["#B5D4F4","#5DCAA5","#7F77DD"], edgecolor="white")
ax2.set_title("R² Score (higher = better)", fontsize=12, fontweight="bold")
ax2.set_xlabel("R² Score")
for i, v in enumerate(r2_vals):
    ax2.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=9)

plt.suptitle("Model Comparison", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("plots/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("📈 Saved: plots/model_comparison.png")

# ── Plot 4: Actual vs Predicted ──
best_preds = best_model.predict(X_test)
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y_test, best_preds, alpha=0.7, color="#7F77DD", edgecolors="white", linewidth=0.5)
ax.plot([1,10], [1,10], "k--", linewidth=1, label="Perfect prediction")
ax.set_xlabel("Actual Rating", fontsize=11)
ax.set_ylabel("Predicted Rating", fontsize=11)
ax.set_title(f"Actual vs Predicted — {best_name}", fontsize=13, fontweight="bold")
ax.set_xlim(1, 10); ax.set_ylim(1, 10)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("plots/actual_vs_predicted.png", dpi=150)
plt.close()
print("📈 Saved: plots/actual_vs_predicted.png")

print("\n✅ Training complete!")
