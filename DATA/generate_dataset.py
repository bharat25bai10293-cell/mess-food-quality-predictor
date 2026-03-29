"""
generate_dataset.py
====================
Creates a synthetic dataset for the Mess Food Quality Predictor project.
We simulate realistic food ratings you'd see in a college mess/canteen.
"""

import pandas as pd
import random

# Set a seed so we always get the same data (reproducibility)
random.seed(42)

# --- Define the possible values ---
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
meals = ["Breakfast", "Lunch", "Dinner"]

# Each meal type has its own realistic menu items
food_items = {
    "Breakfast": ["Poha", "Upma", "Idli-Sambar", "Paratha", "Bread-Butter", "Aloo Puri"],
    "Lunch":     ["Dal Rice", "Rajma Rice", "Chole Bhature", "Veg Biryani", "Roti Sabzi", "Paneer Curry"],
    "Dinner":    ["Khichdi", "Dal Tadka", "Palak Paneer", "Aloo Matar", "Fried Rice", "Chapati Sabzi"]
}

# Realistic base ratings for each food item (out of 10)
base_ratings = {
    # Breakfast items
    "Poha": 7.0, "Upma": 6.0, "Idli-Sambar": 8.0, "Paratha": 8.5,
    "Bread-Butter": 5.5, "Aloo Puri": 7.5,
    # Lunch items
    "Dal Rice": 6.5, "Rajma Rice": 8.0, "Chole Bhature": 8.5,
    "Veg Biryani": 9.0, "Roti Sabzi": 6.5, "Paneer Curry": 8.5,
    # Dinner items
    "Khichdi": 6.0, "Dal Tadka": 7.0, "Palak Paneer": 8.0,
    "Aloo Matar": 6.5, "Fried Rice": 8.0, "Chapati Sabzi": 6.0
}

# Weekend effect: food is slightly better on weekends
weekend_boost = {"Saturday": 0.5, "Sunday": 0.7}

# Build the dataset row by row
rows = []
for _ in range(60):  # 60 rows of data
    day  = random.choice(days)
    meal = random.choice(meals)
    food = random.choice(food_items[meal])

    # Start with base rating for that food item
    rating = base_ratings[food]

    # Add weekend boost if applicable
    rating += weekend_boost.get(day, 0)

    # Add a small random noise so it's not perfectly predictable
    noise = random.uniform(-1.0, 1.0)
    rating += noise

    # Clamp between 1 and 10
    rating = round(max(1.0, min(10.0, rating)), 1)

    rows.append({"Day": day, "Meal": meal, "Food_Item": food, "Rating": rating})

# Create DataFrame and save
df = pd.DataFrame(rows)
df.to_csv("data/mess_food_data.csv", index=False)

print("✅ Dataset created: data/mess_food_data.csv")
print(f"   Shape: {df.shape}")
print("\nSample rows:")
print(df.head(10).to_string(index=False))
