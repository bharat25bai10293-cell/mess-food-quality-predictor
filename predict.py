"""
predict.py
==========
Command-line prediction tool.
Run: python predict.py

You will be asked to enter:
  • Day of the week
  • Meal type
  • Food item
…and the model will predict a rating out of 10.
"""

import pickle
import numpy as np

# ── Load the saved model and encoders ──
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)


def get_valid_input(prompt, valid_options):
    """
    Keep asking the user until they give a valid answer.
    Case-insensitive matching.
    """
    options_lower = {opt.lower(): opt for opt in valid_options}
    while True:
        raw = input(prompt).strip().lower()
        if raw in options_lower:
            return options_lower[raw]
        # Friendly partial-match hint
        matches = [opt for opt in valid_options if raw in opt.lower()]
        if matches:
            print(f"  Did you mean one of: {', '.join(matches)}?")
        else:
            print(f"  ❌ Invalid. Options: {', '.join(valid_options)}")


def predict_rating(day, meal, food_item):
    """
    Encode the inputs and return the model's predicted rating.
    Returns None if any input is not in the training data.
    """
    try:
        day_enc  = encoders["Day"].transform([day])[0]
        meal_enc = encoders["Meal"].transform([meal])[0]
        food_enc = encoders["Food_Item"].transform([food_item])[0]
    except ValueError as e:
        print(f"  ⚠️  {e} — item not seen during training.")
        return None

    X = np.array([[day_enc, meal_enc, food_enc]])
    return round(float(model.predict(X)[0]), 1)


def rating_emoji(r):
    if r >= 8.5: return "🌟 Excellent!"
    if r >= 7.0: return "😊 Good"
    if r >= 5.5: return "😐 Average"
    return "😞 Below average"


# ──────────────────────────────────────────────
# Main interaction loop
# ──────────────────────────────────────────────
print("\n" + "="*50)
print("   🍽️  Mess Food Quality Predictor")
print("="*50)

# Derive valid choices from encoders
valid_days  = list(encoders["Day"].classes_)
valid_meals = list(encoders["Meal"].classes_)

while True:
    print()
    day  = get_valid_input(f"Enter Day {valid_days}: ", valid_days)
    meal = get_valid_input(f"Enter Meal {['Breakfast','Lunch','Dinner']}: ", valid_meals)

    # Show only the food items for the selected meal
    valid_foods = [
        cls for cls in encoders["Food_Item"].classes_
    ]
    # A small lookup so we show only meal-appropriate items
    meal_foods = {
        "Breakfast": ["Poha","Upma","Idli-Sambar","Paratha","Bread-Butter","Aloo Puri"],
        "Lunch":     ["Dal Rice","Rajma Rice","Chole Bhature","Veg Biryani","Roti Sabzi","Paneer Curry"],
        "Dinner":    ["Khichdi","Dal Tadka","Palak Paneer","Aloo Matar","Fried Rice","Chapati Sabzi"]
    }
    food = get_valid_input(
        f"Enter Food Item {meal_foods.get(meal, valid_foods)}: ",
        meal_foods.get(meal, valid_foods)
    )

    rating = predict_rating(day, meal, food)

    if rating is not None:
        print(f"\n  ┌─────────────────────────────────────────┐")
        print(f"  │  {day} | {meal} | {food:<22}│")
        print(f"  │  Predicted Rating : {rating}/10             │")
        print(f"  │  Verdict          : {rating_emoji(rating):<22}│")
        print(f"  └─────────────────────────────────────────┘")

    again = input("\n  Predict another? (yes/no): ").strip().lower()
    if again not in ("yes","y"):
        print("\n  Goodbye! 🍛\n")
        break
