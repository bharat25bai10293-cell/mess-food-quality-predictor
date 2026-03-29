# 🍛 Mess Food Quality Predictor

A beginner-friendly Machine Learning project that predicts canteen/mess food
ratings (out of 10) based on the day of the week, meal type, and food item.

---

## 📁 Project Structure

```
mess_food_predictor/
│
├── data/
│   ├── generate_dataset.py   ← Creates mess_food_data.csv
│   └── mess_food_data.csv    ← Generated dataset (60 rows)
│
├── models/
│   ├── best_model.pkl        ← Trained ML model (created by train_model.py)
│   └── encoders.pkl          ← Label encoders for categorical columns
│
├── plots/
│   ├── avg_rating_by_day.png
│   ├── avg_rating_by_meal.png
│   ├── model_comparison.png
│   └── actual_vs_predicted.png
│
├── train_model.py            ← Full ML pipeline: preprocess → train → evaluate → plot
├── predict.py                ← Command-line prediction tool
├── app.py                    ← Streamlit web UI
├── requirements.txt          ← Python dependencies
└── README.md                 ← This file
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the dataset
```bash
python data/generate_dataset.py
```

### 3. Train the model
```bash
python train_model.py
```

### 4. Make predictions (command-line)
```bash
python predict.py
```

### 5. Launch the web UI
```bash
streamlit run app.py
```

---

## 🧠 How It Works

| Step | What happens |
|------|-------------|
| Dataset | 60 rows of synthetic mess food data with realistic ratings |
| Encoding | Day, Meal, Food_Item converted from text → numbers via LabelEncoder |
| Models | Linear Regression, Decision Tree, and Random Forest all trained |
| Evaluation | Best model chosen by lowest Mean Absolute Error (MAE) |
| Prediction | User inputs day/meal/food → model returns rating out of 10 |

---

## 📊 Key Metrics Explained

- **MAE (Mean Absolute Error)**: On average, how many rating points off is the model?
  - MAE = 0.8 means the model is off by 0.8 points on average
- **RMSE**: Similar to MAE but penalises big errors more
- **R² Score**: How well does the model explain the variation? 1.0 = perfect, 0 = random

---

## 🔮 Future Improvements

1. Collect real ratings from students (Google Form → CSV)
2. Add more features: cook name, weather, ingredients
3. Use a neural network (TensorFlow/Keras) for higher accuracy
4. Add a weekly meal rating report with email notification
5. Deploy on Streamlit Cloud for public access
6. Build a mobile app using Kivy or Flutter + API
