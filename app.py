import streamlit as st
import pandas as pd
from utils import load_model

st.set_page_config(page_title="NutriClass ML Inference", layout="wide")

model, preprocessor, label_encoder = load_model()

st.title("🍽️ NutriClass — Food Health Prediction")

st.sidebar.header("Enter Nutrition Details")

# User Inputs
Calories = st.sidebar.number_input("Calories", 0.0)
Protein = st.sidebar.number_input("Protein", 0.0)
Fat = st.sidebar.number_input("Fat", 0.0)
Carbs = st.sidebar.number_input("Carbs", 0.0)
Sugar = st.sidebar.number_input("Sugar", 0.0)
Fiber = st.sidebar.number_input("Fiber", 0.0)
Sodium = st.sidebar.number_input("Sodium", 0.0)
Cholesterol = st.sidebar.number_input("Cholesterol", 0.0)
Glycemic_Index = st.sidebar.number_input("Glycemic Index", 0.0)
Water_Content = st.sidebar.number_input("Water Content", 0.0)
Serving_Size = st.sidebar.number_input("Serving Size", 0.0)

Meal_Type = st.sidebar.selectbox("Meal Type", ["breakfast", "lunch", "dinner"])
Preparation_Method = st.sidebar.selectbox("Preparation Method", ["fried", "boiled", "raw"])
Is_Vegan = st.sidebar.checkbox("Is Vegan")
Is_Gluten_Free = st.sidebar.checkbox("Is Gluten Free")

if st.sidebar.button("Predict"):
    input_df = pd.DataFrame([{
        "Calories": Calories,
        "Protein": Protein,
        "Fat": Fat,
        "Carbs": Carbs,
        "Sugar": Sugar,
        "Fiber": Fiber,
        "Sodium": Sodium,
        "Cholesterol": Cholesterol,
        "Glycemic_Index": Glycemic_Index,
        "Water_Content": Water_Content,
        "Serving_Size": Serving_Size,
        "Meal_Type": Meal_Type,
        "Preparation_Method": Preparation_Method,
        "Is_Vegan": int(Is_Vegan),
        "Is_Gluten_Free": int(Is_Gluten_Free)
    }])

    X_processed = preprocessor.transform(input_df)
    prediction = model.predict(X_processed)
    result = label_encoder.inverse_transform(prediction)

    st.success(f"🧠 Predicted Food Health: **{result[0]}**")
