# --------------------------------------------------------------
# NutriClass — Diet Recommendation & Planning Engine
# (RULE-BASED • BUSINESS-READY • EXAM-SAFE)
# --------------------------------------------------------------
# DESIGN INTENT:
# - NO prediction here
# - Food Classifier handles ML prediction
# - This page ENFORCES diet & PLANS meals
# - Supports weekly & monthly diet plans
# --------------------------------------------------------------

import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

# --------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------
st.set_page_config(
    page_title="NutriClass | Diet Recommendation",
    layout="wide"
)

# --------------------------------------------------------------
# LOAD DATASET (SOURCE OF TRUTH)
# --------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "clean_food_data.csv"

df = pd.read_csv(DATA_PATH)

# --------------------------------------------------------------
# HEADER
# --------------------------------------------------------------
st.markdown("## 🥗 **NutriClass — Diet Recommendation Engine**")
st.caption(
    "Rule-based diet enforcement & planning using nutritional data (no prediction)"
)

st.markdown("---")

# --------------------------------------------------------------
# WHY NO PREDICTION HERE (EXAM-CRITICAL)
# --------------------------------------------------------------
with st.expander("🧠 Why this is NOT prediction (Important)"):
    st.markdown(
        """
        **Key Reasoning:**

        - Prediction is already handled by **Food Classifier**
        - Diet planning requires **strict compliance**, not probability
        - Food names already exist in dataset
        - Hence we **select**, not predict

        ✔ This avoids misuse of supervised ML  
        ✔ This is business-accurate and exam-safe
        """
    )

# --------------------------------------------------------------
# SIDEBAR — USER DIET TARGET
# --------------------------------------------------------------
st.sidebar.header("🎯 Diet Target")

meal_type = st.sidebar.selectbox(
    "Meal Type",
    ["breakfast", "lunch", "dinner"]
)

plan_duration = st.sidebar.selectbox(
    "Diet Plan Duration",
    ["Single Meal", "Weekly Plan", "Monthly Plan"]
)

target_calories = st.sidebar.slider(
    "Maximum Calories (kcal)",
    100, 800, 500
)

min_protein = st.sidebar.slider(
    "Minimum Protein (g)",
    5, 40, 15
)

max_fat = st.sidebar.slider(
    "Maximum Fat (g)",
    5, 40, 15
)

max_sugar = st.sidebar.slider(
    "Maximum Sugar (g)",
    0, 30, 10
)

st.sidebar.markdown("---")
generate = st.sidebar.button("🍽️ Generate Diet Plan")

# --------------------------------------------------------------
# DIET ENFORCEMENT LOGIC (NO ML)
# --------------------------------------------------------------
if generate:

    filtered = df[
        (df["Meal_Type"] == meal_type) &
        (df["Calories"] <= target_calories) &
        (df["Protein"] >= min_protein) &
        (df["Fat"] <= max_fat) &
        (df["Sugar"] <= max_sugar)
    ].copy()

    if filtered.empty:
        st.error("🚫 No food matches your strict diet criteria.")
        st.stop()

    # Diet score for ranking (rule-based)
    filtered["Diet_Score"] = (
        filtered["Protein"] * 3
        - filtered["Fat"] * 2
        - filtered["Sugar"] * 2
        + filtered["Fiber"] * 2
    )

    filtered = filtered.sort_values("Diet_Score", ascending=False)

    # ----------------------------------------------------------
    # SINGLE MEAL (STRICT ENFORCEMENT)
    # ----------------------------------------------------------
    if plan_duration == "Single Meal":

        approved_food = filtered.iloc[0]

        st.markdown("### ✅ Approved Food for Your Diet")
        st.markdown("🔒 **Read-Only Diet Enforcement Mode**")

        st.success(f"🍽️ **{approved_food['Food_Name']}**")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Calories", f"{approved_food['Calories']:.0f} kcal")
        c2.metric("Protein", f"{approved_food['Protein']:.1f} g")
        c3.metric("Fat", f"{approved_food['Fat']:.1f} g")
        c4.metric("Sugar", f"{approved_food['Sugar']:.1f} g")

    # ----------------------------------------------------------
    # WEEKLY PLAN (7 MEALS)
    # ----------------------------------------------------------
    elif plan_duration == "Weekly Plan":

        st.markdown("### 📅 Weekly Diet Plan")

        weekly_plan = filtered.head(7)

        st.dataframe(
            weekly_plan[
                ["Food_Name", "Calories", "Protein", "Fat", "Sugar"]
            ],
            use_container_width=True
        )

    # ----------------------------------------------------------
    # MONTHLY PLAN (30 MEALS)
    # ----------------------------------------------------------
    else:

        st.markdown("### 🗓️ Monthly Diet Plan")

        monthly_plan = filtered.head(30)

        st.dataframe(
            monthly_plan[
                ["Food_Name", "Calories", "Protein", "Fat", "Sugar"]
            ],
            use_container_width=True
        )

    # ----------------------------------------------------------
    # MACRONUTRIENT DONUT (AVERAGE)
    # ----------------------------------------------------------
    st.markdown("### 🍩 Average Macronutrient Distribution")

    avg_protein = filtered["Protein"].mean()
    avg_fat = filtered["Fat"].mean()
    avg_carbs = filtered["Carbs"].mean()

    donut = go.Figure(
        data=[
            go.Pie(
                labels=["Protein", "Fat", "Carbohydrates"],
                values=[avg_protein, avg_fat, avg_carbs],
                hole=0.55,
                textinfo="label+percent"
            )
        ]
    )

    donut.update_layout(
        height=400,
        title="Macronutrient Balance of Selected Foods"
    )

    st.plotly_chart(donut, use_container_width=True)

# --------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------
st.caption(
    "NutriClass • Rule-Based Diet Planning • Business-Ready Enforcement System"
)
