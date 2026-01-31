# --------------------------------------------------------------
# NutriClass — ML Pipeline Overview
# --------------------------------------------------------------
# PURPOSE:
# - Explain end-to-end system architecture
# - Show correct ML usage
# - Help evaluators understand design decisions
# --------------------------------------------------------------

import streamlit as st

# --------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------
st.set_page_config(
    page_title="NutriClass | Pipeline Overview",
    layout="wide"
)

# --------------------------------------------------------------
# HEADER
# --------------------------------------------------------------
st.markdown("## 🧠 **NutriClass — ML Pipeline Overview**")
st.caption(
    "End-to-end architecture for health classification and strict diet automation"
)

st.markdown("---")

# --------------------------------------------------------------
# SYSTEM ARCHITECTURE
# --------------------------------------------------------------
st.subheader("📐 System Architecture")

st.markdown(
    """
    **NutriClass is designed as a modular ML system:**

    ```
    User Input
        ↓
    Feature Engineering
        ↓
    Trained ML Pipeline (.pkl)
        ↓
    Inference Layer (Read-Only)
        ↓
    Explainability + UI
    ```
    """
)

# --------------------------------------------------------------
# PIPELINE STAGES
# --------------------------------------------------------------
st.subheader("🔁 Pipeline Stages")

st.markdown(
    """
    **1️⃣ Data Preparation**
    - Clean nutritional dataset
    - Standardized numeric features
    - Encoded categorical features (Meal_Type, Prep_Method)

    **2️⃣ Model Training (Offline)**
    - Supervised learning
    - Target column: `Food_Name`
    - Saved as `nutriclass_pipeline.pkl`

    **3️⃣ Inference (Online)**
    - Streamlit loads trained pipeline
    - No retraining
    - Predictions are read-only

    **4️⃣ Explainability Layer**
    - Health balance score
    - Nutrient charts
    - Compliance metrics
    """
)

# --------------------------------------------------------------
# WHY FOOD NAME AS TARGET
# --------------------------------------------------------------
st.subheader("🎯 Why Food Name is the Target Variable")

st.markdown(
    """
    **Business Reasoning:**

    - Each food is a **unique allowed option**
    - Nutrition metrics are **features**
    - Model predicts **exact food** for compliance
    - Eliminates ambiguity in strict diets

    ✔ Ideal for fitness apps, hospitals, dieticians
    """
)

# --------------------------------------------------------------
# CLASSIFICATION VS RECOMMENDATION
# --------------------------------------------------------------
st.subheader("⚖️ Classification vs Recommendation")

st.markdown(
    """
    | Aspect | Recommendation | NutriClass Approach |
    |------|----------------|--------------------|
    | Output | Multiple options | One exact food |
    | Control | Flexible | Strict |
    | Use Case | Discovery | Compliance |
    | ML Type | Ranking | Classification |

    ✔ NutriClass intentionally uses **classification**
    """
)

# --------------------------------------------------------------
# BUSINESS USE CASES
# --------------------------------------------------------------
st.subheader("🏢 Business Use Cases")

st.markdown(
    """
    - Smart Dietary Applications
    - Health Monitoring Systems
    - Food Logging Platforms
    - Fitness & Gym Apps
    - Grocery & Meal Planning Automation

    ✔ Real-time, scalable, production-ready
    """
)

# --------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------
st.caption(
    "NutriClass • ML Architecture • Explainable AI • Production Design"
)
