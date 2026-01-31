# --------------------------------------------------------------
# NutriClass — Main Application Entry Point
# --------------------------------------------------------------
# PURPOSE:
# - Defines project navigation flow
# - Aligns pages with ML reasoning order
# - Does NOT contain ML logic (exam best practice)
#
# DESIGN STORY:
# Raw Data (Unsupervised Insights)
# → Diet Recommendation (Unsupervised Logic)
# → Food Classifier (Supervised Enforcement)
# → Pipeline Overview (System Explanation)
# --------------------------------------------------------------

import streamlit as st

# --------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------
st.set_page_config(
    page_title="NutriClass | Nutrition Intelligence Platform",
    layout="wide"
)
# --------------------------------------------------------------
# HIDE DEFAULT STREAMLIT PAGE LIST (SAFE)
# --------------------------------------------------------------
st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------------------
# APP HEADER
# --------------------------------------------------------------
st.markdown("## 🍽️ **NutriClass — Nutrition Intelligence Platform**")
st.caption(
    "From data understanding → diet recommendation → strict food classification"
)

st.markdown("---")

# --------------------------------------------------------------
# PROJECT STORY (EXAM-CRITICAL)
# --------------------------------------------------------------
st.markdown(
    """
### 🔗 How This Project Is Structured

NutriClass follows a **progressive machine learning design**:

1. **Raw Data Explorer**
   - Uses **unsupervised learning**
   - Discovers nutritional patterns and clusters
   - Validates that foods are separable

2. **Diet Recommendation Engine**
   - Uses **unsupervised logic**
   - Groups and filters foods based on similarity
   - Provides flexible meal planning

3. **Food Classifier**
   - Uses **supervised learning**
   - Predicts **exact food name** for strict diet enforcement
   - Enables real-time, controlled decision-making

4. **Pipeline Overview**
   - Explains the full end-to-end ML system
   - Justifies model and design choices

👉 Navigate pages **top to bottom** to follow the reasoning.
"""
)

st.markdown("---")

# --------------------------------------------------------------
# SIDEBAR NAVIGATION (ORDER MATTERS)
# --------------------------------------------------------------
st.sidebar.markdown("## 📂 NutriClass Modules")

st.sidebar.page_link(
    "pages/4_Raw_Data_Explorer.py",
    label="🔍 Raw Data Explorer",
    help="Unsupervised learning, clustering, PCA, silhouette"
)

st.sidebar.page_link(
    "pages/2_Diet_Recommendation.py",
    label="🥗 Diet Recommendation",
    help="Unsupervised food grouping & meal planning"
)

st.sidebar.page_link(
    "pages/1_Food_Classifier.py",
    label="🎯 Food Classifier",
    help="Supervised ML — exact food prediction"
)

st.sidebar.page_link(
    "pages/3_Pipeline_Overview.py",
    label="⚙️ Pipeline Overview",
    help="End-to-end ML architecture explanation"
)

# --------------------------------------------------------------
# LANDING MESSAGE
# --------------------------------------------------------------
st.info(
    "⬅️ Use the sidebar to explore NutriClass modules in order.\n\n"
    "This application is **read-only inference** and **exam-aligned by design**."
)

# --------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------
st.caption(
    "NutriClass • Unsupervised → Supervised ML • Explainable Nutrition Intelligence"
)
