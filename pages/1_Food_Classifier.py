# --------------------------------------------------------------
# NutriClass — Food Classifier (FINAL • FIXED • EXAM-READY)
# --------------------------------------------------------------
# KEY DESIGN PRINCIPLES:
# 1. Single pre-trained ML pipeline (no training in Streamlit)
# 2. Editable INPUTS only
# 3. Read-only OUTPUTS (anti-pattern avoided)
# 4. Explainability via scores & charts
# 5. Two supervised ML use-cases in ONE page
# --------------------------------------------------------------

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from io import BytesIO

# ---------------- PDF (Output Only) ----------------
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# ---------------- Visualization ----------------
import plotly.graph_objects as go

# --------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------
st.set_page_config(
    page_title="NutriClass | Food Classifier",
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
# CUSTOM NAVBAR (CONSISTENT ACROSS ALL PAGES)
# --------------------------------------------------------------
st.sidebar.markdown("## 📂 NutriClass Modules")

st.sidebar.page_link("app.py", label="🏠 Home")
st.sidebar.page_link("pages/4_Raw_Data_Explorer.py", label="🔍 Raw Data Explorer")
st.sidebar.page_link("pages/2_Diet_Recommendation.py", label="🥗 Diet Recommendation")
st.sidebar.page_link("pages/1_Food_Classifier.py", label="🎯 Food Classifier")
st.sidebar.page_link("pages/3_Pipeline_Overview.py", label="⚙️ Pipeline Overview")


# --------------------------------------------------------------
# LOAD TRAINED PIPELINE (READ-ONLY)
# --------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "nutriclass_pipeline.pkl"

@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)

model = load_pipeline()

# --------------------------------------------------------------
# HEALTH BALANCE SCORE (RULE-BASED, EXPLAINABILITY)
# --------------------------------------------------------------
def calculate_health_score(row):
    score = 100

    if row["Calories"] > 500: score -= 15
    if row["Fat"] > 20: score -= 15
    if row["Sugar"] > 15: score -= 15
    if row["Sodium"] > 800: score -= 10

    if row["Calories"] > 400: score -= 5
    if row["Fat"] > 10: score -= 5
    if row["Sugar"] > 8: score -= 5
    if row["Protein"] < 15: score -= 5
    if row["Fiber"] < 7: score -= 5

    return max(0, min(100, score))


def get_health_label(score):
    if score >= 90:
        return "🌟 Perfect Meal"
    elif score >= 80:
        return "✅ Very Good Meal"
    elif score >= 60:
        return "⚠️ Average Meal"
    else:
        return "❌ Poor Meal"

# --------------------------------------------------------------
# PDF REPORT GENERATOR (OUTPUT EXPORT ONLY)
# --------------------------------------------------------------
def generate_pdf_report(data_dict, title):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(title, styles["Title"]))
    elements.append(Spacer(1, 12))

    table_data = [["Metric", "Value"]]
    for k, v in data_dict.items():
        table_data.append([k, str(v)])

    elements.append(Table(table_data))
    doc.build(elements)
    buffer.seek(0)
    return buffer

# --------------------------------------------------------------
# HEADER
# --------------------------------------------------------------
st.markdown("## 🍽️ **NutriClass — Food Classifier**")
st.caption(
    "Supervised ML • Read-Only Inference • Explainable Nutrition Intelligence"
)


st.markdown("---")

# --------------------------------------------------------------
# MODE SELECTION
# --------------------------------------------------------------
mode = st.radio(
    "Choose Prediction Mode:",
    [
        "Standard Health Classification (Healthy / Unhealthy)",
        "Strict Diet Planning — Exact Food Name Prediction"
    ]
)

st.markdown("---")

# ==============================================================
# MODE 1 — STANDARD HEALTH CLASSIFICATION
# ==============================================================
if mode.startswith("Standard"):

    st.subheader("🥗 Nutrition Input — Health Classification")

    with st.form("health_form"):

        c1, c2, c3 = st.columns(3)
        Calories = c1.number_input("Calories (kcal)", min_value=0.0, step=10.0)
        Protein = c2.number_input("Protein (g)", min_value=0.0)
        Fat = c3.number_input("Fat (g)", min_value=0.0)

        c4, c5, c6 = st.columns(3)
        Carbs = c4.number_input("Carbohydrates (g)", min_value=0.0)
        Sugar = c5.number_input("Sugar (g)", min_value=0.0)
        Fiber = c6.number_input("Fiber (g)", min_value=0.0)

        Sodium = st.number_input("Sodium (mg)", min_value=0.0)
        Meal_Type = st.selectbox("Meal Type", ["breakfast", "lunch", "dinner"])

        submitted = st.form_submit_button("🔍 Predict Food Health")

    if submitted:

        input_df = pd.DataFrame([{
            "Calories": Calories,
            "Protein": Protein,
            "Fat": Fat,
            "Carbs": Carbs,
            "Sugar": Sugar,
            "Fiber": Fiber,
            "Sodium": Sodium,
            "Cholesterol": 0,
            "Glycemic_Index": 0,
            "Water_Content": 0,
            "Serving_Size": 0,
            "Meal_Type": Meal_Type,
            "Preparation_Method": "raw",
            "Is_Vegan": 0,
            "Is_Gluten_Free": 0
        }])

        prediction = model.predict(input_df)[0]
        confidence = model.predict_proba(input_df).max()

        score = calculate_health_score(input_df.iloc[0])
        label = get_health_label(score)

        st.markdown(
            "🔒 **Read-Only Inference Mode**  \n"
            "_All results below are system-generated and cannot be edited._"
        )

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        st.metric("Prediction Confidence", f"{confidence:.2%}")

        if prediction.lower() == "healthy":
            st.success("✅ Healthy Food")
        else:
            st.error("⚠️ Unhealthy Food")

        st.metric("Health Balance Score", f"{score} / 100")
        st.progress(score / 100)
        st.info(f"Meal Quality: **{label}**")

        # ---------------- Visualization ----------------
        st.markdown("### 🧠 Nutrient Contribution")
        bar_df = pd.DataFrame({
            "Nutrient": ["Calories", "Protein", "Fat", "Carbs", "Sugar", "Fiber", "Sodium"],
            "Value": [Calories, Protein, Fat, Carbs, Sugar, Fiber, Sodium]
        })
        st.bar_chart(bar_df.set_index("Nutrient"))

        st.markdown("### 🕸️ Nutrient Balance Radar")
        fig = go.Figure(
            go.Scatterpolar(
                r=[Protein, Fiber, Carbs, Fat, Sugar],
                theta=["Protein", "Fiber", "Carbs", "Fat", "Sugar"],
                fill="toself"
            )
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
        # ------------------------------------------------------
        # MACRONUTRIENT DISTRIBUTION — DONUT CHART (EXPLAINABILITY)
        # ------------------------------------------------------
        st.markdown("### 🥯 Macronutrient Distribution")

        macro_labels = ["Protein (g)", "Fat (g)", "Carbohydrates (g)"]
        macro_values = [Protein, Fat, Carbs]

        # Proper conditional block
        if sum(macro_values) > 0:
            donut_fig = go.Figure(
                data=[
                    go.Pie(
                        labels=macro_labels,
                        values=macro_values,
                        hole=0.5,                 # Donut style
                        textinfo="label+percent"
                    )
                ]
            )

            donut_fig.update_layout(
                height=350,
                showlegend=True,
                margin=dict(t=20, b=20)
            )

            st.plotly_chart(donut_fig, use_container_width=True)

        else:
            st.info("Macronutrient distribution will appear after valid inputs.")

                
                
                
        
        pdf_buffer = generate_pdf_report(
            {
                "Calories": Calories,
                "Protein": Protein,
                "Fat": Fat,
                "Carbs": Carbs,
                "Sugar": Sugar,
                "Fiber": Fiber,
                "Sodium": Sodium,
                "Meal Type": Meal_Type,
                "Prediction": prediction,
                "Confidence": f"{confidence:.2%}",
                "Health Balance Score": f"{score}/100",
                "Meal Quality": label
            },
            "NutriClass — Health Classification Report"
        )

        st.download_button(
            "⬇️ Download PDF Nutrition Report",
            pdf_buffer,
            file_name="nutriclass_health_report.pdf",
            mime="application/pdf"
        )

# ==============================================================
# MODE 2 — STRICT DIET PLANNING (FIXED — NO BLANK PAGE)
# ==============================================================
else:

    st.subheader("🎯 Strict Diet Planning — Exact Food Name Prediction")

    with st.form("strict_form"):

        c1, c2, c3 = st.columns(3)
        target_protein = c1.number_input("Target Protein (g)", min_value=0.0)
        max_fat = c2.number_input("Maximum Fat (g)", min_value=0.0)
        max_sugar = c3.number_input("Maximum Sugar (g)", min_value=0.0)

        calories = st.number_input("Target Calories", min_value=0.0)
        meal = st.selectbox("Meal Type", ["breakfast", "lunch", "dinner"])

        submitted = st.form_submit_button("🍽️ Find Allowed Food")

    if submitted:

        strict_df = pd.DataFrame([{
            "Calories": calories,
            "Protein": target_protein,
            "Fat": max_fat,
            "Sugar": max_sugar,
            "Carbs": 0,
            "Fiber": 0,
            "Sodium": 0,
            "Cholesterol": 0,
            "Glycemic_Index": 0,
            "Water_Content": 0,
            "Serving_Size": 0,
            "Meal_Type": meal,
            "Preparation_Method": "raw",
            "Is_Vegan": 0,
            "Is_Gluten_Free": 0
        }])

        food_name = model.predict(strict_df)[0]

        compliance = min(
            100,
            int((target_protein / (target_protein + max_fat + max_sugar + 1)) * 100)
        )

        st.markdown(
            "🔒 **Read-Only Inference Mode**  \n"
            "_Approved food is system-selected and cannot be modified._"
        )

        st.markdown("---")
        st.success(f"🍽️ **Approved Food:** {food_name}")
        st.metric("Diet Compliance Score", f"{compliance}%")
        st.progress(compliance / 100)

        pdf_buffer = generate_pdf_report(
            {
                "Target Calories": calories,
                "Target Protein": target_protein,
                "Max Fat": max_fat,
                "Max Sugar": max_sugar,
                "Meal Type": meal,
                "Approved Food": food_name,
                "Diet Compliance Score": f"{compliance}%"
            },
            "NutriClass — Strict Diet Approval Report"
        )
        

        st.download_button(
            "⬇️ Download Diet PDF Report",
            pdf_buffer,
            file_name="nutriclass_strict_diet_report.pdf",
            mime="application/pdf"
        )

# --------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------
st.caption(
    "NutriClass • Supervised ML • Read-Only Inference • Explainable Decision System"
)
