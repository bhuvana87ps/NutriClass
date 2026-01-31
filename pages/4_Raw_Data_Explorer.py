# --------------------------------------------------------------
# NutriClass — Raw Data Explorer & Unsupervised Insights
# --------------------------------------------------------------
# MY POINT OF VIEW:
# - Before predicting food, I must understand food
# - Nutrition data has NO labels initially
# - Patterns must be discovered, not assumed
# - Hence, UNSUPERVISED learning is mandatory first
# --------------------------------------------------------------

import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------
st.set_page_config(
    page_title="NutriClass | Raw Data & Unsupervised Insights",
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
# LOAD DATA
# --------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "clean_food_data.csv"
df = pd.read_csv(DATA_PATH)

# --------------------------------------------------------------
# HEADER
# --------------------------------------------------------------
st.markdown("## 📊 **NutriClass — Raw Data & Unsupervised Insights**")
st.caption("Understanding nutrition data before applying machine learning")

st.markdown("---")

# ==============================================================
# WHY THIS PAGE EXISTS (VERY IMPORTANT)
# ==============================================================
st.subheader("🧠 Why Raw Data Exploration Comes First")

st.markdown(
    """
    **My Design Thinking:**

    - Nutrition datasets do not come with decision labels
    - Food health is **contextual**, not binary
    - Before prediction, I must understand:
        - Nutrient ranges
        - Natural groupings
        - Feature relationships

    👉 This page exists to **justify why unsupervised learning was necessary**
    before choosing any supervised model.
    """
)

# --------------------------------------------------------------
# RAW DATA PREVIEW (TRUST BUILDING)
# --------------------------------------------------------------
st.markdown("---")
st.subheader("🔍 Raw Dataset Snapshot")

st.dataframe(df.head(10), use_container_width=True)

st.info(
    "At this stage, no target variable is used. "
    "The data is treated as unlabeled."
)

# --------------------------------------------------------------
# DISTRIBUTION ANALYSIS (PATTERN DISCOVERY)
# --------------------------------------------------------------
st.markdown("---")
st.subheader("📈 Nutrient Distribution Analysis")

c1, c2 = st.columns(2)

with c1:
    fig_cal = px.histogram(
        df, x="Calories", nbins=30,
        title="Calories Distribution"
    )
    st.plotly_chart(fig_cal, use_container_width=True)

with c2:
    fig_pro = px.histogram(
        df, x="Protein", nbins=30,
        title="Protein Distribution"
    )
    st.plotly_chart(fig_pro, use_container_width=True)

st.markdown(
    """
    **Insight:**
    - Calories and protein are not uniformly distributed
    - This confirms that foods vary significantly
    - Simple rule-based labeling would be unreliable
    """
)

# --------------------------------------------------------------
# FEATURE RELATIONSHIPS
# --------------------------------------------------------------
st.markdown("---")
st.subheader("🔗 Nutrient Relationships")

fig_scatter = px.scatter(
    df,
    x="Protein",
    y="Calories",
    color="Meal_Type",
    title="Protein vs Calories by Meal Type",
    opacity=0.7
)

st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown(
    """
    **Insight:**
    - Same calories can map to different protein levels
    - Meal type adds context
    - Reinforces need for pattern discovery
    """
)

# ==============================================================
# WHY UNSUPERVISED LEARNING
# ==============================================================
st.markdown("---")
st.subheader("🧠 Why I Chose Unsupervised Learning")

st.markdown(
    """
    **Key Reasoning:**

    - No predefined food categories exist
    - Health labels are subjective
    - Foods must be grouped by similarity, not prediction

    ✔ Unsupervised learning allows:
    - Pattern discovery
    - Cluster formation
    - Feature validation
    """
)

# --------------------------------------------------------------
# CLUSTERING SETUP
# --------------------------------------------------------------
cluster_features = [
    "Calories", "Protein", "Fat", "Carbs", "Sugar", "Fiber"
]

cluster_df = df[cluster_features].dropna()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(cluster_df)

# --------------------------------------------------------------
# ELBOW METHOD
# --------------------------------------------------------------
st.markdown("---")
st.subheader("📈 Choosing Number of Clusters (Elbow Method)")

inertia = []
for k in range(1, 9):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled_features)
    inertia.append(km.inertia_)

elbow_df = pd.DataFrame({"K": range(1, 9), "Inertia": inertia})

fig_elbow = px.line(
    elbow_df, x="K", y="Inertia",
    markers=True, title="Elbow Method"
)

st.plotly_chart(fig_elbow, use_container_width=True)

st.info("Elbow around K = 4 → chosen for clustering")

# --------------------------------------------------------------
# APPLY KMEANS
# --------------------------------------------------------------
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

cluster_df["Cluster"] = clusters

# --------------------------------------------------------------
# SILHOUETTE SCORE
# --------------------------------------------------------------
sil_score = silhouette_score(scaled_features, clusters)
st.metric("Silhouette Score (Cluster Quality)", f"{sil_score:.3f}")

# --------------------------------------------------------------
# CLUSTER VISUALIZATION
# --------------------------------------------------------------
st.markdown("---")
st.subheader("📌 Food Clusters (Nutritional Similarity)")

fig_cluster = px.scatter(
    cluster_df,
    x="Protein",
    y="Calories",
    color=cluster_df["Cluster"].astype(str),
    title="KMeans Food Clusters",
    opacity=0.7
)

st.plotly_chart(fig_cluster, use_container_width=True)

# --------------------------------------------------------------
# PCA VISUALIZATION
# --------------------------------------------------------------
st.markdown("---")
st.subheader("📊 PCA Projection (High-Dimensional → 2D)")

pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

pca_df = pd.DataFrame(pca_features, columns=["PC1", "PC2"])
pca_df["Cluster"] = clusters

fig_pca = px.scatter(
    pca_df, x="PC1", y="PC2",
    color=pca_df["Cluster"].astype(str),
    title="PCA View of Food Clusters",
    opacity=0.7
)

st.plotly_chart(fig_pca, use_container_width=True)

# --------------------------------------------------------------
# CLUSTER INTERPRETATION
# --------------------------------------------------------------
st.markdown("---")
st.subheader("🧩 Interpreting Clusters")

clustered_foods = df.loc[cluster_df.index].copy()
clustered_foods["Cluster"] = clusters

for cid in sorted(clustered_foods["Cluster"].unique()):
    with st.expander(f"Cluster {cid} — Example Foods"):
        st.dataframe(
            clustered_foods[
                clustered_foods["Cluster"] == cid
            ][["Food_Name", "Calories", "Protein", "Fat", "Carbs"]].head(5),
            use_container_width=True
        )

# --------------------------------------------------------------
# BRIDGE TO SUPERVISED MODEL
# --------------------------------------------------------------
st.markdown("---")
st.subheader("🔗 How This Led to Supervised Classification")

st.markdown(
    """
    **Final Design Decision:**

    - Clusters showed that foods are separable
    - Features are meaningful
    - Food_Name can be safely predicted

    👉 Hence, supervised classification was chosen
    for real-time diet enforcement.
    """
)

# --------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------
st.caption(
    "NutriClass • Unsupervised Learning • Data Understanding • Model Justification"
)
