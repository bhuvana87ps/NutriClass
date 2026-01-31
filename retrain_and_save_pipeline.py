import joblib
import pandas as pd
from pathlib import Path

from pipelines.preprocessing_pipeline import build_preprocessing_pipeline
from pipelines.model_pipelines import random_forest_pipeline

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "clean_food_data.csv"
MODEL_PATH = BASE_DIR / "models" / "nutriclass_pipeline.pkl"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)

# --------------------------------------------------
# CREATE TARGET (RULE-BASED)
# --------------------------------------------------
def label_food(row):
    if row["Calories"] <= 400 and row["Sugar"] <= 10 and row["Fat"] <= 15:
        return "Healthy"
    return "Unhealthy"

df["Food_Category"] = df.apply(label_food, axis=1)

# --------------------------------------------------
# SPLIT FEATURES / TARGET
# --------------------------------------------------
X = df.drop(columns=["Food_Category"])
y = df["Food_Category"]

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

# --------------------------------------------------
# BUILD PIPELINE
# --------------------------------------------------
preprocessor = build_preprocessing_pipeline(
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols
)

pipeline = random_forest_pipeline(preprocessor)

# --------------------------------------------------
# TRAIN & SAVE
# --------------------------------------------------
pipeline.fit(X, y)

MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)

print("✅ Pipeline trained & saved with rule-based labels")
