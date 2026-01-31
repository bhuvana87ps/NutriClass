# Models to Preprocessing Pipelines

# Takes preprocessing pipeline
# Adds ML models (Logistic, RF, etc.)
# Returns full end-to-end pipelines


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def logistic_pipeline(preprocessor):
    return Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

def random_forest_pipeline(preprocessor):
    return Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(random_state=42))
    ])
