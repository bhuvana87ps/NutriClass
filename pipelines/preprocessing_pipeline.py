# Feature Engineering + Data Preparation Pipeline

# Handle numeric scaling
# Encode categorical features
# Ensure the train& inference consistency

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def build_preprocessing_pipeline(numeric_cols, categorical_cols=None):
    numeric_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    if categorical_cols:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_cols)
            ]
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_cols)
            ]
        )

    return preprocessor

# prevents data leakage during model training and inference
