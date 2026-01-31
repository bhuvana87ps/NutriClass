# --------------------------------------------------------------
# grid_search.py — Model Selection & Saving
# --------------------------------------------------------------

import joblib
from sklearn.model_selection import GridSearchCV

from pipelines.model_pipelines import (
    logistic_pipeline,
    random_forest_pipeline
)

def run_grid_search(X_train, y_train, preprocessor):
    pipelines = {
        "logistic": logistic_pipeline(preprocessor),
        "random_forest": random_forest_pipeline(preprocessor)
    }

    param_grids = {
        "logistic": {
            "model__C": [0.1, 1, 10]
        },
        "random_forest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20]
        }
    }

    best_score = 0
    best_pipeline = None

    for name, pipeline in pipelines.items():
        grid = GridSearchCV(
            pipeline,
            param_grids[name],
            cv=5,
            scoring="f1_weighted",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_pipeline = grid.best_estimator_

    # ----------------------------------------------------------
    # SAVE THE BEST MODEL (FINAL STEP)
    # ----------------------------------------------------------
    joblib.dump(
        best_pipeline,
        "models/nutriclass_pipeline.pkl"
    )

    return best_pipeline
