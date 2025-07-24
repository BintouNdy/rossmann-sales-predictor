# MLflow Configuration for Rossmann Sales Prediction

# Basic Configuration
EXPERIMENT_NAME = "rossmann-sales-prediction"
TRACKING_URI = "file:./mlruns"
MODEL_NAME = "rossmann_xgboost"

# Model Registry Configuration
REGISTERED_MODEL_NAME = "RossmannSalesPredictor"

# Default Tags
DEFAULT_TAGS = {
    "project": "rossmann-sales-prediction",
    "model_type": "xgboost",
    "dataset": "rossmann",
    "framework": "sklearn",
    "problem_type": "regression"
}

# Metrics to Track
METRICS_TO_TRACK = [
    "train_rmse",
    "test_rmse", 
    "train_mae",
    "test_mae",
    "train_r2",
    "test_r2",
    "overfitting_score"
]

# Parameters to Log
PARAMS_TO_LOG = [
    "n_estimators",
    "max_depth", 
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "gamma",
    "train_samples",
    "test_samples",
    "features_count"
]

# Artifacts to Save
ARTIFACTS_TO_SAVE = [
    "predictions.csv",
    "feature_importance.csv"
]
