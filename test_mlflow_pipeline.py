"""
Simple test script to validate MLflow pipeline configuration
"""

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Add src to path
sys.path.append('src')

from pipelines.mlflow_utils import train_and_log_model

def test_mlflow_pipeline():
    """Test the MLflow pipeline with synthetic data"""
    print("🧪 Testing MLflow Pipeline...")
    
    # Create synthetic data similar to Rossmann dataset
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Generate features
    X = pd.DataFrame({
        f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
    })
    
    # Generate target with some relationship to features
    y = pd.Series(
        X['feature_0'] * 100 + 
        X['feature_1'] * 50 + 
        np.random.randn(n_samples) * 10 + 1000
    )
    
    print(f"📊 Generated synthetic data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create simple XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    
    # Test MLflow training pipeline
    print("🚀 Running MLflow training pipeline...")
    
    trained_model = train_and_log_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_name="test_model",
        experiment_name="mlflow-pipeline-test",
        run_name="synthetic_data_test",
        tags={"test": "true", "data_type": "synthetic"}
    )
    
    print("✅ MLflow pipeline test completed successfully!")
    print("📊 You can view the results by running: mlflow ui")
    
    return trained_model

if __name__ == "__main__":
    test_mlflow_pipeline()
