import os
import sys
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pipelines.mlflow_utils import train_and_log_model, setup_mlflow, compare_runs
from features import get_features, load_data, preprocess_data, engineer_features

def create_model(hyperparams=None):
    """Create XGBoost model with given hyperparameters"""
    default_params = {
        'n_estimators': 260,
        'max_depth': 7,
        'learning_rate': 0.068,
        'subsample': 0.73,
        'colsample_bytree': 0.98,
        'gamma': 0.42,
        'random_state': 42,
        'verbosity': 0
    }
    
    if hyperparams:
        default_params.update(hyperparams)
    
    return xgb.XGBRegressor(**default_params)

def run_training_pipeline(experiment_name="rossmann-sales-prediction", 
                         run_name=None, 
                         hyperparams=None,
                         test_size=0.2,
                         random_state=42):
    """
    Complete training pipeline with MLflow logging
    
    Args:
        experiment_name: MLflow experiment name
        run_name: Optional run name
        hyperparams: Optional model hyperparameters
        test_size: Test set size (default 0.2)
        random_state: Random state for reproducibility
    """
    print("🚀 Démarrage de la pipeline d'entraînement avec MLflow")
    
    # Setup MLflow
    setup_mlflow(experiment_name)
    
    # Load and preprocess data
    print("📊 Chargement des données...")
    df = load_data()
    
    print("🔧 Préprocessing des données...")
    df = preprocess_data(df)
    
    print("⚙️  Feature engineering...")
    df = engineer_features(df)
    
    # Select features
    feature_columns = get_features(df)
    X = df[feature_columns]
    y = df['Sales']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"📈 Données divisées: {len(X_train)} train, {len(X_test)} test")
    
    # Create model
    model = create_model(hyperparams)
    
    # Prepare tags
    tags = {
        "data_version": "v1.0",
        "features_count": str(len(feature_columns)),
        "test_size": str(test_size)
    }
    
    if hyperparams:
        tags["hyperparams_tuned"] = "true"
    
    # Train and log model
    trained_model = train_and_log_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_name="rossmann_xgboost",
        experiment_name=experiment_name,
        run_name=run_name,
        tags=tags
    )
    
    print("🎉 Pipeline d'entraînement terminée avec succès!")
    return trained_model


def run_hyperparameter_tuning(experiment_name="rossmann-sales-prediction-tuning"):
    """
    Run multiple experiments with different hyperparameters
    """
    print("🔍 Démarrage du tuning d'hyperparamètres...")
    
    # Different hyperparameter combinations to try
    hyperparam_configs = [
        {
            "name": "baseline",
            "params": {}  # Use default parameters
        },
        {
            "name": "high_learning_rate",
            "params": {"learning_rate": 0.1, "n_estimators": 200}
        },
        {
            "name": "deep_trees",
            "params": {"max_depth": 10, "min_child_weight": 3}
        },
        {
            "name": "conservative",
            "params": {"learning_rate": 0.05, "n_estimators": 400, "subsample": 0.8}
        },
        {
            "name": "aggressive_regularization",
            "params": {"gamma": 1.0, "reg_alpha": 0.1, "reg_lambda": 1.0}
        }
    ]
    
    models = []
    for config in hyperparam_configs:
        print(f"\n🧪 Test de la configuration: {config['name']}")
        model = run_training_pipeline(
            experiment_name=experiment_name,
            run_name=config['name'],
            hyperparams=config['params']
        )
        models.append((config['name'], model))
    
    # Compare results
    print("\n📊 Comparaison des résultats...")
    compare_runs(experiment_name, metric="test_rmse")
    
    return models


if __name__ == "__main__":
    # Run the basic training pipeline
    run_training_pipeline()
    
    # Uncomment the line below to run hyperparameter tuning
    # run_hyperparameter_tuning()
