import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
import xgboost as xgb
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import datetime


def setup_mlflow(experiment_name="rossmann-sales-prediction", tracking_uri=None):
    """
    Configure MLflow experiment and tracking
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        # Use local tracking
        mlflow.set_tracking_uri("file:./mlruns")
    
    # Set or create experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"✨ Création de l'expérience MLflow: {experiment_name}")
    else:
        experiment_id = experiment.experiment_id
        print(f"📊 Utilisation de l'expérience existante: {experiment_name}")
    
    mlflow.set_experiment(experiment_name)
    return experiment_id


def train_and_log_model(model, X_train, y_train, X_test, y_test,
                        model_name, experiment_name, run_name=None, tags=None, save_model=True):
    """
    Enregistre un modèle XGBoost Booster dans MLflow avec évaluation.
    
    Args:
        model: Booster entraîné via xgb.train
        X_train, y_train: Données d'entraînement (pandas ou numpy)
        X_test, y_test: Données de test
        model_name: Nom du modèle pour l'enregistrement
        experiment_name: Nom de l'expérience MLflow
        run_name: Nom facultatif de la run
        tags: Dictionnaire de tags MLflow
    Returns:
        model: Booster XGBoost
    """
    # Setup experiment
    setup_mlflow(experiment_name)
    
    with mlflow.start_run(run_name=run_name):
        # Set tags
        if tags:
            mlflow.set_tags(tags)
        
        # Tags par défaut
        mlflow.set_tag("model_type", "xgboost")
        mlflow.set_tag("dataset", "rossmann")
        mlflow.set_tag("training_date", datetime.datetime.now().isoformat())

        print("🚀 Évaluation du modèle booster...")

        # Convert to DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Prédictions
        y_pred_train = model.predict(dtrain)
        y_pred_test = model.predict(dtest)

        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        # Log métriques
        mlflow.log_metrics({
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "overfitting_score": abs(train_rmse - test_rmse) / train_rmse
        })
        
        # Log infos données
        mlflow.log_params({
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features_count": X_train.shape[1]
        })
        
        print(f"✅ Métriques — Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
        print(f"📊 R² Score — Train: {train_r2:.3f}, Test: {test_r2:.3f}")
        
        # Log the model with signature
        if save_model:
            signature = infer_signature(X_train, y_pred_test)
            mlflow.xgboost.log_model(
                model, 
                model_name, 
                signature=signature,
                input_example=X_train.iloc[:5] if hasattr(X_train, 'iloc') else X_train[:5]
            )
            print(f"📦 Booster XGBoost loggé dans MLflow sous '{model_name}'")
        
       # Résultats prédictions
        results_df = pd.DataFrame({
            'actual': y_test.values if hasattr(y_test, 'values') else y_test,
            'predicted': y_pred_test,
            'residual': (y_test.values if hasattr(y_test, 'values') else y_test) - y_pred_test
        })        
        results_path = "predictions.csv"
        results_df.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path)
        os.remove(results_path)  # Clean up local file
        
        # Importance des features
        try:
            importance_dict = model.get_score(importance_type='weight')
            features = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f"f{i}" for i in range(X_train.shape[1])]
            importance_df = pd.DataFrame({
                'feature': features,
                'importance': [importance_dict.get(f"f{i}", 0) for i in range(len(features))]
            }).sort_values('importance', ascending=False)

            importance_path = "feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)
            os.remove(importance_path)
            print("📈 Importance des features enregistrée")
        except Exception as e:
            print(f"⚠️ Impossible de logger l’importance des features : {e}")
        
        return model


def load_model_from_mlflow(model_name, run_id=None, version=None):
    """
    Load a model from MLflow
    
    Args:
        model_name: Name of the model
        run_id: Specific run ID (optional)
        version: Model version (optional)
    """
    if run_id:
        model_uri = f"runs:/{run_id}/{model_name}"
    elif version:
        model_uri = f"models:/{model_name}/{version}"
    else:
        model_uri = f"models:/{model_name}/latest"
    
    try:
        model = mlflow.xgboost.load_model(model_uri)
        print(f"✅ Modèle chargé depuis MLflow: {model_uri}")
        return model
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return None


def register_model(model_name, run_id, registered_model_name=None):
    """
    Register a model in MLflow Model Registry
    """
    if not registered_model_name:
        registered_model_name = model_name
    
    model_uri = f"runs:/{run_id}/{model_name}"
    
    try:
        model_version = mlflow.register_model(model_uri, registered_model_name)
        print(f"🏷️  Modèle enregistré: {registered_model_name}, Version: {model_version.version}")
        return model_version
    except Exception as e:
        print(f"❌ Erreur lors de l'enregistrement: {e}")
        return None


def compare_runs(experiment_name="rossmann-sales-prediction", metric="test_rmse"):
    """
    Compare runs in an experiment
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"❌ Expérience '{experiment_name}' non trouvée")
        return None
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if runs.empty:
        print("❌ Aucun run trouvé")
        return None
    
    # Sort by metric
    if f"metrics.{metric}" in runs.columns:
        runs_sorted = runs.sort_values(f"metrics.{metric}")
        print(f"🏆 Meilleur run basé sur {metric}:")
        best_run = runs_sorted.iloc[0]
        print(f"Run ID: {best_run['run_id']}")
        print(f"{metric}: {best_run[f'metrics.{metric}']:.4f}")
        return runs_sorted
    else:
        print(f"❌ Métrique '{metric}' non trouvée")
        return runs
