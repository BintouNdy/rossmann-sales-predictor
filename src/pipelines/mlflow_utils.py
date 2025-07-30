import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
import xgboost as xgb
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
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


def get_best_previous_model(experiment_name="rossmann-sales-prediction", metric="test_rmse", current_run_id=None):
    """
    Get the best previous model from MLflow experiments
    
    Args:
        experiment_name: Name of the experiment
        metric: Metric to optimize (lower is better for RMSE/MAE)
        current_run_id: Current run ID to exclude from comparison
    
    Returns:
        dict: Best model info with run_id, metrics, and model
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"❌ Expérience '{experiment_name}' non trouvée")
        return None
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if runs.empty:
        print("❌ Aucun run trouvé")
        return None
    
    # Exclude current run if provided
    if current_run_id:
        runs = runs[runs['run_id'] != current_run_id]
    
    # Sort by metric (ascending for RMSE/MAE, descending for R2)
    if f"metrics.{metric}" in runs.columns:
        ascending = metric.lower() in ['rmse', 'mae', 'mse', 'overfitting_score']
        runs_sorted = runs.sort_values(f"metrics.{metric}", ascending=ascending)
        
        if runs_sorted.empty:
            print("❌ Aucun run précédent trouvé")
            return None
            
        best_run = runs_sorted.iloc[0]
        
        # Try to load the model
        try:
            model_uri = f"runs:/{best_run['run_id']}/rossmann_xgboost_booster"
            model = mlflow.xgboost.load_model(model_uri)
            
            return {
                'run_id': best_run['run_id'],
                'metrics': {
                    'test_rmse': best_run.get('metrics.test_rmse', None),
                    'test_mae': best_run.get('metrics.test_mae', None),
                    'test_r2': best_run.get('metrics.test_r2', None),
                    'train_rmse': best_run.get('metrics.train_rmse', None),
                    'overfitting_score': best_run.get('metrics.overfitting_score', None)
                },
                'model': model,
                'run_name': best_run.get('tags.mlflow.runName', 'Unknown'),
                'start_time': best_run.get('start_time', None)
            }
        except Exception as e:
            print(f"⚠️ Impossible de charger le modèle du meilleur run: {e}")
            return {
                'run_id': best_run['run_id'],
                'metrics': {
                    'test_rmse': best_run.get('metrics.test_rmse', None),
                    'test_mae': best_run.get('metrics.test_mae', None),
                    'test_r2': best_run.get('metrics.test_r2', None),
                    'train_rmse': best_run.get('metrics.train_rmse', None),
                    'overfitting_score': best_run.get('metrics.overfitting_score', None)
                },
                'model': None,
                'run_name': best_run.get('tags.mlflow.runName', 'Unknown'),
                'start_time': best_run.get('start_time', None)
            }
    else:
        print(f"❌ Métrique '{metric}' non trouvée")
        return None


def compare_with_previous_model(current_metrics, previous_model_info, X_test, y_test):
    """
    Compare current model performance with previous best model
    
    Args:
        current_metrics: Dict of current model metrics
        previous_model_info: Dict from get_best_previous_model()
        X_test: Test features for prediction comparison
        y_test: Test labels for comparison
    
    Returns:
        dict: Comparison results and improvement metrics
    """
    if not previous_model_info:
        print("❌ Aucun modèle précédent trouvé pour la comparaison")
        return None
    
    print(f"\n🔍 Comparaison avec le meilleur modèle précédent:")
    print(f"   📋 Run précédent: {previous_model_info['run_name']} ({previous_model_info['run_id'][:8]}...)")
    
    comparison = {
        'previous_run_id': previous_model_info['run_id'],
        'previous_run_name': previous_model_info['run_name'],
        'improvements': {},
        'summary': 'unknown'
    }
    
    # Compare metrics
    prev_metrics = previous_model_info['metrics']
    
    for metric_name in ['test_rmse', 'test_mae', 'test_r2', 'overfitting_score']:
        if metric_name in current_metrics and prev_metrics.get(metric_name) is not None:
            current_val = current_metrics[metric_name]
            previous_val = prev_metrics[metric_name]
            
            # Calculate improvement (negative for metrics where lower is better)
            if metric_name in ['test_rmse', 'test_mae', 'overfitting_score']:
                improvement = (previous_val - current_val) / previous_val * 100
                is_better = current_val < previous_val
            else:  # r2_score - higher is better
                improvement = (current_val - previous_val) / abs(previous_val) * 100
                is_better = current_val > previous_val
            
            comparison['improvements'][metric_name] = {
                'current': current_val,
                'previous': previous_val,
                'improvement_pct': improvement,
                'is_better': is_better
            }
            
            status = "✅ Amélioration" if is_better else "❌ Dégradation"
            print(f"   📊 {metric_name.upper()}: {previous_val:.4f} → {current_val:.4f} ({improvement:+.2f}%) {status}")
    
    # Overall assessment
    better_count = sum(1 for imp in comparison['improvements'].values() if imp['is_better'])
    total_count = len(comparison['improvements'])
    
    if better_count > total_count / 2:
        comparison['summary'] = 'improvement'
        print(f"\n🎉 Modèle amélioré! ({better_count}/{total_count} métriques améliorées)")
    elif better_count == total_count / 2:
        comparison['summary'] = 'mixed'
        print(f"\n🤔 Performance mixte ({better_count}/{total_count} métriques améliorées)")
    else:
        comparison['summary'] = 'regression'
        print(f"\n⚠️ Régression de performance ({better_count}/{total_count} métriques améliorées)")
    
    # If we have the previous model, compare predictions
    if previous_model_info['model'] is not None and X_test is not None and y_test is not None:
        try:
            # Make predictions with previous model
            dtest = xgb.DMatrix(X_test)
            prev_predictions = previous_model_info['model'].predict(dtest)
            
            # Calculate prediction correlation
            current_predictions = current_metrics.get('predictions', None)
            if current_predictions is not None:
                correlation, _ = pearsonr(current_predictions, prev_predictions)
                comparison['prediction_correlation'] = correlation
                print(f"   🔗 Corrélation des prédictions: {correlation:.4f}")
        except Exception as e:
            print(f"   ⚠️ Impossible de comparer les prédictions: {e}")
    
    return comparison


def train_and_log_model(model, X_train, y_train, X_test, y_test,
                        model_name, experiment_name, run_name=None, tags=None, save_model=True,
                        compare_with_previous=True, X_val=None, y_val=None):
    """
    Enregistre un modèle XGBoost Booster dans MLflow avec évaluation et comparaison.
    Maintenant supporte les données de validation pour une évaluation plus complète.
    
    Args:
        model: Booster entraîné via xgb.train
        X_train, y_train: Données d'entraînement (pandas ou numpy)
        X_test, y_test: Données de test
        model_name: Nom du modèle pour l'enregistrement
        experiment_name: Nom de l'expérience MLflow
        run_name: Nom facultatif de la run
        tags: Dictionnaire de tags MLflow
        save_model: Sauvegarder le modèle dans MLflow
        compare_with_previous: Comparer avec le meilleur modèle précédent
        X_val, y_val: Données de validation optionnelles
    Returns:
        tuple: (model, comparison_results)
    """
    # Setup experiment
    setup_mlflow(experiment_name)
    
    # Get best previous model before starting new run (if comparison is enabled)
    previous_model_info = None
    if compare_with_previous:
        print("🔍 Recherche du meilleur modèle précédent...")
        previous_model_info = get_best_previous_model(experiment_name, metric="test_rmse")
    
    with mlflow.start_run(run_name=run_name):
        # Get current run info
        current_run = mlflow.active_run()
        current_run_id = current_run.info.run_id
        
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
        
        # Calculate overfitting score
        overfitting_score = abs(train_rmse - test_rmse) / train_rmse

        # If validation data is provided, calculate validation metrics
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            y_pred_val = model.predict(dval)
            
            val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            val_mae = mean_absolute_error(y_val, y_pred_val)
            val_r2 = r2_score(y_val, y_pred_val)
            
            # Better overfitting score using validation set
            overfitting_score = abs(train_rmse - val_rmse) / train_rmse
            generalization_gap = abs(val_rmse - test_rmse) / val_rmse
            
            # Log validation metrics
            mlflow.log_metrics({
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "val_r2": val_r2,
                "generalization_gap": generalization_gap
            })
            
            print(f"✅ Métriques — Train RMSE: {train_rmse:.2f}, Val RMSE: {val_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
            print(f"📊 R² Score — Train: {train_r2:.3f}, Val: {val_r2:.3f}, Test: {test_r2:.3f}")
            print(f"🎯 Overfitting Score: {overfitting_score:.3f}, Generalization Gap: {generalization_gap:.3f}")
            
            # Update current metrics for comparison
            current_metrics = {
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "test_rmse": test_rmse,
                "train_mae": train_mae,
                "val_mae": val_mae,
                "test_mae": test_mae,
                "train_r2": train_r2,
                "val_r2": val_r2,
                "test_r2": test_r2,
                "overfitting_score": overfitting_score,
                "generalization_gap": generalization_gap,
                "predictions": y_pred_test
            }
        else:
            # Original metrics without validation
            current_metrics = {
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "train_mae": train_mae,
                "test_mae": test_mae,
                "train_r2": train_r2,
                "test_r2": test_r2,
                "overfitting_score": overfitting_score,
                "predictions": y_pred_test
            }
            
            print(f"✅ Métriques — Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
            print(f"📊 R² Score — Train: {train_r2:.3f}, Test: {test_r2:.3f}")

        # Log métriques principales
        mlflow.log_metrics({
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "overfitting_score": overfitting_score
        })
        
        # Log infos données
        data_params = {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features_count": X_train.shape[1]
        }
        if X_val is not None:
            data_params["val_samples"] = len(X_val)
        
        mlflow.log_params(data_params)
        
        # Compare with previous model
        comparison_results = None
        if compare_with_previous and previous_model_info:
            comparison_results = compare_with_previous_model(
                current_metrics, previous_model_info, X_test, y_test
            )
            
            # Log comparison metrics
            if comparison_results:
                mlflow.set_tag("comparison.previous_run_id", comparison_results['previous_run_id'])
                mlflow.set_tag("comparison.summary", comparison_results['summary'])
                
                # Log improvement percentages
                for metric_name, improvement_info in comparison_results['improvements'].items():
                    mlflow.log_metric(f"improvement.{metric_name}_pct", improvement_info['improvement_pct'])
                    mlflow.log_metric(f"previous.{metric_name}", improvement_info['previous'])
                
                if 'prediction_correlation' in comparison_results:
                    mlflow.log_metric("comparison.prediction_correlation", comparison_results['prediction_correlation'])
        
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
            print(f"⚠️ Impossible de logger l'importance des features : {e}")
        
        return model, comparison_results


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
