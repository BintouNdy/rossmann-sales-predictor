import os
import sys
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
# Add the parent directory to the path so we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pipelines.training import train_model_dmatrix, load_data, preprocess_data, engineer_features
from pipelines.mlflow_utils import train_and_log_model, setup_mlflow, compare_runs

def get_features(df):
    """Get the list of feature columns to use for training"""
    return [
        'Store', 'DayOfWeek', 'Promo', 'CompetitionDistance',
        'CompetitionOpenSinceMonth', 'Day','StoreType_b','StoreType_c','StoreType_d',
        'WeekOfYear', 'Promo2Since', 'CompetitionOpenSince', 'DaysSinceStart', 'IsPromo2Active',
        'Sales_lag_1', 'Sales_lag_7', 'Sales_lag_14', 'Sales_Mean_3', 'Sales_Mean_7', 'Sales_Mean_14',
        'CompetitionIntensity', 'PromoDuringHoliday', 'IsAfterPromo', 'IsBeforePromo', 'Sales_DOW_Mean', 'Sales_DOW_Deviation'
    ]

def create_xgb_params(hyperparams=None, regularization_level='moderate'):
    """Crée un dictionnaire de paramètres pour xgb.train() avec régularisation améliorée"""
    
    # Parameters with strong regularization to combat overfitting
    if regularization_level == 'light':
        default_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,           # Reduced from 7
            'eta': 0.05,              # Reduced from 0.068
            'subsample': 0.8,         # Reduced from 0.73
            'colsample_bytree': 0.8,  # Reduced from 0.98
            'gamma': 0.5,             # Increased from 0.42
            'reg_alpha': 0.1,         # L1 regularization
            'reg_lambda': 1.0,        # L2 regularization
            'min_child_weight': 3,    # Increased from default 1
            'seed': 42,
            'verbosity': 0,
            'eval_metric': 'rmse'
        }
    elif regularization_level == 'moderate':
        default_params = {
            'objective': 'reg:squarederror',
            'max_depth': 5,           # Further reduced
            'eta': 0.03,              # Lower learning rate
            'subsample': 0.7,         # More subsampling
            'colsample_bytree': 0.7,  # More feature subsampling
            'gamma': 1.0,             # Higher minimum split loss
            'reg_alpha': 0.5,         # Higher L1 regularization
            'reg_lambda': 2.0,        # Higher L2 regularization
            'min_child_weight': 5,    # Higher minimum child weight
            'seed': 42,
            'verbosity': 0,
            'eval_metric': 'rmse'
        }
    elif regularization_level == 'aggressive':
        default_params = {
            'objective': 'reg:squarederror',
            'max_depth': 4,           # Very shallow trees
            'eta': 0.01,              # Very low learning rate
            'subsample': 0.6,         # Heavy subsampling
            'colsample_bytree': 0.6,  # Heavy feature subsampling
            'gamma': 2.0,             # High minimum split loss
            'reg_alpha': 1.0,         # High L1 regularization
            'reg_lambda': 5.0,        # High L2 regularization
            'min_child_weight': 10,   # High minimum child weight
            'seed': 42,
            'verbosity': 0,
            'eval_metric': 'rmse'
        }
    else:  # Default to moderate
        default_params = {
            'objective': 'reg:squarederror',
            'max_depth': 5,
            'eta': 0.03,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'gamma': 1.0,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
            'min_child_weight': 5,
            'seed': 42,
            'verbosity': 0,
            'eval_metric': 'rmse'
        }
    
    if hyperparams:
        default_params.update(hyperparams)
    
    return default_params

def run_training_pipeline(experiment_name="rossmann-sales-prediction", 
                         run_name=None, 
                         hyperparams=None,
                         regularization_level='aggressive',
                         test_size=0.2,
                         val_size=0.15,  # Add validation set
                         random_state=42,
                         compare_with_previous=True,
                         max_rounds=3000,
                         early_stopping_rounds=100):
    """
    Complete training pipeline with MLflow logging and model comparison
    Now includes proper train/validation/test splits and early stopping
    
    Args:
        experiment_name: MLflow experiment name
        run_name: Optional run name
        hyperparams: Optional model hyperparameters
        regularization_level: 'light', 'moderate', 'aggressive'
        test_size: Test set size (default 0.2)
        val_size: Validation set size (default 0.15)
        random_state: Random state for reproducibility
        compare_with_previous: Whether to compare with previous best model
        max_rounds: Maximum number of boosting rounds
        early_stopping_rounds: Early stopping patience
    """
    print("🚀 Démarrage de la pipeline d'entraînement avec MLflow (Anti-Overfitting)")
    
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
    
    # Three-way split: Train/Validation/Test
    print("📊 Division des données en Train/Validation/Test...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Adjust validation size for remaining data
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    print(f"📈 Données divisées: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test")
    
    # Créer les paramètres du modèle XGBoost avec régularisation
    params = create_xgb_params(hyperparams, regularization_level)
    print(f"🎛️  Utilisation du niveau de régularisation: {regularization_level}")
    
    # Train with proper early stopping using validation set
    print("🔄 Entraînement avec early stopping sur le set de validation...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Watchlist for monitoring training and validation
    watchlist = [(dtrain, "train"), (dval, "validation")]
    
    # Train model with early stopping on validation set
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=max_rounds,
        evals=watchlist,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False
    )
    
    print(f"✅ Entraînement terminé avec {booster.best_iteration} rounds (early stopping)")
    
    # Prepare tags
    tags = {
        "data_version": "v1.0",
        "features_count": str(len(feature_columns)),
        "test_size": str(test_size),
        "val_size": str(val_size),
        "regularization_level": regularization_level,
        "early_stopping_rounds": str(early_stopping_rounds),
        "best_iteration": str(booster.best_iteration)
    }
    
    if hyperparams:
        tags["hyperparams_tuned"] = "true"

    # Train and log model with comparison
    trained_model, comparison_results = train_and_log_model(
        model=booster,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,  # Still using test for final evaluation
        y_test=y_test,
        model_name="rossmann_xgboost_booster",
        experiment_name=experiment_name,
        run_name=run_name,
        tags=tags,
        compare_with_previous=compare_with_previous,
        X_val=X_val,  # Pass validation data for additional logging
        y_val=y_val
    )
    
    # Display comparison summary if available
    if comparison_results:
        print("\n" + "="*60)
        print("📋 RÉSUMÉ DE LA COMPARAISON")
        print("="*60)
        if comparison_results['summary'] == 'improvement':
            print("🎉 MODÈLE AMÉLIORÉ!")
        elif comparison_results['summary'] == 'regression':
            print("⚠️  RÉGRESSION DE PERFORMANCE")
        else:
            print("🤔 PERFORMANCE MIXTE")
        
        print(f"📊 Comparé avec: {comparison_results['previous_run_name']}")
        print("="*60)
    
    print("🎉 Pipeline d'entraînement terminée avec succès!")
    return trained_model, comparison_results


def run_hyperparameter_tuning(experiment_name="rossmann-sales-prediction-tuning"):
    """
    Run multiple experiments with different hyperparameters and regularization levels
    """
    print("🔍 Démarrage du tuning d'hyperparamètres et de régularisation...")
    
    # Different hyperparameter configurations to try with focus on reducing overfitting
    hyperparam_configs = [
        {
            "name": "light_regularization",
            "regularization_level": "light",
            "params": {}  # Use default parameters for this level
        },
        {
            "name": "moderate_regularization", 
            "regularization_level": "moderate",
            "params": {}  # Use default parameters for this level
        },
        {
            "name": "aggressive_regularization",
            "regularization_level": "aggressive", 
            "params": {}  # Use default parameters for this level
        },
        {
            "name": "moderate_with_higher_lr",
            "regularization_level": "moderate",
            "params": {"eta": 0.05}  # Slightly higher learning rate
        },
        {
            "name": "aggressive_with_custom_depth",
            "regularization_level": "aggressive",
            "params": {"max_depth": 3, "eta": 0.02}  # Even more conservative
        }
    ]
    
    models = []
    comparison_results = []
    
    for i, config in enumerate(hyperparam_configs):
        print(f"\n🧪 Test de la configuration: {config['name']} ({i+1}/{len(hyperparam_configs)})")
        print(f"   🎛️  Niveau de régularisation: {config['regularization_level']}")
        
        # For the first run, don't compare since there's no previous model in this experiment
        compare_with_previous = i > 0
        
        model, comparison = run_training_pipeline(
            experiment_name=experiment_name,
            run_name=config['name'],
            hyperparams=config['params'],
            regularization_level=config['regularization_level'],
            compare_with_previous=compare_with_previous
        )
        
        models.append((config['name'], model))
        if comparison:
            comparison_results.append(comparison)
    
    # Compare results across all runs
    print("\n📊 Comparaison des résultats de toutes les configurations...")
    runs_comparison = compare_runs(experiment_name, metric="test_rmse")
    
    # Summary of best performing configurations
    if runs_comparison is not None and not runs_comparison.empty:
        print(f"\n🏆 TOP 3 CONFIGURATIONS:")
        for i, (_, row) in enumerate(runs_comparison.head(3).iterrows()):
            run_name = row.get('tags.mlflow.runName', 'Unknown')
            test_rmse = row.get('metrics.test_rmse', 'N/A')
            val_rmse = row.get('metrics.val_rmse', 'N/A')
            overfitting = row.get('metrics.overfitting_score', 'N/A')
            test_r2 = row.get('metrics.test_r2', 'N/A')
            
            print(f"   {i+1}. {run_name}:")
            print(f"      Test RMSE: {test_rmse:.4f}")
            if val_rmse != 'N/A':
                print(f"      Val RMSE: {val_rmse:.4f}")
            print(f"      R²: {test_r2:.4f}")
            print(f"      Overfitting: {overfitting:.4f}")
    
    return models, comparison_results


def get_experiment_summary(experiment_name="rossmann-sales-prediction"):
    """
    Get a summary of all models in the experiment
    """
    from pipelines.mlflow_utils import setup_mlflow
    
    print(f"\n📊 Résumé de l'expérience: {experiment_name}")
    print("="*60)
    
    setup_mlflow(experiment_name)
    
    try:
        runs = mlflow.search_runs(experiment_ids=[mlflow.get_experiment_by_name(experiment_name).experiment_id])
        
        if runs.empty:
            print("❌ Aucun run trouvé dans cette expérience")
            return None
        
        # Sort by test_rmse (best first)
        if 'metrics.test_rmse' in runs.columns:
            runs_sorted = runs.sort_values('metrics.test_rmse', ascending=True)
            
            print(f"🏆 CLASSEMENT DES MODÈLES (par RMSE de test):")
            print("="*60)
            
            for i, (_, row) in enumerate(runs_sorted.head(10).iterrows()):
                run_name = row.get('tags.mlflow.runName', 'Unknown')
                test_rmse = row.get('metrics.test_rmse', 'N/A')
                test_r2 = row.get('metrics.test_r2', 'N/A')
                run_id = row['run_id'][:8]
                start_time = row.get('start_time', 'Unknown')
                
                print(f"{i+1:2d}. {run_name:20s} | RMSE: {test_rmse:8.4f} | R²: {test_r2:6.4f} | ID: {run_id}")
            
            # Show improvement over time
            runs_sorted['start_time'] = pd.to_datetime(runs_sorted['start_time'])
            latest_runs = runs_sorted.sort_values('start_time').tail(5)
            
            print(f"\n🕒 TENDANCE RÉCENTE (5 derniers runs):")
            print("="*60)
            
            for _, row in latest_runs.iterrows():
                run_name = row.get('tags.mlflow.runName', 'Unknown')
                test_rmse = row.get('metrics.test_rmse', 'N/A')
                start_time = row.get('start_time').strftime('%Y-%m-%d %H:%M') if pd.notna(row.get('start_time')) else 'Unknown'
                print(f"📅 {start_time} | {run_name:20s} | RMSE: {test_rmse:8.4f}")
        
        return runs_sorted
        
    except Exception as e:
        print(f"❌ Erreur lors de la récupération du résumé: {e}")
        return None


if __name__ == "__main__":
    print("🎯 Script de training MLflow avec comparaison de modèles")
    print("="*60)
    
    # Get experiment summary first
    get_experiment_summary()
    
    # Run the basic training pipeline with comparison enabled
    print("\n1️⃣ Entraînement du modèle de base avec comparaison...")
    trained_model, comparison_results = run_training_pipeline()
    
    # Display comparison results if available
    if comparison_results:
        print("\n" + "="*60)
        print("📈 ANALYSE DÉTAILLÉE DE LA COMPARAISON")
        print("="*60)
        for metric_name, improvement_info in comparison_results['improvements'].items():
            improvement_pct = improvement_info['improvement_pct']
            is_better = improvement_info['is_better']
            current = improvement_info['current']
            previous = improvement_info['previous']
            
            trend = "📈" if is_better else "📉"
            print(f"{trend} {metric_name.upper()}: {previous:.4f} → {current:.4f} ({improvement_pct:+.2f}%)")
        
        if 'prediction_correlation' in comparison_results:
            correlation = comparison_results['prediction_correlation']
            print(f"🔗 Corrélation des prédictions: {correlation:.4f}")
            if correlation > 0.95:
                print("   ✅ Prédictions très similaires (>95%)")
            elif correlation > 0.8:
                print("   🟡 Prédictions assez similaires (80-95%)")
            else:
                print("   ⚠️ Prédictions assez différentes (<80%)")
    
    print("\n" + "="*60)
    print("💡 Pour tester plusieurs configurations, décommentez la ligne suivante:")
    print("# run_hyperparameter_tuning()")
    print("="*60)
    
    # Uncomment the line below to run hyperparameter tuning
    # models, comparisons = run_hyperparameter_tuning()
