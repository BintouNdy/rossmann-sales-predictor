"""
Overfitting fixes and improved training strategies for Rossmann Sales Prediction

This module contains improved training functions with regularization techniques
to reduce overfitting and improve generalization.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def create_regularized_xgb_params(level='moderate'):
    """
    Create XGBoost parameters with different levels of regularization
    
    Args:
        level: 'light', 'moderate', 'aggressive'
    """
    base_params = {
        'objective': 'reg:squarederror',
        'verbosity': 0,
        'seed': 42,
        'eval_metric': 'rmse'
    }
    
    if level == 'light':
        regularized_params = {
            'max_depth': 6,           # Reduced from 7
            'eta': 0.05,              # Reduced from 0.068
            'subsample': 0.8,         # Reduced from 0.73
            'colsample_bytree': 0.8,  # Reduced from 0.98
            'gamma': 0.5,             # Slightly increased
            'reg_alpha': 0.1,         # L1 regularization
            'reg_lambda': 1.0,        # L2 regularization
            'min_child_weight': 3     # Increased from default 1
        }
    elif level == 'moderate':
        regularized_params = {
            'max_depth': 5,           # Further reduced
            'eta': 0.03,              # Lower learning rate
            'subsample': 0.7,         # More subsampling
            'colsample_bytree': 0.7,  # More feature subsampling
            'gamma': 1.0,             # Higher minimum split loss
            'reg_alpha': 0.5,         # Higher L1 regularization
            'reg_lambda': 2.0,        # Higher L2 regularization
            'min_child_weight': 5     # Higher minimum child weight
        }
    elif level == 'aggressive':
        regularized_params = {
            'max_depth': 4,           # Very shallow trees
            'eta': 0.01,              # Very low learning rate
            'subsample': 0.6,         # Heavy subsampling
            'colsample_bytree': 0.6,  # Heavy feature subsampling
            'gamma': 2.0,             # High minimum split loss
            'reg_alpha': 1.0,         # High L1 regularization
            'reg_lambda': 5.0,        # High L2 regularization
            'min_child_weight': 10    # High minimum child weight
        }
    
    base_params.update(regularized_params)
    return base_params


def train_with_early_stopping(X_train, y_train, X_val, y_val, params, max_rounds=3000):
    """
    Train XGBoost with proper early stopping using validation set
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Watchlist for monitoring training
    watchlist = [(dtrain, "train"), (dval, "validation")]
    
    # Train with early stopping
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=max_rounds,
        evals=watchlist,
        early_stopping_rounds=100,  # Stop if no improvement for 100 rounds
        verbose_eval=False  # Set to True if you want to see training progress
    )
    
    return model


def train_with_cv_early_stopping(X, y, params, test_size=0.2, val_size=0.2, random_state=42):
    """
    Train with proper train/validation/test splits and cross-validation
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate train and validation from remaining data
    val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    print(f"📊 Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Train model with early stopping
    model = train_with_early_stopping(X_train, y_train, X_val, y_val, params)
    
    # Evaluate on all sets
    dtrain = xgb.DMatrix(X_train)
    dval = xgb.DMatrix(X_val)
    dtest = xgb.DMatrix(X_test)
    
    y_pred_train = model.predict(dtrain)
    y_pred_val = model.predict(dval)
    y_pred_test = model.predict(dtest)
    
    # Calculate metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred_val)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'val_mae': mean_absolute_error(y_val, y_pred_val),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'train_r2': r2_score(y_train, y_pred_train),
        'val_r2': r2_score(y_val, y_pred_val),
        'test_r2': r2_score(y_test, y_pred_test)
    }
    
    # Calculate overfitting scores
    metrics['overfitting_rmse'] = abs(metrics['train_rmse'] - metrics['val_rmse']) / metrics['train_rmse']
    metrics['generalization_gap'] = abs(metrics['val_rmse'] - metrics['test_rmse']) / metrics['val_rmse']
    
    return model, metrics, (X_train, X_val, X_test, y_train, y_val, y_test)


def feature_importance_analysis(model, feature_names, top_n=20):
    """
    Analyze feature importance to identify potential overfitting features
    """
    importance_dict = model.get_score(importance_type='weight')
    
    # Create DataFrame with feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': [importance_dict.get(f"f{i}", 0) for i in range(len(feature_names))]
    }).sort_values('importance', ascending=False)
    
    print(f"\n📈 Top {top_n} Feature Importance:")
    print("=" * 50)
    for idx, row in importance_df.head(top_n).iterrows():
        print(f"{row['feature']:30} {row['importance']:>8.0f}")
    
    return importance_df


def suggest_feature_reduction(importance_df, threshold_percentile=70):
    """
    Suggest features to remove based on low importance
    """
    threshold = np.percentile(importance_df['importance'], threshold_percentile)
    low_importance_features = importance_df[importance_df['importance'] < threshold]['feature'].tolist()
    
    print(f"\n🔍 Features with importance below {threshold_percentile}th percentile (threshold={threshold:.1f}):")
    print("Consider removing these features to reduce overfitting:")
    for feature in low_importance_features:
        importance = importance_df[importance_df['feature'] == feature]['importance'].iloc[0]
        print(f"  - {feature}: {importance:.1f}")
    
    return low_importance_features


def run_regularization_experiment(X, y, feature_names):
    """
    Run experiments with different regularization levels
    """
    regularization_levels = ['light', 'moderate', 'aggressive']
    results = {}
    
    print("🧪 Running regularization experiments...")
    print("=" * 60)
    
    for level in regularization_levels:
        print(f"\n🔬 Testing {level} regularization...")
        
        # Get parameters for this level
        params = create_regularized_xgb_params(level)
        
        # Train model
        model, metrics, data_splits = train_with_cv_early_stopping(X, y, params)
        
        # Store results
        results[level] = {
            'model': model,
            'metrics': metrics,
            'params': params,
            'data_splits': data_splits
        }
        
        # Print results
        print(f"   📊 Train RMSE: {metrics['train_rmse']:.2f}")
        print(f"   📊 Val RMSE:   {metrics['val_rmse']:.2f}")
        print(f"   📊 Test RMSE:  {metrics['test_rmse']:.2f}")
        print(f"   🎯 Overfitting Score: {metrics['overfitting_rmse']:.3f}")
        print(f"   🎯 Generalization Gap: {metrics['generalization_gap']:.3f}")
        
        # Feature importance for this model
        importance_df = feature_importance_analysis(model, feature_names, top_n=10)
    
    # Find best model based on validation performance and overfitting
    best_level = min(results.keys(), 
                    key=lambda x: results[x]['metrics']['val_rmse'] + 
                                 results[x]['metrics']['overfitting_rmse'] * 100)
    
    print(f"\n🏆 Best regularization level: {best_level}")
    print("=" * 60)
    
    return results, best_level


def create_overfitting_report(metrics):
    """
    Create a comprehensive overfitting analysis report
    """
    print("\n" + "=" * 60)
    print("📋 OVERFITTING ANALYSIS REPORT")
    print("=" * 60)
    
    # RMSE Analysis
    print(f"📊 RMSE Scores:")
    print(f"   Train: {metrics['train_rmse']:.2f}")
    print(f"   Val:   {metrics['val_rmse']:.2f}")
    print(f"   Test:  {metrics['test_rmse']:.2f}")
    
    # Overfitting indicators
    overfitting_score = metrics['overfitting_rmse']
    generalization_gap = metrics['generalization_gap']
    
    print(f"\n🎯 Overfitting Indicators:")
    print(f"   Overfitting Score: {overfitting_score:.3f}")
    print(f"   Generalization Gap: {generalization_gap:.3f}")
    
    # Interpretation
    print(f"\n💡 Interpretation:")
    if overfitting_score < 0.1:
        print("   ✅ Low overfitting - good balance")
    elif overfitting_score < 0.3:
        print("   ⚠️  Moderate overfitting - consider more regularization")
    else:
        print("   ❌ High overfitting - strong regularization needed")
    
    if generalization_gap < 0.1:
        print("   ✅ Good generalization to test set")
    elif generalization_gap < 0.2:
        print("   ⚠️  Moderate generalization gap")
    else:
        print("   ❌ Poor generalization - model may not work on new data")
    
    # Recommendations
    print(f"\n🔧 Recommendations:")
    if overfitting_score > 0.2:
        print("   • Increase regularization (higher reg_alpha, reg_lambda)")
        print("   • Reduce model complexity (lower max_depth)")
        print("   • Increase subsampling (lower subsample, colsample_bytree)")
        print("   • Lower learning rate with more rounds")
        print("   • Remove low-importance features")
    
    if generalization_gap > 0.15:
        print("   • Consider collecting more training data")
        print("   • Check for data distribution shift")
        print("   • Implement cross-validation")
    
    print("=" * 60)


# Example usage function
def run_improved_training_pipeline():
    """
    Example of how to use the improved training functions
    """
    from training import load_data, preprocess_data, engineer_features
    
    # Load and prepare data
    print("📊 Loading and preparing data...")
    df = load_data()
    df = preprocess_data(df)
    df = engineer_features(df)
    
    # Select features (you can modify this list based on importance analysis)
    feature_columns = [
        'Store', 'DayOfWeek', 'Promo', 'CompetitionDistance',
        'CompetitionOpenSinceMonth', 'Day','StoreType_b','StoreType_c','StoreType_d',
        'WeekOfYear', 'Promo2Since', 'CompetitionOpenSince', 'DaysSinceStart', 'IsPromo2Active',
        'Sales_lag_1', 'Sales_lag_7', 'Sales_lag_14', 'Sales_Mean_3', 'Sales_Mean_7', 'Sales_Mean_14',
        'CompetitionIntensity', 'PromoDuringHoliday', 'IsAfterPromo', 'IsBeforePromo', 
        'Sales_DOW_Mean', 'Sales_DOW_Deviation'
    ]
    
    X = df[feature_columns]
    y = df['Sales']
    
    print(f"📈 Dataset: {len(X)} samples, {len(feature_columns)} features")
    
    # Run regularization experiments
    results, best_level = run_regularization_experiment(X, y, feature_columns)
    
    # Get best model and create report
    best_model = results[best_level]['model']
    best_metrics = results[best_level]['metrics']
    
    create_overfitting_report(best_metrics)
    
    # Feature importance analysis
    importance_df = feature_importance_analysis(best_model, feature_columns)
    low_importance_features = suggest_feature_reduction(importance_df)
    
    return best_model, best_metrics, importance_df, low_importance_features


if __name__ == "__main__":
    # Run the improved training pipeline
    model, metrics, importance_df, low_features = run_improved_training_pipeline()
