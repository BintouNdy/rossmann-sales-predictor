#!/usr/bin/env python3
"""
Anti-Overfitting Training Script for Rossmann Sales Prediction

This script specifically addresses the overfitting issue where:
- Training RMSE: 22.29
- Test RMSE: 121.62

The script tests different regularization levels and provides recommendations.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pipelines.training_with_mlflow import run_training_pipeline, run_hyperparameter_tuning
from pipelines.overfitting_fixes import run_regularization_experiment
from pipelines.training import load_data, preprocess_data, engineer_features


def quick_overfitting_test():
    """
    Quick test to see the impact of regularization on overfitting
    """
    print("🎯 QUICK OVERFITTING TEST")
    print("=" * 60)
    print("Testing different regularization levels to reduce overfitting...")
    print("Current problem: Train RMSE: 22.29, Test RMSE: 121.62")
    print("=" * 60)
    
    # Test configurations from least to most regularized
    configs = [
        {
            "name": "original_settings", 
            "regularization_level": "light",
            "description": "Light regularization (closer to original)"
        },
        {
            "name": "moderate_regularization", 
            "regularization_level": "moderate",
            "description": "Moderate regularization (recommended starting point)"
        },
        {
            "name": "aggressive_regularization", 
            "regularization_level": "aggressive", 
            "description": "Aggressive regularization (if moderate still overfits)"
        }
    ]
    
    results = {}
    experiment_name = "overfitting-fix-experiment"
    
    for i, config in enumerate(configs):
        print(f"\n🧪 Test {i+1}/3: {config['description']}")
        print("-" * 50)
        
        try:
            model, comparison = run_training_pipeline(
                experiment_name=experiment_name,
                run_name=config['name'],
                regularization_level=config['regularization_level'],
                compare_with_previous=(i > 0)
            )
            
            results[config['name']] = {
                'model': model,
                'comparison': comparison,
                'config': config
            }
            
        except Exception as e:
            print(f"❌ Erreur lors du test {config['name']}: {e}")
            continue
    
    # Summary and recommendations
    print("\n" + "=" * 60)
    print("📋 SUMMARY & RECOMMENDATIONS")
    print("=" * 60)
    
    if results:
        print("✅ Tests completed successfully!")
        print("\n🎯 Key Improvements:")
        print("• Separated validation set for proper early stopping")
        print("• Added L1/L2 regularization")
        print("• Reduced tree depth and learning rate")
        print("• Increased subsampling for variance reduction")
        
        print("\n💡 Next Steps:")
        print("1. Check MLflow UI to compare results")
        print("2. Choose the best regularization level")
        print("3. Consider feature selection if still overfitting")
        print("4. Monitor validation vs test performance gap")
        
        print(f"\n🔗 View results in MLflow:")
        print(f"   mlflow ui --backend-store-uri file:./mlruns")
        print(f"   Then open: http://localhost:5000")
        
    else:
        print("❌ No successful tests completed")
    
    return results


def comprehensive_feature_analysis():
    """
    Analyze features to identify potential overfitting sources
    """
    print("\n🔍 COMPREHENSIVE FEATURE ANALYSIS")
    print("=" * 60)
    
    # Load data for analysis
    df = load_data()
    df = preprocess_data(df)
    df = engineer_features(df)
    
    # Get features
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
    
    print(f"📊 Dataset: {len(X)} samples, {len(feature_columns)} features")
    
    # Run feature analysis
    try:
        from pipelines.overfitting_fixes import run_regularization_experiment
        results, best_level = run_regularization_experiment(X, y, feature_columns)
        
        print(f"\n🏆 Best regularization level from analysis: {best_level}")
        return results, best_level
        
    except Exception as e:
        print(f"❌ Error in feature analysis: {e}")
        return None, None


def main():
    """
    Main function to run overfitting fixes
    """
    print("🚀 ROSSMANN OVERFITTING FIX - MAIN SCRIPT")
    print("=" * 60)
    print("Addressing the overfitting issue:")
    print("• Current: Train RMSE: 22.29, Test RMSE: 121.62")
    print("• Target: Reduce the gap between train and test performance")
    print("=" * 60)
    
    # Step 1: Quick regularization test
    print("\n📍 STEP 1: Quick Regularization Test")
    results = quick_overfitting_test()
    
    # Step 2: Feature analysis (optional, more detailed)
    while True:
        choice = input("\n🤔 Run detailed feature analysis? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            print("\n📍 STEP 2: Detailed Feature Analysis")
            feature_results, best_level = comprehensive_feature_analysis()
            break
        elif choice in ['n', 'no']:
            print("\n⏭️  Skipping detailed feature analysis")
            break
        else:
            print("Please enter 'y' or 'n'")
    
    # Final recommendations
    print("\n" + "=" * 60)
    print("🎯 FINAL RECOMMENDATIONS")
    print("=" * 60)
    print("1. Use 'moderate' or 'aggressive' regularization level")
    print("2. Monitor validation RMSE during training")
    print("3. Look for overfitting_score < 0.2 in MLflow")
    print("4. Consider removing lag features if still overfitting")
    print("5. Ensure early stopping on validation set")
    
    print("\n📊 To view results:")
    print("   mlflow ui")
    print("   # Then open http://localhost:5000")
    
    print("\n✅ Overfitting fix script completed!")


if __name__ == "__main__":
    main()
