#!/usr/bin/env python3
"""
Quick comparison script to run both original and improved training
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_comparison():
    """
    Run comparison between original and improved training
    """
    from pipelines.training_with_mlflow import run_training_pipeline
    
    print("🔬 OVERFITTING FIX COMPARISON")
    print("=" * 60)
    
    # Original settings (recreated with original parameters)
    print("\n1️⃣  Testing ORIGINAL settings (high overfitting)...")
    print("-" * 40)
    
    original_hyperparams = {
        'max_depth': 7,
        'eta': 0.068,
        'subsample': 0.73,
        'colsample_bytree': 0.98,
        'gamma': 0.42,
        'reg_alpha': 0.0,   # No L1 regularization
        'reg_lambda': 1.0,  # Minimal L2 regularization
        'min_child_weight': 1
    }
    
    try:
        model1, _ = run_training_pipeline(
            experiment_name="comparison-original-vs-improved",
            run_name="original_parameters",
            hyperparams=original_hyperparams,
            regularization_level="light",  # Override with original params
            compare_with_previous=False
        )
        print("✅ Original settings test completed")
    except Exception as e:
        print(f"❌ Error with original settings: {e}")
    
    # Improved settings
    print("\n2️⃣  Testing IMPROVED settings (reduced overfitting)...")
    print("-" * 40)
    
    try:
        model2, comparison = run_training_pipeline(
            experiment_name="comparison-original-vs-improved",
            run_name="improved_regularization",
            regularization_level="moderate",
            compare_with_previous=True  # Compare with original
        )
        print("✅ Improved settings test completed")
    except Exception as e:
        print(f"❌ Error with improved settings: {e}")
    
    print("\n" + "=" * 60)
    print("📊 COMPARISON SUMMARY")
    print("=" * 60)
    print("Check the output above to compare:")
    print("• Train vs Test RMSE gap")
    print("• Overfitting scores")
    print("• Generalization gaps")
    print("\n💡 View detailed comparison in MLflow UI:")
    print("   mlflow ui")
    print("   Open: http://localhost:5000")
    print("   Navigate to 'comparison-original-vs-improved' experiment")


if __name__ == "__main__":
    run_comparison()
