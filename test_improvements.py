#!/usr/bin/env python3
"""
Simple test script to run the improved training pipeline
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pipelines.training_with_mlflow import run_training_pipeline


def test_improved_training():
    """
    Test the improved training pipeline with regularization
    """
    print("🧪 Testing improved training pipeline...")
    print("=" * 50)
    
    # Test with moderate regularization
    model, comparison = run_training_pipeline(
        experiment_name="test-overfitting-fix",
        run_name="moderate_regularization_test",
        regularization_level="moderate",  # Use aggressive regularization
        compare_with_previous=False  # First run
    )
    
    print("✅ Test completed!")
    print("\n💡 Next steps:")
    print("1. Check the output above for RMSE values")
    print("2. Look for much smaller gap between train/val/test RMSE")
    print("3. Run: mlflow ui to see detailed results")
    
    return model, comparison


if __name__ == "__main__":
    model, comparison = test_improved_training()
