"""
Main Script - Run All Neural Network Examples
==============================================

This script runs all 5 neural network examples in sequence:
1. Binary Classification on Tabular Data
2. Multi-Class Classification on Synthetic Data
3. Text Preprocessing and Classification
4. Keras/TensorFlow Model Building
5. Complete Pipeline (from data loading to evaluation)

To run: python main.py
"""

import warnings
warnings.filterwarnings('ignore')

# Import main functions from all examples
from tabular_classification import main as example_1_main
from multiclass_classification import main as example_2_main
from text_preprocessing import main as example_3_main
from keras_models import main as example_4_main
from complete_pipeline import main as example_5_main


def run_all_examples():
    """Run all 5 neural network examples"""
    
    print("\n" + "="*80)
    print("  NEURAL NETWORK EXAMPLES - MASTER RUNNER")
    print("  Running All 5 Examples in Sequence")
    print("="*80 + "\n")
    
    examples = [
        ("Example 1: Binary Classification on Tabular Data", example_1_main),
        ("Example 2: Multi-Class Classification", example_2_main),
        ("Example 3: Text Preprocessing and Classification", example_3_main),
        ("Example 4: Keras/TensorFlow Models", example_4_main),
        ("Example 5: Complete Pipeline", example_5_main),
    ]
    
    total_examples = len(examples)
    successful = 0
    failed = 0
    
    print(f"📋 Total Examples: {total_examples}\n")
    
    for idx, (name, example_func) in enumerate(examples, 1):
        try:
            print("\n" + "="*80)
            print(f"  [{idx}/{total_examples}] {name}")
            print("="*80 + "\n")
            
            # Run the example
            example_func()
            
            successful += 1
            print(f"\n✅ [{idx}/{total_examples}] {name} - COMPLETED\n")
            
        except Exception as e:
            failed += 1
            print(f"\n❌ [{idx}/{total_examples}] {name} - FAILED")
            print(f"   Error: {str(e)}\n")
    
    # Final summary
    print("\n" + "="*80)
    print("  EXECUTION SUMMARY")
    print("="*80 + "\n")
    
    print(f"📊 Results:")
    print(f"   • Total Examples:    {total_examples}")
    print(f"   • ✅ Successful:     {successful}")
    print(f"   • ❌ Failed:         {failed}")
    print(f"   • Success Rate:      {(successful/total_examples)*100:.1f}%")
    
    print(f"\n📁 Output Files Generated:")
    print(f"   • 01_binary_classification_results.png")
    print(f"   • 02_multiclass_classification_results.png")
    print(f"   • 03_text_classification_results.png")
    print(f"   • 04_keras_comparison_results.png")
    print(f"   • output_1_glucose_age_scatter.png")
    print(f"   • output_2_features_distribution.png")
    print(f"   • output_3_correlation_matrix.png")
    print(f"   • results_metrics.csv")
    print(f"   • results_training_history.csv")
    
    print(f"\n🎉 All examples execution completed!\n")


if __name__ == '__main__':
    run_all_examples()
