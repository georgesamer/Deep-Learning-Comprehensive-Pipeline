"""
Main Script - Run All Neural Network Examples
==============================================
This script runs all 5 neural network examples in sequence:

1. Binary Classification on Tabular Data
2. Multi-Class Classification on Synthetic Data
3. Text Preprocessing and Classification
4. Keras/TensorFlow Model Building
5. Complete Pipeline (from data loading to evaluation)

Output folder structure:
    output/
    ├── examples/            ← outputs from examples 1-4
    ├── data_exploration/    ← from example 5
    ├── training_results/    ← from example 5
    └── reports/             ← from example 5

To run: python main.py
"""

import warnings
warnings.filterwarnings('ignore')

import os
from pathlib import Path

# Import main functions from all examples
from tabular_classification import main as example_1_main
from multiclass_classification import main as example_2_main
from text_preprocessing import main as example_3_main
from keras_models import main as example_4_main
from complete_pipeline import main as example_5_main

# ─── Output folder for examples 1-4 ──────────────────────────────────────────
EXAMPLES_DIR = Path('output') / 'examples'

def create_examples_folder():
    """Create output/examples/ folder if it doesn't exist."""
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Examples output folder ready: {EXAMPLES_DIR}\n")

def run_all_examples():
    """Run all 5 neural network examples"""

    print("\n" + "=" * 80)
    print("  NEURAL NETWORK EXAMPLES - MASTER RUNNER")
    print("  Running All 5 Examples in Sequence")
    print("=" * 80 + "\n")

    # Make sure output/examples/ exists before we start
    create_examples_folder()

    # Save the project root so we can come back after each example
    project_root = Path.cwd()

    examples = [
        ("Example 1: Binary Classification on Tabular Data", example_1_main),
        ("Example 2: Multi-Class Classification",            example_2_main),
        ("Example 3: Text Preprocessing and Classification", example_3_main),
        ("Example 4: Keras/TensorFlow Models",               example_4_main),
    ]

    total_examples = 5
    successful = 0
    failed     = 0

    print(f"📋 Total Examples: {total_examples}\n")

    # ── Examples 1-4: run from inside output/examples/ ────────────────────────
    for idx, (name, example_func) in enumerate(examples, 1):
        try:
            print("\n" + "=" * 80)
            print(f"  [{idx}/{total_examples}] {name}")
            print("=" * 80 + "\n")

            # Switch into the examples folder so savefig() lands here
            os.chdir(EXAMPLES_DIR)
            example_func()
            successful += 1
            print(f"\n✅ [{idx}/{total_examples}] {name} - COMPLETED\n")

        except Exception as e:
            failed += 1
            print(f"\n❌ [{idx}/{total_examples}] {name} - FAILED")
            print(f"   Error: {str(e)}\n")

        finally:
            # Always return to project root before the next example
            os.chdir(project_root)

    # ── Example 5: runs from project root (manages its own folders) ───────────
    try:
        print("\n" + "=" * 80)
        print(f"  [5/{total_examples}] Example 5: Complete Pipeline")
        print("=" * 80 + "\n")

        example_5_main()
        successful += 1
        print(f"\n✅ [5/{total_examples}] Example 5: Complete Pipeline - COMPLETED\n")

    except Exception as e:
        failed += 1
        print(f"\n❌ [5/{total_examples}] Example 5: Complete Pipeline - FAILED")
        print(f"   Error: {str(e)}\n")

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  EXECUTION SUMMARY")
    print("=" * 80 + "\n")
    print(f"📊 Results:")
    print(f"   • Total Examples: {total_examples}")
    print(f"   • ✅ Successful:  {successful}")
    print(f"   • ❌ Failed:      {failed}")
    print(f"   • Success Rate:   {(successful/total_examples)*100:.1f}%")

    print(f"""
📁 output/
├── examples/
│   ├── 01_binary_classification_results.png
│   ├── 02_multiclass_classification_results.png
│   ├── 03_text_classification_results.png
│   └── 04_keras_comparison_results.png
├── data_exploration/
│   ├── output_1_glucose_age_scatter.png
│   ├── output_2_features_distribution.png
│   └── output_3_correlation_matrix.png
├── training_results/
│   ├── output_4_training_history.png
│   ├── output_5_confusion_matrix.png
│   └── output_6_predictions_comparison.png
└── reports/
    ├── output_7_complete_summary.png
    ├── results_metrics.csv
    └── results_training_history.csv
""")
    print("🎉 All examples execution completed!\n")


if __name__ == '__main__':
    run_all_examples()
