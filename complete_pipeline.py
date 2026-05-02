"""
Example 5: Complete Pipeline - From Data Loading to Model Evaluation
==================================================================
This is the complete, production-ready analysis pipeline that demonstrates
the full workflow: data loading, exploration, preprocessing, model training,
evaluation, and comprehensive visualization with results export.

Output folder structure:
    output/
    ├── data_exploration/    → scatter, distribution, correlation plots
    ├── training_results/    → training history, confusion matrix, predictions
    └── reports/             → summary report + CSV files
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from neural_networks import (
    NetworkConfig,
    FeedForwardNN,
    ModelEvaluator,
    one_hot_encode,
    standardize_features,
    set_random_seed
)

# ─── Chart settings ───────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

COLORS = {
    'primary':   '#2E86AB',
    'secondary': '#A23B72',
    'accent':    '#F18F01',
    'success':   '#06A77D',
    'danger':    '#D62246'
}

# ─── Output folders ───────────────────────────────────────────────────────────
OUTPUT_ROOT       = Path('output')
DATA_EXPLORATION  = OUTPUT_ROOT / 'data_exploration'
TRAINING_RESULTS  = OUTPUT_ROOT / 'training_results'
REPORTS           = OUTPUT_ROOT / 'reports'

def create_output_folders():
    """Create the three output folders if they don't exist."""
    for folder in [DATA_EXPLORATION, TRAINING_RESULTS, REPORTS]:
        folder.mkdir(parents=True, exist_ok=True)
    print("📁 Output folders created:")
    print(f"   • {DATA_EXPLORATION}")
    print(f"   • {TRAINING_RESULTS}")
    print(f"   • {REPORTS}\n")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def print_section(title):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")

# ─── Step 1: Load & Explore ───────────────────────────────────────────────────
def load_and_explore_data(
    filepath=Path(r'D:\VS_code\VS_code_WorkSpace\python_projects\nn\data\diabetes.csv')
):
    print_section("📊 Step 1: Load and Explore Data")

    if not Path(filepath).exists():
        print(f"⚠️  Warning: File {filepath} not found!")
        print("📝 Using sample data instead...\n")
        np.random.seed(42)
        n = 768
        data = pd.DataFrame({
            'Glucose':                  np.random.randint(50, 200, n),
            'Age':                      np.random.randint(21, 81,  n),
            'Pregnancies':              np.random.randint(0,  17,  n),
            'BloodPressure':            np.random.randint(40, 120, n),
            'SkinThickness':            np.random.randint(0,  100, n),
            'Insulin':                  np.random.randint(0,  850, n),
            'BMI':                      np.random.uniform(18,  50, n),
            'DiabetesPedigreeFunction': np.random.uniform(0.078, 2.42, n),
            'Outcome':                  np.random.randint(0, 2, n)
        })
    else:
        data = pd.read_csv(filepath)
        print(f"✅ Data loaded from: {filepath}\n")

    print("📋 General Data Information:")
    print(f"   • Number of Rows:    {len(data)}")
    print(f"   • Number of Columns: {len(data.columns)}")
    print(f"\n📊 First 5 Rows:\n{data.head()}")
    print(f"\n📈 Descriptive Statistics:\n{data.describe()}")

    counts = data['Outcome'].value_counts()
    print(f"\n🎯 Outcome Distribution:")
    print(f"   • No Diabetes (0): {counts.get(0,0)} ({counts.get(0,0)/len(data)*100:.1f}%)")
    print(f"   • Has Diabetes (1): {counts.get(1,0)} ({counts.get(1,0)/len(data)*100:.1f}%)")
    return data

# ─── Step 2: Visualize (→ data_exploration/) ──────────────────────────────────
def visualize_data_distribution(data):
    print_section("📊 Step 2: Visualize Data Distribution")

    feature_cols = ['Glucose', 'Age', 'Pregnancies', 'BloodPressure',
                    'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']

    # Plot 1 — Glucose vs Age scatter
    print("🖼️  Plot 1: Glucose vs Age Relationship")
    fig, ax = plt.subplots(figsize=(12, 6))
    for outcome, color, label in [(0, COLORS['primary'], 'No Diabetes'),
                                   (1, COLORS['danger'],  'Has Diabetes')]:
        mask = data['Outcome'] == outcome
        ax.scatter(data[mask]['Glucose'], data[mask]['Age'],
                   c=color, label=label, alpha=0.6, s=50,
                   edgecolors='white', linewidth=0.5)
    ax.set_xlabel('Glucose Level', fontsize=14, fontweight='bold')
    ax.set_ylabel('Age',           fontsize=14, fontweight='bold')
    ax.set_title('Relationship between Glucose Level and Age by Diabetes Status',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = DATA_EXPLORATION / 'output_1_glucose_age_scatter.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved → {out}\n")
    plt.show()

    # Plot 2 — Feature distributions
    print("🖼️  Plot 2: Distribution of All Features")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for idx, col in enumerate(feature_cols):
        ax = axes[idx]
        data[data['Outcome'] == 0][col].hist(ax=ax, bins=20, alpha=0.6,
                                              color=COLORS['primary'], label='No Diabetes')
        data[data['Outcome'] == 1][col].hist(ax=ax, bins=20, alpha=0.6,
                                              color=COLORS['danger'],  label='Has Diabetes')
        ax.set_title(col, fontsize=12, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.suptitle('Feature Distribution by Diabetes Status',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = DATA_EXPLORATION / 'output_2_features_distribution.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved → {out}\n")
    plt.show()

    # Plot 3 — Correlation matrix
    print("🖼️  Plot 3: Correlation Matrix")
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = data[feature_cols + ['Outcome']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Correlation Matrix Between Features',
                 fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    out = DATA_EXPLORATION / 'output_3_correlation_matrix.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved → {out}\n")
    plt.show()

# ─── Step 3: Prepare Data ─────────────────────────────────────────────────────
def prepare_data(data):
    print_section("🔧 Step 3: Prepare Data")

    feature_cols = ['Glucose', 'Age', 'Pregnancies', 'BloodPressure',
                    'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
    X = data[feature_cols].values
    y = data['Outcome'].values

    print(f"📐 Original Data Dimensions:")
    print(f"   • X (Features): {X.shape}")
    print(f"   • y (Target):   {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n✂️  Data Split:")
    print(f"   • Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   • Testing:  {len(X_test)}  samples ({len(X_test)/len(X)*100:.1f}%)")

    print(f"\n🔄 Standardizing Data (Z-score normalization)...")
    X_train, mean, std = standardize_features(X_train)
    X_test,  _,    _   = standardize_features(X_test, mean, std)
    print(f"   ✅ New Mean:               {X_train.mean():.6f}")
    print(f"   ✅ New Standard Deviation: {X_train.std():.6f}")

    print(f"\n🎯 Converting Target to One-Hot Encoding...")
    y_train_encoded = one_hot_encode(y_train)
    y_test_encoded  = one_hot_encode(y_test)
    print(f"   ✅ Target Shape After Conversion: {y_train_encoded.shape}")

    return X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded

# ─── Step 4: Build & Train ────────────────────────────────────────────────────
def build_and_train_model(X_train, y_train_encoded, X_test, y_test_encoded):
    print_section("🧠 Step 4: Build and Train Neural Network")

    set_random_seed(42)

    print("⚙️  Neural Network Configuration:")
    config = NetworkConfig(
        input_dim=X_train.shape[1],
        hidden_dims=[16, 8],
        output_dim=2,
        learning_rate=0.01,
        activation='relu',
        seed=42
    )
    print(f"   • Input Features:    {config.input_dim}")
    print(f"   • Hidden Layers:     {config.hidden_dims}")
    print(f"   • Output Classes:    {config.output_dim}")
    print(f"   • Learning Rate:     {config.learning_rate}")
    print(f"   • Activation:        {config.activation}")

    print(f"\n🏗️  Building Model...")
    model = FeedForwardNN(config)
    print("   ✅ Model built successfully!")

    print(f"\n🎓 Starting Training...")
    print("   " + "-" * 70)
    history = model.fit(
        X_train, y_train_encoded,
        X_val=X_test, y_val=y_test_encoded,
        epochs=300, batch_size=32,
        l2_reg=0.001, early_stopping=20,
        verbose=True
    )
    print("   " + "-" * 70)
    print(f"   ✅ Training completed after {len(history['train_loss'])} epochs!")
    return model, history

# ─── Step 5: Training History (→ training_results/) ───────────────────────────
def plot_training_history(history):
    print_section("📈 Step 5: Analyze Training Process")
    print("🖼️  Plot 4: Training Curves (Loss & Accuracy)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], linewidth=2.5, color=COLORS['primary'],
             label='Training Loss',    marker='o', markersize=4)
    ax1.plot(epochs, history['val_loss'],   linewidth=2.5, color=COLORS['danger'],
             label='Validation Loss',  marker='s', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Loss',  fontsize=13, fontweight='bold')
    ax1.set_title('Loss Curve During Training', fontsize=15, fontweight='bold', pad=15)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], linewidth=2.5, color=COLORS['primary'],
             label='Training Accuracy',   marker='o', markersize=4)
    ax2.plot(epochs, history['val_acc'],   linewidth=2.5, color=COLORS['danger'],
             label='Validation Accuracy', marker='s', markersize=4)
    ax2.set_xlabel('Epoch',    fontsize=13, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax2.set_title('Accuracy Curve During Training', fontsize=15, fontweight='bold', pad=15)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    out = TRAINING_RESULTS / 'output_4_training_history.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved → {out}\n")
    plt.show()

    print("📊 Training Summary:")
    print(f"   • Best Training Loss:       {min(history['train_loss']):.4f}")
    print(f"   • Best Validation Loss:     {min(history['val_loss']):.4f}")
    print(f"   • Best Training Accuracy:   {max(history['train_acc'])*100:.2f}%")
    print(f"   • Best Validation Accuracy: {max(history['val_acc'])*100:.2f}%")

# ─── Step 6: Evaluate ─────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test):
    print_section("🎯 Step 6: Evaluate Model")

    y_pred    = model.predict_classes(X_test)
    evaluator = ModelEvaluator()
    metrics   = evaluator.evaluate_classification(
        y_test, y_pred, class_names=['No Diabetes', 'Has Diabetes']
    )

    print("📊 Evaluation Results on Test Data:")
    print(f"\n   🎯 Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"   📏 Precision: {metrics['precision']*100:.2f}%")
    print(f"   🔍 Recall:    {metrics['recall']*100:.2f}%")
    print(f"   ⚖️  F1-Score:  {metrics['f1_score']*100:.2f}%")
    print(f"\n📋 Confusion Matrix:\n{metrics['confusion_matrix']}")
    print(f"\n📄 Full Classification Report:\n{metrics['classification_report']}")
    return metrics

# ─── Confusion Matrix (→ training_results/) ───────────────────────────────────
def plot_confusion_matrix(model, X_test, y_test):
    print("🖼️  Plot 5: Confusion Matrix")

    y_pred = model.predict_classes(X_test)
    cm     = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=['No Diabetes', 'Has Diabetes'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Confusion Matrix - Classification Results',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted', fontsize=13, fontweight='bold')
    ax.set_ylabel('Actual',    fontsize=13, fontweight='bold')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm[i, j] / cm[i].sum() * 100
            ax.text(j, i + 0.3, f'({pct:.1f}%)',
                    ha='center', va='center', fontsize=11, color='gray')

    plt.tight_layout()
    out = TRAINING_RESULTS / 'output_5_confusion_matrix.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved → {out}\n")
    plt.show()

# ─── Prediction Comparison (→ training_results/) ──────────────────────────────
def plot_prediction_comparison(model, X_test, y_test):
    print("🖼️  Plot 6: Comparing Predictions with Actual Values")

    y_pred_proba = model.predict(X_test)
    y_pred       = model.predict_classes(X_test)
    n            = min(50, len(y_test))
    x            = np.arange(n)
    width        = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    ax1.bar(x - width/2, y_test[:n], width, label='Actual Values',
            color=COLORS['primary'], alpha=0.8)
    ax1.bar(x + width/2, y_pred[:n], width, label='Predictions',
            color=COLORS['danger'],  alpha=0.8)
    ax1.set_xlabel('Sample Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Classification (0=No Diabetes, 1=Has Diabetes)',
                   fontsize=12, fontweight='bold')
    ax1.set_title('Comparing Actual Values with Predictions', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(i+1) for i in x], rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    proba    = y_pred_proba[:n, 1]
    colors   = [COLORS['danger'] if p > 0.5 else COLORS['primary'] for p in proba]
    ax2.bar(x, proba, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=2,
                label='Decision Boundary (50%)')
    ax2.set_xlabel('Sample Number', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Probability of Diabetes', fontsize=12, fontweight='bold')
    ax2.set_title('Prediction Probabilities for Each Sample', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(i+1) for i in x], rotation=45)
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = TRAINING_RESULTS / 'output_6_predictions_comparison.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved → {out}\n")
    plt.show()

# ─── Summary Report (→ reports/) ──────────────────────────────────────────────
def create_summary_report(data, history, metrics):
    print_section("📄 Step 7: Complete Summary Report")

    fig = plt.figure(figsize=(16, 12))
    gs  = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    def text_box(ax, text, color):
        ax.axis('off')
        ax.text(0.1, 0.5, text, fontsize=11, family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.1))

    text_box(fig.add_subplot(gs[0, 0]), f"""
📊 Data Information
{'='*40}
• Number of Samples:  {len(data)}
• Number of Features: {len(data.columns) - 1}
• No Diabetes:        {len(data[data['Outcome']==0])}
• Has Diabetes:       {len(data[data['Outcome']==1])}
• No Diabetes Ratio:  {len(data[data['Outcome']==0])/len(data)*100:.1f}%
• Diabetes Ratio:     {len(data[data['Outcome']==1])/len(data)*100:.1f}%
""", COLORS['primary'])

    text_box(fig.add_subplot(gs[0, 1]), f"""
🧠 Model Information
{'='*40}
• Model Type:         Feed-Forward NN
• Hidden Layers:      [16, 8]
• Activation:         ReLU
• Learning Rate:      0.01
• Epochs Trained:     {len(history['train_loss'])}
• Batch Size:         32
• Early Stopping:     Yes
""", COLORS['secondary'])

    text_box(fig.add_subplot(gs[1, 0]), f"""
📈 Training Results
{'='*40}
• Best Training Loss:       {min(history['train_loss']):.4f}
• Best Validation Loss:     {min(history['val_loss']):.4f}
• Best Training Accuracy:   {max(history['train_acc'])*100:.2f}%
• Best Validation Accuracy: {max(history['val_acc'])*100:.2f}%
• Overall Improvement:      {(history['train_loss'][0]-min(history['train_loss']))/history['train_loss'][0]*100:.1f}%
""", COLORS['accent'])

    text_box(fig.add_subplot(gs[1, 1]), f"""
🎯 Final Evaluation Results
{'='*40}
• Accuracy:   {metrics['accuracy']*100:.2f}%
• Precision:  {metrics['precision']*100:.2f}%
• Recall:     {metrics['recall']*100:.2f}%
• F1-Score:   {metrics['f1_score']*100:.2f}%

✅ Model is ready for use!
""", COLORS['success'])

    epochs = range(1, len(history['train_loss']) + 1)

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(epochs, history['train_loss'], linewidth=2, label='Training',  color=COLORS['primary'])
    ax5.plot(epochs, history['val_loss'],   linewidth=2, label='Validation', color=COLORS['danger'])
    ax5.set_xlabel('Epoch'); ax5.set_ylabel('Loss'); ax5.set_title('Loss Curve')
    ax5.legend(); ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(epochs, history['train_acc'], linewidth=2, label='Training',  color=COLORS['primary'])
    ax6.plot(epochs, history['val_acc'],   linewidth=2, label='Validation', color=COLORS['danger'])
    ax6.set_xlabel('Epoch'); ax6.set_ylabel('Accuracy'); ax6.set_title('Accuracy Curve')
    ax6.legend(); ax6.grid(True, alpha=0.3); ax6.set_ylim([0, 1])

    fig.suptitle('📊 Complete Analysis Report for Diabetes Data',
                 fontsize=18, fontweight='bold', y=0.98)

    out = REPORTS / 'output_7_complete_summary.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"🖼️  Plot 7: Complete Summary Report")
    print(f"   ✅ Saved → {out}\n")
    plt.show()

# ─── Save CSVs (→ reports/) ───────────────────────────────────────────────────
def save_results_to_csv(metrics, history):
    print("💾 Saving CSV Results...")

    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value':  [
            f"{metrics['accuracy']*100:.2f}%",
            f"{metrics['precision']*100:.2f}%",
            f"{metrics['recall']*100:.2f}%",
            f"{metrics['f1_score']*100:.2f}%"
        ]
    })
    out1 = REPORTS / 'results_metrics.csv'
    metrics_df.to_csv(out1, index=False, encoding='utf-8-sig')
    print(f"   ✅ Saved → {out1}")

    history_df = pd.DataFrame(history)
    out2 = REPORTS / 'results_training_history.csv'
    history_df.to_csv(out2, index=False)
    print(f"   ✅ Saved → {out2}")

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 80)
    print("  🚀 Starting Comprehensive Diabetes Data Analysis")
    print("=" * 80)

    # Create folder structure first
    create_output_folders()

    # Pipeline steps
    data                                                        = load_and_explore_data()
    visualize_data_distribution(data)
    X_train, X_test, y_train, y_test, y_train_enc, y_test_enc = prepare_data(data)
    model, history                                              = build_and_train_model(X_train, y_train_enc, X_test, y_test_enc)
    plot_training_history(history)
    metrics                                                     = evaluate_model(model, X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test)
    plot_prediction_comparison(model, X_test, y_test)
    create_summary_report(data, history, metrics)
    save_results_to_csv(metrics, history)

    # Final summary
    print("\n" + "=" * 80)
    print("  ✅ Analysis completed successfully!")
    print("=" * 80)
    print("""
📁 output/
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
    return model, history, metrics


if __name__ == '__main__':
    model, history, metrics = main()
