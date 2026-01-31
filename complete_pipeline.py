"""
Example 5: Complete Pipeline - From Data Loading to Model Evaluation
==================================================================

This is the complete, production-ready analysis pipeline that demonstrates
the full workflow: data loading, exploration, preprocessing, model training,
evaluation, and comprehensive visualization with results export.

This example integrates all aspects:
- Comprehensive data loading and exploration
- Multiple visualizations (scatter plots, distributions, correlations)
- Complete preprocessing pipeline
- Model training with validation
- Detailed evaluation and confusion matrix
- Results export to CSV
- Professional summary reporting
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


# Chart settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# Color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#06A77D',
    'danger': '#D62246'
}


def print_section(title):
    """Print a section title clearly"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def load_and_explore_data(filepath=Path(r'D:\VS_code\VS_code_WorkSpace\python_projects\nn\data\diabetes.csv')):
    """
    Load and explore the dataset
    """
    print_section("📊 Step 1: Load and Explore Data")
    
    # Load the data
    if not Path(filepath).exists():
        print(f"⚠️  Warning: File {filepath} not found!")
        print("📝 Using sample data instead...\n")
        
        # Create sample data
        np.random.seed(42)
        n_samples = 768
        data = pd.DataFrame({
            'Glucose': np.random.randint(50, 200, n_samples),
            'Age': np.random.randint(21, 81, n_samples),
            'Pregnancies': np.random.randint(0, 17, n_samples),
            'BloodPressure': np.random.randint(40, 120, n_samples),
            'SkinThickness': np.random.randint(0, 100, n_samples),
            'Insulin': np.random.randint(0, 850, n_samples),
            'BMI': np.random.uniform(18, 50, n_samples),
            'DiabetesPedigreeFunction': np.random.uniform(0.078, 2.42, n_samples),
            'Outcome': np.random.randint(0, 2, n_samples)
        })
    else:
        data = pd.read_csv(filepath)
        print(f"✅ Data loaded from: {filepath}\n")
    
    # Display data information
    print("📋 General Data Information:")
    print(f"   • Number of Rows: {len(data)}")
    print(f"   • Number of Columns: {len(data.columns)}")
    print(f"\n📊 First 5 Rows:")
    print(data.head())
    
    print(f"\n📈 Descriptive Statistics:")
    print(data.describe())
    
    # Outcome distribution
    outcome_counts = data['Outcome'].value_counts()
    print(f"\n🎯 Outcome Distribution:")
    print(f"   • No Diabetes (0): {outcome_counts.get(0, 0)} ({outcome_counts.get(0, 0)/len(data)*100:.1f}%)")
    print(f"   • Has Diabetes (1): {outcome_counts.get(1, 0)} ({outcome_counts.get(1, 0)/len(data)*100:.1f}%)")
    
    return data


def visualize_data_distribution(data):
    """
    Visualize data distribution
    """
    print_section("📊 Step 2: Visualize Data Distribution")
    
    feature_cols = ['Glucose', 'Age', 'Pregnancies', 'BloodPressure', 
                    'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
    
    # Plot 1: Glucose vs Age scatter plot
    print("🌜 Plot 1: Glucose vs Age Relationship")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for outcome, color, label in [(0, COLORS['primary'], 'No Diabetes'),
                                   (1, COLORS['danger'], 'Has Diabetes')]:
        mask = data['Outcome'] == outcome
        ax.scatter(data[mask]['Glucose'], 
                  data[mask]['Age'],
                  c=color, 
                  label=label,
                  alpha=0.6,
                  s=50,
                  edgecolors='white',
                  linewidth=0.5)
    
    ax.set_xlabel('Glucose Level', fontsize=14, fontweight='bold')
    ax.set_ylabel('Age', fontsize=14, fontweight='bold')
    ax.set_title('Relationship between Glucose Level and Age by Diabetes Status', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output_1_glucose_age_scatter.png', dpi=300, bbox_inches='tight')
    print("   ✅ Plot saved: output_1_glucose_age_scatter.png\n")
    plt.show()
    
    # Plot 2: Distribution of all features
    print("🌜 Plot 2: Distribution of All Features")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, col in enumerate(feature_cols):
        ax = axes[idx]
        
        # Plot histogram for each group
        data[data['Outcome'] == 0][col].hist(ax=ax, bins=20, alpha=0.6, 
                                              color=COLORS['primary'], label='No Diabetes')
        data[data['Outcome'] == 1][col].hist(ax=ax, bins=20, alpha=0.6,
                                              color=COLORS['danger'], label='Has Diabetes')
        
        ax.set_title(col, fontsize=12, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Feature Distribution by Diabetes Status', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('output_2_features_distribution.png', dpi=300, bbox_inches='tight')
    print("   ✅ Plot saved: output_2_features_distribution.png\n")
    plt.show()
    
    # Plot 3: Correlation Matrix
    print("🌜 Plot 3: Correlation Matrix")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    correlation = data[feature_cols + ['Outcome']].corr()
    
    sns.heatmap(correlation, 
                annot=True, 
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8},
                ax=ax)
    
    ax.set_title('Correlation Matrix Between Features', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('output_3_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("   ✅ Plot saved: output_3_correlation_matrix.png\n")
    plt.show()


def prepare_data(data):
    """
    Prepare data for training
    """
    print_section("🔧 Step 3: Prepare Data")
    
    # Select features
    feature_cols = ['Glucose', 'Age', 'Pregnancies', 'BloodPressure',
                    'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
    
    X = data[feature_cols].values
    y = data['Outcome'].values
    
    print(f"📐 Original Data Dimensions:")
    print(f"   • X (Features): {X.shape}")
    print(f"   • y (Target): {y.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n✌️  Data Split:")
    print(f"   • Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   • Testing: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Standardize the data (Z-score normalization)
    print(f"\n🔄 Standardizing Data (Z-score normalization)...")
    X_train, mean, std = standardize_features(X_train)
    X_test, _, _ = standardize_features(X_test, mean, std)
    
    print(f"   ✅ New Mean: {X_train.mean():.6f}")
    print(f"   ✅ New Standard Deviation: {X_train.std():.6f}")
    
    # One-hot encoding for target
    print(f"\n🎯 Converting Target to One-Hot Encoding...")
    y_train_encoded = one_hot_encode(y_train)
    y_test_encoded = one_hot_encode(y_test)
    
    print(f"   ✅ Target Shape After Conversion: {y_train_encoded.shape}")
    
    return X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded


def build_and_train_model(X_train, y_train_encoded, X_test, y_test_encoded):
    """
    Build and train the neural network
    """
    print_section("🦬 Step 4: Build and Train Neural Network")
    
    # Set seed for reproducibility
    set_random_seed(42)
    
    # Create network configuration
    print("⚙️  Neural Network Configuration:")
    config = NetworkConfig(
        input_dim=X_train.shape[1],
        hidden_dims=[16, 8],  # two hidden layers
        output_dim=2,
        learning_rate=0.01,
        activation='relu',
        seed=42
    )
    
    print(f"   • Input Features: {config.input_dim}")
    print(f"   • Hidden Layers: {config.hidden_dims}")
    print(f"   • Output Classes: {config.output_dim}")
    print(f"   • Learning Rate: {config.learning_rate}")
    print(f"   • Activation Function: {config.activation}")
    
    # Build the model
    print(f"\n🏗️  Building Model...")
    model = FeedForwardNN(config)
    print("   ✅ Model built successfully!")
    
    # Training
    print(f"\n🎓 Starting Training...")
    print("   " + "-"*70)
    
    history = model.fit(
        X_train, y_train_encoded,
        X_val=X_test,
        y_val=y_test_encoded,
        epochs=300,
        batch_size=32,
        l2_reg=0.001,
        early_stopping=20,
        verbose=True
    )
    
    print("   " + "-"*70)
    print(f"   ✅ Training completed after {len(history['train_loss'])} epochs!")
    
    return model, history


def plot_training_history(history):
    """
    Plot training history
    """
    print_section("📈 Step 5: Analyze Training Process")
    
    print("🌜 Plot 4: Training Curves (Loss & Accuracy)")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss (Loss)
    ax1.plot(epochs, history['train_loss'], 
             linewidth=2.5, color=COLORS['primary'], 
             label='Training Loss', marker='o', markersize=4)
    ax1.plot(epochs, history['val_loss'], 
             linewidth=2.5, color=COLORS['danger'], 
             label='Validation Loss', marker='s', markersize=4)
    
    ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax1.set_title('Loss Curve During Training', fontsize=15, fontweight='bold', pad=15)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy (Accuracy)
    ax2.plot(epochs, history['train_acc'], 
             linewidth=2.5, color=COLORS['primary'], 
             label='Training Accuracy', marker='o', markersize=4)
    ax2.plot(epochs, history['val_acc'], 
             linewidth=2.5, color=COLORS['danger'], 
             label='Validation Accuracy', marker='s', markersize=4)
    
    ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax2.set_title('Accuracy Curve During Training', fontsize=15, fontweight='bold', pad=15)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('output_4_training_history.png', dpi=300, bbox_inches='tight')
    print("   ✅ Plot saved: output_4_training_history.png\n")
    plt.show()
    
    # Print statistics
    print("📊 Training Summary:")
    print(f"   • Best Training Loss: {min(history['train_loss']):.4f}")
    print(f"   • Best Validation Loss: {min(history['val_loss']):.4f}")
    print(f"   • Best Training Accuracy: {max(history['train_acc']):.4f} ({max(history['train_acc'])*100:.2f}%)")
    print(f"   • Best Validation Accuracy: {max(history['val_acc']):.4f} ({max(history['val_acc'])*100:.2f}%)")


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model
    """
    print_section("🎯 Step 6: Evaluate Model")
    
    # Get predictions
    y_pred = model.predict_classes(X_test)
    
    # Calculate metrics
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_classification(
        y_test, 
        y_pred,
        class_names=['No Diabetes', 'Has Diabetes']
    )
    
    # Print results
    print("📊 Evaluation Results on Test Data:")
    print(f"\n   🎯 Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   📏 Precision:          {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"   🔍 Recall:             {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"   ⚖️  F1-Score:           {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    
    print(f"\n📋 Confusion Matrix:")
    print(metrics['confusion_matrix'])
    
    print(f"\n📄 Full Classification Report:")
    print(metrics['classification_report'])
    
    return metrics


def plot_confusion_matrix(model, X_test, y_test):
    """
    Plot confusion matrix
    """
    print("🎨 Plot 5: Confusion Matrix")
    
    y_pred = model.predict_classes(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['No Diabetes', 'Has Diabetes']
    )
    
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    
    ax.set_title('Confusion Matrix - Classification Results', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted', fontsize=13, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=13, fontweight='bold')
    
    # Add percentages
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percentage = cm[i, j] / cm[i].sum() * 100
            ax.text(j, i + 0.3, f'({percentage:.1f}%)', 
                   ha='center', va='center', fontsize=11, color='gray')
    
    plt.tight_layout()
    plt.savefig('output_5_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("   ✅ Plot saved: output_5_confusion_matrix.png\n")
    plt.show()


def plot_prediction_comparison(model, X_test, y_test):
    """
    Plot prediction comparison
    """
    print("🎨 Plot 6: Comparing Predictions with Actual Values")
    
    # Get probabilities
    y_pred_proba = model.predict(X_test)
    y_pred = model.predict_classes(X_test)
    
    # Select first 50 samples for display
    n_samples = min(50, len(y_test))
    indices = range(n_samples)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Top plot: Actual values vs Predictions
    width = 0.35
    x = np.arange(n_samples)
    
    bars1 = ax1.bar(x - width/2, y_test[:n_samples], width, 
                    label='Actual Values', color=COLORS['primary'], alpha=0.8)
    bars2 = ax1.bar(x + width/2, y_pred[:n_samples], width,
                    label='Predictions', color=COLORS['danger'], alpha=0.8)
    
    ax1.set_xlabel('Sample Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Classification (0=No Diabetes, 1=Has Diabetes)', fontsize=12, fontweight='bold')
    ax1.set_title('Comparing Actual Values with Predictions', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(i+1) for i in x], rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Bottom plot: Probabilities
    proba_diabetes = y_pred_proba[:n_samples, 1]  # Probability of diabetes
    
    colors = [COLORS['danger'] if p > 0.5 else COLORS['primary'] for p in proba_diabetes]
    bars = ax2.bar(x, proba_diabetes, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Decision Boundary (50%)')
    ax2.set_xlabel('Sample Number', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Probability of Diabetes', fontsize=12, fontweight='bold')
    ax2.set_title('Prediction Probabilities for Each Sample', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(i+1) for i in x], rotation=45)
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('output_6_predictions_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✅ Plot saved: output_6_predictions_comparison.png\n")
    plt.show()


def create_summary_report(data, history, metrics):
    """
    Create comprehensive summary report
    """
    print_section("📄 Step 7: Complete Summary Report")
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Data information
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    data_info = f"""
    📊 Data Information
    {'='*40}
    • Number of Samples: {len(data)}
    • Number of Features: {len(data.columns) - 1}
    
    • No Diabetes: {len(data[data['Outcome']==0])}
    • Has Diabetes: {len(data[data['Outcome']==1])}
    
    • No Diabetes Ratio: {len(data[data['Outcome']==0])/len(data)*100:.1f}%
    • Diabetes Ratio: {len(data[data['Outcome']==1])/len(data)*100:.1f}%
    """
    
    ax1.text(0.1, 0.5, data_info, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor=COLORS['primary'], alpha=0.1))
    
    # Model information
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    
    model_info = f"""
    🧠 Model Information
    {'='*40}
    • Model Type: Feed-Forward NN
    • Hidden Layers: [16, 8]
    • Activation Function: ReLU
    • Learning Rate: 0.01
    
    • Number of Epochs: {len(history['train_loss'])}
    • Batch Size: 32
    • Early Stopping: Yes
    """
    
    ax2.text(0.1, 0.5, model_info, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round',
             facecolor=COLORS['secondary'], alpha=0.1))
    
    # Training results
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    
    training_results = f"""
    📈 Training Results
    {'='*40}
    • Best Training Loss: {min(history['train_loss']):.4f}
    • Best Validation Loss: {min(history['val_loss']):.4f}
    
    • Best Training Accuracy: {max(history['train_acc'])*100:.2f}%
    • Best Validation Accuracy: {max(history['val_acc'])*100:.2f}%
    
    • Overall Improvement: {(history['train_loss'][0] - min(history['train_loss']))/history['train_loss'][0]*100:.1f}%
    """
    
    ax3.text(0.1, 0.5, training_results, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round',
             facecolor=COLORS['accent'], alpha=0.1))
    
    # Evaluation results
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    eval_results = f"""
    🎯 Final Evaluation Results
    {'='*40}
    • Accuracy:  {metrics['accuracy']*100:.2f}%
    • Precision: {metrics['precision']*100:.2f}%
    • Recall:    {metrics['recall']*100:.2f}%
    • F1-Score:  {metrics['f1_score']*100:.2f}%
    
    ✅ Model is ready for use!
    """
    
    ax4.text(0.1, 0.5, eval_results, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round',
             facecolor=COLORS['success'], alpha=0.1))
    
    # Small loss curve
    ax5 = fig.add_subplot(gs[2, 0])
    epochs = range(1, len(history['train_loss']) + 1)
    ax5.plot(epochs, history['train_loss'], linewidth=2, label='Training', color=COLORS['primary'])
    ax5.plot(epochs, history['val_loss'], linewidth=2, label='Validation', color=COLORS['danger'])
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss')
    ax5.set_title('Loss Curve')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Small accuracy curve
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(epochs, history['train_acc'], linewidth=2, label='Training', color=COLORS['primary'])
    ax6.plot(epochs, history['val_acc'], linewidth=2, label='Validation', color=COLORS['danger'])
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Accuracy')
    ax6.set_title('Accuracy Curve')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 1])
    
    fig.suptitle('📊 Complete Analysis Report for Diabetes Data', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig('output_7_complete_summary.png', dpi=300, bbox_inches='tight')
    print("🎨 Plot 7: Complete Summary Report")
    print("   ✅ Plot saved: output_7_complete_summary.png\n")
    plt.show()


def save_results_to_csv(metrics, history):
    """
    Save results to CSV file
    """
    print("💾 Saving Results...")
    
    # Save metrics
    results_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [
            f"{metrics['accuracy']*100:.2f}%",
            f"{metrics['precision']*100:.2f}%",
            f"{metrics['recall']*100:.2f}%",
            f"{metrics['f1_score']*100:.2f}%"
        ]
    })
    
    results_df.to_csv('results_metrics.csv', index=False, encoding='utf-8-sig')
    print("   ✅ Metrics saved: results_metrics.csv")
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv('results_training_history.csv', index=False)
    print("   ✅ Training history saved: results_training_history.csv")


def main():
    """
    Main function - Run complete analysis
    """
    print("\n" + "="*80)
    print("  🚀 Starting Comprehensive Diabetes Data Analysis")
    print("  🚀 Analysis Started")
    print("="*80)
    
    # Step 1: Load data
    data = load_and_explore_data()
    
    # Step 2: Visualize distributions
    visualize_data_distribution(data)
    
    # Step 3: Prepare data
    X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded = prepare_data(data)
    
    # Step 4: Build and train model
    model, history = build_and_train_model(X_train, y_train_encoded, X_test, y_test_encoded)
    
    # Step 5: Plot training history
    plot_training_history(history)
    
    # Step 6: Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Additional plots
    plot_confusion_matrix(model, X_test, y_test)
    plot_prediction_comparison(model, X_test, y_test)
    
    # Step 7: Summary report
    create_summary_report(data, history, metrics)
    
    # Save results
    save_results_to_csv(metrics, history)
    
    # Completion message
    print("\n" + "="*80)
    print("  ✅ Analysis completed successfully!")
    print("  ✅ Analysis Finished!")
    print("="*80)
    
    print("\n📁 Saved Files:")
    print("   📊 output_1_glucose_age_scatter.png")
    print("   📊 output_2_features_distribution.png")
    print("   📊 output_3_correlation_matrix.png")
    print("   📊 output_4_training_history.png")
    print("   📊 output_5_confusion_matrix.png")
    print("   📊 output_6_predictions_comparison.png")
    print("   📊 output_7_complete_summary.png")
    print("   📄 results_metrics.csv")
    print("   📄 results_training_history.csv")
    
    print("\n💡 Tip: Open the images to view the detailed results!")
    print("\n")
    
    return model, history, metrics


if __name__ == '__main__':
    model, history, metrics = main()
