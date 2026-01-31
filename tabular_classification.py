"""
Example 1: Binary Classification on Tabular Data
================================================

This example demonstrates binary classification using a custom neural network
on tabular data (Diabetes dataset). It includes data loading, preprocessing,
model training, and evaluation with confusion matrix analysis.

Features:
- Binary classification (Diabetes/No Diabetes)
- Data standardization and encoding
- Simple neural network with validation
- Performance metrics and confusion matrix visualization
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from neural_networks import (
    NetworkConfig,
    FeedForwardNN,
    ModelEvaluator,
    one_hot_encode,
    standardize_features,
    set_random_seed
)


def set_random_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)


def load_data(filepath=Path(r'D:\VS_code\VS_code_WorkSpace\python_projects\nn\data\diabetes.csv')):
    """Load and explore the diabetes dataset"""
    print("\n" + "="*80)
    print("  EXAMPLE 1: Binary Classification on Tabular Data")
    print("="*80 + "\n")
    
    print("📊 Loading Data...")
    
    if not Path(filepath).exists():
        print(f"⚠️  File {filepath} not found. Creating sample data...\n")
        
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
    
    print(f"✅ Data loaded: {data.shape[0]} samples, {data.shape[1]} features\n")
    
    # Display basic statistics
    print("📋 Dataset Overview:")
    print(f"   • Shape: {data.shape}")
    print(f"   • Classes: {data['Outcome'].value_counts().to_dict()}")
    print(f"   • Class 0 (No Diabetes): {(data['Outcome']==0).sum()} samples")
    print(f"   • Class 1 (Has Diabetes): {(data['Outcome']==1).sum()} samples")
    
    return data


def preprocess_data(data):
    """Preprocess and split the data"""
    print("\n" + "="*80)
    print("  Data Preprocessing")
    print("="*80 + "\n")
    
    # Select features
    feature_cols = ['Glucose', 'Age', 'Pregnancies', 'BloodPressure',
                    'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
    
    X = data[feature_cols].values
    y = data['Outcome'].values
    
    print(f"📐 Feature matrix shape: {X.shape}")
    print(f"📐 Target vector shape: {y.shape}")
    
    # Split data
    print(f"\n✂️  Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   • Training set: {X_train.shape[0]} samples")
    print(f"   • Test set: {X_test.shape[0]} samples")
    
    # Standardize features
    print(f"\n🔄 Standardizing features (Z-score normalization)...")
    X_train, mean, std = standardize_features(X_train)
    X_test, _, _ = standardize_features(X_test, mean, std)
    
    print(f"   • Mean after standardization: {X_train.mean():.6f}")
    print(f"   • Std after standardization: {X_train.std():.6f}")
    
    # One-hot encode targets
    print(f"\n🎯 Encoding targets to one-hot format...")
    y_train_encoded = one_hot_encode(y_train)
    y_test_encoded = one_hot_encode(y_test)
    
    print(f"   • Encoded shape: {y_train_encoded.shape}")
    
    return X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded


def build_and_train(X_train, y_train_encoded, X_test, y_test_encoded):
    """Build and train the neural network"""
    print("\n" + "="*80)
    print("  Model Building and Training")
    print("="*80 + "\n")
    
    # Set seed for reproducibility
    set_random_seed(42)
    
    # Create config
    print("⚙️  Neural Network Configuration:")
    config = NetworkConfig(
        input_dim=X_train.shape[1],
        hidden_dims=[16, 8],
        output_dim=2,
        learning_rate=0.01,
        activation='relu',
        seed=42
    )
    
    print(f"   • Input dimensions: {config.input_dim}")
    print(f"   • Hidden layers: {config.hidden_dims}")
    print(f"   • Output classes: {config.output_dim}")
    print(f"   • Learning rate: {config.learning_rate}")
    print(f"   • Activation: {config.activation}")
    
    # Build model
    print(f"\n🏗️  Building model...")
    model = FeedForwardNN(config)
    print("   ✅ Model built successfully!")
    
    # Train model
    print(f"\n🎓 Training model...")
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
    print(f"✅ Training completed after {len(history['train_loss'])} epochs!")
    
    return model, history


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    print("\n" + "="*80)
    print("  Model Evaluation")
    print("="*80 + "\n")
    
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
    print("📊 Classification Metrics:")
    print(f"\n   🎯 Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   📏 Precision:   {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"   🔍 Recall:      {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"   ⚖️  F1-Score:    {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    
    print(f"\n📋 Confusion Matrix:")
    print(metrics['confusion_matrix'])
    
    print(f"\n📄 Classification Report:")
    print(metrics['classification_report'])
    
    return metrics


def plot_results(model, X_test, y_test, history):
    """Create visualizations"""
    print("\n" + "="*80)
    print("  Visualization")
    print("="*80 + "\n")
    
    y_pred = model.predict_classes(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training history - Loss
    ax = axes[0, 0]
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], label='Training Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History - Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Training history - Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], label='Training Accuracy', linewidth=2)
    ax.plot(epochs, history['val_acc'], label='Validation Accuracy', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training History - Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Confusion Matrix
    ax = axes[1, 0]
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = ax.text(j, i, cm[i, j], ha='center', va='center', color='white', fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    
    # Prediction distribution
    ax = axes[1, 1]
    y_pred_proba = model.predict(X_test)
    ax.hist(y_pred_proba[:, 0], bins=30, alpha=0.6, label='Probability No Diabetes')
    ax.hist(y_pred_proba[:, 1], bins=30, alpha=0.6, label='Probability Has Diabetes')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Frequency')
    ax.set_title('Prediction Probability Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('01_binary_classification_results.png', dpi=300, bbox_inches='tight')
    print("✅ Results saved to: 01_binary_classification_results.png")
    plt.show()


def main():
    """Main execution"""
    # Load data
    data = load_data()
    
    # Preprocess
    X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded = preprocess_data(data)
    
    # Build and train
    model, history = build_and_train(X_train, y_train_encoded, X_test, y_test_encoded)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Visualize
    plot_results(model, X_test, y_test, history)
    
    print("\n" + "="*80)
    print("  ✅ Example 1 Completed Successfully!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()