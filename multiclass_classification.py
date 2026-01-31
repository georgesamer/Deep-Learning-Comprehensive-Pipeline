"""
Example 2: Multi-Class Classification on Synthetic Data
========================================================

This example demonstrates multi-class classification (3+ classes) using a custom
neural network on synthetically generated data. It shows how to handle multiple
output classes and includes performance analysis across all classes.

Features:
- Multi-class classification (3 classes)
- Synthetic data generation
- One-hot encoding for multiple classes
- Per-class performance metrics
- Confusion matrix visualization
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def generate_multiclass_data(n_samples=1000, n_features=10, n_classes=3):
    """Generate synthetic multi-class classification data"""
    print("\n" + "="*80)
    print("  EXAMPLE 2: Multi-Class Classification on Synthetic Data")
    print("="*80 + "\n")
    
    print(f"📊 Generating synthetic data...")
    print(f"   • Samples: {n_samples}")
    print(f"   • Features: {n_features}")
    print(f"   • Classes: {n_classes}\n")
    
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate targets with some separation between classes
    y = np.zeros(n_samples, dtype=int)
    samples_per_class = n_samples // n_classes
    
    for class_idx in range(n_classes):
        start_idx = class_idx * samples_per_class
        end_idx = (class_idx + 1) * samples_per_class if class_idx < n_classes - 1 else n_samples
        
        # Add class-specific patterns
        X[start_idx:end_idx] += np.random.randn(1, n_features) * (class_idx + 1)
        y[start_idx:end_idx] = class_idx
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    print(f"✅ Data generated successfully!")
    print(f"   • Feature shape: {X.shape}")
    print(f"   • Target shape: {y.shape}")
    print(f"   • Class distribution: {np.bincount(y).tolist()}\n")
    
    return X, y


def preprocess_data(X, y):
    """Preprocess and split the data"""
    print("="*80)
    print("  Data Preprocessing")
    print("="*80 + "\n")
    
    print(f"✂️  Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   • Training set: {X_train.shape[0]} samples")
    print(f"   • Test set: {X_test.shape[0]} samples")
    
    # Standardize features
    print(f"\n🔄 Standardizing features...")
    X_train, mean, std = standardize_features(X_train)
    X_test, _, _ = standardize_features(X_test, mean, std)
    
    print(f"   • Mean: {X_train.mean():.6f}")
    print(f"   • Std: {X_train.std():.6f}")
    
    # One-hot encode targets
    print(f"\n🎯 One-hot encoding targets...")
    y_train_encoded = one_hot_encode(y_train)
    y_test_encoded = one_hot_encode(y_test)
    
    print(f"   • Encoded shape: {y_train_encoded.shape}")
    print(f"   • Classes: {y_train_encoded.shape[1]}")
    
    return X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded


def build_and_train(X_train, y_train_encoded, X_test, y_test_encoded):
    """Build and train the multi-class neural network"""
    print("\n" + "="*80)
    print("  Model Building and Training")
    print("="*80 + "\n")
    
    set_random_seed(42)
    
    # Network configured for multi-class
    n_classes = y_train_encoded.shape[1]
    print(f"⚙️  Neural Network Configuration:")
    
    config = NetworkConfig(
        input_dim=X_train.shape[1],
        hidden_dims=[32, 16],  # Deeper network for multi-class
        output_dim=n_classes,
        learning_rate=0.01,
        activation='relu',
        seed=42
    )
    
    print(f"   • Input dimensions: {config.input_dim}")
    print(f"   • Hidden layers: {config.hidden_dims}")
    print(f"   • Output classes: {config.output_dim}")
    print(f"   • Learning rate: {config.learning_rate}")
    
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
    """Evaluate the multi-class model"""
    print("\n" + "="*80)
    print("  Model Evaluation")
    print("="*80 + "\n")
    
    # Get predictions
    y_pred = model.predict_classes(X_test)
    
    # Determine class names
    n_classes = len(np.unique(y_test))
    class_names = [f'Class {i}' for i in range(n_classes)]
    
    # Calculate metrics
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_classification(
        y_test,
        y_pred,
        class_names=class_names
    )
    
    # Print results
    print("📊 Multi-Class Classification Metrics:")
    print(f"\n   🎯 Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   📏 Precision:   {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"   🔍 Recall:      {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"   ⚖️  F1-Score:    {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    
    print(f"\n📋 Confusion Matrix ({n_classes}x{n_classes}):")
    print(metrics['confusion_matrix'])
    
    print(f"\n📄 Per-Class Classification Report:")
    print(metrics['classification_report'])
    
    return metrics


def plot_results(model, X_test, y_test, history):
    """Visualize multi-class results"""
    print("\n" + "="*80)
    print("  Visualization")
    print("="*80 + "\n")
    
    n_classes = len(np.unique(y_test))
    y_pred = model.predict_classes(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training Loss
    ax = axes[0, 0]
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], label='Training', linewidth=2)
    ax.plot(epochs, history['val_loss'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History - Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Training Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], label='Training', linewidth=2)
    ax.plot(epochs, history['val_acc'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training History - Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Confusion Matrix
    ax = axes[1, 0]
    im = ax.imshow(cm, interpolation='nearest', cmap='viridis')
    ax.set_title(f'Confusion Matrix ({n_classes} Classes)')
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = ax.text(j, i, cm[i, j], ha='center', va='center',
                          color='white' if cm[i, j] > cm.max() / 2 else 'black',
                          fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    
    # Prediction confidence
    ax = axes[1, 1]
    y_pred_proba = model.predict(X_test)
    max_probs = y_pred_proba.max(axis=1)
    
    ax.hist(max_probs, bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Maximum Predicted Probability')
    ax.set_ylabel('Frequency')
    ax.set_title('Prediction Confidence Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('02_multiclass_classification_results.png', dpi=300, bbox_inches='tight')
    print("✅ Results saved to: 02_multiclass_classification_results.png")
    plt.show()


def main():
    """Main execution"""
    # Generate synthetic data
    X, y = generate_multiclass_data(n_samples=1000, n_features=10, n_classes=3)
    
    # Preprocess
    X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded = preprocess_data(X, y)
    
    # Build and train
    model, history = build_and_train(X_train, y_train_encoded, X_test, y_test_encoded)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Visualize
    plot_results(model, X_test, y_test, history)
    
    print("\n" + "="*80)
    print("  ✅ Example 2 Completed Successfully!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
