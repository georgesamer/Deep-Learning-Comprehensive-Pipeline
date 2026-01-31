"""
Example 4: Keras/TensorFlow Model Building and Training
=======================================================

This example demonstrates building neural networks using Keras/TensorFlow
and compares them with custom neural network implementations. It shows
how to use high-level deep learning frameworks for rapid prototyping.

Features:
- Keras Sequential API for model building
- Multiple layer types (Dense, Dropout, BatchNormalization)
- Model compilation and training
- Comparison with custom implementations
- Performance visualization
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("⚠️  Warning: TensorFlow/Keras not available.")

from neural_networks import (
    NetworkConfig,
    FeedForwardNN,
    ModelEvaluator,
    one_hot_encode,
    standardize_features,
    set_random_seed as custom_seed
)


def load_data(filepath=Path(r'D:\VS_code\VS_code_WorkSpace\python_projects\nn\data\diabetes.csv')):
    """Load the diabetes dataset"""
    print("\n" + "="*80)
    print("  EXAMPLE 4: Keras/TensorFlow Model Building")
    print("="*80 + "\n")
    
    print("📊 Loading data...")
    
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
    
    return data


def prepare_data(data):
    """Prepare data for both Keras and custom models"""
    print("="*80)
    print("  Data Preparation")
    print("="*80 + "\n")
    
    feature_cols = ['Glucose', 'Age', 'Pregnancies', 'BloodPressure',
                    'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
    
    X = data[feature_cols].values
    y = data['Outcome'].values
    
    print(f"✂️  Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   • Training: {X_train.shape[0]} samples")
    print(f"   • Test: {X_test.shape[0]} samples")
    
    # Standardize
    print(f"\n🔄 Standardizing features...")
    X_train, mean, std = standardize_features(X_train)
    X_test, _, _ = standardize_features(X_test, mean, std)
    
    # One-hot encode for custom model
    y_train_encoded = one_hot_encode(y_train)
    y_test_encoded = one_hot_encode(y_test)
    
    print(f"   ✅ Data preparation complete\n")
    
    return X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded


def build_keras_model(input_dim):
    """Build a Keras/TensorFlow model"""
    if not KERAS_AVAILABLE:
        print("⚠️  TensorFlow/Keras not available. Skipping Keras model.\n")
        return None
    
    print("="*80)
    print("  Building Keras Model")
    print("="*80 + "\n")
    
    # Set seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("🏗️  Building Sequential model...")
    
    model = keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=(input_dim,),
                    kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(8, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.2),
        layers.Dense(2, activation='softmax')
    ])
    
    print("✅ Model architecture:")
    model.summary()
    
    # Compile
    print(f"\n⚙️  Compiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("   ✅ Model compiled\n")
    
    return model


def train_keras_model(model, X_train, y_train, X_test, y_test):
    """Train Keras model"""
    if model is None:
        return None
    
    print("="*80)
    print("  Training Keras Model")
    print("="*80 + "\n")
    
    print(f"🎓 Training model...")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=300,
        batch_size=32,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )],
        verbose=0
    )
    
    print(f"✅ Training completed after {len(history.history['loss'])} epochs\n")
    
    return history


def build_and_train_custom_model(X_train, y_train_encoded, X_test, y_test_encoded):
    """Build and train custom neural network"""
    print("="*80)
    print("  Building Custom Neural Network")
    print("="*80 + "\n")
    
    custom_seed(42)
    
    print("⚙️  Configuring custom model...")
    config = NetworkConfig(
        input_dim=X_train.shape[1],
        hidden_dims=[16, 8],
        output_dim=2,
        learning_rate=0.01,
        activation='relu',
        seed=42
    )
    
    print("🏗️  Building model...")
    model = FeedForwardNN(config)
    print("   ✅ Custom model built\n")
    
    print("🎓 Training custom model...")
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
    print(f"✅ Training completed after {len(history['train_loss'])} epochs\n")
    
    return model, history


def evaluate_models(keras_model, custom_model, X_test, y_test):
    """Evaluate both models"""
    print("="*80)
    print("  Model Evaluation")
    print("="*80 + "\n")
    
    evaluator = ModelEvaluator()
    
    # Custom model evaluation
    print("📊 Custom Neural Network Results:")
    y_pred_custom = custom_model.predict_classes(X_test)
    metrics_custom = evaluator.evaluate_classification(
        y_test,
        y_pred_custom,
        class_names=['No Diabetes', 'Has Diabetes']
    )
    
    print(f"   • Accuracy:  {metrics_custom['accuracy']*100:.2f}%")
    print(f"   • Precision: {metrics_custom['precision']*100:.2f}%")
    print(f"   • Recall:    {metrics_custom['recall']*100:.2f}%")
    print(f"   • F1-Score:  {metrics_custom['f1_score']*100:.2f}%")
    
    # Keras model evaluation
    if keras_model is not None:
        print(f"\n📊 Keras/TensorFlow Model Results:")
        y_pred_keras = keras_model.predict(X_test, verbose=0).argmax(axis=1)
        metrics_keras = evaluator.evaluate_classification(
            y_test,
            y_pred_keras,
            class_names=['No Diabetes', 'Has Diabetes']
        )
        
        print(f"   • Accuracy:  {metrics_keras['accuracy']*100:.2f}%")
        print(f"   • Precision: {metrics_keras['precision']*100:.2f}%")
        print(f"   • Recall:    {metrics_keras['recall']*100:.2f}%")
        print(f"   • F1-Score:  {metrics_keras['f1_score']*100:.2f}%")
        
        print(f"\n📊 Comparison:")
        print(f"   • Custom Model Accuracy:  {metrics_custom['accuracy']*100:.2f}%")
        print(f"   • Keras Model Accuracy:   {metrics_keras['accuracy']*100:.2f}%")
        print(f"   • Difference: {abs(metrics_custom['accuracy'] - metrics_keras['accuracy'])*100:.2f}%\n")
    
    return metrics_custom, metrics_keras if keras_model is not None else None


def plot_comparison(keras_history, custom_history):
    """Visualize model comparison"""
    print("="*80)
    print("  Visualization")
    print("="*80 + "\n")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Custom model training
    ax = axes[0]
    epochs_custom = range(1, len(custom_history['train_loss']) + 1)
    ax.plot(epochs_custom, custom_history['train_loss'], label='Training Loss', linewidth=2)
    ax.plot(epochs_custom, custom_history['val_loss'], label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Custom Neural Network - Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Keras model training
    if keras_history is not None:
        ax = axes[1]
        epochs_keras = range(1, len(keras_history.history['loss']) + 1)
        ax.plot(epochs_keras, keras_history.history['loss'], label='Training Loss', linewidth=2)
        ax.plot(epochs_keras, keras_history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Keras/TensorFlow Model - Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax = axes[1]
        epochs_custom = range(1, len(custom_history['train_acc']) + 1)
        ax.plot(epochs_custom, custom_history['train_acc'], label='Training Accuracy', linewidth=2)
        ax.plot(epochs_custom, custom_history['val_acc'], label='Validation Accuracy', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Custom Neural Network - Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('04_keras_comparison_results.png', dpi=300, bbox_inches='tight')
    print("✅ Results saved to: 04_keras_comparison_results.png")
    plt.show()


def main():
    """Main execution"""
    # Load data
    data = load_data()
    
    # Prepare data
    X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded = prepare_data(data)
    
    # Build Keras model
    keras_model = None
    keras_history = None
    if KERAS_AVAILABLE:
        keras_model = build_keras_model(X_train.shape[1])
        if keras_model is not None:
            keras_history = train_keras_model(keras_model, X_train, y_train_encoded, X_test, y_test_encoded)
    
    # Build custom model
    custom_model, custom_history = build_and_train_custom_model(
        X_train, y_train_encoded, X_test, y_test_encoded
    )
    
    # Evaluate
    metrics_custom, metrics_keras = evaluate_models(keras_model, custom_model, X_test, y_test)
    
    # Visualize
    plot_comparison(keras_history, custom_history)
    
    print("\n" + "="*80)
    print("  ✅ Example 4 Completed Successfully!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
