"""
Example 3: Text Preprocessing and Classification
================================================

This example demonstrates text preprocessing including tokenization, lemmatization,
and vectorization, followed by neural network classification on text data.
It shows NLP preprocessing techniques before feeding text to a neural network.

Features:
- Text tokenization
- Lemmatization with NLTK
- TF-IDF vectorization
- Text classification with neural network
- Per-class text analysis
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    # Download required NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: NLTK not available. Using basic preprocessing.")
    NLTK_AVAILABLE = False

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


def generate_text_data():
    """Generate synthetic text classification data"""
    print("\n" + "="*80)
    print("  EXAMPLE 3: Text Preprocessing and Classification")
    print("="*80 + "\n")
    
    print("📊 Generating synthetic text data...\n")
    
    # Sample texts for different classes
    texts = [
        # Class 0: Positive reviews
        ("This product is amazing and works great", 0),
        ("Excellent quality, highly recommended", 0),
        ("Love it, best purchase ever made", 0),
        ("Outstanding service and product quality", 0),
        ("Fantastic experience, very satisfied", 0),
        ("Great value for money, very happy", 0),
        ("Perfect item, exceeded expectations", 0),
        ("Wonderful quality, fast shipping", 0),
        
        # Class 1: Negative reviews
        ("Terrible product, completely useless", 1),
        ("Very disappointed with this purchase", 1),
        ("Poor quality, waste of money", 1),
        ("Horrible experience, do not recommend", 1),
        ("Awful service and bad product", 1),
        ("Completely broken, total disaster", 1),
        ("Extremely disappointed, very poor", 1),
        ("Not worth it, very unhappy", 1),
    ]
    
    # Expand dataset
    expanded_texts = texts * 10  # Repeat to get more samples
    
    texts = [t[0] for t in expanded_texts]
    labels = [t[1] for t in expanded_texts]
    
    print(f"✅ Generated {len(texts)} text samples")
    print(f"   • Positive reviews: {labels.count(0)}")
    print(f"   • Negative reviews: {labels.count(1)}\n")
    
    return texts, np.array(labels)


def preprocess_texts(texts):
    """Preprocess text data: tokenization, lemmatization, vectorization"""
    print("="*80)
    print("  Text Preprocessing")
    print("="*80 + "\n")
    
    if NLTK_AVAILABLE:
        print("🔤 Tokenizing and lemmatizing text...")
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        processed_texts = []
        for text in texts:
            # Tokenize
            tokens = word_tokenize(text.lower())
            
            # Remove stopwords and lemmatize
            tokens = [lemmatizer.lemmatize(token) for token in tokens
                     if token.isalpha() and token not in stop_words]
            
            processed_texts.append(' '.join(tokens))
    else:
        print("🔤 Using basic text preprocessing...")
        processed_texts = [text.lower() for text in texts]
    
    print(f"✅ Text preprocessing complete\n")
    
    # Vectorize with TF-IDF
    print("📊 Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=50, min_df=1, max_df=0.9)
    X = vectorizer.fit_transform(processed_texts).toarray()
    
    print(f"   • Feature vector shape: {X.shape}")
    print(f"   • Number of features: {X.shape[1]}")
    print(f"   • Vocabulary size: {len(vectorizer.get_feature_names_out())}\n")
    
    feature_names = vectorizer.get_feature_names_out()
    print(f"   • Top features: {', '.join(feature_names[:10])}\n")
    
    return X, vectorizer


def prepare_data(X, y):
    """Prepare and split the data"""
    print("="*80)
    print("  Data Preparation")
    print("="*80 + "\n")
    
    print(f"✂️  Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   • Training samples: {X_train.shape[0]}")
    print(f"   • Test samples: {X_test.shape[0]}")
    
    # Standardize features
    print(f"\n🔄 Standardizing features...")
    X_train, mean, std = standardize_features(X_train)
    X_test, _, _ = standardize_features(X_test, mean, std)
    
    # One-hot encode targets
    print(f"\n🎯 One-hot encoding targets...")
    y_train_encoded = one_hot_encode(y_train)
    y_test_encoded = one_hot_encode(y_test)
    
    print(f"   • Encoded shape: {y_train_encoded.shape}\n")
    
    return X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded


def build_and_train(X_train, y_train_encoded, X_test, y_test_encoded):
    """Build and train the text classification neural network"""
    print("="*80)
    print("  Model Building and Training")
    print("="*80 + "\n")
    
    set_random_seed(42)
    
    print(f"⚙️  Neural Network Configuration:")
    config = NetworkConfig(
        input_dim=X_train.shape[1],
        hidden_dims=[16, 8],
        output_dim=2,
        learning_rate=0.01,
        activation='relu',
        seed=42
    )
    
    print(f"   • Input features: {config.input_dim}")
    print(f"   • Hidden layers: {config.hidden_dims}")
    print(f"   • Output classes: {config.output_dim}")
    
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
        epochs=200,
        batch_size=16,
        l2_reg=0.001,
        early_stopping=15,
        verbose=True
    )
    
    print("   " + "-"*70)
    print(f"✅ Training completed after {len(history['train_loss'])} epochs!")
    
    return model, history


def evaluate_model(model, X_test, y_test):
    """Evaluate text classification"""
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
        class_names=['Negative', 'Positive']
    )
    
    # Print results
    print("📊 Text Classification Results:")
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
    """Visualize text classification results"""
    print("\n" + "="*80)
    print("  Visualization")
    print("="*80 + "\n")
    
    y_pred = model.predict_classes(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training Loss
    ax = axes[0, 0]
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], label='Training', linewidth=2, marker='o')
    ax.plot(epochs, history['val_loss'], label='Validation', linewidth=2, marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History - Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Training Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], label='Training', linewidth=2, marker='o')
    ax.plot(epochs, history['val_acc'], label='Validation', linewidth=2, marker='s')
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
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'])
    
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j], ha='center', va='center',
                          color='white' if cm[i, j] > cm.max() / 2 else 'black',
                          fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    
    # Prediction Distribution
    ax = axes[1, 1]
    y_pred_proba = model.predict(X_test)
    negative_probs = y_pred_proba[:, 0]
    positive_probs = y_pred_proba[:, 1]
    
    ax.hist(negative_probs, bins=20, alpha=0.6, label='Negative Score', edgecolor='black')
    ax.hist(positive_probs, bins=20, alpha=0.6, label='Positive Score', edgecolor='black')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Frequency')
    ax.set_title('Sentiment Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('03_text_classification_results.png', dpi=300, bbox_inches='tight')
    print("✅ Results saved to: 03_text_classification_results.png")
    plt.show()


def main():
    """Main execution"""
    # Generate text data
    texts, y = generate_text_data()
    
    # Preprocess texts
    X, vectorizer = preprocess_texts(texts)
    
    # Prepare data
    X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded = prepare_data(X, y)
    
    # Build and train
    model, history = build_and_train(X_train, y_train_encoded, X_test, y_test_encoded)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Visualize
    plot_results(model, X_test, y_test, history)
    
    print("\n" + "="*80)
    print("  ✅ Example 3 Completed Successfully!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
