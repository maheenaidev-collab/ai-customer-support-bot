"""
Training pipeline for AI Customer Support Bot.
Author: Maheen Riaz
"""

import json
import os
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from preprocessor import TextPreprocessor


def load_intents(filepath='training_data/intents.json'):
    """Load training intents from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['intents']


def prepare_training_data(intents, preprocessor):
    """Prepare texts and labels for training."""
    texts = []
    labels = []

    for intent in intents:
        tag = intent['tag']
        for pattern in intent['patterns']:
            cleaned = preprocessor.clean(pattern)
            texts.append(cleaned)
            labels.append(tag)

    return texts, labels


def train_model(data_path='training_data/intents.json'):
    """Train the chatbot model."""
    print("🚀 Starting training pipeline...\n")

    # Initialize
    preprocessor = TextPreprocessor()
    intents = load_intents(data_path)

    print(f"📚 Loaded {len(intents)} intents")
    total_patterns = sum(len(i['patterns']) for i in intents)
    print(f"📝 Total training patterns: {total_patterns}\n")

    # Prepare data
    texts, labels = prepare_training_data(intents, preprocessor)

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    print(f"📊 Feature matrix shape: {X.shape}")
    print(f"🏷️  Classes: {list(label_encoder.classes_)}\n")

    # Train SVM classifier
    model = SVC(kernel='linear', probability=True, C=1.0)
    model.fit(X, y)

    # Cross validation
    scores = cross_val_score(model, X, y, cv=min(5, len(set(labels))), scoring='accuracy')
    print(f"✅ Cross-validation accuracy: {scores.mean():.2%} (+/- {scores.std():.2%})\n")

    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/chatbot_model.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    joblib.dump(label_encoder, 'models/label_encoder.pkl')

    # Save intents for response lookup
    with open('models/intents.json', 'w') as f:
        json.dump({'intents': intents}, f, indent=2)

    print("💾 Model saved to models/")
    print("🎉 Training complete!\n")

    # Test predictions
    print("🧪 Sample predictions:")
    test_queries = [
        "Where is my order?",
        "I want a refund",
        "Hello!",
        "Your service is terrible",
        "Talk to a real person"
    ]
    for query in test_queries:
        cleaned = preprocessor.clean(query)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec).max()
        intent = label_encoder.inverse_transform([pred])[0]
        print(f"  '{query}' → {intent} ({prob:.2%})")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train the chatbot model')
    parser.add_argument('--data', type=str, default='training_data/intents.json', help='Path to intents JSON')
    args = parser.parse_args()
    train_model(args.data)
