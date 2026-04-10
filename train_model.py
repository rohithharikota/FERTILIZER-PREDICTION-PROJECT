"""
Fertilizer Recommendation Model Training Script
=================================================
Trains Random Forest, Decision Tree, and SVM classifiers on the fertilizer dataset.
Saves the best performing model and encoders to the models/ directory.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ─── Configuration ───────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'dataset', 'fertilizer_dataset.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ─── Load Dataset ────────────────────────────────────────────────────────────
print("=" * 60)
print("  Fertilizer Recommendation - Model Training")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)
print(f"\nDataset loaded: {len(df)} rows, {len(df.columns)} columns")
print(f"   Features: {list(df.columns[:-1])}")
print(f"   Target:   {df.columns[-1]}")
print(f"   Classes:  {df['Fertilizer'].nunique()} -> {list(df['Fertilizer'].unique())}")

# ─── Encode Categorical Features ────────────────────────────────────────────
label_encoders = {}
categorical_cols = ['Soil_Type', 'Crop_Type']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"   Encoded {col}: {list(le.classes_)}")

# Encode the target
fertilizer_encoder = LabelEncoder()
df['Fertilizer'] = fertilizer_encoder.fit_transform(df['Fertilizer'])

# ─── Split Features & Target ────────────────────────────────────────────────
feature_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorus', 'Potassium', 'Soil_Type', 'Crop_Type']
X = df[feature_cols]
y = df['Fertilizer']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

print(f"\nTrain/Test Split: {len(X_train)} train, {len(X_test)} test")

# ─── Train Models ────────────────────────────────────────────────────────────
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=RANDOM_STATE
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=None,
        random_state=RANDOM_STATE
    ),
    'SVM': SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        probability=True,
        random_state=RANDOM_STATE
    )
}

results = {}

print("\n" + "-" * 60)
print("  Training Models...")
print("-" * 60)

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {'model': model, 'accuracy': accuracy}

    print(f"   Accuracy: {accuracy * 100:.2f}%")
    print(f"\n   Classification Report:")
    report = classification_report(y_test, y_pred, target_names=fertilizer_encoder.classes_, zero_division=0)
    for line in report.split('\n'):
        print(f"   {line}")

# ─── Select Best Model ──────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = results[best_name]['model']
best_accuracy = results[best_name]['accuracy']

print("\n" + "=" * 60)
print(f"  Best Model: {best_name} ({best_accuracy * 100:.2f}%)")
print("=" * 60)

# ─── Save Models & Encoders ─────────────────────────────────────────────────
os.makedirs(MODELS_DIR, exist_ok=True)

joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.pkl'))
joblib.dump(label_encoders, os.path.join(MODELS_DIR, 'label_encoders.pkl'))
joblib.dump(fertilizer_encoder, os.path.join(MODELS_DIR, 'fertilizer_encoder.pkl'))

# Save model metadata
metadata = {
    'best_model': best_name,
    'accuracy': round(best_accuracy * 100, 2),
    'feature_columns': feature_cols,
    'categorical_columns': categorical_cols,
    'soil_types': list(label_encoders['Soil_Type'].classes_),
    'crop_types': list(label_encoders['Crop_Type'].classes_),
    'fertilizer_types': list(fertilizer_encoder.classes_),
    'all_results': {name: round(r['accuracy'] * 100, 2) for name, r in results.items()}
}

with open(os.path.join(MODELS_DIR, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved to {MODELS_DIR}/")
print(f"   - best_model.pkl ({best_name})")
print(f"   - label_encoders.pkl")
print(f"   - fertilizer_encoder.pkl")
print(f"   - metadata.json")
print(f"\nTraining complete! Run 'python app.py' to start the server.")
