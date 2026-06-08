"""
Abhivyakti — Model Training Script
Trains a Dense Neural Network on ISL hand landmark data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import pickle

# ── Configuration ──────────────────────────
DATA_PATH = "data/Indian Sign Language Gesture Landmarks.csv"
MODEL_PATH = "models/isl_model.h5"
ENCODER_PATH = "models/label_encoder.pkl"
BATCH_SIZE = 32
EPOCHS = 50

# A-Z mapping
SIGN_LABELS = {i: chr(65 + i) for i in range(26)}

print("🤟 Abhivyakti — Model Training")
print("=" * 40)

# ── Load Data ───────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"   Shape: {df.shape}")
print(f"   Signs: {sorted(df['target'].unique())}")

# ── Prepare Features & Labels ───────────────
# Drop non-landmark columns
feature_cols = [c for c in df.columns if c not in ['target', 'uses_two_hands']]
X = df[feature_cols].values
y = df['target'].values

print(f"   Features: {X.shape[1]}")
print(f"   Samples: {X.shape[0]}")

# ── Handle Missing Values ───────────────────
X = np.nan_to_num(X, nan=0.0)

# ── Encode Labels ───────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)
print(f"   Classes: {num_classes}")

# Save encoder for inference
os.makedirs("models", exist_ok=True)
with open(ENCODER_PATH, 'wb') as f:
    pickle.dump(le, f)
print("   ✅ Label encoder saved")

# ── Train/Test Split ────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
print(f"\n📊 Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── Normalize Features ──────────────────────
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler for inference
scaler_path = "models/scaler.pkl"
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print("   ✅ Scaler saved")

# ── Build Model ─────────────────────────────
print("\n🏗️  Building model...")
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── Callbacks ───────────────────────────────
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ── Train ───────────────────────────────────
print("\n🚀 Training started...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ── Evaluate ────────────────────────────────
print("\n📊 Final Evaluation:")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"   Test Accuracy: {accuracy * 100:.2f}%")
print(f"   Test Loss: {loss:.4f}")

# ── Save Final Model ────────────────────────
model.save(MODEL_PATH)
print(f"\n✅ Model saved to: {MODEL_PATH}")
print("\n🎉 Training complete!")
print(f"   Accuracy: {accuracy * 100:.2f}%")
print("\n▶️  Next step: Run inference with python src/inference.py")