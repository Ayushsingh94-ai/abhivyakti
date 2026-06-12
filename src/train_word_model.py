"""
Abhivyakti — Word Level Model Training
Trains an LSTM model on ISL word sequences
"""

import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle

# ── Configuration ──────────────────────────
DATA_DIR = "data/word_keypoints"
MODEL_PATH = "models/isl_word_model.h5"
ENCODER_PATH = "models/word_label_encoder.pkl"
SEQUENCE_LENGTH = 30
BATCH_SIZE = 16
EPOCHS = 100

ISL_WORDS = [
    "hello", "bye", "help", "water", "food",
    "thank_you", "yes", "no", "please", "stop",
    "come", "go", "name", "me", "you",
    "good", "bad", "more", "time", "pain"
]

print("🤟 Abhivyakti — Word Model Training")
print("=" * 40)

# ── Load Data ───────────────────────────────
print("📂 Loading word sequences...")
X = []
y = []

for word in ISL_WORDS:
    word_dir = os.path.join(DATA_DIR, word)
    if not os.path.exists(word_dir):
        print(f"   ⚠️ Missing: {word}")
        continue
    files = [f for f in os.listdir(word_dir) if f.endswith('.npy')]
    for file in files:
        sequence = np.load(os.path.join(word_dir, file))
        if sequence.shape == (SEQUENCE_LENGTH, 126):
            X.append(sequence)
            y.append(word)

X = np.array(X)
print(f"   ✅ Loaded {len(X)} sequences")
print(f"   Shape: {X.shape}")

# ── Encode Labels ───────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)
print(f"   Classes: {num_classes} words")

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
print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")

# ── Build LSTM Model ────────────────────────
print("\n🏗️  Building LSTM model...")
model = Sequential([
    LSTM(128, return_sequences=True,
         input_shape=(SEQUENCE_LENGTH, 126)),
    Dropout(0.3),

    LSTM(64, return_sequences=False),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),
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
        patience=15,
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
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n📊 Results:")
print(f"   Test Accuracy: {accuracy * 100:.2f}%")
print(f"   Test Loss: {loss:.4f}")

model.save(MODEL_PATH)
print(f"\n✅ Model saved: {MODEL_PATH}")
print("\n🎉 Word model training complete!")
print(f"   Accuracy: {accuracy * 100:.2f}%")