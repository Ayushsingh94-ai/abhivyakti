"""
Abhivyakti — Combined Real-Time Inference Engine
Recognizes both A-Z alphabets (static) and ISL words (dynamic gestures)
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import os
import urllib.request
from collections import deque

# ── Configuration ──────────────────────────
ALPHA_MODEL_PATH = "models/isl_model.h5"
ALPHA_ENCODER_PATH = "models/label_encoder.pkl"
ALPHA_SCALER_PATH = "models/scaler.pkl"

WORD_MODEL_PATH = "models/isl_word_model.h5"
WORD_ENCODER_PATH = "models/word_label_encoder.pkl"

SEQUENCE_LENGTH = 30
ALPHA_CONFIDENCE_THRESHOLD = 0.85
WORD_CONFIDENCE_THRESHOLD = 0.70
STABLE_FRAMES = 15
MP_MODEL_PATH = "hand_landmarker.task"

SIGN_LABELS = {i: chr(65 + i) for i in range(26)}

# ── Download MediaPipe Model ────────────────
if not os.path.exists(MP_MODEL_PATH):
    print("📥 Downloading MediaPipe model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    urllib.request.urlretrieve(url, MP_MODEL_PATH)
    print("✅ Downloaded!")

# ── Load Alphabet Model ─────────────────────
print("📂 Loading alphabet model...")
alpha_model = tf.keras.models.load_model(ALPHA_MODEL_PATH)
with open(ALPHA_ENCODER_PATH, 'rb') as f:
    alpha_le = pickle.load(f)
with open(ALPHA_SCALER_PATH, 'rb') as f:
    alpha_scaler = pickle.load(f)
print("✅ Alphabet model loaded!")

# ── Load Word Model ──────────────────────────
print("📂 Loading word model...")
word_model = tf.keras.models.load_model(WORD_MODEL_PATH)
with open(WORD_ENCODER_PATH, 'rb') as f:
    word_le = pickle.load(f)
print("✅ Word model loaded!")

# ── MediaPipe Setup ─────────────────────────
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MP_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# ── Helper Functions ────────────────────────
def extract_keypoints(result):
    """Extract 126 features — both hands normalized."""
    left_hand = np.zeros(63)
    right_hand = np.zeros(63)
    if result.hand_landmarks:
        for idx, hand in enumerate(result.hand_landmarks):
            wrist_x = hand[0].x
            wrist_y = hand[0].y
            wrist_z = hand[0].z
            keypoints = []
            for lm in hand:
                keypoints.extend([
                    lm.x - wrist_x,
                    lm.y - wrist_y,
                    lm.z - wrist_z
                ])
            if idx == 0:
                left_hand = np.array(keypoints)
            else:
                right_hand = np.array(keypoints)
    return np.concatenate([left_hand, right_hand])

def draw_landmarks(frame, result):
    if not result.hand_landmarks:
        return frame
    h, w, _ = frame.shape
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17)
    ]
    for hand in result.hand_landmarks:
        for lm in hand:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        for start, end in connections:
            x1, y1 = int(hand[start].x * w), int(hand[start].y * h)
            x2, y2 = int(hand[end].x * w), int(hand[end].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
    return frame

def predict_alphabet(keypoints):
    """Predict A-Z from single frame keypoints."""
    keypoints_scaled = alpha_scaler.transform([keypoints])
    predictions = alpha_model.predict(keypoints_scaled, verbose=0)
    confidence = float(np.max(predictions))
    class_idx = np.argmax(predictions)
    label = alpha_le.inverse_transform([class_idx])[0]
    sign = SIGN_LABELS.get(label, str(label))
    return sign, confidence

def predict_word(sequence):
    """Predict ISL word from a 30-frame sequence."""
    sequence = np.array(sequence).reshape(1, SEQUENCE_LENGTH, 126)
    predictions = word_model.predict(sequence, verbose=0)
    confidence = float(np.max(predictions))
    class_idx = np.argmax(predictions)
    word = word_le.inverse_transform([class_idx])[0]
    return word, confidence

def calculate_motion(sequence):
    """Calculate average motion between consecutive frames."""
    if len(sequence) < 2:
        return 0
    diffs = []
    for i in range(1, len(sequence)):
        diff = np.mean(np.abs(sequence[i] - sequence[i-1]))
        diffs.append(diff)
    return np.mean(diffs)

# ── Main Inference Loop ─────────────────────
def run_inference():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam!")
        return

    print("\n🤟 Abhivyakti — Live Inference (Alphabets + Words)")
    print("=" * 50)
    print("Q = Quit | C = Clear sentence")

    # State
    current_sign = ""
    sentence = []
    last_added = ""
    stable_count = 0

    # Rolling buffer for word detection
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    MOTION_THRESHOLD = 0.005  # Above this = word gesture, below = static alphabet

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Webcam read failed!")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )
                result = landmarker.detect(mp_image)
                frame = draw_landmarks(frame, result)

                if result.hand_landmarks:
                    keypoints = extract_keypoints(result)
                    frame_buffer.append(keypoints)

                    # Calculate motion to decide alphabet vs word
                    motion = calculate_motion(list(frame_buffer))

                    if motion < MOTION_THRESHOLD:
                        # ── STATIC → Alphabet Detection ──
                        sign, confidence = predict_alphabet(keypoints)
                        mode = "ALPHABET"
                        threshold = ALPHA_CONFIDENCE_THRESHOLD

                    elif len(frame_buffer) == SEQUENCE_LENGTH:
                        # ── DYNAMIC → Word Detection ──
                        sign, confidence = predict_word(list(frame_buffer))
                        mode = "WORD"
                        threshold = WORD_CONFIDENCE_THRESHOLD
                    else:
                        sign, confidence = "...", 0.0
                        mode = "BUFFERING"
                        threshold = 1.0

                    # ── Stability check ──────
                    if sign == current_sign:
                        stable_count += 1
                    else:
                        stable_count = 0
                        current_sign = sign

                    # ── Add to sentence ──────
                    if (stable_count == STABLE_FRAMES and
                            confidence > threshold and
                            sign != last_added):
                        sentence.append(sign)
                        last_added = sign
                        print(f"✅ [{mode}] {sign} ({confidence*100:.1f}%)")

                    # ── Display ──────────────
                    color = (0, 255, 0) if confidence > threshold else (0, 165, 255)
                    mode_color = (0, 255, 255) if mode == "ALPHABET" else (255, 100, 255)

                    cv2.putText(frame, f"[{mode}]", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
                    cv2.putText(frame, f"{sign}", (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 4)
                    cv2.putText(frame, f"{confidence*100:.1f}%", (20, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    # Motion indicator
                    cv2.putText(frame, f"Motion: {motion:.4f}", (20, 165),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

                    # Stability bar
                    bar = int((stable_count / STABLE_FRAMES) * 200)
                    cv2.rectangle(frame, (20, 175), (220, 190), (50,50,50), -1)
                    cv2.rectangle(frame, (20, 175), (20+bar, 190), (0,255,0), -1)

                else:
                    stable_count = 0
                    last_added = ""
                    frame_buffer.clear()
                    cv2.putText(frame, "Show hand...", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            except Exception as e:
                print(f"⚠️ {e}")

            # ── Sentence bar ─────────────────
            sentence_text = " ".join(sentence[-12:])
            cv2.rectangle(frame, (0, h-55), (w, h), (20, 20, 20), -1)
            cv2.putText(frame, f"Sentence: {sentence_text}",
                        (10, h-18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)

            cv2.putText(frame, "Q:Quit | C:Clear",
                        (w-190, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (180, 180, 180), 1)

            cv2.imshow("Abhivyakti — Live Sign Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('c') or key == ord('C'):
                sentence = []
                last_added = ""
                print("🗑️  Cleared")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n👋 Done! Sentence: {' '.join(sentence)}")

if __name__ == "__main__":
    run_inference()