"""
Abhivyakti — Data Collection Script
Press 'S' to start recording, 'Q' to quit
"""

import cv2
import numpy as np
import os
import mediapipe as mp
import urllib.request

# ── Configuration ──────────────────────────
ISL_WORDS = [
    "hello", "help", "water", "thank_you", "yes",
    "no", "hospital", "food", "pain", "please",
    "stop", "come", "go", "name", "me",
    "you", "good", "bad", "more", "time"
]

SEQUENCE_LENGTH = 30
SAMPLES_PER_WORD = 50
DATA_DIR = "data/keypoints"

# ── Download Model ──────────────────────────
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading MediaPipe model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("✅ Model downloaded!")

# ── MediaPipe Setup ─────────────────────────
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_keypoints(result):
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
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
        return np.array(keypoints)
    return np.zeros(63)

def draw_landmarks(frame, result):
    if not result.hand_landmarks:
        return frame
    h, w, _ = frame.shape
    hand = result.hand_landmarks[0]
    for lm in hand:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17)
    ]
    for start, end in connections:
        x1, y1 = int(hand[start].x * w), int(hand[start].y * h)
        x2, y2 = int(hand[end].x * w), int(hand[end].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
    return frame

def create_directories():
    for word in ISL_WORDS:
        os.makedirs(os.path.join(DATA_DIR, word), exist_ok=True)
    print(f"✅ Directories created for {len(ISL_WORDS)} signs")

def collect_data():
    create_directories()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam!")
        return

    print("\n🤟 Abhivyakti — Data Collection")
    print("=" * 40)

    with HandLandmarker.create_from_options(options) as landmarker:
        for word_idx, word in enumerate(ISL_WORDS):
            print(f"\n📌 Sign: '{word.upper()}' ({word_idx+1}/{len(ISL_WORDS)})")
            print("   Press 'S' to start, 'Q' to quit")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"Sign: {word.upper()}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 2)
                cv2.putText(frame, "Press S to start, Q to quit", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                cv2.imshow("Abhivyakti", frame)
                key = cv2.waitKey(10) & 0xFF
                if key == ord('s') or key == ord('S'):
                    break
                elif key == ord('q') or key == ord('Q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            for sample_idx in range(SAMPLES_PER_WORD):
                sequence = []

                for countdown in range(3, 0, -1):
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    cv2.putText(frame, f"Starting in {countdown}...", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                    cv2.imshow("Abhivyakti", frame)
                    cv2.waitKey(500)

                for frame_num in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb
                    )
                    result = landmarker.detect(mp_image)
                    frame = draw_landmarks(frame, result)
                    sequence.append(extract_keypoints(result))

                    cv2.putText(frame, f"Recording: {word.upper()}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame,
                                f"Sample {sample_idx+1}/{SAMPLES_PER_WORD} | Frame {frame_num+1}/{SEQUENCE_LENGTH}",
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                    cv2.imshow("Abhivyakti", frame)
                    cv2.waitKey(1)

                if len(sequence) == SEQUENCE_LENGTH:
                    save_path = os.path.join(DATA_DIR, word, f"{sample_idx}.npy")
                    np.save(save_path, np.array(sequence))

            print(f"   ✅ Done: {SAMPLES_PER_WORD} samples for '{word}'")

    cap.release()
    cv2.destroyAllWindows()
    print("\n🎉 Data collection complete!")

if __name__ == "__main__":
    collect_data()