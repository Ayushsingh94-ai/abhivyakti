"""
Abhivyakti — Word Level Data Collection Script
Auto-records 20 ISL word signs with on-screen instructions
Press 'Q' to quit, 'P' to pause/resume
"""

import cv2
import numpy as np
import os
import mediapipe as mp
import urllib.request
import time

# ── Configuration ──────────────────────────
ISL_WORDS = [
    "hello", "bye", "help", "water", "food",
    "thank_you", "yes", "no", "please", "stop",
    "come", "go", "name", "me", "you",
    "good", "bad", "more", "time", "pain"
]

# On-screen instructions for each sign
SIGN_INSTRUCTIONS = {
    "hello":     "Open palm, wave side to side",
    "bye":       "Wave hand left to right",
    "help":      "Fist on open palm, move up",
    "water":     "3 fingers tap chin twice",
    "food":      "Fingers together, tap lips",
    "thank_you": "Flat hand from chin, move forward",
    "yes":       "Closed fist, nod up and down",
    "no":        "Index+middle tap thumb twice",
    "please":    "Flat hand, circle on chest",
    "stop":      "One hand chops down on other",
    "come":      "Index finger curl toward you",
    "go":        "Both index fingers point forward",
    "name":      "Two fingers tap on two fingers",
    "me":        "Point index finger to chest",
    "you":       "Point index finger to camera",
    "good":      "Flat hand chin, move forward+down",
    "bad":       "Flat hand from chin, flip down",
    "more":      "Pinched fingers tap together twice",
    "time":      "Point index, tap on wrist",
    "pain":      "Both index fingers twist toward each other",
}

SEQUENCE_LENGTH = 30
SAMPLES_PER_WORD = 30
DATA_DIR = "data/word_keypoints"
GAP_BETWEEN_SAMPLES = 1

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
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# ── Helper Functions ────────────────────────
def extract_keypoints(result):
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

def get_completed_words():
    completed = []
    partial = {}
    for word in ISL_WORDS:
        word_dir = os.path.join(DATA_DIR, word)
        if os.path.exists(word_dir):
            count = len([f for f in os.listdir(word_dir) if f.endswith('.npy')])
            if count >= SAMPLES_PER_WORD:
                completed.append(word)
            elif count > 0:
                partial[word] = count
    return completed, partial

def create_directories():
    for word in ISL_WORDS:
        os.makedirs(os.path.join(DATA_DIR, word), exist_ok=True)

# ── Main Loop ───────────────────────────────
def collect_data():
    create_directories()
    completed, partial = get_completed_words()

    print("\n🤟 Abhivyakti — Word Data Collection")
    print("=" * 45)
    print(f"✅ Completed: {len(completed)}/{len(ISL_WORDS)}")
    if partial:
        for word, count in partial.items():
            print(f"   Resuming '{word}' from {count}/50")
    print("Controls: Q = Quit | P = Pause/Resume")
    print("=" * 45)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam!")
        return

    with HandLandmarker.create_from_options(options) as landmarker:
        for word_idx, word in enumerate(ISL_WORDS):

            # ── Skip completed ──────────────
            if word in completed:
                print(f"⏭️  Skipping '{word}' — complete")
                continue

            word_dir = os.path.join(DATA_DIR, word)
            existing = len([f for f in os.listdir(word_dir) if f.endswith('.npy')])
            start_sample = existing
            instruction = SIGN_INSTRUCTIONS.get(word, "Perform the sign")

            print(f"\n📌 Word: '{word.upper()}' ({word_idx+1}/{len(ISL_WORDS)})")
            print(f"   How to sign: {instruction}")

            # ── 5 second prep window ────────
            prep_start = time.time()
            while time.time() - prep_start < 5:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                remaining = int(5 - (time.time() - prep_start))

                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0),
                              (frame.shape[1], frame.shape[0]), (0,0,0), -1)
                frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

                # Word name
                cv2.putText(frame, f"{word.upper()}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 3)

                # Instruction
                cv2.putText(frame, f"How: {instruction}", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Countdown
                cv2.putText(frame, f"Starting in {remaining}s...", (20, 175),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

                # Progress
                cv2.putText(frame,
                            f"Word {word_idx+1}/{len(ISL_WORDS)} | Done: {start_sample}/50",
                            (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.putText(frame, "P=Pause  Q=Quit", (20, 255),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

                cv2.imshow("Abhivyakti — Word Collection", frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # ── Collect samples ─────────────
            paused = False
            sample_idx = start_sample

            while sample_idx < SAMPLES_PER_WORD:

                # ── Pause check ─────────────
                if paused:
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    cv2.putText(frame, "PAUSED", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 3)
                    cv2.putText(frame, "Press P to resume", (20, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow("Abhivyakti — Word Collection", frame)
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('p') or key == ord('P'):
                        paused = False
                    elif key == ord('q') or key == ord('Q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    continue

                # ── Countdown ───────────────
                for countdown in range(3, 0, -1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = landmarker.detect(mp_image)
                    frame = draw_landmarks(frame, result)

                    cv2.putText(frame, f"{word.upper()}", (20, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 3)
                    cv2.putText(frame, f"{instruction}", (20, 115),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    cv2.putText(frame, f"Get ready... {countdown}", (20, 155),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                    cv2.putText(frame, f"Sample {sample_idx+1}/{SAMPLES_PER_WORD}",
                                (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

                    # Progress bar
                    bar = int((sample_idx / SAMPLES_PER_WORD) * 400)
                    cv2.rectangle(frame, (20, 210), (420, 225), (50,50,50), -1)
                    cv2.rectangle(frame, (20, 210), (20+bar, 225), (0,255,0), -1)

                    cv2.putText(frame, "P=Pause  Q=Quit", (20, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
                    cv2.imshow("Abhivyakti — Word Collection", frame)
                    cv2.waitKey(700)

                # ── Record frames ────────────
                sequence = []
                for frame_num in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = landmarker.detect(mp_image)
                    frame = draw_landmarks(frame, result)
                    sequence.append(extract_keypoints(result))

                    # REC indicator
                    cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)
                    cv2.putText(frame, "REC", (50, 38),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.putText(frame, f"{word.upper()}", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 3)
                    cv2.putText(frame, f"{instruction}", (20, 125),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
                    cv2.putText(frame,
                                f"Sample {sample_idx+1}/{SAMPLES_PER_WORD} | Frame {frame_num+1}/{SEQUENCE_LENGTH}",
                                (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

                    # Progress bar
                    bar = int((sample_idx / SAMPLES_PER_WORD) * 400)
                    cv2.rectangle(frame, (20, 180), (420, 195), (50,50,50), -1)
                    cv2.rectangle(frame, (20, 180), (20+bar, 195), (0,255,0), -1)

                    cv2.imshow("Abhivyakti — Word Collection", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    elif key == ord('p') or key == ord('P'):
                        paused = True
                        break

                # ── Save ────────────────────
                if len(sequence) == SEQUENCE_LENGTH:
                    save_path = os.path.join(DATA_DIR, word, f"{sample_idx}.npy")
                    np.save(save_path, np.array(sequence))
                    sample_idx += 1
                    if sample_idx % 10 == 0:
                        print(f"   📊 {sample_idx}/{SAMPLES_PER_WORD} done for '{word}'")

            print(f"   ✅ '{word.upper()}' complete!")

    cap.release()
    cv2.destroyAllWindows()
    completed, _ = get_completed_words()
    print(f"\n🎉 Session complete!")
    print(f"   Progress: {len(completed)}/{len(ISL_WORDS)} words done")
    print(f"\n▶️  Next: python src/train_word_model.py")

if __name__ == "__main__":
    collect_data()