"""
Interactive Hand Gesture Dataset Collection Tool
================================================
Captures real-time webcam hand landmarks, computes invariant 63D features,
and writes labeled records directly into 'ml/dataset/gesture_landmarks.csv'.

Key Bindings:
    [0] - Closed Fist
    [1] - Open Palm
    [2] - Pinch
    [3] - Two Fingers Up
    [4] - Two Fingers Down
    [5] - Thumb Up
    [6] - Thumb Down
    [7] - Three Fingers
    [8] - Screenshot Pose
    [B] - Toggle Continuous Burst Mode
    [Q] - Quit and Save
"""

import sys
import csv
import time
from pathlib import Path
import cv2
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
from src.vision.camera import CameraStream
from src.vision.hand_detector import HandDetector
from src.vision.landmark_utils import LandmarkProcessor

# Key-to-Gesture Mapping for collection
DATASET_KEY_MAPPING = {
    ord("0"): config.GESTURE_FIST,
    ord("1"): config.GESTURE_OPEN_PALM,
    ord("2"): config.GESTURE_PINCH,
    ord("3"): config.GESTURE_TWO_FINGERS_UP,
    ord("4"): config.GESTURE_TWO_FINGERS_DOWN,
    ord("5"): config.GESTURE_THUMB_UP,
    ord("6"): config.GESTURE_THUMB_DOWN,
    ord("7"): config.GESTURE_THREE_FINGERS,
    ord("8"): config.GESTURE_SCREENSHOT,
}


def initialize_csv(csv_path: Path):
    """Creates CSV file with 63 feature columns + label if not already present."""
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        headers = []
        for i in range(21):
            headers.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z"])
        headers.append("label")

        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        print(f"[*] Initialized new dataset at: {csv_path}")


def count_existing_samples(csv_path: Path) -> dict:
    """Returns dictionary of sample counts per gesture."""
    counts = {v: 0 for v in DATASET_KEY_MAPPING.values()}
    if not csv_path.exists():
        return counts

    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if row and len(row) >= 64:
                    label = row[-1]
                    counts[label] = counts.get(label, 0) + 1
    except Exception as e:
        print(f"[!] Warning reading CSV counts: {e}")

    return counts


def main():
    print("=" * 65)
    print("  GESTURE DATASET COLLECTOR")
    print("=" * 65)
    print("  Hold a hand pose in camera view and press:")
    for key_code, gesture_name in DATASET_KEY_MAPPING.items():
        print(f"    [{chr(key_code)}] - {gesture_name}")
    print("    [B] - Toggle Continuous Burst Mode (auto records every 50ms)")
    print("    [Q] - Save and Exit")
    print("=" * 65 + "\n")

    csv_path = config.DATASET_CSV_PATH
    initialize_csv(csv_path)
    sample_counts = count_existing_samples(csv_path)

    camera = CameraStream()
    if not camera.start():
        print("[ERROR] Could not start webcam.")
        return

    detector = HandDetector()
    proc = LandmarkProcessor

    window_name = "Dataset Collector - Press 0-8 to Record"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.FRAME_WIDTH, config.FRAME_HEIGHT)

    burst_mode = False
    burst_gesture = None
    last_burst_time = 0.0

    try:
        with open(csv_path, mode="a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)

            while True:
                success, frame = camera.read_frame()
                if not success or frame is None:
                    continue

                h, w = frame.shape[:2]
                detector.find_hands(frame)
                hand_data = detector.get_hand_data((h, w), hand_index=0)

                # Draw skeleton
                if hand_data:
                    frame = detector.draw_hand_mesh(frame, hand_index=0, modern_style=True)

                # Handle continuous burst recording
                now = time.time()
                if burst_mode and burst_gesture and hand_data:
                    if (now - last_burst_time) >= 0.06:  # Every 60ms
                        features = proc.normalize_landmarks(hand_data["landmarks_norm"])
                        row = list(features) + [burst_gesture]
                        writer.writerow(row)
                        csv_file.flush()
                        sample_counts[burst_gesture] = sample_counts.get(burst_gesture, 0) + 1
                        last_burst_time = now

                # Render Collector HUD
                cv2.rectangle(frame, (10, 10), (w - 10, 45), (20, 24, 30), -1)
                cv2.putText(
                    frame,
                    f"DATASET RECORDER | Total Samples: {sum(sample_counts.values())}",
                    (20, 32),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    (0, 230, 255),
                    1,
                    cv2.LINE_AA,
                )

                # Burst indicator
                if burst_mode:
                    cv2.putText(
                        frame,
                        f"BURST ACTIVE: {burst_gesture}",
                        (w - 260, 32),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.45,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

                # Show sample counts per gesture on the side
                y_offset = 75
                cv2.rectangle(frame, (10, 60), (220, 60 + len(DATASET_KEY_MAPPING) * 22 + 10), (15, 18, 24), -1)
                for key_code, g_name in DATASET_KEY_MAPPING.items():
                    cnt = sample_counts.get(g_name, 0)
                    line = f"[{chr(key_code)}] {g_name[:12]}: {cnt}"
                    cv2.putText(frame, line, (18, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 205, 215), 1, cv2.LINE_AA)
                    y_offset += 22

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    break
                elif key == ord("b") or key == ord("B"):
                    burst_mode = not burst_mode
                    print(f"[*] Burst Mode: {'ENABLED' if burst_mode else 'DISABLED'}")
                elif key in DATASET_KEY_MAPPING:
                    target_gesture = DATASET_KEY_MAPPING[key]
                    if burst_mode:
                        burst_gesture = target_gesture
                        print(f"[*] Burst target set to: {burst_gesture}")
                    elif hand_data:
                        # Single record
                        features = proc.normalize_landmarks(hand_data["landmarks_norm"])
                        row = list(features) + [target_gesture]
                        writer.writerow(row)
                        csv_file.flush()
                        sample_counts[target_gesture] = sample_counts.get(target_gesture, 0) + 1
                        print(f"[+] Saved 1 sample for '{target_gesture}' (Total: {sample_counts[target_gesture]})")
                    else:
                        print("[!] No hand detected in frame to record.")

    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()
        print("\n[*] Data collection closed. CSV saved successfully.")


if __name__ == "__main__":
    main()
