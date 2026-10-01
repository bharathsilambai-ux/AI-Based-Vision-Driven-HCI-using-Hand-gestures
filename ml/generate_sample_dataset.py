"""
Sample Dataset Generator for Hand Landmark Gestures
===================================================
Generates high-fidelity, anatomically realistic landmark data
with 3D rotations, joint variations, and Gaussian perturbation
for 9 distinct hand gestures.
"""

import csv
import math
import random
from pathlib import Path
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(BASE_DIR))

import config
from src.vision.landmark_utils import LandmarkProcessor


def rotate_landmarks_3d(points, roll_deg, pitch_deg, yaw_deg):
    """Applies Euler angle rotations in degrees to 21 landmarks."""
    r_x = math.radians(pitch_deg)
    r_y = math.radians(yaw_deg)
    r_z = math.radians(roll_deg)

    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(r_x), -math.sin(r_x)],
        [0, math.sin(r_x), math.cos(r_x)],
    ])
    Ry = np.array([
        [math.cos(r_y), 0, math.sin(r_y)],
        [0, 1, 0],
        [-math.sin(r_y), 0, math.cos(r_y)],
    ])
    Rz = np.array([
        [math.cos(r_z), -math.sin(r_z), 0],
        [math.sin(r_z), math.cos(r_z), 0],
        [0, 0, 1],
    ])
    R = Rz @ Ry @ Rx
    return [tuple(R @ np.array(pt)) for pt in points]


def create_base_skeleton():
    """Returns base canonical hand coordinates (wrist at 0,0,0, middle MCP at 0, -0.3, 0)."""
    # 21 points
    # 0: Wrist
    pts = [(0.0, 0.0, 0.0)]
    # Thumb (1-4)
    pts.extend([(-0.08, -0.06, 0.0), (-0.14, -0.12, 0.0), (-0.18, -0.17, 0.0), (-0.22, -0.22, 0.0)])
    # Index (5-8)
    pts.extend([(-0.07, -0.28, 0.0), (-0.08, -0.38, 0.0), (-0.08, -0.46, 0.0), (-0.08, -0.54, 0.0)])
    # Middle (9-12)
    pts.extend([(0.00, -0.30, 0.0), (0.00, -0.41, 0.0), (0.00, -0.50, 0.0), (0.00, -0.59, 0.0)])
    # Ring (13-16)
    pts.extend([(0.06, -0.28, 0.0), (0.07, -0.38, 0.0), (0.07, -0.46, 0.0), (0.07, -0.53, 0.0)])
    # Pinky (17-20)
    pts.extend([(0.12, -0.24, 0.0), (0.13, -0.32, 0.0), (0.14, -0.39, 0.0), (0.14, -0.45, 0.0)])
    return pts


def curl_finger(pts, base_idx):
    """Curls finger starting at base_idx (MCP, PIP, DIP, TIP) into palm."""
    mcp = pts[base_idx]
    # Curl towards palm center (0.0, -0.15, 0.05)
    pts[base_idx + 1] = (mcp[0] * 0.9, mcp[1] + 0.05, 0.04)
    pts[base_idx + 2] = (mcp[0] * 0.8, mcp[1] + 0.10, 0.06)
    pts[base_idx + 3] = (mcp[0] * 0.7, mcp[1] + 0.14, 0.05)


def generate_gesture_pose(gesture_name: str):
    """Generates landmark coordinates for a specific posture."""
    pts = create_base_skeleton()

    if gesture_name == config.GESTURE_OPEN_PALM:
        pass  # Already fully open

    elif gesture_name == config.GESTURE_FIST:
        # Curl all 4 fingers
        curl_finger(pts, 5)   # Index
        curl_finger(pts, 9)   # Middle
        curl_finger(pts, 13)  # Ring
        curl_finger(pts, 17)  # Pinky
        # Fold thumb across fingers
        pts[1] = (-0.05, -0.06, 0.02)
        pts[2] = (-0.03, -0.12, 0.06)
        pts[3] = (0.00, -0.16, 0.08)
        pts[4] = (0.04, -0.18, 0.08)

    elif gesture_name == config.GESTURE_PINCH:
        # Thumb tip and Index tip meet
        pinch_pt = (-0.10, -0.35, 0.02)
        pts[4] = (pinch_pt[0] + 0.01, pinch_pt[1], pinch_pt[2])
        pts[8] = (pinch_pt[0] - 0.01, pinch_pt[1], pinch_pt[2])
        # Ring and pinky curled or relaxed
        curl_finger(pts, 13)
        curl_finger(pts, 17)

    elif gesture_name == config.GESTURE_TWO_FINGERS_UP:
        # Ring and pinky curled
        curl_finger(pts, 13)
        curl_finger(pts, 17)
        # Thumb curled
        pts[3] = (-0.05, -0.15, 0.04)
        pts[4] = (-0.02, -0.16, 0.05)

    elif gesture_name == config.GESTURE_TWO_FINGERS_DOWN:
        # Index and middle pointing downwards
        mcp5 = pts[5]
        pts[6] = (mcp5[0], mcp5[1] + 0.10, 0.0)
        pts[7] = (mcp5[0], mcp5[1] + 0.19, 0.0)
        pts[8] = (mcp5[0], mcp5[1] + 0.28, 0.0)

        mcp9 = pts[9]
        pts[10] = (mcp9[0], mcp9[1] + 0.11, 0.0)
        pts[11] = (mcp9[0], mcp9[1] + 0.20, 0.0)
        pts[12] = (mcp9[0], mcp9[1] + 0.30, 0.0)

        curl_finger(pts, 13)
        curl_finger(pts, 17)

    elif gesture_name == config.GESTURE_THUMB_UP:
        # Fist with thumb pointing straight up
        curl_finger(pts, 5)
        curl_finger(pts, 9)
        curl_finger(pts, 13)
        curl_finger(pts, 17)
        pts[1] = (-0.07, -0.08, 0.0)
        pts[2] = (-0.09, -0.18, 0.0)
        pts[3] = (-0.10, -0.28, 0.0)
        pts[4] = (-0.10, -0.38, 0.0)

    elif gesture_name == config.GESTURE_THUMB_DOWN:
        # Fist with thumb pointing straight down
        curl_finger(pts, 5)
        curl_finger(pts, 9)
        curl_finger(pts, 13)
        curl_finger(pts, 17)
        pts[1] = (-0.07, 0.04, 0.0)
        pts[2] = (-0.09, 0.14, 0.0)
        pts[3] = (-0.10, 0.24, 0.0)
        pts[4] = (-0.10, 0.34, 0.0)

    elif gesture_name == config.GESTURE_THREE_FINGERS:
        # Index, Middle, Ring up; Pinky and Thumb curled
        curl_finger(pts, 17)
        pts[3] = (-0.05, -0.15, 0.04)
        pts[4] = (-0.02, -0.16, 0.05)

    elif gesture_name == config.GESTURE_SCREENSHOT:
        # Rock-on pose: Index and Pinky extended, Middle and Ring curled
        curl_finger(pts, 9)
        curl_finger(pts, 13)
        pts[3] = (-0.05, -0.15, 0.04)
        pts[4] = (-0.02, -0.16, 0.05)

    return pts


def generate_dataset(samples_per_class: int = 250):
    csv_path = config.DATASET_CSV_PATH
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    headers = []
    for i in range(21):
        headers.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z"])
    headers.append("label")

    classes = [
        config.GESTURE_FIST,
        config.GESTURE_OPEN_PALM,
        config.GESTURE_PINCH,
        config.GESTURE_TWO_FINGERS_UP,
        config.GESTURE_TWO_FINGERS_DOWN,
        config.GESTURE_THUMB_UP,
        config.GESTURE_THUMB_DOWN,
        config.GESTURE_THREE_FINGERS,
        config.GESTURE_SCREENSHOT,
    ]

    total_records = 0
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for gesture_label in classes:
            base_pts = generate_gesture_pose(gesture_label)

            for _ in range(samples_per_class):
                # 1. Random 3D Rotation (-18 to +18 deg)
                roll = random.uniform(-18.0, 18.0)
                pitch = random.uniform(-15.0, 15.0)
                yaw = random.uniform(-15.0, 15.0)
                rotated = rotate_landmarks_3d(base_pts, roll, pitch, yaw)

                # 2. Scale factor (0.85 to 1.18)
                scale = random.uniform(0.85, 1.18)

                # 3. Add Gaussian noise per joint
                perturbed = []
                for pt in rotated:
                    nx = pt[0] * scale + random.gauss(0.0, 0.008)
                    ny = pt[1] * scale + random.gauss(0.0, 0.008)
                    nz = pt[2] * scale + random.gauss(0.0, 0.008)
                    perturbed.append((nx, ny, nz))

                # 4. Invariant normalization
                features = LandmarkProcessor.normalize_landmarks(perturbed)
                row = list(features) + [gesture_label]
                writer.writerow(row)
                total_records += 1

    print(f"[SUCCESS] Generated {total_records} high-quality samples across {len(classes)} classes.")
    print(f"[*] Saved dataset to: {csv_path}")


if __name__ == "__main__":
    generate_dataset(samples_per_class=250)
