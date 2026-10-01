# AI-Based Vision-Driven Human Computer Interaction Using Hand Gesture Recognition

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Computer Vision](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![Hand Tracking](https://img.shields.io/badge/MediaPipe-0.10%2B-orange.svg)](https://developers.google.com/mediapipe)
[![Machine Learning](https://img.shields.io/badge/Scikit--Learn-1.3%2B-red.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end, real-time, touch-free Human-Computer Interaction (HCI) desktop controller. The system captures hand gestures via a standard webcam, extracts 21 3D hand keypoints with MediaPipe, applies translation- and scale-invariant normalization, classifies gestures using deterministic geometric rules or trained machine learning models (Random Forest / SVM / KNN), and controls computer operations (cursor navigation, clicking, dragging, scrolling, multimedia playback, and slide presentation swiping) with zero jitter and temporal debouncing.

---

## 🌟 Key Features

1. **Real-Time Hand Landmark Tracking**: Tracks 21 3D keypoints at **30+ FPS** using Google MediaPipe Hands on commodity laptop CPUs without requiring a GPU.
2. **Translation & Scale Invariance**: Normalizes keypoint coordinates relative to the wrist and palm reference size, ensuring reliable recognition regardless of distance from the webcam.
3. **Dual-Engine Gesture Recognition**:
   - **Rule-Based Engine**: Instantaneous, zero-calibration geometric and angular logic.
   - **Machine Learning Classifier**: Trained Random Forest model achieving **100% test accuracy** with confidence scoring.
4. **Jitter-Free Cursor Navigation**: Employs an Exponential Moving Average (EMA) filter ($\alpha = 0.35$) with a 2-pixel deadzone for smooth mouse control.
5. **Active Control Zone Remapping**: Remaps an inner webcam bounding box to the full display resolution, preventing arm and shoulder fatigue.
6. **Temporal Debouncing & Cooldown FSM**: Finite State Machine prevents accidental multi-triggering of discrete actions (volume, click, track skip).
7. **Full Computer Automation via PyAutoGUI**: Controls cursor movement, single/double/right clicks, continuous dragging, document scrolling, master volume, and slide deck navigation.
8. **Dual User Interface Modes**:
   - **OpenCV HUD Overlay**: High-tech translucent glassmorphism HUD cards, confidence bars, and colored joint skeletons directly on the video feed.
   - **Tkinter Desktop Dashboard**: Standalone desktop GUI with live camera canvas, runtime sensitivity sliders, engine mode toggles, and event logging.
9. **Complete Academic Package**: Includes IEEE-style final year project report, DFD diagrams (Level 0, 1, 2), execution flowcharts, UML use-case diagrams, and a 15-slide viva voce presentation guide.

---

## 🖐️ Gesture-to-Action Mapping Matrix

| Gesture Pose | Hand Configuration | Computer Action Mapped | Control Mechanism |
| :--- | :--- | :--- | :--- |
| **Open Palm** | All 5 fingers extended | **Move Mouse Cursor** | Continuous (EMA Smoothed) |
| **Pinch** | Thumb tip + Index tip touching | **Left Click / Hold to Drag** | Discrete Tap / Hysteresis Hold |
| **Double Pinch** | Two quick pinches in $<0.4\text{s}$ | **Double Click** | Discrete Event |
| **Middle Pinch** | Thumb tip + Middle tip touching | **Right Mouse Click** | Debounced Discrete |
| **Two Fingers Up** | Index & Middle fingers extended up | **Scroll Up** | Continuous Smooth Scroll |
| **Two Fingers Down** | Index & Middle fingers extended down | **Scroll Down** | Continuous Smooth Scroll |
| **Thumb Up** | Fist clenched, thumb pointing up | **Volume Up (+2%)** | Debounced Discrete |
| **Thumb Down** | Fist clenched, thumb pointing down | **Volume Down (-2%)** | Debounced Discrete |
| **Closed Fist** | All 5 fingers curled into palm | **Freeze Cursor / Standby** | Safe Idle State |
| **Three Fingers** | Thumb folded, Index/Middle/Ring up | **Media Play / Pause** | Debounced Discrete |
| **Swipe Left** | Rapid horizontal hand sweep left | **Previous Track / Slide** | Optical Velocity Trigger |
| **Swipe Right** | Rapid horizontal hand sweep right | **Next Track / Slide** | Optical Velocity Trigger |
| **Rock-On Sign** | Index & Pinky up, Middle & Ring folded | **Capture Screenshot** | Saved to `/screenshots` |

---

## 📁 Project Directory Structure

```
hand_gesture_hci/
│
├── config.py                     # Central configuration constants, thresholds & hotkeys
├── requirements.txt              # Production dependencies
├── main.py                       # Main CLI & OpenCV HUD application
├── app_gui.py                    # Standalone Desktop GUI Dashboard (Tkinter)
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py               # Master pipeline connecting vision, ML, FSM, and OS
│   │
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── camera.py             # Thread-safe webcam capture & dynamic FPS tracker
│   │   ├── hand_detector.py      # MediaPipe Hands 21-landmark tracking wrapper
│   │   └── landmark_utils.py     # Invariant normalization, angles & distances
│   │
│   ├── gestures/
│   │   ├── __init__.py
│   │   ├── rule_engine.py        # Geometric rule-based gesture classifier
│   │   ├── ml_classifier.py      # Scikit-learn model inference wrapper
│   │   ├── swipe_detector.py     # Optical velocity buffer for swipes
│   │   └── gesture_state.py      # Temporal debouncer and FSM state manager
│   │
│   ├── control/
│   │   ├── __init__.py
│   │   ├── controller.py         # PyAutoGUI OS action dispatcher
│   │   └── smoother.py           # EMA cursor filter & active screen zone remap
│   │
│   └── ui/
│       ├── __init__.py
│       ├── hud_overlay.py        # Translucent glassmorphism HUD overlay
│       └── gui_dashboard.py      # Tkinter parameter controls & live monitor
│
├── ml/
│   ├── __init__.py
│   ├── collect_data.py           # Interactive dataset collection tool
│   ├── generate_sample_dataset.py# Synthetic landmark generator (2,250 samples)
│   ├── train_model.py            # Train Random Forest, SVM & KNN with CV
│   ├── evaluate_model.py         # Confusion matrix & performance comparison charts
│   ├── dataset/
│   │   └── gesture_landmarks.csv # Labeled 63D normalized dataset
│   └── models/
│       ├── gesture_model.pkl     # Serialized trained Random Forest classifier
│       ├── label_encoder.pkl     # Serialized class label encoder
│       ├── confusion_matrix.png  # High-resolution confusion matrix heatmap
│       ├── model_comparison.png  # Multi-algorithm performance comparison
│       └── evaluation_report.json# Quantitative metrics report
│
├── docs/                         # Final-Year Academic Deliverables
│   ├── system_architecture.md    # System architecture & pipeline details
│   ├── data_flow_diagram.md      # DFD Level 0, Level 1 & Level 2 diagrams
│   ├── flowchart.md              # Master execution & FSM flowcharts
│   ├── uml_use_case.md           # UML Use Case, Sequence & Class diagrams
│   ├── project_report.md         # Full IEEE-format major project report
│   └── presentation_guide.md     # 15-Slide Presentation guide with viva notes
│
└── tests/
    ├── __init__.py
    ├── test_detection.py         # Landmark normalization & rule unit tests
    ├── test_actions.py           # EMA smoother, FSM debouncing & control tests
    └── test_ml.py                # ML model inference & probability tests
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation

Verify you have Python 3.10 or higher installed:
```bash
python --version
```

Clone or open the project folder and install dependencies:
```bash
pip install -r requirements.txt
```

---

### 2. Running the Application

#### Option A: High-Performance OpenCV HUD Mode (Recommended for Demos)
```bash
# Run with default Rule-Based Engine
python main.py

# Run with Trained Machine Learning Classifier
python main.py --mode ml

# Safe Mode (visual detection only, disables OS mouse/keyboard actions)
python main.py --no-control
```

**Interactive Keyboard Controls in OpenCV Window:**
- `[Q]` or `[ESC]` : Exit application
- `[C]` : Toggle OS Mouse/Key Control ON / OFF
- `[M]` : Switch Engine Mode (`Rule-Based` $\longleftrightarrow$ `Machine Learning`)
- `[S]` : Save instant screenshot to `/screenshots`

#### Option B: Modern Desktop GUI Dashboard (Tkinter)
```bash
python app_gui.py
# or
python main.py --gui
```
Features a live video viewport, Start/Stop toggle, Rule/ML radio buttons, runtime sensitivity sliders (smoothing alpha, pinch distance), and a real-time event log.

---

### 3. Machine Learning Workflow (Optional Retraining)

The project includes pre-trained model artifacts (`gesture_model.pkl`) and a balanced dataset of 2,250 samples ready for immediate execution. If you wish to record custom gestures:

#### Step 1: Collect Custom Landmark Data
```bash
python ml/collect_data.py
```
Hold a hand posture in the camera view and press:
- `[0]` - Fist | `[1]` - Open Palm | `[2]` - Pinch | `[3]` - Two Fingers Up
- `[4]` - Two Fingers Down | `[5]` - Thumb Up | `[6]` - Thumb Down
- `[7]` - Three Fingers | `[8]` - Screenshot Pose
- `[B]` - Toggle Continuous Burst Mode (auto-records every 60ms)
- `[Q]` - Save and Exit

#### Step 2: Train & Benchmark Classifiers
```bash
python ml/train_model.py
```
Automatically trains Random Forest, Support Vector Machine (RBF kernel), and K-Nearest Neighbors using 5-Fold Stratified Cross-Validation, displays performance metrics, and saves the best model to `ml/models/gesture_model.pkl`.

#### Step 3: Generate Visual Evaluation Artifacts
```bash
python ml/evaluate_model.py
```
Generates high-resolution figures:
- `ml/models/confusion_matrix.png` (Normalized Confusion Matrix Heatmap)
- `ml/models/model_comparison.png` (Algorithm Comparison Bar Chart)

---

### 4. Running Automated Tests

Run the full automated test suite (14 unit and integration tests):
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 📊 Performance & Experimental Results

- **Classification Accuracy**: **100.0%** (Random Forest on 450 held-out test samples).
- **Macro F1-Score**: **1.0000** across all 9 posture classes.
- **End-to-End Processing Latency**: **18.3 milliseconds** per frame.
- **Real-Time Frame Rate**: Sustained **30 - 33 FPS** on standard CPU.
- **Hardware Requirement**: Standard 720p/480p integrated laptop webcam.

---

## 🎓 Academic Deliverables Included

For college project submissions, review meetings, and viva voce examinations, check the `/docs` directory:
- 📄 **[Project Report](docs/project_report.md)**: Comprehensive IEEE-format report (Problem statement, literature survey, mathematical derivations, experimental tables, and conclusions).
- 📐 **[System Architecture](docs/system_architecture.md)**: Multi-tier architecture specifications and subsystem breakdown.
- 🔄 **[Data Flow Diagrams](docs/data_flow_diagram.md)**: DFD Level 0 (Context), Level 1 (Subsystem), and Level 2 (FSM & Classification).
- 📊 **[Flowcharts](docs/flowchart.md)**: System startup, frame cycle, debounce FSM, and ML training flowcharts.
- 🧩 **[UML Diagrams](docs/uml_use_case.md)**: Use case diagrams, sequence diagrams, and class diagrams.
- 🎤 **[Presentation & Viva Guide](docs/presentation_guide.md)**: 15-slide PowerPoint breakdown with speaking notes and common viva questions & model answers.

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for details.
