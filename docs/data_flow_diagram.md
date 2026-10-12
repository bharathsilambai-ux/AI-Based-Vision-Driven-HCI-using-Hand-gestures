# Data Flow Diagrams (DFD)

This document presents the multi-level Data Flow Diagrams (DFD Level 0, Level 1, and Level 2) for the **AI-Based Vision-Driven Human Computer Interaction System**.

---

## 1. DFD Level 0 — Context Diagram
The Level 0 diagram shows the overall system boundary, external entities (User/Webcam and Host Operating System), and high-level data exchanges.

```mermaid
flowchart LR
    USER["User / Webcam Environment"] -- "(1) Video Stream (Optical Hand Gestures)" --> SYS(("AI-Based Vision HCI System"))
    SYS -- "(2) Synthesized OS Actions (Mouse, Keyboard, Media)" --> OS["Host Operating System (Windows)"]
    SYS -- "(3) Visual Telemetry & HUD Feed" --> USER
```

### Data Dictionary for Level 0:
- **Optical Hand Gestures**: Raw physical hand postures and movements captured at 30 FPS.
- **Synthesized OS Actions**: Cursor coordinates $(X, Y)$, mouse click/release signals, scroll deltas, and multimedia virtual keycodes.
- **Visual Telemetry & HUD Feed**: Mirrored camera stream overlaid with 21-joint skeleton, bounding boxes, gesture classification labels, and FPS counters.

---

## 2. DFD Level 1 — Subsystem Data Flow

The Level 1 diagram breaks the system into its 5 primary functional processes and associated data stores.

```mermaid
flowchart TD
    EXT_CAM["Webcam Device"] --> |Raw Video Frames| P1["1.0 Frame Acquisition & Mirroring"]
    P1 --> |Mirrored BGR Frames| P2["2.0 Hand Landmark Extraction"]
    
    P2 --> |21 3D Keypoints| P3["3.0 Feature Normalization & Invariance"]
    P2 --> |Palm Centroid History| P4["4.0 Gesture Classification & Temporal FSM"]
    
    P3 --> |63D Invariant Feature Vector| P4
    
    DS_MODEL[("D1: Trained ML Model Artifacts")] --> |Model Weights & Label Encoders| P4
    
    P4 --> |Validated Action Events| P5["5.0 OS Command Automation"]
    P3 --> |Index Keypoint Coordinates| P5
    
    P5 --> |OS Signals| EXT_OS["Host Operating System"]
    
    P4 --> |Telemetry & HUD Metrics| P6["6.0 UI & HUD Display Engine"]
    P1 --> |Background Video Frames| P6
    P6 --> |Rendered Display Output| EXT_USER["User Display Screen"]

    DS_LOG[("D2: Dataset Storage (CSV)")] <--> |Landmark Data Ingestion| P3
```

### Process Descriptions for Level 1:
1. **Process 1.0 (Frame Acquisition)**: Handles hardware negotiation, frame polling, horizontal flip, and frame rate calculation.
2. **Process 2.0 (Hand Landmark Extraction)**: Employs MediaPipe neural pipeline to localize hand region and compute 21 landmark positions.
3. **Process 3.0 (Feature Normalization)**: Translates coordinates relative to wrist and normalizes by palm reference scale ($d_{ref}$).
4. **Process 4.0 (Gesture Classification & FSM)**: Identifies gesture via geometric rules or trained ML classifier, and validates via debouncing FSM.
5. **Process 5.0 (OS Command Automation)**: Transforms normalized coordinates to screen pixel targets with EMA smoothing and dispatches PyAutoGUI events.
6. **Process 6.0 (UI & HUD Display Engine)**: Composites translucent HUD cards, confidence bars, and skeletons onto the output canvas.

---

## 3. DFD Level 2 — Detailed Process 4.0 (Gesture Classification & State FSM)

The Level 2 diagram details the exact internal decision and transformation logic within the classification subsystem.

```mermaid
flowchart TD
    NORM_DATA["Input: 63D Normalized Landmark Vector"] --> DUAL_SWITCH{"Engine Mode Selector"}
    
    DUAL_SWITCH --> |Mode == 'Rule'| RULE_PROC["4.1 Rule-Based Evaluation (Curled Fingers, Angles, Distances)"]
    DUAL_SWITCH --> |Mode == 'ML'| ML_PROC["4.2 Random Forest Inference (predict_proba)"]
    
    CENTROID_DATA["Input: Temporal Palm Centroid Buffer"] --> SWIPE_PROC["4.3 Optical Velocity Tracker (dx/dt)"]
    
    SWIPE_PROC --> |Velocity > Threshold| SWIPE_FOUND{"Swipe Detected?"}
    SWIPE_FOUND --> |Yes| GEST_CAND["Gesture Candidate"]
    SWIPE_FOUND --> |No| GEST_STATIC["Static Posture Candidate"]
    
    RULE_PROC --> GEST_STATIC
    ML_PROC --> GEST_STATIC
    
    GEST_STATIC --> GEST_CAND
    
    GEST_CAND --> FSM_STAB["4.4 Consecutive Frame Stability Filter (Count >= 3)"]
    FSM_STAB --> |Stable Gesture| FSM_COOL["4.5 Cooldown Verification (Time Elapsed >= Cooldown_Sec)"]
    FSM_STAB --> |Unstable / Flickering| REJECT["Suppress Trigger (Return Idle)"]
    
    FSM_COOL --> |Cooldown Satisfied| PINCH_CHECK{"Pinch Event State?"}
    FSM_COOL --> |In Cooldown| REJECT
    
    PINCH_CHECK --> |Continuous Hold| DRAG_EVENT["Action Event: drag_hold"]
    PINCH_CHECK --> |Quick Tap| CLICK_EVENT["Action Event: click"]
    PINCH_CHECK --> |Double Tap| DBL_EVENT["Action Event: double_click"]
    PINCH_CHECK --> |Discrete Gesture| DISCRETE_EVENT["Action Event: discrete"]
    PINCH_CHECK --> |Open Palm| CURSOR_EVENT["Action Event: continuous"]
    
    DRAG_EVENT --> ACTION_OUT["Output: Action Event & Target Coords to OS Controller"]
    CLICK_EVENT --> ACTION_OUT
    DBL_EVENT --> ACTION_OUT
    DISCRETE_EVENT --> ACTION_OUT
    CURSOR_EVENT --> ACTION_OUT
```
