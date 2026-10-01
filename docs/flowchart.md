# System Flowcharts

This document illustrates the execution and logic flowcharts for the **AI-Based Vision-Driven HCI System**.

---

## 1. Master Application Flowchart

```mermaid
flowchart TD
    START(["Start Application"]) --> INIT["Parse CLI Arguments (--mode, --gui, --no-control)"]
    INIT --> LOAD_CONF["Load Hyper-Parameters from config.py"]
    LOAD_CONF --> INIT_CAM["Initialize Webcam (cv2.VideoCapture)"]
    
    INIT_CAM --> CAM_OK{"Camera Initialized?"}
    CAM_OK --> |No| ERR_LOG["Log Error & Exit"] --> STOP(["End Application"])
    CAM_OK --> |Yes| INIT_MP["Initialize MediaPipe Hands & Controllers"]
    
    INIT_MP --> GUI_CHECK{"GUI Mode Enabled?"}
    GUI_CHECK --> |Yes| LAUNCH_GUI["Launch Tkinter Desktop Dashboard"]
    GUI_CHECK --> |No| LAUNCH_CV["Launch High-Performance OpenCV HUD Window"]
    
    LAUNCH_GUI --> MAIN_LOOP["Enter Frame Processing Loop"]
    LAUNCH_CV --> MAIN_LOOP
    
    MAIN_LOOP --> READ_FRAME["Capture Frame from Webcam"]
    READ_FRAME --> FLIP_FRAME["Apply Horizontal Mirror Flip"]
    FLIP_FRAME --> DETECT_HAND["Detect Hand Landmarks (MediaPipe)"]
    
    DETECT_HAND --> HAND_FOUND{"Hand Detected?"}
    HAND_FOUND --> |Yes| PROCESS_GESTURE["Extract Invariant Features & Classify Gesture"]
    HAND_FOUND --> |No| RESET_STATE["Reset Motion & Stability State Buffers"]
    
    PROCESS_GESTURE --> FSM_DEBOUNCE["Evaluate Temporal Debounce & Cooldown FSM"]
    RESET_STATE --> RENDER_HUD["Render HUD Telemetry & Skeleton"]
    
    FSM_DEBOUNCE --> DISPATCH_ACTION["Execute PyAutoGUI OS Action (if active)"]
    DISPATCH_ACTION --> RENDER_HUD
    
    RENDER_HUD --> SHOW_OUTPUT["Display Processed Video Frame"]
    SHOW_OUTPUT --> KEY_CHECK{"Keypress / Exit Event?"}
    
    KEY_CHECK --> |Quit 'Q' or Close| CLEANUP["Release Webcam & PyAutoGUI States"]
    KEY_CHECK --> |Toggle Mode 'M'| SW_MODE["Switch Rule <-> ML Engine"] --> MAIN_LOOP
    KEY_CHECK --> |Toggle Control 'C'| SW_CTRL["Toggle OS Mouse/Key Control"] --> MAIN_LOOP
    KEY_CHECK --> |No Key / Continue| MAIN_LOOP
    
    CLEANUP --> STOP
```

---

## 2. Gesture Classification & Temporal FSM Flowchart

```mermaid
flowchart TD
    SUB_START(["Input: Normalized Landmarks & Centroid"]) --> SWIPE_CHECK{"Swipe Velocity Spike?"}
    
    SWIPE_CHECK --> |dx/dt > Thresh| SET_SWIPE["Candidate = Swipe Gesture"]
    SWIPE_CHECK --> |No| ENG_CHECK{"Active Engine Mode?"}
    
    ENG_CHECK --> |Mode == 'ML'| ML_INFER["Run Random Forest Inference on 63D Vector"]
    ENG_CHECK --> |Mode == 'Rule'| RULE_INFER["Evaluate Geometric Angles & Finger States"]
    
    ML_INFER --> CONF_CHECK{"Prediction Confidence >= Min Thresh?"}
    CONF_CHECK --> |Yes| SET_ML["Candidate = ML Predicted Class"]
    CONF_CHECK --> |No| SET_IDLE["Candidate = Idle"]
    
    RULE_INFER --> SET_RULE["Candidate = Rule Gesture Class"]
    
    SET_SWIPE --> STAB_ACCUM["Candidate Stability Counter ++"]
    SET_ML --> STAB_ACCUM
    SET_RULE --> STAB_ACCUM
    SET_IDLE --> STAB_ACCUM
    
    STAB_ACCUM --> COUNT_CHECK{"Candidate Count >= Stability Frames?"}
    COUNT_CHECK --> |No| RET_NO_ACTION["Suppress Trigger (Return Idle / No-Op)"]
    COUNT_CHECK --> |Yes| ACTIVATE_GEST["Set Active Gesture = Candidate"]
    
    ACTIVATE_GEST --> TYPE_SWITCH{"Gesture Category?"}
    
    TYPE_SWITCH --> |Open Palm| CURSOR_MOVE["Apply EMA Smoothing -> Move Cursor"]
    TYPE_SWITCH --> |Pinch (Index+Thumb)| PINCH_FSM{"Pinch Duration?"}
    TYPE_SWITCH --> |Middle Pinch| RIGHT_CLICK["Trigger Right Click Event"]
    TYPE_SWITCH --> |Two Fingers (Up/Down)| SCROLL_FSM["Trigger Scroll Event"]
    TYPE_SWITCH --> |Thumb / Three Fingers / Screenshot| COOL_CHECK{"Cooldown Expired?"}
    TYPE_SWITCH --> |Closed Fist| FREEZE["Freeze Cursor Movement"]
    
    PINCH_FSM --> |< 0.28s (Tap)| SINGLE_CLICK["Trigger Left Click"]
    PINCH_FSM --> |Double Tap in 0.4s| DBL_CLICK["Trigger Double Click"]
    PINCH_FSM --> |Held > 0.28s| DRAG_HOLD["Trigger Mouse Down (Drag Mode)"]
    
    COOL_CHECK --> |Yes| TRIGGER_DISCRETE["Dispatch OS Keystroke & Reset Cooldown"]
    COOL_CHECK --> |No| RET_NO_ACTION
    
    CURSOR_MOVE --> SUB_END(["Action Complete"])
    SINGLE_CLICK --> SUB_END
    DBL_CLICK --> SUB_END
    DRAG_HOLD --> SUB_END
    RIGHT_CLICK --> SUB_END
    SCROLL_FSM --> SUB_END
    TRIGGER_DISCRETE --> SUB_END
    FREEZE --> SUB_END
    RET_NO_ACTION --> SUB_END
```

---

## 3. Machine Learning Training & Evaluation Flowchart

```mermaid
flowchart TD
    ML_START(["Start Training Pipeline"]) --> READ_CSV["Load gesture_landmarks.csv via Pandas"]
    READ_CSV --> VALIDATE["Validate 63 Numerical Features + 1 String Label"]
    VALIDATE --> ENCODE["Encode Class Labels with LabelEncoder"]
    
    ENCODE --> SPLIT["Stratified Split (80% Train, 20% Test)"]
    SPLIT --> K_FOLD["Initialize 5-Fold Stratified Cross-Validation"]
    
    K_FOLD --> TRAIN_RF["Train Random Forest (100 Trees, Depth=15)"]
    K_FOLD --> TRAIN_SVM["Train Support Vector Machine (RBF Kernel)"]
    K_FOLD --> TRAIN_KNN["Train K-Nearest Neighbors (k=5)"]
    
    TRAIN_RF --> EVAL_CV["Compute Mean CV Accuracy & Variance"]
    TRAIN_SVM --> EVAL_CV
    TRAIN_KNN --> EVAL_CV
    
    EVAL_CV --> TEST_EVAL["Evaluate on Held-Out 20% Test Set"]
    TEST_EVAL --> METRICS["Calculate Accuracy, Macro Precision, Recall, F1"]
    
    METRICS --> SELECT_BEST["Identify Winning Algorithm (Highest F1-Score)"]
    SELECT_BEST --> DUMP_MODEL["Serialize Winning Model to gesture_model.pkl"]
    DUMP_MODEL --> DUMP_ENC["Serialize LabelEncoder to label_encoder.pkl"]
    DUMP_ENC --> PLOT_CM["Generate Normalized Confusion Matrix Heatmap"]
    PLOT_CM --> PLOT_BAR["Generate Multi-Algorithm Metric Comparison Bar Chart"]
    PLOT_BAR --> DUMP_JSON["Export evaluation_report.json"]
    DUMP_JSON --> ML_END(["Training Pipeline Complete"])
```
