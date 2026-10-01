# UML Diagrams: Use Case, Sequence & Class Architecture

This document provides formal Unified Modeling Language (UML) specifications for the **AI-Based Vision-Driven HCI System**.

---

## 1. UML Use Case Diagram

```mermaid
flowchart LR
    USER["User / Operator"]
    OS_ACTOR["Host Operating System"]

    subgraph "AI Vision HCI System Boundary"
        UC1(["UC1: Move Mouse Pointer"])
        UC2(["UC2: Perform Mouse Clicks (Left/Right/Double)"])
        UC3(["UC3: Drag and Drop Items"])
        UC4(["UC4: Scroll Documents / Web Pages"])
        UC5(["UC5: Adjust System Audio Volume"])
        UC6(["UC6: Control Multimedia Play / Pause"])
        UC7(["UC7: Navigate Slide Decks / Tracks"])
        UC8(["UC8: Capture Screen Snapshot"])
        UC9(["UC9: Switch Classifier Engine (Rule <-> ML)"])
        UC10(["UC10: Tune Cursor Sensitivity & Margins"])
        UC11(["UC11: Collect Landmark Dataset"])
        UC12(["UC12: Train & Evaluate Classifiers"])
    end

    USER --> UC1
    USER --> UC2
    USER --> UC3
    USER --> UC4
    USER --> UC5
    USER --> UC6
    USER --> UC7
    USER --> UC8
    USER --> UC9
    USER --> UC10
    USER --> UC11
    USER --> UC12

    UC1 --> OS_ACTOR
    UC2 --> OS_ACTOR
    UC3 --> OS_ACTOR
    UC4 --> OS_ACTOR
    UC5 --> OS_ACTOR
    UC6 --> OS_ACTOR
    UC7 --> OS_ACTOR
    UC8 --> OS_ACTOR
```

### Use Case Specifications:
- **UC1 (Move Mouse Pointer)**: User extends open palm and shifts hand position within the camera's active zone. System remaps coordinates through the EMA filter and dispatches `pyautogui.moveTo()`.
- **UC2 (Mouse Clicks)**: User touches thumb tip to index tip (left click), thumb tip to middle tip (right click), or performs a quick double pinch (double click).
- **UC3 (Drag and Drop)**: User pinches and holds thumb and index fingertips for $>0.28$ seconds. System triggers `pyautogui.mouseDown()` and keeps it locked until released.
- **UC4 (Scroll)**: User points index and middle fingers up or down. System dispatches `pyautogui.scroll()`.
- **UC5 (Volume)**: User clenches a fist with thumb straight up or down. System increments or decrements master system volume via virtual keycodes.
- **UC7 (Navigate Slides)**: User sweeps hand quickly left or right. System detects horizontal optical velocity spike and fires `nexttrack`/`prevtrack` and arrow keys.

---

## 2. UML Sequence Diagram (Frame Execution Cycle)

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Cam as CameraStream
    participant Det as HandDetector
    participant Proc as LandmarkProcessor
    participant Engine as Classifier (Rule/ML)
    participant FSM as GestureStateManager
    participant Smooth as CoordinateSmoother
    participant Ctrl as ComputerController
    participant OS as Host OS
    participant HUD as HUDOverlay

    User->>Cam: Physical Hand Posture
    Cam->>Cam: read_frame(), mirror flip, calc FPS
    Cam->>Det: BGR Video Frame
    Det->>Det: mp_hands.process(RGB_Frame)
    Det-->>Proc: 21 3D Landmarks (x, y, z)
    
    Proc->>Proc: normalize_landmarks() -> 63D Vector
    Proc-->>Engine: Normalized Landmarks & Features
    
    Engine->>Engine: classify() / predict()
    Engine-->>FSM: (raw_gesture, confidence)
    
    FSM->>FSM: Check stability & cooldown FSM
    FSM-->>Ctrl: (stable_gesture, trigger_action, event_type)
    
    opt When Gesture is Continuous or Drag
        Proc-->>Smooth: Index Tip Coordinates (x, y)
        Smooth->>Smooth: Linear Remap + EMA Filter
        Smooth-->>Ctrl: Smoothed (screen_x, screen_y)
    end
    
    opt When Action Triggered
        Ctrl->>OS: pyautogui.moveTo() / click() / press()
        OS-->>Ctrl: Action Executed
    end
    
    Ctrl-->>HUD: Telemetry (Gesture, Action, Conf, FPS)
    HUD->>HUD: Render Glassmorphism Cards & Skeleton
    HUD-->>User: Display Output on Screen / Canvas
```

---

## 3. UML Class Architecture Diagram

```mermaid
classDiagram
    class CameraStream {
        +int camera_id
        +int width
        +int height
        +float fps
        +start() bool
        +read_frame() Tuple[bool, ndarray]
        +get_fps() float
        +release() void
    }

    class HandDetector {
        +Hands mp_hands
        +find_hands(frame)
        +get_hand_data(shape, index) dict
        +draw_hand_mesh(frame, index, modern) ndarray
        +close() void
    }

    class LandmarkProcessor {
        +normalize_landmarks(landmarks) ndarray
        +get_finger_states(landmarks, hand) List[int]
        +get_pinch_distance(landmarks) float
        +get_middle_pinch_distance(landmarks) float
        +get_palm_center(landmarks) Tuple[float, float]
    }

    class RuleGestureEngine {
        +float pinch_threshold
        +float middle_pinch_threshold
        +classify(landmarks, handedness) Tuple[str, float]
    }

    class MLGestureClassifier {
        +Path model_path
        +bool is_ready
        +load_model() bool
        +predict(landmarks_norm) Tuple[str, float]
    }

    class GestureStateManager {
        +int stability_frames
        +float discrete_cooldown
        +bool is_dragging
        +process(raw_gesture, raw_conf) Tuple[str, bool, str]
        +reset() void
    }

    class CoordinateSmoother {
        +int screen_width
        +int screen_height
        +float alpha
        +smooth_and_map(raw_x, raw_y) Tuple[int, int]
        +reset() void
    }

    class ComputerController {
        +bool enabled
        +bool is_mouse_down
        +set_enabled(state) void
        +move_cursor(x, y) void
        +click(button) void
        +scroll(direction, amount) void
        +volume_up() void
        +volume_down() void
        +dispatch(gesture, event, coords) str
    }

    class HUDOverlay {
        +draw_glass_rect(img, pt1, pt2, color, alpha) ndarray
        +draw_active_region(frame, is_active) ndarray
        +render_hud(...) ndarray
    }

    class HCIPipeline {
        +CameraStream camera
        +HandDetector detector
        +RuleGestureEngine rule_engine
        +MLGestureClassifier ml_classifier
        +GestureStateManager state_manager
        +CoordinateSmoother smoother
        +ComputerController controller
        +HUDOverlay hud
        +start() bool
        +process_one_frame() Tuple[bool, ndarray, dict]
        +stop() void
    }

    HCIPipeline *-- CameraStream
    HCIPipeline *-- HandDetector
    HCIPipeline *-- LandmarkProcessor
    HCIPipeline *-- RuleGestureEngine
    HCIPipeline *-- MLGestureClassifier
    HCIPipeline *-- GestureStateManager
    HCIPipeline *-- CoordinateSmoother
    HCIPipeline *-- ComputerController
    HCIPipeline *-- HUDOverlay
```
