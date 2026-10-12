"""
Phase 14: Custom Gesture-to-Command Mapping Engine & Profile Management Subsystem (Module 14)
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Key Architectural Features:
1. Dynamic Custom Action Types (Module 14):
   - HOTKEY: Multi-key combinations (e.g., ['ctrl', 'c'], ['alt', 'tab'], ['win', 'd']).
   - KEY_PRESS: Single key triggers with repeats (e.g., 'enter', 'space', 'escape').
   - TEXT_INPUT: Text typing automation for boilerplate templates and canned responses.
   - SHELL_CMD: Safe application launching and terminal scripts with security filtering.
   - CANONICAL_ACTION: Reference to built-in system/mouse/media/browser/presentation actions.
   - PYTHON_CALLBACK: Programmatic callable hooks for extensions and plugins.

2. Comprehensive Profile Schema & Persistence:
   - Versioned JSON schema (v1.0) with metadata (name, author, description, timestamps).
   - Atomic disk writes (write to .tmp then replace) to prevent config corruption during power loss.
   - Automated .bak backup creation prior to modifying existing profile files.
   - Schema validation with descriptive error reporting on malformed configs.

3. Conflict Analysis & Coverage Reporting:
   - Identifies duplicate actions assigned to conflicting gestures within the same mode.
   - Generates coverage reports showing mapped vs unmapped gestures across all 14 recognized classes.

4. Safe Custom Action Executor:
   - Command blacklist rejecting destructive commands (rm -rf, format, del, shutdown, etc.).
   - Mock mode support with zero host OS side effects for rapid automated unit testing.
   - Telemetry logging of executed actions, timestamps, and return codes.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pyautogui
from gestures.dynamic_gestures import DYNAMIC_GESTURE_CLASSES
from gestures.gesture_mapping import (
    DEFAULT_MAPPING_PATH,
    DEFAULT_MODE_MAPPINGS,
    OperationalMode,
)
from gestures.static_gestures import STATIC_GESTURE_CLASSES
from utils.logger import logger

# PyAutoGUI Safety Configuration
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.001

ALL_KNOWN_GESTURES: List[str] = sorted(list(STATIC_GESTURE_CLASSES) + list(DYNAMIC_GESTURE_CLASSES))

# Dangerous command blacklist to prevent security issues through custom shell bindings
DANGEROUS_COMMAND_PATTERNS = [
    r"\bformat\b",
    r"\bdel\s+/[fFqQsS]",
    r"\brm\s+-rf",
    r"\bsudo\s+rm",
    r":\(\)\s*\{\s*:\|:&\s*\};:",
    r"\bshutdown\s+/[sSrR]",
    r"\bdrop\s+database",
    r"\bdiskpart\b",
    r"\breg\s+delete",
]


class CustomActionType:
    """Supported types of customizable gesture actions."""
    HOTKEY = "hotkey"
    KEY_PRESS = "key_press"
    TEXT_INPUT = "text_input"
    SHELL_CMD = "shell_cmd"
    CANONICAL_ACTION = "canonical_action"
    PYTHON_CALLBACK = "python_callback"

    ALL_TYPES = [HOTKEY, KEY_PRESS, TEXT_INPUT, SHELL_CMD, CANONICAL_ACTION, PYTHON_CALLBACK]


class CustomAction:
    """Defines a single customizable gesture action."""

    def __init__(
        self,
        name: str,
        action_type: str,
        payload: Any,
        description: str = "",
        callback: Optional[Callable] = None,
    ):
        self.name = name.strip()
        self.action_type = action_type.lower().strip()
        self.payload = payload
        self.description = description
        self.callback = callback

        if self.action_type not in CustomActionType.ALL_TYPES:
            raise ValueError(f"Unknown action type: '{self.action_type}'. Must be one of {CustomActionType.ALL_TYPES}")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes action definition to a JSON-compatible dict."""
        return {
            "name": self.name,
            "action_type": self.action_type,
            "payload": self.payload,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomAction":
        """Deserializes action from a dictionary."""
        return cls(
            name=data.get("name", "UNNAMED"),
            action_type=data.get("action_type", CustomActionType.CANONICAL_ACTION),
            payload=data.get("payload", ""),
            description=data.get("description", ""),
        )

    def __repr__(self) -> str:
        return f"<CustomAction '{self.name}' ({self.action_type}): {self.payload}>"


class CustomActionExecutor:
    """Safely executes custom actions with security validation and mock support."""

    def __init__(self, mock_mode: bool = False):
        self.mock_mode = mock_mode
        self.execution_history: List[Dict[str, Any]] = []
        self._registered_callbacks: Dict[str, Callable] = {}

    def register_callback(self, name: str, fn: Callable):
        """Registers a Python callable by name."""
        self._registered_callbacks[name] = fn

    def execute(self, action: Union[CustomAction, str], **kwargs) -> bool:
        """
        Executes a custom or canonical action.

        Args:
            action: CustomAction instance or canonical action string (e.g. 'VOLUME_UP')
            kwargs: Optional arguments passed to callbacks or commands.

        Returns:
            bool indicating execution success.
        """
        if isinstance(action, str):
            # Treat plain string as canonical action
            action = CustomAction(
                name=action,
                action_type=CustomActionType.CANONICAL_ACTION,
                payload=action,
                description=f"Canonical action {action}",
            )

        t_start = time.perf_counter()
        success = False
        error_msg = None

        try:
            if action.action_type == CustomActionType.HOTKEY:
                success = self._exec_hotkey(action.payload)

            elif action.action_type == CustomActionType.KEY_PRESS:
                success = self._exec_key_press(action.payload)

            elif action.action_type == CustomActionType.TEXT_INPUT:
                success = self._exec_text_input(action.payload)

            elif action.action_type == CustomActionType.SHELL_CMD:
                success = self._exec_shell_cmd(action.payload)

            elif action.action_type == CustomActionType.PYTHON_CALLBACK:
                success = self._exec_callback(action, **kwargs)

            elif action.action_type == CustomActionType.CANONICAL_ACTION:
                # Canonical actions are handled directly or logged for router dispatch
                success = True

            else:
                error_msg = f"Unsupported action type: {action.action_type}"
                logger.warning(f"[CustomActionExecutor] {error_msg}")

        except PermissionError as e:
            error_msg = str(e)
            duration_ms = (time.perf_counter() - t_start) * 1000.0
            self._record_execution(action, False, duration_ms, error_msg)
            raise
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[CustomActionExecutor] Failed executing '{action.name}': {e}")
            success = False

        duration_ms = (time.perf_counter() - t_start) * 1000.0
        self._record_execution(action, success, duration_ms, error_msg)
        return success

    def _exec_hotkey(self, keys: Union[List[str], str]) -> bool:
        key_list = [k.strip().lower() for k in keys] if isinstance(keys, list) else [k.strip().lower() for k in keys.split("+")]
        if not key_list:
            return False

        if self.mock_mode:
            return True

        pyautogui.hotkey(*key_list)
        return True

    def _exec_key_press(self, key: str) -> bool:
        key_name = str(key).strip().lower()
        if not key_name:
            return False

        if self.mock_mode:
            return True

        pyautogui.press(key_name)
        return True

    def _exec_text_input(self, text: str) -> bool:
        if not text:
            return False

        if self.mock_mode:
            return True

        pyautogui.write(str(text))
        return True

    def _exec_shell_cmd(self, cmd: str) -> bool:
        cmd_str = str(cmd).strip()
        if not cmd_str:
            return False

        # Security validation against dangerous commands
        for pattern in DANGEROUS_COMMAND_PATTERNS:
            if re.search(pattern, cmd_str, re.IGNORECASE):
                logger.error(f"[CustomActionExecutor] Command blocked by security policy: '{cmd_str}'")
                raise PermissionError(f"Command '{cmd_str}' matches security blacklist.")

        if self.mock_mode:
            return True

        # Launch detached process without blocking
        subprocess.Popen(
            cmd_str,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        return True

    def _exec_callback(self, action: CustomAction, **kwargs) -> bool:
        fn = action.callback or self._registered_callbacks.get(action.name)
        if fn is None:
            raise ValueError(f"No callable registered for '{action.name}'")

        fn(**kwargs)
        return True

    def _record_execution(self, action: CustomAction, success: bool, duration_ms: float, error: Optional[str]):
        self.execution_history.append({
            "timestamp": time.time(),
            "action_name": action.name,
            "action_type": action.action_type,
            "success": success,
            "duration_ms": round(duration_ms, 3),
            "error": error,
        })
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-500:]


class MappingProfile:
    """Represents a structured profile containing metadata, custom actions, and mode mappings."""

    def __init__(
        self,
        name: str = "Default Profile",
        version: str = "1.0",
        author: str = "System",
        description: str = "Default HCI gesture mapping profile",
    ):
        self.name = name
        self.version = version
        self.author = author
        self.description = description
        self.created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.updated_at = self.created_at

        # Custom actions repository: action_name -> CustomAction
        self.custom_actions: Dict[str, CustomAction] = {}

        # Mappings: mode -> {gesture -> action_name}
        self.mappings: Dict[str, Dict[str, str]] = {
            mode: dict(m) for mode, m in DEFAULT_MODE_MAPPINGS.items()
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializes entire profile to JSON dictionary."""
        return {
            "metadata": {
                "name": self.name,
                "version": self.version,
                "author": self.author,
                "description": self.description,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            },
            "custom_actions": {k: act.to_dict() for k, act in self.custom_actions.items()},
            "mappings": self.mappings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MappingProfile":
        """Constructs a MappingProfile instance from a dictionary with validation."""
        meta = data.get("metadata", {})
        profile = cls(
            name=meta.get("name", "Loaded Profile"),
            version=meta.get("version", "1.0"),
            author=meta.get("author", "User"),
            description=meta.get("description", ""),
        )
        profile.created_at = meta.get("created_at", profile.created_at)
        profile.updated_at = meta.get("updated_at", profile.updated_at)

        # Deserialize custom actions
        raw_actions = data.get("custom_actions", {})
        if isinstance(raw_actions, dict):
            for name, act_data in raw_actions.items():
                if isinstance(act_data, dict):
                    profile.custom_actions[name] = CustomAction.from_dict(act_data)

        # Deserialize mode mappings
        raw_mappings = data.get("mappings", {})
        if isinstance(raw_mappings, dict):
            for mode, binds in raw_mappings.items():
                if isinstance(binds, dict):
                    profile.mappings[mode] = {str(k): str(v) for k, v in binds.items()}

        return profile


class CustomMappingEngine:
    """
    Central manager for dynamic gesture mapping, profile import/export,
    atomic JSON persistence, conflict analysis, and runtime action execution.
    """

    def __init__(
        self,
        config_path: str = DEFAULT_MAPPING_PATH,
        mock_mode: bool = False,
    ):
        self.config_path = os.path.abspath(config_path)
        self.mock_mode = mock_mode
        self.executor = CustomActionExecutor(mock_mode=mock_mode)
        self.profile = MappingProfile()
        self.active_mode = OperationalMode.MOUSE

        # Attempt loading config or save default
        if os.path.exists(self.config_path):
            self.load_from_disk()
        else:
            self.save_to_disk()

    # =========================================================================
    # Binding & Action Management (Module 14)
    # =========================================================================

    def define_custom_action(
        self,
        name: str,
        action_type: str,
        payload: Any,
        description: str = "",
        callback: Optional[Callable] = None,
    ) -> CustomAction:
        """Defines and registers a new custom action in the profile."""
        action = CustomAction(name, action_type, payload, description, callback)
        self.profile.custom_actions[action.name] = action
        self.profile.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[CustomMappingEngine] Registered custom action '{action.name}' ({action.action_type})")
        return action

    def bind_gesture(
        self,
        gesture: str,
        action_name: str,
        mode: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Binds a recognized gesture to an action name within a given mode.

        Args:
            gesture: Recognized gesture name (e.g. 'PEACE', 'SWIPE_UP')
            action_name: Action name (canonical or custom action)
            mode: Operational mode name (defaults to active_mode)

        Returns:
            Tuple[bool, str]: (Success, message)
        """
        g = gesture.upper().strip()
        target_mode = (mode or self.active_mode).upper().strip()

        if target_mode not in self.profile.mappings:
            self.profile.mappings[target_mode] = {}

        self.profile.mappings[target_mode][g] = action_name.strip()
        self.profile.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[CustomMappingEngine] Bound [{g}] -> [{action_name}] in [{target_mode}]")
        return True, "Success"

    def unbind_gesture(self, gesture: str, mode: Optional[str] = None) -> bool:
        """Removes a gesture mapping from the specified mode."""
        g = gesture.upper().strip()
        target_mode = (mode or self.active_mode).upper().strip()

        if target_mode in self.profile.mappings and g in self.profile.mappings[target_mode]:
            del self.profile.mappings[target_mode][g]
            self.profile.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
            return True
        return False

    def get_action_for_gesture(
        self,
        gesture: str,
        mode: Optional[str] = None,
    ) -> Optional[Union[CustomAction, str]]:
        """
        Resolves the mapped action for a gesture.
        Returns CustomAction instance if custom, or canonical string if standard.
        """
        g = gesture.upper().strip()
        target_mode = (mode or self.active_mode).upper().strip()

        action_name = self.profile.mappings.get(target_mode, {}).get(g)
        if not action_name:
            # Fallback to MOUSE mode
            action_name = self.profile.mappings.get(OperationalMode.MOUSE, {}).get(g)

        if not action_name:
            return None

        # Check if custom action exists
        if action_name in self.profile.custom_actions:
            return self.profile.custom_actions[action_name]

        return action_name

    def execute_gesture(self, gesture: str, mode: Optional[str] = None, **kwargs) -> bool:
        """Resolves and immediately executes the action assigned to a gesture."""
        resolved = self.get_action_for_gesture(gesture, mode)
        if resolved is None:
            return False
        return self.executor.execute(resolved, **kwargs)

    # =========================================================================
    # Conflict Detection & Coverage Analytics (Module 14)
    # =========================================================================

    def detect_conflicts(self, mode: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Scans bindings for conflicts:
        1. Duplicate actions mapped to different gestures.
        2. Unrecognized gesture names.
        """
        target_mode = (mode or self.active_mode).upper().strip()
        mappings = self.profile.mappings.get(target_mode, {})

        conflicts = []
        action_to_gestures: Dict[str, List[str]] = {}

        for g, act in mappings.items():
            if g not in ALL_KNOWN_GESTURES:
                conflicts.append({
                    "type": "UNKNOWN_GESTURE",
                    "mode": target_mode,
                    "gesture": g,
                    "action": act,
                    "message": f"Gesture '{g}' is not in recognized gesture repertoire.",
                })

            action_to_gestures.setdefault(act, []).append(g)

        for act, gestures in action_to_gestures.items():
            # Exclude benign passive actions
            if len(gestures) > 1 and act not in ["HOVER", "NONE"]:
                conflicts.append({
                    "type": "DUPLICATE_ACTION",
                    "mode": target_mode,
                    "action": act,
                    "gestures": gestures,
                    "message": f"Multiple gestures {gestures} map to identical action '{act}'.",
                })

        return conflicts

    def get_coverage_report(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """Analyzes gesture coverage for a given mode."""
        target_mode = (mode or self.active_mode).upper().strip()
        mappings = self.profile.mappings.get(target_mode, {})

        mapped_gestures = set(mappings.keys())
        all_gestures_set = set(ALL_KNOWN_GESTURES)

        valid_mapped = mapped_gestures.intersection(all_gestures_set)
        unmapped = all_gestures_set - mapped_gestures

        return {
            "mode": target_mode,
            "total_supported_gestures": len(ALL_KNOWN_GESTURES),
            "mapped_count": len(valid_mapped),
            "unmapped_count": len(unmapped),
            "coverage_pct": round((len(valid_mapped) / len(ALL_KNOWN_GESTURES)) * 100.0, 1),
            "mapped_gestures": sorted(list(valid_mapped)),
            "unmapped_gestures": sorted(list(unmapped)),
        }

    # =========================================================================
    # Schema Validation & Atomic File Persistence
    # =========================================================================

    def validate_schema(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validates that a dictionary conforms to the Custom Mapping Profile Schema."""
        errors = []
        if not isinstance(data, dict):
            return False, ["Root must be a JSON object."]

        if "metadata" not in data or not isinstance(data["metadata"], dict):
            errors.append("Missing or invalid 'metadata' object.")

        if "mappings" not in data or not isinstance(data["mappings"], dict):
            errors.append("Missing or invalid 'mappings' dictionary.")

        if "custom_actions" in data and not isinstance(data["custom_actions"], dict):
            errors.append("'custom_actions' must be a dictionary if present.")

        # Check action payloads
        for name, act_data in data.get("custom_actions", {}).items():
            if not isinstance(act_data, dict):
                errors.append(f"Custom action '{name}' definition is not an object.")
                continue
            act_type = act_data.get("action_type")
            if act_type not in CustomActionType.ALL_TYPES:
                errors.append(f"Custom action '{name}' has invalid action_type '{act_type}'.")

        return len(errors) == 0, errors

    def save_to_disk(self, target_path: Optional[str] = None) -> bool:
        """
        Saves profile to disk atomically with automatic backup.
        """
        dest_path = os.path.abspath(target_path or self.config_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # 1. Create .bak backup if target exists
        if os.path.exists(dest_path):
            bak_path = dest_path + ".bak"
            try:
                shutil.copy2(dest_path, bak_path)
            except Exception as e:
                logger.warning(f"[CustomMappingEngine] Failed to create backup: {e}")

        # 2. Write to temp file first (atomic write)
        temp_path = dest_path + ".tmp"
        try:
            data = self.profile.to_dict()
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            # Atomic replace
            os.replace(temp_path, dest_path)
            logger.info(f"[CustomMappingEngine] Saved profile to '{dest_path}'")
            return True
        except Exception as e:
            logger.error(f"[CustomMappingEngine] Failed atomic save: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False

    def load_from_disk(self, source_path: Optional[str] = None) -> bool:
        """Loads profile from disk with schema validation and safe fallback."""
        src_path = os.path.abspath(source_path or self.config_path)
        if not os.path.exists(src_path):
            logger.warning(f"[CustomMappingEngine] Config file not found at: {src_path}")
            return False

        try:
            with open(src_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            is_valid, errors = self.validate_schema(data)
            if not is_valid:
                logger.error(f"[CustomMappingEngine] Schema validation failed for '{src_path}': {errors}")
                return False

            self.profile = MappingProfile.from_dict(data)
            logger.info(f"[CustomMappingEngine] Loaded profile '{self.profile.name}' from '{src_path}'")
            return True
        except Exception as e:
            logger.error(f"[CustomMappingEngine] Error parsing profile: {e}")
            return False

    def export_profile(self, export_path: str) -> bool:
        """Exports the active profile to an external file."""
        return self.save_to_disk(target_path=export_path)

    def import_profile(self, import_path: str) -> bool:
        """Imports an external profile, validates it, and replaces the current profile."""
        return self.load_from_disk(source_path=import_path)

    def reset_to_defaults(self, mode: Optional[str] = None):
        """Resets mappings for a specific mode or all modes to defaults."""
        if mode:
            target_mode = mode.upper().strip()
            if target_mode in DEFAULT_MODE_MAPPINGS:
                self.profile.mappings[target_mode] = dict(DEFAULT_MODE_MAPPINGS[target_mode])
        else:
            self.profile.mappings = {m: dict(binds) for m, binds in DEFAULT_MODE_MAPPINGS.items()}
            self.profile.custom_actions.clear()
        self.profile.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")


# =============================================================================
# Interactive Terminal REPL Interface (Module 14 CLI)
# =============================================================================

def interactive_cli():
    """Terminal CLI menu for interactive gesture re-mapping."""
    engine = CustomMappingEngine(mock_mode=True)
    print("=" * 60)
    print(" HCI Custom Gesture Mapping Configuration CLI (Module 14)")
    print("=" * 60)

    while True:
        print(f"\nActive Profile: {engine.profile.name} (v{engine.profile.version})")
        print(f"Active Mode:    {engine.active_mode}")
        print("1. List Mappings for Active Mode")
        print("2. Switch Active Mode")
        print("3. Bind Gesture to Custom Hotkey")
        print("4. Bind Gesture to Shell Command / App")
        print("5. Bind Gesture to Text Typing")
        print("6. Detect Mapping Conflicts")
        print("7. View Gesture Coverage Report")
        print("8. Save Profile to Disk")
        print("9. Reset to Defaults")
        print("0. Exit CLI")

        choice = input("\nSelect an option [0-9]: ").strip()
        if choice == "0":
            break

        elif choice == "1":
            mappings = engine.profile.mappings.get(engine.active_mode, {})
            print(f"\n--- Mappings for Mode [{engine.active_mode}] ---")
            for g, act in sorted(mappings.items()):
                print(f"  {g:<16} -> {act}")

        elif choice == "2":
            print(f"Available modes: {', '.join(OperationalMode.ALL_MODES)}")
            new_mode = input("Enter mode name: ").strip().upper()
            if new_mode in OperationalMode.ALL_MODES:
                engine.active_mode = new_mode
                print(f"Switched active mode to: {new_mode}")
            else:
                print("Invalid mode!")

        elif choice == "3":
            g = input(f"Enter gesture name (e.g. PEACE, POINT, SWIPE_RIGHT): ").strip().upper()
            act_name = input("Enter custom action name (e.g. COPY_CLIPBOARD): ").strip()
            keys_input = input("Enter keys separated by '+' (e.g. ctrl+c, alt+tab): ").strip()
            keys = [k.strip() for k in keys_input.split("+")]
            engine.define_custom_action(act_name, CustomActionType.HOTKEY, keys, f"Hotkey {keys_input}")
            engine.bind_gesture(g, act_name)
            print(f"Bound [{g}] -> Hotkey {keys_input}")

        elif choice == "4":
            g = input(f"Enter gesture name: ").strip().upper()
            act_name = input("Enter custom action name (e.g. LAUNCH_NOTEPAD): ").strip()
            cmd = input("Enter executable or command (e.g. notepad.exe): ").strip()
            engine.define_custom_action(act_name, CustomActionType.SHELL_CMD, cmd, f"Shell {cmd}")
            engine.bind_gesture(g, act_name)
            print(f"Bound [{g}] -> Shell '{cmd}'")

        elif choice == "5":
            g = input(f"Enter gesture name: ").strip().upper()
            act_name = input("Enter custom action name: ").strip()
            txt = input("Enter text string to type: ")
            engine.define_custom_action(act_name, CustomActionType.TEXT_INPUT, txt, f"Type: {txt}")
            engine.bind_gesture(g, act_name)
            print(f"Bound [{g}] -> Text typing")

        elif choice == "6":
            conflicts = engine.detect_conflicts()
            if not conflicts:
                print("\n[OK] No conflicts detected in active mode.")
            else:
                print(f"\n[WARNING] {len(conflicts)} conflict(s) found:")
                for c in conflicts:
                    print(f"  - {c['message']}")

        elif choice == "7":
            rep = engine.get_coverage_report()
            print(f"\n--- Coverage for Mode [{rep['mode']}] ---")
            print(f"  Coverage: {rep['coverage_pct']}% ({rep['mapped_count']}/{rep['total_supported_gestures']})")
            print(f"  Unmapped gestures: {', '.join(rep['unmapped_gestures'])}")

        elif choice == "8":
            if engine.save_to_disk():
                print("Profile successfully saved!")
            else:
                print("Save failed!")

        elif choice == "9":
            engine.reset_to_defaults()
            print("Reset all mappings to defaults.")


if __name__ == "__main__":
    interactive_cli()
