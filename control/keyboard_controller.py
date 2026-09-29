"""
Keyboard and Shortcut Control Module using PyAutoGUI.
Handles operating system keyboard shortcuts, navigation keys,
clipboard actions (Copy, Paste, Cut, Select All), and history (Undo, Redo).
"""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pyautogui
from utils.logger import logger

try:
    from .dispatcher import action_dispatcher
except ImportError:
    from dispatcher import action_dispatcher


class KeyboardController:
    """Dispatches keyboard strokes and standard operating system hotkeys."""

    @staticmethod
    def page_up():
        action_dispatcher.submit(pyautogui.press, "pageup", priority=True)
        logger.info("[Keyboard] Page Up pressed.")

    @staticmethod
    def page_down():
        action_dispatcher.submit(pyautogui.press, "pagedown", priority=True)
        logger.info("[Keyboard] Page Down pressed.")

    @staticmethod
    def browser_back():
        action_dispatcher.submit(pyautogui.hotkey, "alt", "left", priority=True)
        logger.info("[Keyboard] Browser Back (Alt+Left) dispatched.")

    @staticmethod
    def browser_forward():
        action_dispatcher.submit(pyautogui.hotkey, "alt", "right", priority=True)
        logger.info("[Keyboard] Browser Forward (Alt+Right) dispatched.")

    @staticmethod
    def app_switch():
        action_dispatcher.submit(pyautogui.hotkey, "alt", "tab", priority=True)
        logger.info("[Keyboard] Application Switch (Alt+Tab) dispatched.")

    @staticmethod
    def task_view():
        action_dispatcher.submit(pyautogui.hotkey, "win", "tab", priority=True)
        logger.info("[Keyboard] Task View (Win+Tab) dispatched.")

    @staticmethod
    def show_desktop():
        action_dispatcher.submit(pyautogui.hotkey, "win", "d", priority=True)
        logger.info("[Keyboard] Show Desktop (Win+D) dispatched.")

    @staticmethod
    def window_minimize():
        action_dispatcher.submit(pyautogui.hotkey, "win", "down", priority=True)
        logger.info("[Keyboard] Window Minimize (Win+Down) dispatched.")

    @staticmethod
    def window_maximize():
        action_dispatcher.submit(pyautogui.hotkey, "win", "up", priority=True)
        logger.info("[Keyboard] Window Maximize (Win+Up) dispatched.")

    @staticmethod
    def copy():
        action_dispatcher.submit(pyautogui.hotkey, "ctrl", "c", priority=True)
        logger.info("[Keyboard] Copy (Ctrl+C) dispatched.")

    @staticmethod
    def paste():
        action_dispatcher.submit(pyautogui.hotkey, "ctrl", "v", priority=True)
        logger.info("[Keyboard] Paste (Ctrl+V) dispatched.")

    @staticmethod
    def cut():
        action_dispatcher.submit(pyautogui.hotkey, "ctrl", "x", priority=True)
        logger.info("[Keyboard] Cut (Ctrl+X) dispatched.")

    @staticmethod
    def select_all():
        action_dispatcher.submit(pyautogui.hotkey, "ctrl", "a", priority=True)
        logger.info("[Keyboard] Select All (Ctrl+A) dispatched.")

    @staticmethod
    def undo():
        action_dispatcher.submit(pyautogui.hotkey, "ctrl", "z", priority=True)
        logger.info("[Keyboard] Undo (Ctrl+Z) dispatched.")

    @staticmethod
    def redo():
        action_dispatcher.submit(pyautogui.hotkey, "ctrl", "y", priority=True)
        logger.info("[Keyboard] Redo (Ctrl+Y) dispatched.")

    @staticmethod
    def enter():
        action_dispatcher.submit(pyautogui.press, "enter", priority=True)
        logger.info("[Keyboard] Enter key pressed.")

    @staticmethod
    def escape():
        action_dispatcher.submit(pyautogui.press, "escape", priority=True)
        logger.info("[Keyboard] Escape key pressed.")
