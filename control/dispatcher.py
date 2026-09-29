"""
Asynchronous Action Dispatcher Module.
Decouples operating system input commands (PyAutoGUI, PyCaw, WMI)
from the real-time computer vision thread using a prioritized worker queue.
Provides emergency stop safety override.
"""

import os
import queue
import sys
import threading
import time
from typing import Any, Callable, Optional

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config
from utils.logger import logger


class ActionDispatcher:
    """Non-blocking queue-based action dispatcher with emergency stop safety."""

    def __init__(self, maxsize: int = 15):
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self.is_active = True
        self.is_emergency_stopped = False
        self._running = True

        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logger.info("ActionDispatcher worker thread initialized.")

    def submit(self, action_func: Callable, *args, priority: bool = False, **kwargs) -> bool:
        """
        Submits an action function to be executed asynchronously.
        If priority is False and queue is full, stale requests are dropped to prevent backlog.
        """
        if not self.is_active or self.is_emergency_stopped:
            return False

        try:
            if priority and self._queue.full():
                # Discard oldest non-priority item to make room
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put_nowait((action_func, args, kwargs))
            return True
        except queue.Full:
            return False

    def emergency_stop(self):
        """Immediately halts all action execution and clears pending tasks."""
        self.is_emergency_stopped = True
        self.is_active = False
        # Drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        logger.warning("EMERGENCY STOP TRIGGERED: Action queue flushed and disabled.")

    def reset_emergency_stop(self):
        """Clears emergency stop and restores action dispatching."""
        self.is_emergency_stopped = False
        self.is_active = True
        logger.success("Emergency stop cleared. Action dispatching resumed.")

    def toggle_active(self) -> bool:
        """Toggles active state on/off."""
        if self.is_emergency_stopped:
            self.reset_emergency_stop()
            return True
        self.is_active = not self.is_active
        logger.info(f"Action dispatcher state: {'ACTIVE' if self.is_active else 'PAUSED'}")
        return self.is_active

    def _worker_loop(self):
        """Worker thread executing system actions."""
        while self._running:
            try:
                action_func, args, kwargs = self._queue.get(timeout=0.1)
                if self.is_active and not self.is_emergency_stopped:
                    try:
                        action_func(*args, **kwargs)
                    except Exception as e:
                        logger.error(f"Error executing dispatched action {getattr(action_func, '__name__', 'unknown')}: {e}")
                self._queue.task_done()
            except queue.Empty:
                continue

    def stop(self):
        """Shuts down the dispatcher cleanly."""
        self._running = False
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.5)


action_dispatcher = ActionDispatcher()
