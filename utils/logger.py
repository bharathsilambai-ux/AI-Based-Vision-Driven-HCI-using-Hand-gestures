"""
Thread-safe system logging module with ring buffer for GUI console integration.
"""

from collections import deque
from datetime import datetime
import threading


class SystemLogger:
    """Thread-safe logger storing recent log records for UI presentation."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SystemLogger, cls).__new__(cls)
                cls._instance._buffer = deque(maxlen=200)
                cls._instance._callbacks = []
            return cls._instance

    def log(self, level: str, message: str):
        now_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted = f"[{now_str}] [{level.upper()}] {message}"
        with self._lock:
            self._buffer.append(formatted)
            # Invoke registered UI callbacks
            for cb in self._callbacks:
                try:
                    cb(formatted)
                except Exception:
                    pass
        print(formatted)

    def info(self, msg: str):
        self.log("INFO", msg)

    def warning(self, msg: str):
        self.log("WARN", msg)

    def error(self, msg: str):
        self.log("ERROR", msg)

    def success(self, msg: str):
        self.log("SUCCESS", msg)

    def register_callback(self, cb):
        with self._lock:
            if cb not in self._callbacks:
                self._callbacks.append(cb)

    def get_recent_logs(self):
        with self._lock:
            return list(self._buffer)


logger = SystemLogger()
