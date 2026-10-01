"""Control module for cursor smoothing, screen mapping, and OS automation."""
from .smoother import CoordinateSmoother
from .controller import ComputerController

__all__ = ["CoordinateSmoother", "ComputerController"]
