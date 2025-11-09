"""
Lightweight ONNX-only inference helpers for VoxCPM.

This package factors out utilities, constants, and functions
from the original infer.py to enable reuse across CLI and server.
"""

__all__ = [
    "constants",
    "runtime",
]