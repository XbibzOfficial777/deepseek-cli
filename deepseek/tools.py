"""
DeepSeek CLI — tools.py is DEPRECATED.
All tool definitions moved to toolkit.py.
This file kept for backward compatibility.
"""
import warnings
warnings.warn(
    "tools.py is deprecated. Use toolkit.py instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .toolkit import ToolRegistry

__all__ = ['ToolRegistry']
