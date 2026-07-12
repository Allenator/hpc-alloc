"""Core package for hpc-alloc.

The executable is intentionally kept separate from these importable services so
configuration, persistence, transport, and lifecycle policy can be tested
without invoking the CLI.
"""

from .config import Config
from .context import RuntimeContext
from .state import StateRepository

__all__ = ["Config", "RuntimeContext", "StateRepository"]

__version__ = "2.0.0"
