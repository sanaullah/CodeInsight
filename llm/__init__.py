"""Compatibility imports for :mod:`infrastructure.llm`."""

from pathlib import Path

from infrastructure import llm as _target
from infrastructure.llm import *

__all__ = _target.__all__
__version__ = _target.__version__
__path__ = [str(Path(__file__).resolve().parent), *list(_target.__path__)]
