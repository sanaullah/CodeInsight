"""Compatibility imports for :mod:`analysis.chunking`."""

from pathlib import Path

from analysis import chunking as _target
from analysis.chunking import *

__all__ = _target.__all__
__path__ = [str(Path(__file__).resolve().parent), *list(_target.__path__)]
