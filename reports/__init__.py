"""Compatibility imports for :mod:`analysis.reports`."""

from pathlib import Path

from analysis import reports as _target
from analysis.reports import *

__all__ = _target.__all__
__path__ = [str(Path(__file__).resolve().parent), *list(_target.__path__)]
