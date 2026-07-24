"""Compatibility imports for the relocated :mod:`analysis.agents` package."""

from pathlib import Path

from analysis import agents as _target
from analysis.agents import *

__all__ = _target.__all__
__path__ = [str(Path(__file__).resolve().parent), *list(_target.__path__)]
