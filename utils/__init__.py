"""Compatibility imports for :mod:`infrastructure.utils`."""

from pathlib import Path

from infrastructure import utils as _target
from infrastructure.utils import *

__all__ = _target.__all__
__path__ = [str(Path(__file__).resolve().parent), *list(_target.__path__)]
