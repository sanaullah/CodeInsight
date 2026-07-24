"""Compatibility imports for :mod:`infrastructure.services`."""

from pathlib import Path

from infrastructure import services as _target
from infrastructure.services import *

__all__ = _target.__all__
__path__ = [str(Path(__file__).resolve().parent), *list(_target.__path__)]
