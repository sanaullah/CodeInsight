"""Compatibility imports for :mod:`indexing.scanners`."""

from pathlib import Path

from indexing import scanners as _target
from indexing.scanners import *

__all__ = _target.__all__
__path__ = [str(Path(__file__).resolve().parent), *list(_target.__path__)]
