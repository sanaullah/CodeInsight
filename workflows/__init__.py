"""Compatibility imports for the relocated :mod:`workflow` package."""

from pathlib import Path

import workflow as _target
from workflow import *

__all__ = _target.__all__
__version__ = _target.__version__
__path__ = [str(Path(__file__).resolve().parent), *list(_target.__path__)]
