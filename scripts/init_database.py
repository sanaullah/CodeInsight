"""Compatibility entry point for database initialization."""

from infrastructure.scripts.init_database import *
from infrastructure.scripts.init_database import main


if __name__ == "__main__":
    raise SystemExit(main())
