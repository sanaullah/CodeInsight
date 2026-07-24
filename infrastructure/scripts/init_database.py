"""Initialize or upgrade the single CodeInsight SQLite database."""

from __future__ import annotations

import argparse
from pathlib import Path

from infrastructure.db.database import initialize_database


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "database",
        nargs="?",
        type=Path,
        default=Path(".codeinsight") / "codeinsight.db",
        help="single application database path",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    version = initialize_database(args.database)
    print(f"CodeInsight database ready at {args.database} (schema {version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
