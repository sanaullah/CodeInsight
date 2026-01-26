"""Report loading utilities for Swarm Analysis (v3).

Provides functions to load previous analysis reports from file paths
or scan history for comparison with current analysis.

This module is adapted from the CodeInsight v2 implementation and
supports both legacy and v3 result formats.
"""

import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def load_report_from_file(file_path: str) -> Optional[str]:
    """Load previous report from file path.

    Args:
        file_path: Path to report file (.md or .txt)

    Returns:
        Report content as string, or None if error
    """
    try:
        path = Path(file_path)
        if not path.exists():
            logger.warning("Report file not found: %s", file_path)
            return None

        if not path.is_file():
            logger.warning("Path is not a file: %s", file_path)
            return None

        content = path.read_text(encoding="utf-8")
        logger.info("Loaded report from file: %s (%d chars)", file_path, len(content))
        return content

    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Error loading report from file %s: %s", file_path, exc, exc_info=True)
        return None


def extract_synthesized_report_from_result(result: Dict[str, Any]) -> Optional[str]:
    """Extract ``synthesized_report`` from a scan result dictionary.

    This is tolerant of both v2 and v3 storage formats.

    Args:
        result: Result dictionary from scan history or experience storage

    Returns:
        Synthesized report string, or None if not found
    """
    try:
        if not isinstance(result, dict):
            logger.warning("Result is not a dictionary when extracting synthesized_report")
            return None

        # 1) Direct access (v3 typical shape)
        report = result.get("synthesized_report")
        if isinstance(report, str):
            return report

        # 2) Nested under "result" field
        inner_result = result.get("result")
        if isinstance(inner_result, dict):
            report = inner_result.get("synthesized_report")
            if isinstance(report, str):
                return report

        # 3) JSON string stored under "result_json" (v2-style)
        result_json = result.get("result_json")
        if isinstance(result_json, str):
            try:
                parsed = json.loads(result_json)
            except json.JSONDecodeError as json_exc:
                logger.warning("Failed to parse result_json while extracting report: %s", json_exc)
            else:
                if isinstance(parsed, dict):
                    report = parsed.get("synthesized_report")
                    if isinstance(report, str):
                        return report

        logger.warning("synthesized_report not found in result payload")
        return None

    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Error extracting synthesized report from result: %s", exc, exc_info=True)
        return None


def load_report_from_scan_history(scan_id: int) -> Optional[str]:
    """Load previous report from scan history by ID.

    Args:
        scan_id: ID of scan in history database

    Returns:
        Report content as string, or None if error
    """
    try:
        # Local import to avoid circular dependency at module import time
        from utils.scan_history import get_scan_by_id

        scan = get_scan_by_id(scan_id)
        if not scan:
            logger.warning("Scan not found in history: %s", scan_id)
            return None

        report = extract_synthesized_report_from_result(scan)
        if report:
            logger.info("Loaded report from scan history: %s (%d chars)", scan_id, len(report))
        else:
            logger.warning("No synthesized report found in scan %s", scan_id)

        return report

    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Error loading report from scan history %s: %s", scan_id, exc, exc_info=True)
        return None
