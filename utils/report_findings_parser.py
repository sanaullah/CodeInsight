"""Structured findings parser for Swarm Analysis reports.

This module parses markdown swarm analysis reports (v2/v3) and
extracts structured findings for comparison across runs.

The parser is intentionally heuristic and resilient:
- It works best with reports that follow the standard Swarm template
  (Executive Summary, Key Findings, etc.).
- It degrades gracefully when sections are missing or formatted
  differently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


SEVERITY_LEVELS = ("CRITICAL", "HIGH", "MEDIUM", "LOW")


@dataclass
class ParsedIssue:
    """Represents a single parsed issue/finding from a report."""

    severity: str
    title: str
    description: str
    file_references: List[str]


def _extract_report_date(text: str) -> Optional[str]:
    """Best-effort extraction of a report date from the header.

    Looks for lines like "**Date**: 2025-12-13" or "Date: ...".
    """

    date_pattern = re.compile(r"^\s*\*\*?Date\*\*?\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
    match = date_pattern.search(text)
    if match:
        return match.group(1).strip()
    return None


def extract_file_references(report_text: str) -> List[str]:
    """Extract file:line references from a report.

    Matches common patterns like:
    - `path/to/file.py:123`
    - `path/to/file.py` (without line)
    - In markdown code fences with comments such as `# file: path.py`.
    """

    # Simple heuristic: look for .py, .js, .ts, .java, etc. followed by optional :line
    file_pattern = re.compile(
        r"(?P<path>[\w./\\-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|rb|php|cs|cpp|c|md))(?::(?P<line>\d+))?",
        re.IGNORECASE,
    )

    references: List[str] = []
    for match in file_pattern.finditer(report_text):
        path = match.group("path")
        line = match.group("line")
        if line:
            references.append(f"{path}:{line}")
        else:
            references.append(path)

    # Deduplicate while preserving order
    seen = set()
    unique_refs: List[str] = []
    for ref in references:
        if ref not in seen:
            seen.add(ref)
            unique_refs.append(ref)
    return unique_refs


def _split_sections_by_severity(report_text: str) -> Dict[str, str]:
    """Split report text into sections keyed by severity.

    This looks for headings containing severity names and captures the
    text until the next severity heading or end of document.
    """

    # Build a pattern like (CRITICAL|HIGH|MEDIUM|LOW)
    sev_group = "|".join(SEVERITY_LEVELS)
    # Match markdown headings that contain a severity token
    heading_pattern = re.compile(rf"^(#+)\s+.*\b({sev_group})\b.*$", re.MULTILINE)

    sections: Dict[str, str] = {lvl: "" for lvl in SEVERITY_LEVELS}
    matches = list(heading_pattern.finditer(report_text))
    if not matches:
        return sections

    for idx, match in enumerate(matches):
        severity = match.group(2).upper()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(report_text)
        body = report_text[start:end].strip()
        if body:
            existing = sections.get(severity, "")
            sections[severity] = (existing + "\n\n" + body).strip() if existing else body

    return sections


def extract_issues_by_severity(report_text: str) -> Dict[str, List[Dict[str, Any]]]:
    """Extract issues grouped by severity level.

    For now we treat each bullet or numbered item in a severity section
    as a separate issue. Titles are taken from the first line; the rest
    is treated as description.
    """

    sections = _split_sections_by_severity(report_text)
    issues: Dict[str, List[Dict[str, Any]]] = {lvl: [] for lvl in SEVERITY_LEVELS}

    bullet_pattern = re.compile(r"^\s*[-*+]\s+(.+)$", re.MULTILINE)

    for severity, body in sections.items():
        if not body:
            continue

        # Split by bullets; if no bullets, treat entire section as one issue
        bullets = [m.group(1).strip() for m in bullet_pattern.finditer(body)]
        if not bullets:
            if body.strip():
                issues[severity].append(
                    {
                        "severity": severity,
                        "title": body.splitlines()[0].strip(),
                        "description": body.strip(),
                        "file_references": extract_file_references(body),
                    }
                )
            continue

        for bullet in bullets:
            title = bullet.split("-", 1)[0].strip() if "-" in bullet else bullet
            description = bullet
            issues[severity].append(
                {
                    "severity": severity,
                    "title": title,
                    "description": description,
                    "file_references": extract_file_references(bullet),
                }
            )

    return issues


def _extract_executive_summary(report_text: str) -> Optional[str]:
    """Extract the Executive Summary section if present."""

    pattern = re.compile(r"^##\s+Executive Summary\s*$", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(report_text)
    if not match:
        return None

    start = match.end()
    next_heading = re.search(r"^##\s+", report_text[start:], re.MULTILINE)
    end = start + next_heading.start() if next_heading else len(report_text)
    summary = report_text[start:end].strip()
    return summary or None


def _extract_recommendations(report_text: str) -> List[str]:
    """Extract recommendations from common sections.

    We look for headings like "Recommendations" or "Prioritized
    Recommendations" and treat list items under them as recommendations.
    """

    pattern = re.compile(
        r"^##\s+(Prioritized\s+Recommendations|Recommendations)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(report_text)
    if not match:
        return []

    start = match.end()
    next_heading = re.search(r"^##\s+", report_text[start:], re.MULTILINE)
    end = start + next_heading.start() if next_heading else len(report_text)
    body = report_text[start:end]

    bullet_pattern = re.compile(r"^\s*[-*+]\s+(.+)$", re.MULTILINE)
    return [m.group(1).strip() for m in bullet_pattern.finditer(body)]


def parse_report_findings(report_text: str) -> Dict[str, Any]:
    """Parse markdown report and extract structured findings.

    Returns a dictionary suitable for storing in SwarmAnalysisState
    under ``previous_report_findings``.
    """

    if not report_text:
        return {
            "report_date": None,
            "issues": {lvl: [] for lvl in SEVERITY_LEVELS},
            "file_references": [],
            "recommendations": [],
            "executive_summary": None,
            "metrics": {
                "total_issues": 0,
                "by_severity": {lvl: 0 for lvl in SEVERITY_LEVELS},
            },
        }

    report_date = _extract_report_date(report_text)
    issues = extract_issues_by_severity(report_text)
    file_refs = extract_file_references(report_text)
    exec_summary = _extract_executive_summary(report_text)
    recommendations = _extract_recommendations(report_text)

    by_severity_counts = {lvl: len(issues.get(lvl, [])) for lvl in SEVERITY_LEVELS}
    total_issues = sum(by_severity_counts.values())

    return {
        "report_date": report_date,
        "issues": issues,
        "file_references": file_refs,
        "recommendations": recommendations,
        "executive_summary": exec_summary,
        "metrics": {
            "total_issues": total_issues,
            "by_severity": by_severity_counts,
        },
    }
