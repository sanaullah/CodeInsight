"""Workflow node for loading and parsing previous Swarm Analysis reports.

This node:
- Reads previous_report_path / previous_report_id from state
- Loads the corresponding report content
- Parses structured findings for comparison
- Stores both raw text and structured findings back into state
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from workflows.swarm_analysis_state import SwarmAnalysisState
from workflows.swarm_analysis_state_decorators import validate_state_fields
from workflows.state_keys import StateKeys
from utils.report_loader import load_report_from_file, load_report_from_scan_history
from utils.report_findings_parser import parse_report_findings

logger = logging.getLogger(__name__)


@validate_state_fields([], "load_previous_reports")
def load_previous_reports_node(
    state: SwarmAnalysisState,
    config: Optional[Dict[str, Any]] = None,
) -> SwarmAnalysisState:
    """Load previous report and extract structured findings.

    This node is safe to run even when no previous report is configured.
    In that case it simply returns the state unchanged.
    """

    previous_report_path = state.get(StateKeys.PREVIOUS_REPORT_PATH)
    previous_report_id = state.get(StateKeys.PREVIOUS_REPORT_ID)

    updated_state = state.copy()

    # Pre-fetch all relevant skills for use in subsequent nodes (ACE Phase 2)
    # This happens regardless of whether a previous report exists
    try:
        from utils.swarm_skillbook import get_relevant_skills
        
        architecture_model = state.get(StateKeys.ARCHITECTURE_MODEL, {})
        architecture_type = architecture_model.get("system_type", "unknown")
        goal = state.get(StateKeys.GOAL)
        
        # Retrieve all relevant skills
        relevant_skills = get_relevant_skills(
            architecture_type=architecture_type,
            goal=goal
        )
        
        # Convert SwarmSkill objects to dictionaries for state storage
        learned_skills_dict = {}
        for skill_type, skills_list in relevant_skills.items():
            learned_skills_dict[skill_type] = [
                {
                    "skill_id": skill.skill_id,
                    "skill_type": skill.skill_type,
                    "skill_category": skill.skill_category,
                    "content": skill.content,
                    "context": skill.context,
                    "confidence": skill.confidence,
                    "success_rate": skill.success_rate,
                    "usage_count": skill.usage_count
                }
                for skill in skills_list
            ]
        
        updated_state[StateKeys.LEARNED_SKILLS] = learned_skills_dict
        logger.info(f"Pre-fetched skills for {len(learned_skills_dict)} skill types")
        
    except Exception as skill_error:
        # Non-blocking: log but continue without skills
        logger.warning(f"Failed to pre-fetch skills: {skill_error}")
        updated_state["learned_skills"] = {}

    # Now handle previous report loading (if configured)
    if not previous_report_path and previous_report_id is None:
        # No previous report to load, but skills were already pre-fetched above
        logger.debug("No previous report configured; skipping report loading")
        return updated_state

    raw_report: Optional[str] = None

    try:
        if previous_report_path:
            logger.info("Loading previous report from file: %s", previous_report_path)
            raw_report = load_report_from_file(previous_report_path)
        elif previous_report_id is not None:
            logger.info("Loading previous report from scan history id=%s", previous_report_id)
            raw_report = load_report_from_scan_history(previous_report_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Unexpected error while loading previous report: %s", exc, exc_info=True)
        raw_report = None

    updated_state[StateKeys.PREVIOUS_REPORT_TEXT] = raw_report

    if raw_report:
        try:
            findings = parse_report_findings(raw_report)
            updated_state[StateKeys.PREVIOUS_REPORT_FINDINGS] = findings
            logger.info(
                "Extracted %d total issues from previous report",
                findings.get("metrics", {}).get("total_issues", 0),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to parse previous report findings: %s", exc, exc_info=True)
            updated_state["previous_report_findings"] = None
    else:
        updated_state["previous_report_findings"] = None

    return updated_state
