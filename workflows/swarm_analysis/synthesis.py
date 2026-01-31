"""
Result synthesis node for Swarm Analysis workflow.

Synthesizes results from all agents into a unified report.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from workflows.swarm_analysis_state import SwarmAnalysisState
from workflows.swarm_analysis_state_decorators import validate_state_fields
from workflows.state_keys import StateKeys
from workflows.nodes import llm_node
from utils.error_context import create_error_context
from services.experience_models import Experience

logger = logging.getLogger(__name__)


def _get_severity_count(by_severity: Optional[Dict[str, Any]], severity: str) -> str:
    """Get severity count, handling both int and list formats."""
    if not isinstance(by_severity, dict):
        return 'n/a'
    value = by_severity.get(severity, 0)
    if isinstance(value, int):
        return str(value)
    elif isinstance(value, list):
        return str(len(value))
    else:
        return str(value) if value else '0'


@validate_state_fields(["agent_results"], "synthesize_results")
async def synthesize_results_node(state: SwarmAnalysisState, config: Optional[Dict[str, Any]] = None) -> SwarmAnalysisState:
    """
    Synthesize results from all agents.
    
    Uses llm_node for synthesis.
    
    Args:
        state: Current swarm analysis state
        config: Optional LangGraph config
        
    Returns:
        Updated state with synthesized_report
    """
    agent_results: Dict[str, Dict[str, Any]] = state.get(StateKeys.AGENT_RESULTS, {})
    architecture_model: Dict[str, Any] = state.get(StateKeys.ARCHITECTURE_MODEL, {})
    role_names: List[str] = state.get(StateKeys.ROLE_NAMES, [])
    goal: Optional[str] = state.get(StateKeys.GOAL)
    model_name: Optional[str] = state.get(StateKeys.MODEL_NAME)
    previous_report_findings: Optional[Dict[str, Any]] = state.get(StateKeys.PREVIOUS_REPORT_FINDINGS)
    
    if not agent_results:
        error_msg = (
            "agent_results are required for result synthesis. "
            "Please ensure agents were executed successfully before synthesizing results."
        )
        state[StateKeys.ERROR] = error_msg
        state[StateKeys.ERROR_STAGE] = "synthesize_results"
        logger.error(error_msg)
        return state
    
    from workflows.swarm_analysis.utils import get_stream_callback
    stream_callback = get_stream_callback(state)
    if stream_callback:
        stream_callback("result_synthesis_start", {
            "message": "Synthesizing results..."
        })
    
    try:
        # Format agent results for synthesis
        formatted_results: List[str] = []
        for role_name in role_names:
            result = agent_results.get(role_name, {})
            if "error" not in result:
                report = result.get("synthesized_report", "")
                formatted_results.append(f"## {role_name} Analysis\n\n{report}")
            else:
                formatted_results.append(f"## {role_name} Analysis\n\nError: {result.get('error', 'Unknown error')}")
        
        formatted_text = "\n\n---\n\n".join(formatted_results)
        
        # Optional previous-report comparison section
        comparison_section = ""
        if previous_report_findings and isinstance(previous_report_findings, dict):
            # Safely extract metrics with defensive checks
            metrics = previous_report_findings.get("metrics")
            if not isinstance(metrics, dict):
                metrics = {}
            
            # Safely extract total_issues
            total_issues_val = metrics.get("total_issues", 0)
            if isinstance(total_issues_val, (int, float)):
                total_issues = int(total_issues_val)
            else:
                total_issues = 0
            
            # Safely extract by_severity
            by_severity = metrics.get("by_severity", {})
            if not isinstance(by_severity, dict):
                by_severity = {}
            
            report_date = previous_report_findings.get("report_date", "Unknown")
            if not isinstance(report_date, str):
                report_date = "Unknown"

            comparison_section = f"""

Previous Analysis Report (For Comparison)
----------------------------------------

- Report Date: {report_date}
- Total Issues: {total_issues}
- By Severity: {by_severity}

Key Findings from Previous Report (structured summary):
- CRITICAL: {_get_severity_count(by_severity, 'CRITICAL')}
- HIGH: {_get_severity_count(by_severity, 'HIGH')}
- MEDIUM: {_get_severity_count(by_severity, 'MEDIUM')}
- LOW: {_get_severity_count(by_severity, 'LOW')}

Use this previous report to:
1. Identify what was fixed: issues present before but no longer found now.
2. Identify new issues: issues in the current analysis not present previously.
3. Calibrate severity: reassess whether previously CRITICAL/HIGH issues are still at the same level.
4. Track progress: clearly call out improvements and remaining risk.
"""

        # Prepare date context for the report
        current_date = datetime.now()
        date_context = f"""
## Report Date Context

IMPORTANT: Use the following date and time when generating this report:
- Current Date: {current_date.strftime('%B %d, %Y')}
- Current Date (ISO): {current_date.strftime('%Y-%m-%d')}
- Current Time: {current_date.strftime('%H:%M:%S')}

When you write phrases like "Prepared by Senior Technical Mentor | [DATE]", use: {current_date.strftime('%B %Y')}
When you reference the report generation date, use: {current_date.strftime('%B %d, %Y')}
"""

        # Retrieve synthesis skills (ACE Phase 2)
        skills_section = ""
        harmful_section = ""
        try:
            from utils.skill_integration import format_skills_for_prompt, format_harmful_patterns
            from agents.swarm_skillbook_models import SwarmSkill
            
            # Try to get from pre-fetched skills first
            learned_skills = state.get("learned_skills", {})
            synthesis_skills_data = learned_skills.get("synthesis", [])
            
            if synthesis_skills_data:
                # Convert dicts back to SwarmSkill objects for formatting
                synthesis_skills = [
                    SwarmSkill(
                        skill_id=s["skill_id"],
                        skill_type=s["skill_type"],
                        skill_category=s["skill_category"],
                        content=s["content"],
                        context=s.get("context", {}),
                        confidence=s.get("confidence", 0.5),
                        success_rate=s.get("success_rate", 0.0),
                        usage_count=s.get("usage_count", 0)
                    )
                    for s in synthesis_skills_data
                ]
            else:
                # Fallback: retrieve directly if not pre-fetched
                from utils.swarm_skillbook import get_relevant_skills
                architecture_type = architecture_model.get("system_type", "unknown")
                relevant_skills = get_relevant_skills(
                    architecture_type=architecture_type,
                    goal=goal
                )
                synthesis_skills = relevant_skills.get("synthesis", [])
            
            # Separate helpful vs harmful
            helpful_skills = [s for s in synthesis_skills if s.skill_category == "helpful"]
            harmful_skills = [s for s in synthesis_skills if s.skill_category == "harmful"]
            
            # Format for prompt
            skills_section = format_skills_for_prompt(
                helpful_skills,
                max_skills=5,
                category_label="Synthesis Best Practices"
            )
            
            harmful_section = format_harmful_patterns(harmful_skills, max_skills=3)
            
            if skills_section or harmful_section:
                logger.info(f"Retrieved {len(helpful_skills)} helpful and {len(harmful_skills)} harmful synthesis skills")
                
        except Exception as skill_error:
            # Non-blocking: log but continue without skills
            logger.warning(f"Failed to retrieve synthesis skills: {skill_error}")
            skills_section = ""
            harmful_section = ""

        # Prepare synthesis prompt
        synthesis_prompt = f"""Synthesize the following analysis results from multiple specialized agents into a comprehensive, constructive report.

Architecture Context:
- System: {architecture_model.get('system_name', 'Unknown')}
- Type: {architecture_model.get('system_type', 'Unknown')}
- Pattern: {architecture_model.get('architecture_pattern', 'Unknown')}

Analysis Goal: {goal or 'General code analysis'}

{comparison_section}

{skills_section}

{harmful_section}

Individual Agent Results:
{formatted_text}

{date_context}

## Report Structure and Tone

Create a balanced, constructive report that helps the development team improve their codebase. Your role is to be a supportive technical mentor, not just a critic.

### Tone Guidelines:
- **Constructive and Supportive**: Frame findings as learning opportunities and improvement areas, not just problems
- **Honest but Encouraging**: Be truthful about issues while acknowledging what's working well
- **Educational**: Help developers understand why changes matter and how to implement them
- **Balanced**: Always include strengths alongside areas for improvement

### Language Patterns:
- Instead of "Critical vulnerability found" → "Security enhancement opportunity: Consider implementing..."
- Instead of "Poor performance" → "Performance optimization opportunity: This could be improved by..."
- Instead of "Bad architecture" → "Architectural enhancement: Consider refactoring to..."

Synthesize these results into a unified report with:

1. **Executive Summary**
   - Provide a balanced overview acknowledging both strengths and areas for improvement
   - Frame the overall codebase health constructively
   - Highlight key positive aspects alongside important improvement opportunities

2. **Strengths and Good Practices** (REQUIRED)
   - Acknowledge what's working well in the codebase
   - Highlight good design patterns, security practices, performance optimizations, or code quality
   - Recognize areas where the code demonstrates best practices
   - This section should be substantial - every codebase has strengths worth acknowledging

3. **Progress Since Last Analysis** (if previous report is provided):
   - Frame improvements positively: "Successfully resolved..." or "Good progress on..."
   - Acknowledge remaining areas that need attention (frame as ongoing improvement opportunities)
   - Note new areas for improvement (frame as evolving codebase needs)
   - Provide an encouraging overall progress assessment

4. **Areas for Improvement** (renamed from "Key Findings")
   - Prioritize by impact and severity, but frame as learning opportunities
   - Distinguish between:
     * **Confirmed Issues**: Problems with clear evidence (mark with [HIGH] or [MEDIUM] confidence)
     * **Potential Concerns**: Patterns that might be problematic (mark as [MEDIUM] or [LOW], frame as "worth reviewing")
     * **Enhancement Suggestions**: Improvements without evidence of problems (mark as "SUGGESTION" or "ENHANCEMENT")
   - For each item, provide context on why it matters and how to address it

5. **Cross-Cutting Concerns**
   - Frame as opportunities for systematic improvement
   - Show how addressing these can have broad positive impact

6. **Guidance and Recommendations**
   - Provide constructive, actionable steps
   - Prioritize recommendations by impact and effort
   - Include guidance on implementation approach
   - Frame as "Consider..." or "You might benefit from..." rather than "You must..."

7. **Overall Assessment**
   - Provide a balanced, honest but supportive conclusion
   - Acknowledge the codebase's current state while being encouraging about improvement potential
   - Frame next steps as a journey of continuous improvement

### Critical Requirements:
- **Always include a "Strengths and Good Practices" section** - this is mandatory, not optional
- **Distinguish between issues and suggestions**: Clearly separate confirmed problems from improvement opportunities
- **Use constructive language**: Frame everything as guidance and learning opportunities
- **Be honest but supportive**: Don't sugarcoat real issues, but present them in a way that helps rather than discourages
- **Acknowledge uncertainty**: When confidence is low, frame as "worth reviewing" rather than "critical issue"

Format as a well-structured Markdown document.
"""
        
        # Use llm_node for synthesis
        llm_state = {
            "messages": [
                {"role": "system", "content": "You are a Senior Technical Mentor and Advisor synthesizing analysis results from multiple specialized agents. Your role is to provide constructive, balanced, and supportive guidance that helps development teams improve their codebase. You acknowledge strengths, frame issues as learning opportunities, and offer actionable guidance rather than just criticism. Be honest but encouraging, educational but not condescending."},
                {"role": "user", "content": synthesis_prompt}
            ],
            "model": model_name,
            "temperature": 0.5
        }
        
        # Call llm_node
        llm_result = await llm_node(llm_state, config=config)
        
        # Extract synthesized report
        synthesized_report = llm_result.get("last_response", "")
        
        # Update state first
        updated_state = state.copy()
        updated_state[StateKeys.SYNTHESIZED_REPORT] = synthesized_report
        
        # Emit synthesis complete event
        if stream_callback:
            stream_callback("result_synthesis_complete", {
                "message": "Results synthesized successfully",
                "report_length": len(synthesized_report),
                "agents_count": len(role_names)
            })
        
        # Evaluate prompt effectiveness for each agent
        try:
            from utils.langfuse_integration import evaluate_prompt_effectiveness
            from utils import flush_langfuse
            
            validation_results = state.get(StateKeys.VALIDATION_RESULTS, {})
            langfuse_prompt_ids = state.get(StateKeys.LANGFUSE_PROMPT_IDS, {})
            trace_ids = state.get(StateKeys.TRACE_IDS, {})
            
            evaluation_results = {}
            
            for role_name in role_names:
                agent_result = agent_results.get(role_name, {})
                validation_result = validation_results.get(role_name)
                langfuse_prompt_id = langfuse_prompt_ids.get(role_name)
                trace_id = trace_ids.get(role_name)
                
                # Calculate evaluation scores (only if trace_id is available)
                if trace_id:
                    scores = evaluate_prompt_effectiveness(
                        trace_id=trace_id,
                        role_name=role_name,
                        validation_results=validation_result,
                        agent_result=agent_result,
                        prompt_id=langfuse_prompt_id
                    )
                    evaluation_results[role_name] = scores
                    logger.debug(f"Evaluated prompt effectiveness for {role_name} with trace_id: {trace_id}")
                else:
                    logger.debug(f"Trace ID not available for {role_name}, skipping score attachment")
                    # Still calculate scores for metadata, but don't attach to trace
                    evaluation_results[role_name] = {
                        "validation_score": 0.0,
                        "quality_score": 0.0,
                        "success_score": 0.0,
                        "overall_effectiveness": 0.0,
                        "note": "trace_id_not_available"
                    }
            
            # Store evaluation results in state metadata
            if "metadata" not in updated_state:
                updated_state[StateKeys.METADATA] = {}
            updated_state[StateKeys.METADATA]["evaluation_results"] = evaluation_results
            
            # Flush Langfuse to ensure scores are sent
            try:
                flush_langfuse()
            except Exception as flush_error:
                logger.debug(f"Could not flush Langfuse: {flush_error}")
                
        except Exception as eval_error:
            # Non-blocking: log but continue
            logger.warning(f"Failed to evaluate prompt effectiveness: {eval_error}")
        
        # Aggregate costs from analysis
        try:
            from utils.langfuse.cost_tracking import aggregate_costs_from_state, calculate_token_cost
            
            # Aggregate costs from state metadata
            cost_analysis = aggregate_costs_from_state(
                state=updated_state,
                model=model_name
            )
            
            # Store cost analysis in metadata
            if "metadata" not in updated_state:
                updated_state[StateKeys.METADATA] = {}
            updated_state[StateKeys.METADATA]["cost_analysis"] = cost_analysis
            
            logger.info(
                f"Cost analysis: {cost_analysis['total_tokens']} tokens, "
                f"${cost_analysis['total_cost_usd']:.4f} USD"
            )
            
        except Exception as cost_error:
            # Non-blocking: log but continue
            logger.warning(f"Failed to aggregate costs: {cost_error}")
        
        # Store experience for learning
        try:
            from agents.swarm_analysis_experience import store_swarm_experience, create_swarm_experience

            # Calculate actual execution time from workflow start
            workflow_start_time = state.get("workflow_start_time")
            execution_time = (
                datetime.now().timestamp() - workflow_start_time
            ) if workflow_start_time else 0.0
            
            # Aggregate token usage from Langfuse traces
            from utils.langfuse.cost_tracking import aggregate_costs_from_traces
            
            # Get main trace ID (the LangGraph trace that includes everything)
            main_trace_id = state.get("trace_id")
            trace_ids_list = []
            
            if main_trace_id:
                # Use the main LangGraph trace (includes all nested observations)
                trace_ids_list = [main_trace_id]
                logger.debug(f"Using main LangGraph trace ID for token aggregation: {main_trace_id}")
            else:
                # Fallback: collect individual trace IDs if main trace not available
                trace_ids_dict = state.get(StateKeys.TRACE_IDS, {})
                if trace_ids_dict:
                    trace_ids_list = list(trace_ids_dict.values())
                    logger.debug(f"Using individual trace IDs for token aggregation: {len(trace_ids_list)} traces")
                else:
                    logger.warning("No trace IDs found in state, cannot track token usage from Langfuse")
            
            # Aggregate token usage from trace(s)
            token_usage = 0
            if trace_ids_list:
                try:
                    trace_usage = aggregate_costs_from_traces(
                        trace_ids=trace_ids_list,
                        model=model_name
                    )
                    token_usage = trace_usage.get("total_tokens", 0)
                    logger.info(
                        f"Aggregated {token_usage} tokens "
                        f"({trace_usage.get('total_prompt_tokens', 0)} prompt + "
                        f"{trace_usage.get('total_completion_tokens', 0)} completion) "
                        f"from {len(trace_ids_list)} trace(s)"
                    )
                    
                    # Also log cost if available
                    if trace_usage.get("total_cost_usd", 0) > 0:
                        logger.info(f"Total cost: ${trace_usage['total_cost_usd']:.4f} USD")
                    
                    # Log any failed traces for debugging
                    failed_traces = trace_usage.get("failed_traces", [])
                    if failed_traces:
                        logger.warning(f"Failed to aggregate tokens from {len(failed_traces)} trace(s): {failed_traces}")
                        
                except Exception as token_error:
                    logger.warning(f"Failed to aggregate token usage from traces: {token_error}")
                    token_usage = 0
            else:
                logger.warning("No trace IDs found in state, cannot track token usage")
            
            # Create experience object first (needed for skill extraction)
            experience = create_swarm_experience(
                project_path=updated_state.get(StateKeys.PROJECT_PATH, ""),
                architecture_model=updated_state.get(StateKeys.ARCHITECTURE_MODEL, {}),
                roles_selected=updated_state.get(StateKeys.ROLE_NAMES, []),
                prompts_generated=updated_state.get(StateKeys.GENERATED_PROMPTS, {}),
                validation_results=updated_state.get(StateKeys.VALIDATION_RESULTS, {}),
                agent_results=updated_state.get(StateKeys.AGENT_RESULTS, {}),
                synthesized_report=updated_state.get(StateKeys.SYNTHESIZED_REPORT, ""),
                goal=updated_state.get(StateKeys.GOAL),
                files_scanned=len(updated_state.get(StateKeys.FILES, [])),
                chunks_analyzed=updated_state.get("chunks_analyzed", 0),
                execution_time=execution_time,
                token_usage=token_usage
            )
            
            # Store experience (non-blocking)
            import asyncio
            experience_id = await asyncio.to_thread(
                store_swarm_experience,
                state=updated_state,
                execution_time=execution_time,
                token_usage=token_usage
            )
            
            if experience_id:
                logger.info(f"Stored swarm analysis experience: {experience_id}")
                # Add experience_id to state metadata
                if "metadata" not in updated_state:
                    updated_state[StateKeys.METADATA] = {}
                updated_state[StateKeys.METADATA]["experience_id"] = experience_id
                
                # Extract skills from experience (ACE learning loop)
                try:
                    from agents.swarm_analysis_skill_extraction import experience_to_skills
                    
                    # Extract skills from experience (only if it's an Experience object, not a dict)
                    if isinstance(experience, Experience):
                        skills = await experience_to_skills(
                            experience=experience,
                            architecture_model=updated_state.get("architecture_model", {}),
                            model_name=model_name
                        )
                    else:
                        logger.debug("Experience is dict-based, skipping skill extraction")
                        skills = []
                    
                    if skills:
                        logger.info(f"Extracted {len(skills)} skills from experience {experience_id}")
                        if "metadata" not in updated_state:
                            updated_state[StateKeys.METADATA] = {}
                        updated_state[StateKeys.METADATA]["skills_extracted"] = len(skills)
                        updated_state[StateKeys.METADATA]["skill_ids"] = [skill.skill_id for skill in skills]
                    else:
                        logger.debug(f"No skills extracted from experience {experience_id}")
                        
                except Exception as skill_error:
                    # Non-blocking: log but continue
                    logger.warning(f"Failed to extract skills from experience: {skill_error}")
                    # Continue without skill extraction
                    
        except Exception as e:
            logger.warning(f"Failed to store experience: {e}")
            # Continue without experience storage
        
        # Track skill usage and update effectiveness (ACE Phase 3)
        try:
            from utils.swarm_skillbook import update_skill_usage
            from utils.skill_effectiveness import calculate_skill_effectiveness
            
            learned_skills = state.get("learned_skills", {})
            experience_id = updated_state.get("metadata", {}).get("experience_id")
            
            # Use experience_id as analysis_id, or generate one if not available
            analysis_id = experience_id or f"analysis_{datetime.now().isoformat()}"
            
            # Get evaluation results for effectiveness calculation
            evaluation_results = updated_state.get("metadata", {}).get("evaluation_results", {})
            validation_results = state.get("validation_results", {})
            agent_results = state.get("agent_results", {})
            synthesized_report = updated_state.get("synthesized_report", "")
            role_names = state.get("role_names", [])
            
            # Track usage for all skills that were retrieved and potentially used
            skills_tracked = 0
            skills_updated = 0
            
            for skill_type, skills_list in learned_skills.items():
                for skill_dict in skills_list:
                    skill_id = skill_dict.get("skill_id")
                    skill_category = skill_dict.get("skill_category", "helpful")
                    
                    if not skill_id:
                        logger.warning(f"Skipping skill without skill_id in {skill_type}")
                        continue
                    
                    skills_tracked += 1
                    
                    try:
                        # Calculate effectiveness for this skill
                        effectiveness = calculate_skill_effectiveness(
                            skill_type=skill_type,
                            skill_category=skill_category,
                            evaluation_results=evaluation_results,
                            validation_results=validation_results,
                            agent_results=agent_results,
                            synthesized_report=synthesized_report,
                            role_names=role_names
                        )
                        
                        # Update skill usage in database
                        update_skill_usage(
                            skill_id=skill_id,
                            effectiveness=effectiveness,
                            analysis_id=analysis_id
                        )
                        
                        skills_updated += 1
                        logger.debug(
                            f"Updated skill usage: {skill_id} ({skill_type}) "
                            f"with effectiveness {effectiveness:.2f}"
                        )
                        
                    except Exception as skill_update_error:
                        # Non-blocking: log but continue with other skills
                        logger.warning(
                            f"Failed to update skill usage for {skill_id}: {skill_update_error}"
                        )
            
            if skills_tracked > 0:
                logger.info(
                    f"Tracked skill usage: {skills_updated}/{skills_tracked} skills updated "
                    f"for analysis {analysis_id}"
                )
                
                # Store tracking metadata
                if "metadata" not in updated_state:
                    updated_state[StateKeys.METADATA] = {}
                updated_state[StateKeys.METADATA]["skills_tracked"] = skills_tracked
                updated_state[StateKeys.METADATA]["skills_updated"] = skills_updated
                updated_state[StateKeys.METADATA]["skill_tracking_analysis_id"] = analysis_id
            
        except Exception as tracking_error:
            # Non-blocking: log but continue
            logger.warning(f"Failed to track skill usage: {tracking_error}")
        
        if stream_callback:
            stream_callback("swarm_analysis_completed", {
                "project_path": state.get(StateKeys.PROJECT_PATH),
                "agents_executed": len(role_names),
                "timestamp": datetime.now().isoformat()
            })
        
        logger.info("Results synthesized successfully")
        return updated_state
        
    except Exception as e:
        error_context = create_error_context("synthesize_results", state, {
            "agent_count": len(role_names),
            "successful_agents": len([r for r in agent_results.values() if "error" not in r]),
            "model_name": model_name
        })
        error_msg = (
            f"Failed to synthesize results from {len(role_names)} agents: {str(e)}. "
            f"Successfully processed {len([r for r in agent_results.values() if 'error' not in r])} agents before error. "
            f"Check that agent results are valid and the LLM model is accessible."
        )
        logger.error(error_msg, extra=error_context, exc_info=True)
        updated_state = state.copy()
        updated_state[StateKeys.ERROR] = error_msg
        updated_state[StateKeys.ERROR_STAGE] = "synthesize_results"
        updated_state[StateKeys.ERROR_CONTEXT] = error_context
        if stream_callback:
            stream_callback("swarm_analysis_error", {"error": error_msg})
        return updated_state



