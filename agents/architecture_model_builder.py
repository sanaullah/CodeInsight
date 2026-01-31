"""
Architecture Model Builder for analyzing codebases and building architecture models.

Uses LLM analysis to extract architecture information from codebases and construct
structured architecture models for use in prompt generation.

Adapted for CodeInsight with Pydantic models and LangGraph integration.
"""

import json
import logging
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

from scanners.project_scanner import FileInfo
from scanners import create_project_scanner
from agents.architecture_models import (
    ArchitectureModel, Module, DataFlow, Endpoint, Dependency, DesignPattern
)
# Import llm_node - handle import conflicts with tests/workflows
try:
    # Try direct import first
    from workflows.nodes import llm_node
except (ImportError, ModuleNotFoundError):
    # If that fails (e.g., in test environment), try importing from parent
    import sys
    import importlib.util
    from pathlib import Path
    
    # Get the actual workflows module path
    project_root = Path(__file__).parent.parent
    workflows_path = project_root / "workflows" / "nodes.py"
    
    if workflows_path.exists():
        spec = importlib.util.spec_from_file_location("workflows.nodes", workflows_path)
        workflows_nodes = importlib.util.module_from_spec(spec)
        sys.modules["workflows.nodes"] = workflows_nodes
        spec.loader.exec_module(workflows_nodes)
        llm_node = workflows_nodes.llm_node
    else:
        # Fallback: create a mock for test environments
        def llm_node(state):
            """Mock llm_node for test environments."""
            return {"last_response": "", "messages": []}
from utils.knowledge_base import get_architecture_model, store_architecture_model
from utils.io.file_hash import compute_file_hash

logger = logging.getLogger(__name__)


class ArchitectureModelBuilder:
    """
    Builds architecture models from codebases using LLM analysis.
    
    Analyzes project files to extract:
    - System structure and organization
    - Modules/components and their purposes
    - Dependencies and relationships
    - Design patterns and anti-patterns
    - Technology stack
    - Security and performance characteristics
    """
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        auto_detect_languages: Optional[bool] = None,
        file_extensions: Optional[List[str]] = None
    ):
        """
        Initialize architecture model builder.
        
        Args:
            model_name: Optional model name for LLM calls
            auto_detect_languages: Optional override for auto-detect setting (None = use config default)
            file_extensions: Optional override for file extensions (None = use config default)
        """
        self.model_name = model_name
        self.scanner = create_project_scanner(
            auto_detect=auto_detect_languages,
            file_extensions=file_extensions
        )
        
        # Load architecture discovery prompt
        self.discovery_prompt = self._load_discovery_prompt()
        
        logger.info("ArchitectureModelBuilder initialized")
    
    def _load_discovery_prompt(self) -> str:
        """Load architecture discovery prompt from file."""
        try:
            prompt_path = Path(__file__).parent.parent / "prompts" / "core" / "phases" / "planning" / "architecture-discovery.md"
            if prompt_path.exists():
                return prompt_path.read_text(encoding="utf-8")
            else:
                logger.warning(f"Architecture discovery prompt not found at {prompt_path}")
                return "Analyze the codebase and extract architecture information."
        except Exception as e:
            logger.warning(f"Error loading architecture discovery prompt: {e}")
            return "Analyze the codebase and extract architecture information."
    
    async def build_architecture_model(
        self,
        project_path: str,
        files: Optional[List[FileInfo]] = None,
        model_name: Optional[str] = None,
        use_cache: bool = True
    ) -> Optional[ArchitectureModel]:
        """
        Build architecture model from project.
        
        Args:
            project_path: Path to project directory
            files: Optional list of FileInfo objects (will scan if not provided)
            model_name: Optional model name override
            use_cache: Whether to check for cached models
        
        Returns:
            ArchitectureModel (Pydantic) or None if building fails
        """
        try:
            # Scan files if not provided
            if files is None:
                logger.info(f"Scanning project: {project_path}")
                files = self.scanner.scan_directory(project_path)
                if not files:
                    logger.warning(f"No files found in project: {project_path}")
                    return None
            
            # Compute file hash for cache lookup
            file_hash = compute_file_hash(files)
            
            # Check cache first
            if use_cache:
                try:
                    knowledge_obj = get_architecture_model(project_path, file_hash)
                    if knowledge_obj:
                        # Extract content from Knowledge object
                        cached_model_dict = knowledge_obj.content if hasattr(knowledge_obj, 'content') else knowledge_obj
                        if cached_model_dict:
                            logger.info(f"Using cached architecture model for {project_path}")
                            # Reconstruct Pydantic model from dict
                            try:
                                return ArchitectureModel.model_validate(cached_model_dict)
                            except Exception as e:
                                logger.warning(f"Error reconstructing cached model: {e}, rebuilding...")
                except Exception as e:
                    logger.warning(f"Error checking cache for architecture model: {e}")
                    # Continue to build new model
            
            # Step 1: Initial scan (file structure analysis)
            logger.info("Performing initial scan...")
            initial_hypothesis = self._initial_scan(files)
            
            # Step 2: Deep analysis (LLM-based)
            logger.info("Performing deep analysis with LLM...")
            analysis_results = await self._deep_analysis(
                initial_hypothesis,
                files,
                model_name or self.model_name
            )
            
            if not analysis_results:
                logger.error("Deep analysis failed")
                return None
            
            # Step 3: Build architecture model from results
            logger.info("Building architecture model...")
            model = self._build_model_from_analysis(
                analysis_results,
                project_path
            )
            
            # Step 4: Refine model
            logger.info("Refining architecture model...")
            refined_model = self._refine_model(model, analysis_results, files)
            
            # Store in knowledge base for caching
            if use_cache:
                try:
                    # Convert Pydantic model to dict for storage
                    model_dict = refined_model.model_dump()
                    store_architecture_model(project_path, model_dict, file_hash)
                    logger.debug(f"Stored architecture model in cache for {project_path}")
                except Exception as e:
                    logger.warning(f"Error storing architecture model in cache: {e}")
            
            logger.info(f"Architecture model built: {refined_model.system_name} ({refined_model.system_type})")
            return refined_model
            
        except Exception as e:
            logger.error(f"Error building architecture model: {e}", exc_info=True)
            return None
    
    def _initial_scan(
        self,
        files: List[FileInfo]
    ) -> Dict[str, Any]:
        """
        Perform initial scan of file structure.
        
        Args:
            files: List of FileInfo objects
        
        Returns:
            Initial hypothesis dictionary
        """
        # Analyze file structure
        file_paths = [f.relative_path for f in files]
        file_names = {Path(f.relative_path).name for f in files}
        
        # Detect entry points
        entry_points = []
        for name in ["app.py", "main.py", "run.py", "application.py", "__main__.py"]:
            if name in file_names:
                entry_points.append(name)
        
        # Group files by directory structure
        modules_by_dir = {}
        for file_info in files:
            path_parts = Path(file_info.relative_path).parts
            if len(path_parts) > 1:
                module_dir = path_parts[0]
                if module_dir not in modules_by_dir:
                    modules_by_dir[module_dir] = []
                modules_by_dir[module_dir].append(file_info.relative_path)
        
        # Detect imports and dependencies
        all_imports = set()
        for file_info in files:
            content = file_info.content
            # Simple import detection (basic regex)
            imports = re.findall(r'^import\s+(\w+)', content, re.MULTILINE)
            imports.extend(re.findall(r'^from\s+(\w+)', content, re.MULTILINE))
            all_imports.update(imports)
        
        # Simple project type detection
        project_type = "unknown"
        if any("app.py" in f or "main.py" in f for f in file_paths):
            project_type = "web_app"
        elif any("setup.py" in f or "pyproject.toml" in f for f in file_paths):
            project_type = "library"
        
        # Simple complexity assessment
        complexity = "simple"
        if len(files) > 100:
            complexity = "very_complex"
        elif len(files) > 50:
            complexity = "complex"
        elif len(files) > 20:
            complexity = "medium"
        
        return {
            "file_count": len(files),
            "entry_points": entry_points,
            "modules_by_dir": modules_by_dir,
            "detected_imports": list(all_imports),
            "project_type": project_type,
            "complexity": complexity
        }
    
    async def _deep_analysis(
        self,
        initial_hypothesis: Dict[str, Any],
        files: List[FileInfo],
        model_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Perform deep analysis using LLM.
        
        Args:
            initial_hypothesis: Initial scan results
            files: List of FileInfo objects
            model_name: Model name for LLM call
        
        Returns:
            Analysis results dictionary or None
        """
        if not model_name:
            from llm.config import ConfigManager
            config_manager = ConfigManager()
            config = config_manager.load_config()
            model_name = config.default_model
        
        # Prepare codebase content for analysis
        # Limit to key files to avoid token limits
        key_files = self._select_key_files(files, max_files=20)
        
        codebase_summary = self._prepare_codebase_summary(key_files, initial_hypothesis)
        
        # Create messages for LLM
        messages = [
            {"role": "system", "content": self.discovery_prompt},
            {"role": "user", "content": f"""
Analyze this codebase and extract architecture information.

## Initial Analysis
{json.dumps(initial_hypothesis, indent=2)}

## Codebase Summary
{codebase_summary}

Please provide a comprehensive architecture model in JSON format as specified in the prompt.
"""}
        ]
        
        # Use llm_node for LLM call (LangGraph integration)
        llm_state = {
            "messages": messages,
            "model": model_name,
            "temperature": 0.3
        }
        
        try:
            # Call llm_node (this will be tracked by Langfuse)
            llm_result = await llm_node(llm_state)
            
            # Extract response
            response_text = llm_result.get("last_response", "")
            if not response_text:
                # Try to extract from messages
                if llm_result.get("messages"):
                    last_message = llm_result["messages"][-1]
                    if hasattr(last_message, 'content'):
                        response_text = last_message.content
                    elif isinstance(last_message, dict):
                        response_text = last_message.get("content", "")
            
            if not response_text:
                logger.error("No response from LLM")
                return None
            
            # Parse JSON response
            json_text = self._extract_json_from_response(response_text)
            
            if json_text:
                try:
                    analysis_results = json.loads(json_text)
                    return analysis_results
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse extracted JSON: {e}")
                    logger.debug(f"Extracted JSON text (first 500 chars): {json_text[:500]}")
                    # Try to extract just the first valid JSON object if there's trailing text
                    last_brace = json_text.rfind('}')
                    if last_brace > 0:
                        try:
                            truncated_json = json_text[:last_brace + 1]
                            analysis_results = json.loads(truncated_json)
                            logger.warning("Successfully parsed truncated JSON (removed trailing text)")
                            return analysis_results
                        except json.JSONDecodeError:
                            pass
                    return None
            else:
                logger.error("Could not extract JSON from LLM response")
                logger.debug(f"Response text (first 500 chars): {response_text[:500]}")
                return None
                
        except Exception as e:
            logger.error(f"Error in deep analysis: {e}", exc_info=True)
            return None
    
    def _select_key_files(self, files: List[FileInfo], max_files: int = 20) -> List[FileInfo]:
        """Select key files for analysis (entry points, main modules, etc.)."""
        key_files = []
        
        # Priority order
        priority_names = ["app.py", "main.py", "run.py", "application.py", "__init__.py"]
        priority_dirs = ["api", "routes", "views", "controllers", "services", "models"]
        
        # First, add entry points
        for file_info in files:
            file_name = Path(file_info.relative_path).name
            if file_name in priority_names:
                key_files.append(file_info)
        
        # Then, add files from priority directories
        for file_info in files:
            if len(key_files) >= max_files:
                break
            path_parts = Path(file_info.relative_path).parts
            if any(priority_dir in path_parts for priority_dir in priority_dirs):
                if file_info not in key_files:
                    key_files.append(file_info)
        
        # Fill remaining slots with largest files (likely to contain important code)
        remaining = max_files - len(key_files)
        if remaining > 0:
            sorted_files = sorted(
                [f for f in files if f not in key_files],
                key=lambda f: f.line_count,
                reverse=True
            )
            key_files.extend(sorted_files[:remaining])
        
        return key_files[:max_files]
    
    def _prepare_codebase_summary(
        self,
        files: List[FileInfo],
        initial_hypothesis: Dict[str, Any]
    ) -> str:
        """Prepare codebase summary for LLM analysis."""
        lines = []
        
        lines.append(f"## Project Overview")
        lines.append(f"- Total files: {initial_hypothesis['file_count']}")
        lines.append(f"- Project type: {initial_hypothesis['project_type']}")
        lines.append(f"- Complexity: {initial_hypothesis['complexity']}")
        lines.append("")
        
        lines.append("## Key Files")
        for file_info in files[:10]:  # Limit to first 10
            lines.append(f"### {file_info.relative_path}")
            lines.append(f"Lines: {file_info.line_count}")
            # Include first 50 lines of code
            code_lines = file_info.content.split('\n')[:50]
            lines.append("```python")
            lines.append('\n'.join(code_lines))
            lines.append("```")
            lines.append("")
        
        return '\n'.join(lines)
    
    def _extract_json_from_response(self, response_text: str) -> Optional[str]:
        """Extract JSON from LLM response (may be in markdown code blocks)."""
        # Try to find JSON in code blocks first
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
            # Validate it's valid JSON
            try:
                json.loads(json_text)
                return json_text
            except json.JSONDecodeError:
                # If code block extraction failed, fall through to direct extraction
                pass
        
        # Try to find JSON object directly using brace counting
        start_idx = response_text.find('{')
        if start_idx == -1:
            return None
        
        # Count braces to find the matching closing brace
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i in range(start_idx, len(response_text)):
            char = response_text[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found matching closing brace
                        json_text = response_text[start_idx:i+1]
                        # Validate it's valid JSON
                        try:
                            json.loads(json_text)
                            return json_text
                        except json.JSONDecodeError:
                            # Invalid JSON, continue searching
                            start_idx = response_text.find('{', i + 1)
                            if start_idx == -1:
                                return None
                            brace_count = 0
                            continue
        
        return None
    
    def _build_model_from_analysis(
        self,
        analysis_results: Dict[str, Any],
        project_path: str
    ) -> ArchitectureModel:
        """Build ArchitectureModel from analysis results."""
        # Extract system name
        system_name = analysis_results.get("system_name", Path(project_path).name)
        
        # Extract modules
        modules = []
        for module_data in analysis_results.get("modules", []):
            try:
                module = Module(
                    name=module_data.get("name", "Unknown"),
                    purpose=module_data.get("purpose", ""),
                    dependencies=module_data.get("dependencies", []),
                    files=module_data.get("files", []),
                    complexity=module_data.get("complexity", "medium"),
                    description=module_data.get("description"),
                    entry_points=module_data.get("entry_points", []),
                    exposed_apis=module_data.get("exposed_apis", [])
                )
                modules.append(module)
            except Exception as e:
                logger.warning(f"Error creating module: {e}, skipping")
                continue
        
        # Extract data flow
        data_flows = []
        for flow_data in analysis_results.get("data_flow", []):
            try:
                flow = DataFlow(
                    source=flow_data.get("source", ""),
                    target=flow_data.get("target", ""),
                    data_type=flow_data.get("data_type", ""),
                    direction=flow_data.get("direction", "unidirectional"),
                    description=flow_data.get("description"),
                    protocol=flow_data.get("protocol")
                )
                data_flows.append(flow)
            except Exception as e:
                logger.warning(f"Error creating data flow: {e}, skipping")
                continue
        
        # Extract API endpoints
        endpoints = []
        for endpoint_data in analysis_results.get("api_endpoints", []):
            try:
                endpoint = Endpoint(
                    path=endpoint_data.get("path", ""),
                    method=endpoint_data.get("method", "GET"),
                    description=endpoint_data.get("description"),
                    parameters=endpoint_data.get("parameters", []),
                    response_type=endpoint_data.get("response_type"),
                    authentication_required=endpoint_data.get("authentication_required", False),
                    rate_limited=endpoint_data.get("rate_limited", False)
                )
                endpoints.append(endpoint)
            except Exception as e:
                logger.warning(f"Error creating endpoint: {e}, skipping")
                continue
        
        # Extract technologies
        frameworks = analysis_results.get("frameworks", [])
        libraries = analysis_results.get("libraries", [])
        tech_stack = analysis_results.get("tech_stack", {})
        
        # Normalize and deduplicate technologies
        frameworks_dict = {}
        for fw in frameworks:
            frameworks_dict[fw.lower()] = fw
        frameworks = list(frameworks_dict.values())
        
        libraries_dict = {}
        for lib in libraries:
            libraries_dict[lib.lower()] = lib
        libraries = list(libraries_dict.values())
        
        # Also normalize tech_stack
        normalized_tech_stack = {}
        for category, techs in tech_stack.items():
            techs_dict = {}
            for tech in techs:
                techs_dict[tech.lower()] = tech
            normalized_tech_stack[category] = list(techs_dict.values())
        tech_stack = normalized_tech_stack
        
        # Build model
        model = ArchitectureModel(
            system_name=system_name,
            system_type=analysis_results.get("system_type", "unknown"),
            architecture_pattern=analysis_results.get("architecture_pattern", "unknown"),
            modules=modules,
            dependencies=analysis_results.get("dependencies", {}),
            data_flow=data_flows,
            api_endpoints=endpoints,
            database_schema=analysis_results.get("database_schema"),
            design_patterns=analysis_results.get("design_patterns", []),
            anti_patterns=analysis_results.get("anti_patterns", []),
            architectural_smells=analysis_results.get("architectural_smells", []),
            tech_stack=tech_stack,
            frameworks=frameworks,
            libraries=libraries,
            security_architecture=analysis_results.get("security_architecture", {}),
            performance_characteristics=analysis_results.get("performance_characteristics", {}),
            version=1
        )
        
        return model
    
    def _refine_model(
        self,
        model: ArchitectureModel,
        analysis_results: Dict[str, Any],
        files: List[FileInfo]
    ) -> ArchitectureModel:
        """Refine architecture model based on additional analysis."""
        # Validate and enhance modules
        for module in model.modules:
            # Ensure files exist
            valid_files = [f for f in module.files if any(fi.relative_path == f for fi in files)]
            if valid_files != module.files:
                logger.debug(f"Updated files for module {module.name}: {len(valid_files)} valid files")
                module.files = valid_files
        
        # Enhance tech stack if empty
        if not model.frameworks and not model.libraries:
            all_techs = set()
            for category, techs in model.tech_stack.items():
                all_techs.update(techs)
            
            # Common frameworks
            frameworks_list = ["django", "flask", "fastapi", "streamlit", "gradio", "tornado", "bottle"]
            model.frameworks = [t for t in all_techs if t.lower() in frameworks_list]
            
            # Everything else is a library
            model.libraries = [t for t in all_techs if t.lower() not in frameworks_list]
        
        return model

