"""
Architecture model data structures for representing application architecture.

These models capture the high-level structure, components, relationships,
and patterns of software applications for use in prompt generation.

Uses Pydantic v2 for validation and serialization.
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

logger = logging.getLogger(__name__)


class Module(BaseModel):
    """Represents a module or component in the architecture."""
    name: str
    purpose: str
    dependencies: List[str] = Field(default_factory=list)
    files: List[str] = Field(default_factory=list)
    complexity: str = Field(default="medium", description="simple, medium, complex, very_complex")
    description: Optional[str] = None
    entry_points: List[str] = Field(default_factory=list)
    exposed_apis: List[str] = Field(default_factory=list)
    
    model_config = ConfigDict(extra="forbid")
    
    @field_validator('complexity')
    @classmethod
    def validate_complexity(cls, v: str) -> str:
        """Validate complexity value."""
        valid_values = ["simple", "medium", "complex", "very_complex"]
        if v not in valid_values:
            logger.warning(f"Invalid complexity value: {v}, defaulting to 'medium'")
            return "medium"
        return v


class DataFlow(BaseModel):
    """Represents data flow between components."""
    source: str
    target: str
    data_type: str
    direction: str = Field(default="unidirectional", description="unidirectional or bidirectional")
    description: Optional[str] = None
    protocol: Optional[str] = Field(default=None, description="HTTP, gRPC, message_queue, etc.")
    
    model_config = ConfigDict(extra="forbid")
    
    @field_validator('direction')
    @classmethod
    def validate_direction(cls, v: str) -> str:
        """Validate direction value."""
        valid_values = ["unidirectional", "bidirectional"]
        if v not in valid_values:
            logger.warning(f"Invalid direction value: {v}, defaulting to 'unidirectional'")
            return "unidirectional"
        return v


class Endpoint(BaseModel):
    """Represents an API endpoint."""
    path: str
    method: str = Field(default="GET", description="GET, POST, PUT, DELETE, etc.")
    description: Optional[str] = None
    parameters: List[Dict[str, Any]] = Field(default_factory=list)
    response_type: Optional[str] = None
    authentication_required: bool = Field(default=False)
    rate_limited: bool = Field(default=False)
    
    model_config = ConfigDict(extra="forbid")
    
    @field_validator('method')
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate HTTP method."""
        valid_methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
        if v.upper() not in valid_methods:
            logger.warning(f"Invalid HTTP method: {v}, defaulting to 'GET'")
            return "GET"
        return v.upper()


class Dependency(BaseModel):
    """Represents a dependency between modules."""
    from_module: str
    to_module: str
    dependency_type: str = Field(default="import", description="import, call, data, event")
    description: Optional[str] = None
    
    model_config = ConfigDict(extra="forbid")
    
    @field_validator('dependency_type')
    @classmethod
    def validate_dependency_type(cls, v: str) -> str:
        """Validate dependency type."""
        valid_types = ["import", "call", "data", "event"]
        if v not in valid_types:
            logger.warning(f"Invalid dependency type: {v}, defaulting to 'import'")
            return "import"
        return v


class DesignPattern(BaseModel):
    """Represents a detected design pattern."""
    pattern_name: str
    pattern_type: str = Field(description="creational, structural, behavioral, architectural")
    location: str = Field(description="module or file where pattern is used")
    description: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="0.0 to 1.0")
    
    model_config = ConfigDict(extra="forbid")
    
    @field_validator('pattern_type')
    @classmethod
    def validate_pattern_type(cls, v: str) -> str:
        """Validate pattern type."""
        valid_types = ["creational", "structural", "behavioral", "architectural"]
        if v not in valid_types:
            logger.warning(f"Invalid pattern type: {v}, defaulting to 'architectural'")
            return "architectural"
        return v


class ArchitectureModel(BaseModel):
    """Main architecture model container."""
    system_name: str
    system_type: str = Field(description="web_app, library, api_service, cli_tool, data_science, unknown")
    architecture_pattern: str = Field(description="MVC, microservices, monolith, layered, etc.")
    modules: List[Module] = Field(default_factory=list)
    dependencies: Dict[str, List[str]] = Field(default_factory=dict, description="module -> [dependencies]")
    data_flow: List[DataFlow] = Field(default_factory=list)
    api_endpoints: List[Endpoint] = Field(default_factory=list)
    database_schema: Optional[Dict[str, Any]] = None
    design_patterns: List[str] = Field(default_factory=list)
    anti_patterns: List[str] = Field(default_factory=list)
    architectural_smells: List[str] = Field(default_factory=list)
    tech_stack: Dict[str, List[str]] = Field(default_factory=dict, description="category -> [technologies]")
    frameworks: List[str] = Field(default_factory=list)
    libraries: List[str] = Field(default_factory=list)
    security_architecture: Dict[str, Any] = Field(default_factory=dict)
    performance_characteristics: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    version: int = Field(default=1)
    
    model_config = ConfigDict(extra="forbid")
    
    @property
    def system_technologies(self) -> List[str]:
        """
        Get all technologies from the architecture model as a single deduplicated list.
        
        Consolidates frameworks, libraries, and tech_stack into a single source of truth.
        
        Returns:
            Sorted list of unique technology names
        """
        technologies = set()
        
        # Add frameworks
        technologies.update(self.frameworks)
        
        # Add libraries
        technologies.update(self.libraries)
        
        # Add technologies from tech_stack (which is Dict[str, List[str]])
        for category, techs in self.tech_stack.items():
            technologies.update(techs)
        
        # Return sorted, deduplicated list
        return sorted(list(technologies))
    
    def to_natural_language(self) -> str:
        """Convert architecture model to human-readable description for prompts."""
        lines = []
        
        # System overview
        lines.append(f"# System Architecture: {self.system_name}")
        lines.append("")
        lines.append(f"**System Type**: {self.system_type}")
        lines.append(f"**Architecture Pattern**: {self.architecture_pattern}")
        lines.append("")
        
        # Technology stack
        if self.frameworks or self.libraries or self.tech_stack:
            lines.append("## Technology Stack")
            if self.frameworks:
                lines.append(f"- **Frameworks**: {', '.join(self.frameworks)}")
            if self.libraries:
                lines.append(f"- **Libraries**: {', '.join(self.libraries)}")
            if self.tech_stack:
                for category, techs in self.tech_stack.items():
                    lines.append(f"- **{category.title()}**: {', '.join(techs)}")
            lines.append("")
        
        # Modules
        if self.modules:
            lines.append("## Modules/Components")
            for module in self.modules:
                lines.append(f"### {module.name}")
                lines.append(f"- **Purpose**: {module.purpose}")
                if module.description:
                    lines.append(f"- **Description**: {module.description}")
                if module.files:
                    lines.append(f"- **Files**: {', '.join(module.files[:5])}")  # Limit to first 5
                    if len(module.files) > 5:
                        lines.append(f"  (and {len(module.files) - 5} more files)")
                if module.dependencies:
                    lines.append(f"- **Dependencies**: {', '.join(module.dependencies)}")
                lines.append(f"- **Complexity**: {module.complexity}")
                lines.append("")
        
        # Dependencies
        if self.dependencies:
            lines.append("## Module Dependencies")
            for module, deps in self.dependencies.items():
                if deps:
                    lines.append(f"- **{module}** depends on: {', '.join(deps)}")
            lines.append("")
        
        # Data flow
        if self.data_flow:
            lines.append("## Data Flow")
            for flow in self.data_flow:
                direction_symbol = "→" if flow.direction == "unidirectional" else "↔"
                lines.append(f"- {flow.source} {direction_symbol} {flow.target} ({flow.data_type})")
                if flow.description:
                    lines.append(f"  - {flow.description}")
            lines.append("")
        
        # API endpoints
        if self.api_endpoints:
            lines.append("## API Endpoints")
            for endpoint in self.api_endpoints[:10]:  # Limit to first 10
                auth_marker = "🔒" if endpoint.authentication_required else ""
                lines.append(f"- {endpoint.method} {endpoint.path} {auth_marker}")
                if endpoint.description:
                    lines.append(f"  - {endpoint.description}")
            if len(self.api_endpoints) > 10:
                lines.append(f"  (and {len(self.api_endpoints) - 10} more endpoints)")
            lines.append("")
        
        # Design patterns
        if self.design_patterns:
            lines.append("## Design Patterns")
            for pattern in self.design_patterns:
                lines.append(f"- {pattern}")
            lines.append("")
        
        # Anti-patterns and smells
        if self.anti_patterns:
            lines.append("## Anti-Patterns Detected")
            for anti_pattern in self.anti_patterns:
                lines.append(f"- ⚠️ {anti_pattern}")
            lines.append("")
        
        if self.architectural_smells:
            lines.append("## Architectural Smells")
            for smell in self.architectural_smells:
                lines.append(f"- ⚠️ {smell}")
            lines.append("")
        
        # Security architecture
        if self.security_architecture:
            lines.append("## Security Architecture")
            for key, value in self.security_architecture.items():
                if isinstance(value, list):
                    lines.append(f"- **{key}**: {', '.join(str(v) for v in value)}")
                elif isinstance(value, dict):
                    lines.append(f"- **{key}**:")
                    for sub_key, sub_value in value.items():
                        lines.append(f"  - {sub_key}: {sub_value}")
                else:
                    lines.append(f"- **{key}**: {value}")
            lines.append("")
        
        # Performance characteristics
        if self.performance_characteristics:
            lines.append("## Performance Characteristics")
            for key, value in self.performance_characteristics.items():
                if isinstance(value, list):
                    lines.append(f"- **{key}**: {', '.join(str(v) for v in value)}")
                elif isinstance(value, dict):
                    lines.append(f"- **{key}**:")
                    for sub_key, sub_value in value.items():
                        lines.append(f"  - {sub_key}: {sub_value}")
                else:
                    lines.append(f"- **{key}**: {value}")
            lines.append("")
        
        return "\n".join(lines)

