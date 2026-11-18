from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from enum import Enum

class TaskDomain(str, Enum):
    CIRCUIT_SCHEMATIC = "circuit_schematic"
    FLOOR_PLAN = "floor_plan"
    PIPING_INSTRUMENTATION = "piping_instrumentation"

class QuestionType(str, Enum):
    CONNECTIVITY_EXISTS = "connectivity_exists"      # Boolean (Yes/No)
    COMPONENT_COUNT = "component_count"              # Integer
    SHORTEST_PATH_LENGTH = "shortest_path_length"    # Integer (number of hops/components)
    NEIGHBOR_IDENTIFICATION = "neighbor_id"          # Exact component identifier
    PATH_SEQUENCE = "path_sequence"                  # Ordered list of nodes

class RenderStyle(str, Enum):
    BLUEPRINT = "blueprint"       # Classic blueprint cyan on navy
    CAD_WHITE = "cad_white"       # Clean CAD white background with black lines
    AGED_PARCHMENT = "parchment"  # Industrial engineering manual style

@dataclass
class VisualArtifact:
    """Metadata detailing exact bounding coordinates and ground-truth graphics."""
    nodes: Dict[str, Dict[str, Any]]
    edges: List[Dict[str, Any]]
    canvas_dimensions: tuple[int, int]
    grid_size: int
    style: RenderStyle

@dataclass
class SyntheticTask:
    task_id: str
    domain: TaskDomain
    question_type: QuestionType
    question_text: str
    reference_answer: Any
    answer_format: str  # e.g., "boolean", "integer", "string", "list"
    image_filename: str
    visual_evidence_summary: str
    graph_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['domain'] = self.domain.value
        data['question_type'] = self.question_type.value
        return data

@dataclass
class ModelEvaluationResult:
    task_id: str
    model_name: str
    predicted_answer: str
    parsed_answer: Any
    reference_answer: Any
    is_correct: bool
    latency_seconds: float
    error_category: Optional[str] = None
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
