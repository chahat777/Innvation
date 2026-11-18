import os
import pytest
import networkx as nx
from src.schema import QuestionType, TaskDomain
from src.ground_truth_engine import GroundTruthEngine
from src.verifier import AnswerVerifier
from src.visual_renderer import VisualRenderer

def test_ground_truth_engine_consistency():
    engine = GroundTruthEngine(seed=123)
    G = engine.generate_circuit_topology(num_nodes=7)
    
    assert len(G.nodes) == 7
    assert nx.is_connected(G), "Generated graph should be fully connected"
    
    tasks = engine.formulate_circuit_tasks(G, "test_task")
    assert len(tasks) >= 3, "Should generate multiple tasks per schematic"
    
    for t in tasks:
        assert t.reference_answer is not None
        assert t.question_text != ""

def test_verifier_boolean_logic():
    # Test affirmative matches
    ok, val, err = AnswerVerifier.verify("Yes, they are connected", "Yes", QuestionType.CONNECTIVITY_EXISTS)
    assert ok is True
    assert val == "Yes"

    # Test negative matches
    ok, val, err = AnswerVerifier.verify("No, there is no direct wire", "No", QuestionType.CONNECTIVITY_EXISTS)
    assert ok is True
    assert val == "No"

    # Test mismatch
    ok, val, err = AnswerVerifier.verify("Yes", "No", QuestionType.CONNECTIVITY_EXISTS)
    assert ok is False

def test_verifier_integer_counting():
    ok, val, err = AnswerVerifier.verify("I counted 3 resistors in the image.", 3, QuestionType.COMPONENT_COUNT)
    assert ok is True
    assert val == 3

    ok, val, err = AnswerVerifier.verify("There are 5 inductors.", 2, QuestionType.COMPONENT_COUNT)
    assert ok is False
    assert val == 5
    assert "COUNT_ERROR" in err

def test_renderer_output(tmp_path):
    output_dir = str(tmp_path)
    renderer = VisualRenderer(output_dir)
    engine = GroundTruthEngine(seed=456)
    G = engine.generate_circuit_topology(num_nodes=6)
    
    img_path = renderer.render_schematic(G, "test_render.png")
    assert os.path.exists(img_path)
    assert os.path.getsize(img_path) > 1000
