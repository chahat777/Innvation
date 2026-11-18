import os
import pytest
import networkx as nx
from src.schema import QuestionType, TaskDomain
from src.ground_truth_engine import GroundTruthEngine
from src.verifier import AnswerVerifier
from src.visual_renderer import VisualRenderer

def test_ground_truth_circuit_consistency():
    engine = GroundTruthEngine(seed=123)
    G = engine.generate_circuit_topology(num_nodes=7)
    
    assert len(G.nodes) == 7
    assert nx.is_connected(G), "Generated circuit should be connected"
    
    tasks = engine.formulate_circuit_tasks(G, "test_circuit")
    assert len(tasks) >= 3, "Should generate multiple circuit tasks"
    for t in tasks:
        assert t.reference_answer is not None
        assert t.question_text != ""

def test_ground_truth_floor_plan_consistency():
    engine = GroundTruthEngine(seed=789)
    G = engine.generate_floor_plan_topology(num_rooms=6)
    
    assert len(G.nodes) == 6
    assert nx.is_connected(G), "Generated floor plan should be fully connected"
    
    tasks = engine.formulate_floor_plan_tasks(G, "test_floor")
    assert len(tasks) >= 2
    for t in tasks:
        assert t.domain == TaskDomain.FLOOR_PLAN
        assert t.reference_answer is not None

def test_verifier_boolean_logic():
    ok, val, err = AnswerVerifier.verify("Yes, they are connected", "Yes", QuestionType.CONNECTIVITY_EXISTS)
    assert ok is True
    assert val == "Yes"

    ok, val, err = AnswerVerifier.verify("No, there is no direct wire", "No", QuestionType.CONNECTIVITY_EXISTS)
    assert ok is True
    assert val == "No"

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

def test_renderer_outputs(tmp_path):
    output_dir = str(tmp_path)
    renderer = VisualRenderer(output_dir)
    engine = GroundTruthEngine(seed=456)
    
    # Circuit
    G_circ = engine.generate_circuit_topology(num_nodes=6)
    c_path = renderer.render_schematic(G_circ, "test_circuit.png")
    assert os.path.exists(c_path)
    
    # Floor Plan
    G_floor = engine.generate_floor_plan_topology(num_rooms=5)
    f_path = renderer.render_floor_plan(G_floor, "test_floor.png")
    assert os.path.exists(f_path)
