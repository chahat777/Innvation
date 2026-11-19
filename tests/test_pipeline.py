import os
import pytest
import networkx as nx
from src.schema import QuestionType, TaskDomain
from src.ground_truth_engine import GroundTruthEngine
from src.verifier import AnswerVerifier
from src.visual_renderer import VisualRenderer
from src.physics_simulator import PhysicsSimulator
from src.rlvf_exporter import RLVFDatasetExporter

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

def test_physics_simulation_causality():
    G = nx.Graph()
    G.add_node(0, label="BAT1", type="Battery", pos=(1, 1))
    G.add_node(1, label="SW1", type="Switch", pos=(2, 1))
    G.add_node(2, label="D1", type="Diode", pos=(3, 1))
    G.add_node(3, label="GND1", type="Ground", pos=(4, 1))
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 3)

    # When switch is closed, power flows
    state_closed = PhysicsSimulator.simulate_circuit_power(G, switch_closed=True)
    assert state_closed["has_power_loop"] is True
    assert "D1" in state_closed["energized_diodes"]

    # When switch is open, power is severed
    state_open = PhysicsSimulator.simulate_circuit_power(G, switch_closed=False)
    assert state_open["has_power_loop"] is False

def test_rlvf_dpo_export(tmp_path):
    mock_tasks = [{
        "task_id": "test_rlvf_1",
        "question_text": "Is R1 wired to C1?",
        "image_filename": "test.png",
        "reference_answer": "Yes",
        "visual_evidence_summary": "Direct wire exists",
        "domain": "circuit_schematic"
    }]
    out_file = os.path.join(str(tmp_path), "test_dpo.jsonl")
    result_path = RLVFDatasetExporter.convert_tasks_to_dpo(mock_tasks, out_file)
    assert os.path.exists(result_path)
    assert os.path.getsize(result_path) > 100

def test_verifier_boolean_logic():
    ok, val, err = AnswerVerifier.verify("Yes, they are connected", "Yes", QuestionType.CONNECTIVITY_EXISTS)
    assert ok is True
    assert val == "Yes"

    ok, val, err = AnswerVerifier.verify("No, there is no direct wire", "No", QuestionType.CONNECTIVITY_EXISTS)
    assert ok is True
    assert val == "No"

def test_renderer_outputs(tmp_path):
    output_dir = str(tmp_path)
    renderer = VisualRenderer(output_dir)
    engine = GroundTruthEngine(seed=456)
    
    G_circ = engine.generate_circuit_topology(num_nodes=6)
    c_path = renderer.render_schematic(G_circ, "test_circuit.png")
    assert os.path.exists(c_path)
