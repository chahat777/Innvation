import os
import pytest
import json
from PIL import Image
import networkx as nx

from src.schema import QuestionType, TaskDomain
from src.ground_truth_engine import GroundTruthEngine
from src.verifier import AnswerVerifier
from src.visual_renderer import VisualRenderer
from src.physics_simulator import PhysicsSimulator
from src.rlvf_exporter import RLVFDatasetExporter
from src.symbolic_proof import SymbolicProofEngine
from src.adversarial_engine import AdversarialStressEngine

# --- Existing Tests Preserved ---

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
        assert "serialized_graph" in t.graph_metadata

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
        assert "serialized_graph" in t.graph_metadata

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


# --- New Phase 1 Tests Added ---

def test_graph_serialization_and_reconstruction():
    engine = GroundTruthEngine(seed=999)
    G_orig = engine.generate_circuit_topology(num_nodes=7)
    
    # 1. Serialize
    serialized = GroundTruthEngine.serialize_graph(G_orig)
    assert "nodes" in serialized
    assert "edges" in serialized
    assert serialized["num_nodes"] == 7
    assert len(serialized["nodes"]) == 7
    
    # Verify node structure
    sample_node = serialized["nodes"][0]
    assert "id" in sample_node
    assert "label" in sample_node
    assert "type" in sample_node
    assert "pos" in sample_node
    assert len(sample_node["pos"]) == 2
    
    # 2. Deserialize
    G_reconstructed = GroundTruthEngine.deserialize_graph(serialized)
    assert len(G_reconstructed.nodes) == len(G_orig.nodes)
    assert len(G_reconstructed.edges) == len(G_orig.edges)
    
    for n in G_orig.nodes:
        assert G_reconstructed.nodes[n]["label"] == G_orig.nodes[n]["label"]
        assert G_reconstructed.nodes[n]["type"] == G_orig.nodes[n]["type"]
        assert G_reconstructed.nodes[n]["pos"] == G_orig.nodes[n]["pos"]

    for u, v in G_orig.edges:
        assert G_reconstructed.has_edge(u, v)

def test_symbolic_proof_multiple_task_types():
    engine = GroundTruthEngine(seed=321)
    G = engine.generate_circuit_topology(num_nodes=7)
    tasks = engine.formulate_circuit_tasks(G, "test_proof")
    
    for t in tasks:
        task_dict = t.to_dict()
        sg = task_dict["graph_metadata"]["serialized_graph"]
        G_recon = GroundTruthEngine.deserialize_graph(sg)
        proof = SymbolicProofEngine.generate_proof_for_task(task_dict, G_recon)
        
        assert "VALIDATION_ERROR" not in proof
        assert "PROOF_ERROR" not in proof
        assert "Q.E.D." in proof

def test_symbolic_proof_invalid_node_handling():
    G = nx.Graph()
    G.add_node(0, label="R1", type="Resistor", pos=(1.0, 1.0))
    res = SymbolicProofEngine.generate_connectivity_proof(G, "INVALID_NODE", "R1")
    assert "VALIDATION_ERROR" in res
    assert "'INVALID_NODE'" in res

def test_perception_overlay_graph_consistency(tmp_path):
    output_dir = str(tmp_path)
    renderer = VisualRenderer(output_dir)
    engine = GroundTruthEngine(seed=777)
    G = engine.generate_circuit_topology(num_nodes=6)
    
    serialized = GroundTruthEngine.serialize_graph(G)
    G_recon = GroundTruthEngine.deserialize_graph(serialized)
    
    task_meta = {
        "node_u": G_recon.nodes[0]["label"],
        "node_v": G_recon.nodes[1]["label"],
        "is_direct": G_recon.has_edge(0, 1)
    }
    
    out_file = renderer.render_ai_perception_overlay(G_recon, "test_perception.png", task_meta)
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 1000

def test_radar_metrics_calculation_safety():
    # Populated metrics
    b_data = {
        "connectivity_exists": {"accuracy_pct": 25.0},
        "component_count": {"accuracy_pct": 18.75},
        "shortest_path_length": {"accuracy_pct": 20.0},
        "neighbor_id": {"accuracy_pct": 30.0}
    }
    rvals = [
        float(b_data.get("connectivity_exists", {}).get("accuracy_pct", 0.0)),
        float(b_data.get("component_count", {}).get("accuracy_pct", 0.0)),
        float(b_data.get("shortest_path_length", {}).get("accuracy_pct", 0.0)),
        float(b_data.get("neighbor_id", {}).get("accuracy_pct", 0.0))
    ]
    assert rvals == [25.0, 18.75, 20.0, 30.0]
    
    # Missing / empty metrics
    empty_data = {}
    rvals_empty = [
        float(empty_data.get("connectivity_exists", {}).get("accuracy_pct", 0.0)),
        float(empty_data.get("component_count", {}).get("accuracy_pct", 0.0)),
        float(empty_data.get("shortest_path_length", {}).get("accuracy_pct", 0.0)),
        float(empty_data.get("neighbor_id", {}).get("accuracy_pct", 0.0))
    ]
    assert rvals_empty == [0.0, 0.0, 0.0, 0.0]

def test_adversarial_level4_overpass_transformation(tmp_path):
    # Create base dummy image
    base_img_path = os.path.join(str(tmp_path), "base.png")
    out_img_path = os.path.join(str(tmp_path), "overpass_out.png")
    
    img = Image.new("RGB", (400, 300), color=(15, 23, 42))
    img.save(base_img_path)
    
    orig_bytes = open(base_img_path, "rb").read()
    res_path = AdversarialStressEngine.apply_overpass_bridge_trap(base_img_path, out_img_path)
    
    assert os.path.exists(out_img_path)
    assert res_path == out_img_path
    
    # Original is untouched
    assert open(base_img_path, "rb").read() == orig_bytes
    # Output is modified
    assert open(out_img_path, "rb").read() != orig_bytes
