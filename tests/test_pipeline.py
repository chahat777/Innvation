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


# --- Phase 2: Multi-Model Vision Integration Tests ---
from unittest.mock import patch, MagicMock
from src.model_providers import (
    mask_secrets,
    ProviderEvaluationResult,
    GeminiVisionProvider,
    OpenAIVisionProvider,
    AnthropicVisionProvider,
    MultiModelArenaRunner
)

def test_mask_secrets():
    raw_msg_gemini = "API Error at endpoint with key AIzaSyD3x9FakeKey1234567890abcdefgh: quota exceeded"
    sanitized = mask_secrets(raw_msg_gemini)
    assert "AIzaSyD3x9FakeKey1234567890abcdefgh" not in sanitized
    assert "REDACTED" in sanitized

    raw_msg_openai = "Failed authentication for key sk-abc123456789012345678901234567890"
    sanitized_openai = mask_secrets(raw_msg_openai)
    assert "sk-abc123456789012345678901234567890" not in sanitized_openai
    assert "REDACTED" in sanitized_openai

    custom_key = "MY_SPECIAL_SUPER_SECRET_KEY_123"
    sanitized_custom = mask_secrets(f"Error with key {custom_key}", specific_key=custom_key)
    assert custom_key not in sanitized_custom
    assert "***REDACTED_KEY***" in sanitized_custom

def test_provider_unconfigured_states(tmp_path):
    dummy_img = os.path.join(str(tmp_path), "dummy.png")
    Image.new("RGB", (100, 100)).save(dummy_img)

    with patch.dict(os.environ, {}, clear=True):
        # Gemini
        gp = GeminiVisionProvider()
        assert not gp.is_configured()
        res_g = gp.evaluate(dummy_img, "Is R1 wired to SW1?", "Yes", QuestionType.CONNECTIVITY_EXISTS)
        assert res_g.status == "NOT_CONFIGURED"
        assert res_g.parsed_answer is None
        assert res_g.is_correct is None
        assert "GEMINI_API_KEY" in res_g.error_message

        # OpenAI
        op = OpenAIVisionProvider()
        assert not op.is_configured()
        res_o = op.evaluate(dummy_img, "Is R1 wired to SW1?", "Yes", QuestionType.CONNECTIVITY_EXISTS)
        assert res_o.status == "NOT_CONFIGURED"
        assert res_o.parsed_answer is None
        assert res_o.is_correct is None
        assert "OPENAI_API_KEY" in res_o.error_message

        # Anthropic
        ap = AnthropicVisionProvider()
        assert not ap.is_configured()
        res_a = ap.evaluate(dummy_img, "Is R1 wired to SW1?", "Yes", QuestionType.CONNECTIVITY_EXISTS)
        assert res_a.status == "NOT_CONFIGURED"
        assert res_a.parsed_answer is None
        assert res_a.is_correct is None
        assert "ANTHROPIC_API_KEY" in res_a.error_message

def test_gemini_vision_provider_mocked(tmp_path):
    dummy_img = os.path.join(str(tmp_path), "dummy_gemini.png")
    Image.new("RGB", (100, 100)).save(dummy_img)

    fake_key = "AIzaSyD_TestFakeKey_1234567890abcdef"
    with patch.dict(os.environ, {"GEMINI_API_KEY": fake_key}):
        gp = GeminiVisionProvider()
        assert gp.is_configured()

        mock_resp = MagicMock()
        mock_resp.text = "Yes, component R1 is directly wired."

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            res = gp.evaluate(dummy_img, "Is R1 wired to SW1?", "Yes", QuestionType.CONNECTIVITY_EXISTS)
            assert res.status == "REAL_API"
            assert res.is_correct is True
            assert res.parsed_answer == "Yes"
            assert res.latency_seconds >= 0.0

def test_openai_vision_provider_mocked(tmp_path):
    dummy_img = os.path.join(str(tmp_path), "dummy_openai.png")
    Image.new("RGB", (100, 100)).save(dummy_img)

    fake_key = "sk-testOpenAIKey123456789012345678901234567890"
    with patch.dict(os.environ, {"OPENAI_API_KEY": fake_key}):
        op = OpenAIVisionProvider()
        assert op.is_configured()

        mock_choice = MagicMock()
        mock_choice.message.content = "No, there is no direct path."
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_openai_cls.return_value = mock_client

            res = op.evaluate(dummy_img, "Is R1 wired to SW1?", "Yes", QuestionType.CONNECTIVITY_EXISTS)
            assert res.status == "REAL_API"
            assert res.is_correct is False
            assert res.parsed_answer == "No"

def test_anthropic_vision_provider_mocked(tmp_path):
    dummy_img = os.path.join(str(tmp_path), "dummy_anthropic.png")
    Image.new("RGB", (100, 100)).save(dummy_img)

    fake_key = "sk-ant-testAnthropicKey12345678901234567890"
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": fake_key}):
        ap = AnthropicVisionProvider()
        assert ap.is_configured()

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "There are 3 hops."
        mock_msg = MagicMock()
        mock_msg.content = [mock_block]

        with patch("anthropic.Anthropic") as mock_anthropic_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_msg
            mock_anthropic_cls.return_value = mock_client

            res = ap.evaluate(dummy_img, "What is the hop count?", 3, QuestionType.SHORTEST_PATH_LENGTH)
            assert res.status == "REAL_API"
            assert res.is_correct is True
            assert res.parsed_answer == 3

def test_provider_api_error_secret_redaction(tmp_path):
    dummy_img = os.path.join(str(tmp_path), "dummy_err.png")
    Image.new("RGB", (100, 100)).save(dummy_img)

    secret_key = "sk-ant-api03-VERY_CONFIDENTIAL_KEY_XYZ123456"
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": secret_key}):
        ap = AnthropicVisionProvider()
        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = RuntimeError(f"Auth failure with token {secret_key}")
            mock_cls.return_value = mock_client

            res = ap.evaluate(dummy_img, "test prompt", "Yes", QuestionType.CONNECTIVITY_EXISTS)
            assert res.status == "API_ERROR"
            assert res.error_message is not None
            assert secret_key not in res.error_message
            assert "REDACTED" in res.error_message

def test_multi_model_arena_runner_overview_and_simulation():
    runner = MultiModelArenaRunner()
    overview = runner.get_provider_status_overview()
    assert "Google DeepMind" in overview
    assert "OpenAI" in overview
    assert "Anthropic" in overview

    sim_res = runner.get_simulated_result(
        task_id="circuit_001_conn_direct",
        provider_name="Simulated-Model",
        model_name="sim-v1",
        reference_answer="Yes",
        question_type=QuestionType.CONNECTIVITY_EXISTS
    )
    assert sim_res.status == "SIMULATED"
    assert "simulation" in sim_res.error_message.lower()


# --- Phase 3: Physical Law Matrix Engine Tests (KCL, KVL, MNA, Ohm's Law) ---
from src.physics_simulator import CircuitBranch, CircuitSolveResult, ElectricalNetwork

def test_voltage_divider_analytical_solution():
    net = ElectricalNetwork(ground_node="GND")
    net.add_voltage_source("BAT1", "N1", "GND", 10.0)
    net.add_resistor("R1", "N1", "N2", 100.0)
    net.add_resistor("R2", "N2", "GND", 100.0)
    
    res = net.solve()
    assert res.status == "SOLVED"
    assert abs(res.node_voltages["N1"] - 10.0) < 1e-9
    assert abs(res.node_voltages["N2"] - 5.0) < 1e-9
    assert abs(res.node_voltages["GND"] - 0.0) < 1e-9
    assert abs(res.branch_currents["R1"] - 0.05) < 1e-9
    assert abs(res.branch_currents["R2"] - 0.05) < 1e-9
    assert abs(res.branch_currents["BAT1"] - (-0.05)) < 1e-9
    assert res.max_kcl_residual < 1e-12
    assert res.max_ohms_residual < 1e-12
    assert res.matrix_rank == res.matrix_size

def test_parallel_resistors_current_division():
    net = ElectricalNetwork(ground_node="GND")
    net.add_voltage_source("BAT1", "N1", "GND", 12.0)
    net.add_resistor("R1", "N1", "GND", 60.0)
    net.add_resistor("R2", "N1", "GND", 30.0)
    
    res = net.solve()
    assert res.status == "SOLVED"
    assert abs(res.branch_currents["R1"] - 0.2) < 1e-9
    assert abs(res.branch_currents["R2"] - 0.4) < 1e-9
    assert abs(res.branch_currents["BAT1"] - (-0.6)) < 1e-9
    assert res.max_kcl_residual < 1e-12

def test_bridge_network_kvl_cycles():
    net = ElectricalNetwork(ground_node="GND")
    net.add_voltage_source("BAT1", "N1", "GND", 10.0)
    net.add_resistor("R1", "N1", "N2", 100.0)
    net.add_resistor("R2", "N2", "GND", 200.0)
    net.add_resistor("R3", "N1", "N3", 300.0)
    net.add_resistor("R4", "N3", "GND", 400.0)
    net.add_resistor("R5", "N2", "N3", 50.0)
    
    res = net.solve()
    assert res.status == "SOLVED"
    assert len(res.kvl_residuals) == 3
    assert res.max_kvl_residual < 1e-12
    assert res.max_kcl_residual < 1e-12
    assert res.max_ohms_residual < 1e-12
    assert res.condition_number < 1e6

def test_ideal_switch_open_and_closed():
    # Closed switch: ideal conduction, 0V drop
    net_c = ElectricalNetwork(ground_node="GND")
    net_c.add_voltage_source("BAT1", "N1", "GND", 5.0)
    net_c.add_switch("SW1", "N1", "N2", is_closed=True)
    net_c.add_resistor("R1", "N2", "GND", 10.0)
    res_c = net_c.solve()
    assert res_c.status == "SOLVED"
    assert abs(res_c.node_voltages["N2"] - 5.0) < 1e-9
    assert abs(res_c.branch_currents["R1"] - 0.5) < 1e-9
    assert abs(res_c.branch_voltages["SW1"] - 0.0) < 1e-9

    # Open switch: 0 current, downstream unpowered
    net_o = ElectricalNetwork(ground_node="GND")
    net_o.add_voltage_source("BAT1", "N1", "GND", 5.0)
    net_o.add_switch("SW1", "N1", "N2", is_closed=False)
    net_o.add_resistor("R1", "N2", "GND", 10.0)
    res_o = net_o.solve()
    assert res_o.status == "SOLVED"
    assert abs(res_o.node_voltages["N2"] - 0.0) < 1e-9
    assert abs(res_o.branch_currents["SW1"] - 0.0) < 1e-9
    assert abs(res_o.branch_currents["R1"] - 0.0) < 1e-9

def test_invalid_circuit_handling():
    # 1. Missing ground reference
    net_nognd = ElectricalNetwork(ground_node="MISSING_GND")
    net_nognd.add_resistor("R1", "N1", "N2", 10.0)
    res_nognd = net_nognd.solve()
    assert res_nognd.status == "NO_GROUND_REFERENCE"

    # 2. Negative resistance
    net_neg = ElectricalNetwork(ground_node="GND")
    net_neg.add_resistor("R1", "N1", "GND", -10.0)
    res_neg = net_neg.solve()
    assert res_neg.status == "INVALID_RESISTANCE"

    # 3. Floating node
    net_float = ElectricalNetwork(ground_node="GND")
    net_float.add_voltage_source("BAT1", "N1", "GND", 5.0)
    net_float.add_resistor("R1", "N1", "GND", 10.0)
    net_float.add_resistor("R2", "FLOAT_A", "FLOAT_B", 50.0)
    res_float = net_float.solve()
    assert res_float.status == "FLOATING_NODE"

    # 4. Conflicting parallel voltage sources
    net_conflict = ElectricalNetwork(ground_node="GND")
    net_conflict.add_voltage_source("BAT1", "N1", "GND", 5.0)
    net_conflict.add_voltage_source("BAT2", "N1", "GND", 12.0)
    res_conflict = net_conflict.solve()
    assert res_conflict.status == "CONFLICTING_VOLTAGE_SOURCES"


def test_ground_node_kcl_and_tellegen_conservation():
    net = ElectricalNetwork(ground_node="GND")
    net.add_voltage_source("BAT1", "N1", "GND", 15.0)
    net.add_resistor("R1", "N1", "N2", 150.0)
    net.add_resistor("R2", "N2", "GND", 300.0)
    net.add_resistor("R3", "N1", "GND", 50.0)
    
    res = net.solve()
    assert res.status == "SOLVED"
    # Ground node current balance
    assert "GND" in res.kcl_residuals
    assert res.kcl_residuals["GND"] < 1e-12
    # Tellegen power conservation: sum(V_b * I_b) == 0
    assert res.tellegen_power_residual < 1e-12

def test_non_circular_kvl_constitutive_sum():
    net = ElectricalNetwork(ground_node="GND")
    net.add_voltage_source("BAT1", "N1", "GND", 10.0)
    net.add_resistor("R1", "N1", "N2", 100.0)
    net.add_resistor("R2", "N2", "GND", 100.0)
    # Add parallel branch across R2
    net.add_resistor("R2_PAR", "N2", "GND", 100.0)
    
    res = net.solve()
    assert res.status == "SOLVED"
    # KVL loops should include both fundamental cycle and parallel branch 2-cycle
    assert len(res.kvl_residuals) >= 2
    assert res.max_kvl_residual < 1e-12
    # Parallel branches must have matching voltage drop
    assert abs(res.branch_voltages["R2"] - res.branch_voltages["R2_PAR"]) < 1e-12

def test_current_source_circuit():
    net = ElectricalNetwork(ground_node="GND")
    net.add_current_source("IS1", "N1", "GND", 2.0)
    net.add_resistor("R1", "N1", "GND", 10.0)
    
    res = net.solve()
    assert res.status == "SOLVED"
    assert abs(res.node_voltages["N1"] - (-20.0)) < 1e-9
    assert abs(res.branch_currents["R1"] - (-2.0)) < 1e-9
    assert abs(res.branch_currents["IS1"] - 2.0) < 1e-9
    assert res.max_kcl_residual < 1e-12

def test_singular_identical_voltage_sources():
    net = ElectricalNetwork(ground_node="GND")
    net.add_voltage_source("BAT1", "N1", "GND", 5.0)
    net.add_voltage_source("BAT2", "N1", "GND", 5.0)
    net.add_resistor("R1", "N1", "GND", 10.0)
    
    res = net.solve()
    assert res.status == "SINGULAR_MATRIX"
    assert res.matrix_rank < res.matrix_size


# --- Phase 4 Step 1: Strict Final-Answer Extraction Tests ---

def test_verifier_integer_with_numbered_rationale():
    # Numbered list '1.', '2.', '3.' with final answer 0
    text_with_cot = (
        "### Step-by-Step Topological Verification:\n"
        "1. Target Components: Identified from inquiry.\n"
        "2. Physical Trace Analysis: Tracing planar conductive pathways.\n"
        "3. Mathematical Graph Evidence: Count of Diode components in graph: 0.\n"
        "4. Verification Outcome: Continuity confirmed.\n\n"
        "### Final Answer:\n0"
    )
    ok, parsed, err = AnswerVerifier.verify(text_with_cot, 0, QuestionType.COMPONENT_COUNT)
    assert ok is True
    assert parsed == 0
    assert err is None

    # Numbered list with shortest path hops = 3
    text_hops = (
        "1. Start at BAT1.\n"
        "2. Hop through R1.\n"
        "3. Hop through SW1.\n"
        "4. Reach Ground.\n\n"
        "### Final Answer:\n3"
    )
    ok, parsed, err = AnswerVerifier.verify(text_hops, 3, QuestionType.SHORTEST_PATH_LENGTH)
    assert ok is True
    assert parsed == 3
    assert err is None

def test_verifier_boolean_misleading_rationale_keywords():
    # Rationale contains the word "connected", but the final answer is "No"
    misleading_text = (
        "Based on visual appearance, the components look connected because of proximity in the diagram. "
        "However, physical trace analysis reveals the branch is broken.\n\n"
        "### Final Answer:\nNo"
    )
    ok, parsed, err = AnswerVerifier.verify(misleading_text, "No", QuestionType.CONNECTIVITY_EXISTS)
    assert ok is True
    assert parsed == "No"
    assert err is None

    # Mismatch check: Reference is "Yes", but model answered "No" despite "connected" in text
    ok_mismatch, parsed_m, err_m = AnswerVerifier.verify(misleading_text, "Yes", QuestionType.CONNECTIVITY_EXISTS)
    assert ok_mismatch is False
    assert parsed_m == "No"
    assert err_m == "CONNECTIVITY_MISMATCH"

def test_verifier_missing_final_answer_marker_plain_compat():
    # Plain string without marker continues to work (backward compatibility)
    ok1, val1, _ = AnswerVerifier.verify("Yes", "Yes", QuestionType.CONNECTIVITY_EXISTS)
    assert ok1 is True
    assert val1 == "Yes"

    ok2, val2, _ = AnswerVerifier.verify("There are 5 components present", 5, QuestionType.COMPONENT_COUNT)
    assert ok2 is True
    assert val2 == 5

    ok3, val3, _ = AnswerVerifier.verify("R1", ["R1", "C2"], QuestionType.NEIGHBOR_IDENTIFICATION)
    assert ok3 is True
    assert val3 == "R1"

def test_verifier_empty_final_answer_section():
    # Marker present, but answer section is empty
    empty_marker = "Some detailed reasoning steps...\n\n### Final Answer:\n   \n"
    ok, parsed, err = AnswerVerifier.verify(empty_marker, "Yes", QuestionType.CONNECTIVITY_EXISTS)
    assert ok is False
    assert parsed is None
    assert err == "EMPTY_FINAL_ANSWER"

def test_verifier_multiline_whitespace_final_answer():
    multiline_text = (
        "Analysis complete.\n\n"
        "### Final Answer:\n\n"
        "   Yes   \n\n"
    )
    ok, parsed, err = AnswerVerifier.verify(multiline_text, "Yes", QuestionType.CONNECTIVITY_EXISTS)
    assert ok is True
    assert parsed == "Yes"

    # Multiline list for neighbor
    multiline_neighbor = (
        "Identified neighbors:\n\n"
        "### Final Answer:\n"
        "['SW1', 'C2', 'R3']\n"
    )
    ok_n, parsed_n, err_n = AnswerVerifier.verify(multiline_neighbor, ["SW1", "C2", "R3"], QuestionType.NEIGHBOR_IDENTIFICATION)
    assert ok_n is True
    assert parsed_n == "SW1"


# --- Phase 4 Step 2: Dual-Format DPO & Verifier-Backed Exporter Tests ---

def test_rlvf_dpo_legacy_format_and_verification(tmp_path):
    mock_tasks = [
        {
            "task_id": "legacy_bool_yes",
            "question_type": "connectivity_exists",
            "question_text": "Is R1 wired to C1?",
            "image_filename": "images\\circuit_01.png",
            "reference_answer": "Yes",
            "visual_evidence_summary": "Direct wire exists",
            "domain": "circuit_schematic"
        },
        {
            "task_id": "legacy_bool_no",
            "question_type": "connectivity_exists",
            "question_text": "Is R1 wired to GND1?",
            "image_filename": "images/circuit_02.png",
            "reference_answer": "No",
            "visual_evidence_summary": "No wire path",
            "domain": "circuit_schematic"
        },
        {
            "task_id": "legacy_count",
            "question_type": "component_count",
            "question_text": "How many resistors are in this circuit?",
            "image_filename": "images/circuit_03.png",
            "reference_answer": 4,
            "visual_evidence_summary": "4 resistor glyphs found",
            "domain": "circuit_schematic"
        },
        {
            "task_id": "legacy_neighbor",
            "question_type": "neighbor_id",
            "question_text": "Name one component connected to R1.",
            "image_filename": "images/circuit_04.png",
            "reference_answer": ["SW1", "C2"],
            "visual_evidence_summary": "SW1 and C2 connected",
            "domain": "circuit_schematic",
            "graph_metadata": {
                "source": "R1",
                "serialized_graph": {
                    "nodes": [
                        {"id": 0, "label": "R1"},
                        {"id": 1, "label": "SW1"},
                        {"id": 2, "label": "C2"},
                        {"id": 3, "label": "BAT1"}
                    ]
                }
            }
        }
    ]

    out_file = os.path.join(str(tmp_path), "test_legacy_dpo.jsonl")
    result_path = RLVFDatasetExporter.convert_tasks_to_dpo(mock_tasks, out_file)

    assert isinstance(result_path, str)
    assert os.path.exists(result_path)

    with open(result_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    assert len(lines) == 4
    expected_keys = {"task_id", "image_path", "domain", "prompt", "chosen", "rejected", "verification_source"}

    for item, src_t in zip(lines, mock_tasks):
        assert set(item.keys()) == expected_keys
        # Check POSIX path normalization (no backslashes)
        assert "\\" not in item["image_path"]
        assert item["image_path"].startswith("images/")
        assert item["prompt"].startswith("<image>\n")

        # Verify chosen passes and rejected fails AnswerVerifier
        q_type = QuestionType(src_t["question_type"])
        ref_ans = src_t["reference_answer"]

        c_ok, c_val, c_err = AnswerVerifier.verify(item["chosen"], ref_ans, q_type)
        assert c_ok is True, f"Chosen failed verification for {item['task_id']}: {c_err}"

        r_ok, r_val, r_err = AnswerVerifier.verify(item["rejected"], ref_ans, q_type)
        assert r_ok is False, f"Rejected unexpectedly passed verification for {item['task_id']}: {r_val}"

def test_rlvf_dpo_conversational_export_schema(tmp_path):
    mock_tasks = [
        {
            "task_id": "conv_task_1",
            "question_type": "connectivity_exists",
            "question_text": "Is R1 connected to C1?",
            "image_filename": "images\\circuit_conv.png",
            "reference_answer": "Yes",
            "answer_format": "boolean",
            "visual_evidence_summary": "Direct wire",
            "domain": "circuit_schematic"
        }
    ]

    out_file = os.path.join(str(tmp_path), "test_conv_dpo.jsonl")
    report = RLVFDatasetExporter.export_dpo_dataset(mock_tasks, out_file, dataset_version="2.0.0")

    assert report["total_source_tasks"] == 1
    assert report["exported_preference_pairs"] == 1
    assert report["excluded_tasks"] == 0
    assert report["chosen_verification_pass_count"] == 1
    assert report["rejected_verification_failure_count"] == 1
    assert report["duplicate_ids_count"] == 0
    assert report["dataset_version"] == "2.0.0"

    with open(out_file, "r", encoding="utf-8") as f:
        record = json.loads(f.readline())

    expected_record_keys = {
        "id", "domain", "task_type", "images", "prompt",
        "chosen", "rejected", "chosen_text", "rejected_text", "metadata"
    }
    assert set(record.keys()) == expected_record_keys
    assert record["id"] == "conv_task_1"
    assert record["domain"] == "circuit_schematic"
    assert record["task_type"] == "connectivity_exists"
    assert record["images"] == ["images/circuit_conv.png"]
    assert "\\" not in record["images"][0]

    # Check prompt structure
    assert isinstance(record["prompt"], list) and len(record["prompt"]) == 1
    assert record["prompt"][0]["role"] == "user"
    assert record["prompt"][0]["content"] == [
        {"type": "image"},
        {"type": "text", "text": "Is R1 connected to C1?"}
    ]

    # Check chosen / rejected role messages
    assert record["chosen"][0]["role"] == "assistant"
    assert record["rejected"][0]["role"] == "assistant"
    assert record["chosen_text"] == record["chosen"][0]["content"]
    assert record["rejected_text"] == record["rejected"][0]["content"]

    # Check metadata
    meta = record["metadata"]
    assert meta["dataset_version"] == "2.0.0"
    assert meta["verifier_version"] == "AnswerVerifier-v2"
    assert meta["chosen_verification"]["is_correct"] is True
    assert meta["rejected_verification"]["is_correct"] is False

def test_rlvf_dpo_full_56_benchmark_tasks(tmp_path):
    with open("dataset/tasks.json", "r", encoding="utf-8") as f:
        tasks = json.load(f)

    assert len(tasks) == 56

    out_file = os.path.join(str(tmp_path), "bench_56_dpo.jsonl")
    report = RLVFDatasetExporter.export_dpo_dataset(tasks, out_file)

    assert report["total_source_tasks"] == 56
    assert report["exported_preference_pairs"] == 56
    assert report["excluded_tasks"] == 0
    assert report["chosen_verification_pass_count"] == 56
    assert report["rejected_verification_failure_count"] == 56
    assert report["duplicate_ids_count"] == 0

    with open(out_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    assert len(records) == 56
    task_types_found = {r["task_type"] for r in records}
    assert "connectivity_exists" in task_types_found
    assert "component_count" in task_types_found
    assert "shortest_path_length" in task_types_found
    assert "neighbor_id" in task_types_found

    # Spot-check every single record passes verifier
    for r in records:
        assert r["metadata"]["chosen_verification"]["is_correct"] is True
        assert r["metadata"]["rejected_verification"]["is_correct"] is False
        assert "\\" not in r["images"][0]

def test_rlvf_dpo_path_traversal_and_safety(tmp_path):
    malicious_tasks = [
        {
            "task_id": "mal_1",
            "question_type": "connectivity_exists",
            "question_text": "Traverse up",
            "image_filename": "../../secret.txt",
            "reference_answer": "Yes"
        },
        {
            "task_id": "mal_2",
            "question_type": "connectivity_exists",
            "question_text": "Absolute root",
            "image_filename": "/etc/shadow",
            "reference_answer": "Yes"
        },
        {
            "task_id": "mal_3",
            "question_type": "connectivity_exists",
            "question_text": "Drive letter",
            "image_filename": "C:\\Windows\\System32\\cmd.exe",
            "reference_answer": "Yes"
        },
        {
            "task_id": "safe_1",
            "question_type": "connectivity_exists",
            "question_text": "Safe relative",
            "image_filename": "images/valid.png",
            "reference_answer": "Yes"
        }
    ]

    out_file = os.path.join(str(tmp_path), "traversal_test.jsonl")
    report = RLVFDatasetExporter.export_dpo_dataset(malicious_tasks, out_file)

    assert report["total_source_tasks"] == 4
    assert report["exported_preference_pairs"] == 1
    assert report["excluded_tasks"] == 3
    assert len(report["exclusion_details"]) == 3
    for exc in report["exclusion_details"]:
        assert "PATH_TRAVERSAL_DETECTED" in exc["reason"]

    # Verify legacy convert_tasks_to_dpo raises ValueError on path traversal
    with pytest.raises(ValueError, match="Unsafe image path detected"):
        RLVFDatasetExporter.convert_tasks_to_dpo(malicious_tasks[:1], os.path.join(str(tmp_path), "unsafe.jsonl"))

def test_rlvf_dpo_train_test_split_and_determinism(tmp_path):
    tasks = [
        {
            "task_id": f"split_task_{i}",
            "question_type": "connectivity_exists",
            "question_text": f"Is node {i} connected?",
            "image_filename": f"images/task_{i}.png",
            "reference_answer": "Yes" if i % 2 == 0 else "No"
        }
        for i in range(10)
    ]

    out_file = os.path.join(str(tmp_path), "all.jsonl")
    report1 = RLVFDatasetExporter.export_dpo_dataset(tasks, out_file, split_ratio=0.8)

    assert report1["exported_preference_pairs"] == 10
    assert report1["train_count"] == 8
    assert report1["test_count"] == 2
    assert os.path.exists(report1["train_file"])
    assert os.path.exists(report1["test_file"])

    with open(report1["train_file"], "r", encoding="utf-8") as f:
        train_ids = {json.loads(line)["id"] for line in f}
    with open(report1["test_file"], "r", encoding="utf-8") as f:
        test_ids = {json.loads(line)["id"] for line in f}

    assert len(train_ids) == 8
    assert len(test_ids) == 2
    assert train_ids.isdisjoint(test_ids)

    # Determinism check: Second run produces identical content
    out_file2 = os.path.join(str(tmp_path), "all_run2.jsonl")
    report2 = RLVFDatasetExporter.export_dpo_dataset(tasks, out_file2, split_ratio=0.8)

    assert open(out_file, "r", encoding="utf-8").read() == open(out_file2, "r", encoding="utf-8").read()
    assert open(report1["train_file"], "r", encoding="utf-8").read() == open(report2["train_file"], "r", encoding="utf-8").read()
    assert open(report1["test_file"], "r", encoding="utf-8").read() == open(report2["test_file"], "r", encoding="utf-8").read()

def test_rlvf_dpo_duplicate_id_and_missing_id_handling(tmp_path):
    tasks = [
        {
            "task_id": "dup_task_1",
            "question_type": "connectivity_exists",
            "question_text": "First occurrence",
            "image_filename": "images/task_1.png",
            "reference_answer": "Yes"
        },
        {
            "task_id": "dup_task_1",
            "question_type": "connectivity_exists",
            "question_text": "Duplicate occurrence",
            "image_filename": "images/task_1_dup.png",
            "reference_answer": "Yes"
        },
        {
            "task_id": "",
            "question_type": "connectivity_exists",
            "question_text": "Missing id",
            "image_filename": "images/task_empty.png",
            "reference_answer": "Yes"
        }
    ]

    out_file = os.path.join(str(tmp_path), "dup_test.jsonl")
    report = RLVFDatasetExporter.export_dpo_dataset(tasks, out_file)

    assert report["total_source_tasks"] == 3
    assert report["exported_preference_pairs"] == 1
    assert report["duplicate_ids_count"] == 1
    assert report["excluded_tasks"] == 2
    reasons = [e["reason"] for e in report["exclusion_details"]]
    assert "DUPLICATE_TASK_ID" in reasons
    assert "MISSING_TASK_ID" in reasons

