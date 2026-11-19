import networkx as nx
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field

from src.schema import SyntheticTask, TaskDomain, QuestionType


@dataclass
class CircuitBranch:
    """
    Constitutive electrical branch connecting two nodal terminals.
    """
    branch_id: str
    node_pos: str
    node_neg: str
    element_type: str  # "Resistor", "VoltageSource", "CurrentSource", "Switch"
    value: float       # Ohms for Resistor, Volts for VoltageSource, Amperes for CurrentSource
    state: Optional[str] = "CLOSED"  # "CLOSED" or "OPEN" for Switch elements


@dataclass
class CircuitSolveResult:
    """
    Complete physical and numerical state of the solved electrical network.
    """
    status: str  # "SOLVED", "SINGULAR_MATRIX", "NO_GROUND_REFERENCE", "FLOATING_NODE", "CONFLICTING_VOLTAGE_SOURCES", "INVALID_RESISTANCE"
    node_voltages: Dict[str, float] = field(default_factory=dict)
    branch_currents: Dict[str, float] = field(default_factory=dict)
    branch_voltages: Dict[str, float] = field(default_factory=dict)
    kcl_residuals: Dict[str, float] = field(default_factory=dict)
    max_kcl_residual: float = 0.0
    kvl_residuals: List[Dict[str, Any]] = field(default_factory=list)
    max_kvl_residual: float = 0.0
    ohms_law_residuals: Dict[str, float] = field(default_factory=dict)
    max_ohms_residual: float = 0.0
    tellegen_power_residual: float = 0.0
    matrix_rank: int = 0
    matrix_size: int = 0
    condition_number: float = 0.0
    mna_matrix: Optional[List[List[float]]] = None
    mna_vector: Optional[List[float]] = None
    node_list: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "node_voltages": {k: round(v, 6) for k, v in self.node_voltages.items()},
            "branch_currents": {k: round(v, 6) for k, v in self.branch_currents.items()},
            "branch_voltages": {k: round(v, 6) for k, v in self.branch_voltages.items()},
            "max_kcl_residual": float(self.max_kcl_residual),
            "max_kvl_residual": float(self.max_kvl_residual),
            "max_ohms_residual": float(self.max_ohms_residual),
            "tellegen_power_residual": float(self.tellegen_power_residual),
            "matrix_rank": self.matrix_rank,
            "matrix_size": self.matrix_size,
            "condition_number": float(self.condition_number) if not np.isinf(self.condition_number) else "Infinity",
            "error_message": self.error_message
        }


class ElectricalNetwork:
    """
    Rigorous Modified Nodal Analysis (MNA) linear matrix solver for DC electrical circuits.
    Enforces Kirchhoff's Current Law (KCL), Kirchhoff's Voltage Law (KVL), and Ohm's Law (V = IR).
    """

    def __init__(self, ground_node: str = "GND"):
        self.ground_node = ground_node
        self.branches: Dict[str, CircuitBranch] = {}

    def set_ground(self, ground_node: str):
        self.ground_node = ground_node

    def add_branch(self, branch: CircuitBranch):
        self.branches[branch.branch_id] = branch

    def add_resistor(self, branch_id: str, node_pos: str, node_neg: str, resistance: float):
        self.add_branch(CircuitBranch(
            branch_id=branch_id,
            node_pos=str(node_pos),
            node_neg=str(node_neg),
            element_type="Resistor",
            value=float(resistance)
        ))

    def add_voltage_source(self, branch_id: str, node_pos: str, node_neg: str, voltage: float):
        self.add_branch(CircuitBranch(
            branch_id=branch_id,
            node_pos=str(node_pos),
            node_neg=str(node_neg),
            element_type="VoltageSource",
            value=float(voltage)
        ))

    def add_current_source(self, branch_id: str, node_pos: str, node_neg: str, current: float):
        self.add_branch(CircuitBranch(
            branch_id=branch_id,
            node_pos=str(node_pos),
            node_neg=str(node_neg),
            element_type="CurrentSource",
            value=float(current)
        ))

    def add_switch(self, branch_id: str, node_pos: str, node_neg: str, is_closed: bool = True):
        self.add_branch(CircuitBranch(
            branch_id=branch_id,
            node_pos=str(node_pos),
            node_neg=str(node_neg),
            element_type="Switch",
            value=0.0,
            state="CLOSED" if is_closed else "OPEN"
        ))

    def get_nodes(self) -> List[str]:
        nodes = set()
        for b in self.branches.values():
            nodes.add(b.node_pos)
            nodes.add(b.node_neg)
        return sorted(list(nodes))

    def validate_network(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Pre-flight electrical network sanity check.
        Returns: (is_valid, status_code, error_message)
        """
        nodes = self.get_nodes()
        if not nodes:
            return False, "EMPTY_CIRCUIT", "Circuit has no branches or nodes."

        if self.ground_node not in nodes:
            return False, "NO_GROUND_REFERENCE", f"Ground reference node '{self.ground_node}' not found in circuit nodes: {nodes}."

        # Check resistance validity
        for b in self.branches.values():
            if b.element_type == "Resistor" and b.value <= 0.0:
                return False, "INVALID_RESISTANCE", f"Resistor '{b.branch_id}' has non-positive resistance: {b.value} Ohms."

        # Check connectivity to ground (floating subcircuits)
        G_conn = nx.Graph()
        for n in nodes:
            G_conn.add_node(n)
        for b in self.branches.values():
            if b.element_type == "Switch" and b.state == "OPEN":
                continue
            G_conn.add_edge(b.node_pos, b.node_neg)

        # Check for isolated nodes with degree 0
        for n in nodes:
            if G_conn.degree(n) == 0:
                return False, "FLOATING_NODE", f"Node '{n}' is completely floating (unconnected)."

        # Check if ground component is connected to all active subcircuits
        if not nx.is_connected(G_conn):
            connected_to_gnd = nx.node_connected_component(G_conn, self.ground_node)
            floating_nodes = set(nodes) - connected_to_gnd
            if floating_nodes:
                return False, "FLOATING_NODE", f"Subcircuit containing nodes {sorted(list(floating_nodes))} has no DC reference path to ground."

        # Check for direct parallel voltage source conflicts
        v_sources = [b for b in self.branches.values() if b.element_type == "VoltageSource" or (b.element_type == "Switch" and b.state == "CLOSED")]
        for i in range(len(v_sources)):
            for j in range(i + 1, len(v_sources)):
                v1, v2 = v_sources[i], v_sources[j]
                if {v1.node_pos, v1.node_neg} == {v2.node_pos, v2.node_neg}:
                    # Direct parallel branch
                    # Check orientation
                    expected_val = v1.value if v1.node_pos == v2.node_pos else -v1.value
                    if abs(expected_val - v2.value) > 1e-9:
                        return False, "CONFLICTING_VOLTAGE_SOURCES", (
                            f"Parallel voltage conflict between '{v1.branch_id}' ({v1.value}V) "
                            f"and '{v2.branch_id}' ({v2.value}V) across nodes {v1.node_pos} and {v1.node_neg}."
                        )

        return True, "VALID", None

    def solve(self) -> CircuitSolveResult:
        """
        Executes Modified Nodal Analysis (MNA):
        A * x = z
        where A = [[G, B], [C, D]], x = [v, j]^T, z = [i, e]^T.
        """
        is_valid, status_code, err_msg = self.validate_network()
        if not is_valid:
            return CircuitSolveResult(status=status_code, error_message=err_msg)

        all_nodes = self.get_nodes()
        non_gnd_nodes = [n for n in all_nodes if n != self.ground_node]
        n_count = len(non_gnd_nodes)
        node_to_idx = {n: i for i, n in enumerate(non_gnd_nodes)}

        # Active voltage sources: explicit VoltageSource, plus CLOSED switches (ideal 0V source)
        # Note: Open switches are completely open-circuit (current = 0).
        v_branches: List[CircuitBranch] = []
        for b in self.branches.values():
            if b.element_type == "VoltageSource":
                v_branches.append(b)
            elif b.element_type == "Switch" and b.state == "CLOSED":
                v_branches.append(b)

        m_count = len(v_branches)
        total_dim = n_count + m_count

        # Build MNA submatrices
        G_mat = np.zeros((n_count, n_count), dtype=np.float64)
        B_mat = np.zeros((n_count, m_count), dtype=np.float64)
        D_mat = np.zeros((m_count, m_count), dtype=np.float64)
        i_vec = np.zeros(n_count, dtype=np.float64)
        e_vec = np.zeros(m_count, dtype=np.float64)

        # 1. Fill Conductance matrix G_mat and current vector i_vec
        for b in self.branches.values():
            if b.element_type == "Resistor":
                conductance = 1.0 / b.value
                p, q = b.node_pos, b.node_neg
                if p != self.ground_node:
                    G_mat[node_to_idx[p], node_to_idx[p]] += conductance
                if q != self.ground_node:
                    G_mat[node_to_idx[q], node_to_idx[q]] += conductance
                if p != self.ground_node and q != self.ground_node:
                    G_mat[node_to_idx[p], node_to_idx[q]] -= conductance
                    G_mat[node_to_idx[q], node_to_idx[p]] -= conductance

            elif b.element_type == "CurrentSource":
                # Current leaves pos node, enters neg node
                p, q = b.node_pos, b.node_neg
                if p != self.ground_node:
                    i_vec[node_to_idx[p]] -= b.value
                if q != self.ground_node:
                    i_vec[node_to_idx[q]] += b.value

        # 2. Fill B_mat, C_mat (= B_mat^T), and e_vec for voltage sources
        for v_idx, b in enumerate(v_branches):
            p, q = b.node_pos, b.node_neg
            if p != self.ground_node:
                B_mat[node_to_idx[p], v_idx] = 1.0
            if q != self.ground_node:
                B_mat[node_to_idx[q], v_idx] = -1.0
            e_vec[v_idx] = b.value  # 0.0 for closed switch, source voltage for VoltageSource

        C_mat = B_mat.T

        # Assemble global block matrix A and right-hand side z
        # A = [[G, B], [C, D]]
        # z = [i, e]
        A_top = np.hstack([G_mat, B_mat]) if m_count > 0 else G_mat
        A_bot = np.hstack([C_mat, D_mat]) if m_count > 0 else np.empty((0, n_count))
        A_mat = np.vstack([A_top, A_bot]) if m_count > 0 else A_top
        z_vec = np.concatenate([i_vec, e_vec]) if m_count > 0 else i_vec

        # Matrix diagnostics
        matrix_rank = int(np.linalg.matrix_rank(A_mat))
        try:
            cond_num = float(np.linalg.cond(A_mat))
        except Exception:
            cond_num = float("inf")

        if matrix_rank < total_dim or np.isinf(cond_num) or cond_num > 1e15:
            return CircuitSolveResult(
                status="SINGULAR_MATRIX",
                matrix_rank=matrix_rank,
                matrix_size=total_dim,
                condition_number=cond_num,
                mna_matrix=A_mat.tolist(),
                mna_vector=z_vec.tolist(),
                error_message=f"Singular MNA matrix: rank={matrix_rank}/{total_dim}, cond={cond_num:.2e}."
            )

        # Solve system A * x = z
        try:
            x_sol = np.linalg.solve(A_mat, z_vec)
        except np.linalg.LinAlgError as e:
            return CircuitSolveResult(
                status="SINGULAR_MATRIX",
                matrix_rank=matrix_rank,
                matrix_size=total_dim,
                condition_number=cond_num,
                error_message=f"Linear algebra solver failed: {e}"
            )

        # Extract node voltages
        node_voltages = {self.ground_node: 0.0}
        for n, idx in node_to_idx.items():
            node_voltages[n] = float(x_sol[idx])

        # Extract branch currents
        # Note: Sign convention is current flowing from node_pos to node_neg
        branch_currents: Dict[str, float] = {}
        branch_voltages: Dict[str, float] = {}

        # 1. Resistors: Ohm's Law I = (V_pos - V_neg) / R
        for b in self.branches.values():
            v_drop = node_voltages[b.node_pos] - node_voltages[b.node_neg]
            branch_voltages[b.branch_id] = float(v_drop)

            if b.element_type == "Resistor":
                branch_currents[b.branch_id] = float(v_drop / b.value)
            elif b.element_type == "CurrentSource":
                branch_currents[b.branch_id] = float(b.value)
            elif b.element_type == "Switch" and b.state == "OPEN":
                branch_currents[b.branch_id] = 0.0

        # 2. Voltage sources (including closed switches): Currents are in x_sol[n_count + v_idx]
        for v_idx, b in enumerate(v_branches):
            # In MNA, j_k is defined as current leaving node_pos through source to node_neg
            j_current = float(x_sol[n_count + v_idx])
            branch_currents[b.branch_id] = j_current

        # --- Independent Verification: KCL, KVL, and Ohm's Law ---

        # 1. KCL Verification: For every node k, sum(I_leaving) - sum(I_entering) == 0
        kcl_residuals: Dict[str, float] = {}
        for n in all_nodes:
            net_leaving = 0.0
            for b in self.branches.values():
                i_val = branch_currents[b.branch_id]
                if b.node_pos == n:
                    net_leaving += i_val
                elif b.node_neg == n:
                    net_leaving -= i_val
            kcl_residuals[n] = abs(net_leaving)

        max_kcl_residual = max(kcl_residuals.values()) if kcl_residuals else 0.0

        # 2. Ohm's Law Verification: For every resistor, |(V_pos - V_neg) - I*R| == 0
        ohms_residuals: Dict[str, float] = {}
        for b in self.branches.values():
            if b.element_type == "Resistor":
                v_drop = branch_voltages[b.branch_id]
                i_flow = branch_currents[b.branch_id]
                ohms_residuals[b.branch_id] = abs(v_drop - (i_flow * b.value))

        max_ohms_residual = max(ohms_residuals.values()) if ohms_residuals else 0.0

        # 3. KVL Verification: Non-circular verification summing constitutive branch voltage drops along closed loops
        active_branches = [b for b in self.branches.values() if not (b.element_type == "Switch" and b.state == "OPEN")]
        branches_by_pair: Dict[Tuple[str, str], List[CircuitBranch]] = {}
        for b in active_branches:
            pair = tuple(sorted([b.node_pos, b.node_neg]))
            branches_by_pair.setdefault(pair, []).append(b)

        def get_constitutive_step_delta_v(b: CircuitBranch, u: str, v: str) -> float:
            i_val = branch_currents[b.branch_id]
            is_forward = (b.node_pos == u and b.node_neg == v)
            sign = -1.0 if is_forward else 1.0

            if b.element_type == "Resistor":
                return sign * (i_val * b.value)
            elif b.element_type == "VoltageSource":
                return sign * b.value
            elif b.element_type == "Switch":
                return 0.0
            elif b.element_type == "CurrentSource":
                return sign * (node_voltages[b.node_pos] - node_voltages[b.node_neg])
            return 0.0

        kvl_residuals: List[Dict[str, Any]] = []
        cyc_counter = 0

        # A. Check parallel branch pairs (2-cycles between same node pair)
        for pair, b_list in branches_by_pair.items():
            if len(b_list) >= 2:
                u, v = pair[0], pair[1]
                for i_b in range(len(b_list)):
                    for j_b in range(i_b + 1, len(b_list)):
                        b1, b2 = b_list[i_b], b_list[j_b]
                        # Loop: u -> v via b1, v -> u via b2
                        delta_v1 = get_constitutive_step_delta_v(b1, u, v)
                        delta_v2 = get_constitutive_step_delta_v(b2, v, u)
                        loop_sum = delta_v1 + delta_v2
                        kvl_residuals.append({
                            "loop_index": cyc_counter,
                            "nodes": [u, v],
                            "branches": [b1.branch_id, b2.branch_id],
                            "voltage_sum": float(loop_sum),
                            "residual": abs(float(loop_sum))
                        })
                        cyc_counter += 1

        # B. Check fundamental cycles of length >= 3
        G_loop = nx.Graph()
        for pair in branches_by_pair:
            G_loop.add_edge(pair[0], pair[1])

        try:
            cycles = nx.cycle_basis(G_loop)
            for cyc in cycles:
                loop_nodes = cyc + [cyc[0]]
                directed_v_sum = 0.0
                loop_branches = []
                for step in range(len(loop_nodes) - 1):
                    u = loop_nodes[step]
                    v = loop_nodes[step + 1]
                    step_pair = tuple(sorted([u, v]))
                    chosen_b = branches_by_pair[step_pair][0]
                    directed_v_sum += get_constitutive_step_delta_v(chosen_b, u, v)
                    loop_branches.append(chosen_b.branch_id)

                kvl_residuals.append({
                    "loop_index": cyc_counter,
                    "nodes": cyc,
                    "branches": loop_branches,
                    "voltage_sum": float(directed_v_sum),
                    "residual": abs(float(directed_v_sum))
                })
                cyc_counter += 1
        except Exception:
            pass

        max_kvl_residual = max([r["residual"] for r in kvl_residuals], default=0.0)

        # 4. Tellegen's Power Conservation: sum(V_b * I_b) == 0
        tellegen_power = sum(
            branch_voltages[b.branch_id] * branch_currents[b.branch_id]
            for b in self.branches.values()
        )
        tellegen_power_residual = abs(float(tellegen_power))

        return CircuitSolveResult(
            status="SOLVED",
            node_voltages=node_voltages,
            branch_currents=branch_currents,
            branch_voltages=branch_voltages,
            kcl_residuals=kcl_residuals,
            max_kcl_residual=float(max_kcl_residual),
            kvl_residuals=kvl_residuals,
            max_kvl_residual=float(max_kvl_residual),
            ohms_law_residuals=ohms_residuals,
            max_ohms_residual=float(max_ohms_residual),
            tellegen_power_residual=tellegen_power_residual,
            matrix_rank=matrix_rank,
            matrix_size=total_dim,
            condition_number=cond_num,
            mna_matrix=A_mat.tolist(),
            mna_vector=z_vec.tolist(),
            node_list=all_nodes
        )


class PhysicsSimulator:
    """
    Physical causality and electrical signal propagation engine.
    Simulates physical state (closed loops, active components, power delivery)
    to generate visual tasks that test physical law comprehension in VLMs.
    """

    @staticmethod
    def create_network(ground_node: str = "GND") -> ElectricalNetwork:
        """Factory method to create an ElectricalNetwork solver instance."""
        return ElectricalNetwork(ground_node=ground_node)

    @staticmethod
    def simulate_circuit_power(G: nx.Graph, switch_closed: bool = True) -> Dict[str, Any]:
        """
        Calculates whether power flows from Battery (BAT) to Ground (GND),
        and identifies which active components (Diodes, Resistors) are energized.
        Preserves backward compatibility with Phase 1 and Phase 2.
        """
        bat_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'Battery']
        gnd_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'Ground']
        sw_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'Switch']
        diode_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'Diode']

        # Create active circuit copy
        H = G.copy()
        
        # If switch is open, remove switch edges to simulate open circuit
        if not switch_closed and sw_nodes:
            for sw in sw_nodes:
                H.remove_node(sw)

        has_closed_loop = False
        energized_diodes = []

        if bat_nodes and gnd_nodes:
            bat = bat_nodes[0]
            gnd = gnd_nodes[0]
            if nx.has_path(H, bat, gnd):
                has_closed_loop = True
                path = nx.shortest_path(H, bat, gnd)
                energized_diodes = [H.nodes[n]['label'] for n in path if H.nodes[n].get('type') == 'Diode']

        return {
            "has_power_loop": has_closed_loop,
            "energized_diodes": energized_diodes,
            "switch_state": "CLOSED" if switch_closed else "OPEN"
        }

    @classmethod
    def generate_causality_task(cls, G: nx.Graph, task_id: str) -> SyntheticTask:
        """
        Formulates a high-level physical reasoning task:
        'If Switch SW1 is CLOSED, will current reach the Ground terminal to complete the circuit?'
        """
        sim_closed = cls.simulate_circuit_power(G, switch_closed=True)
        has_loop = sim_closed["has_power_loop"]

        q_text = (
            "PHYSICAL CAUSALITY INQUIRY: If all mechanical switches in this schematic are CLOSED, "
            "does an unbroken electrical path exist from the Battery power source ('BAT1') to the Ground terminal ('GND1')? "
            "Answer strictly with 'Yes' or 'No'."
        )

        return SyntheticTask(
            task_id=f"{task_id}_causality_loop",
            domain=TaskDomain.CIRCUIT_SCHEMATIC,
            question_type=QuestionType.CONNECTIVITY_EXISTS,
            question_text=q_text,
            reference_answer="Yes" if has_loop else "No",
            answer_format="boolean",
            image_filename="",
            visual_evidence_summary=f"Kirchhoff loop calculation: Battery to Ground path status={has_loop}.",
            graph_metadata={"is_closed_loop": has_loop, "type": "physical_causality"}
        )
