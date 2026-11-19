import networkx as nx
from typing import Dict, Any, List, Tuple
from src.schema import SyntheticTask, TaskDomain, QuestionType

class PhysicsSimulator:
    """
    Physical causality and electrical signal propagation engine.
    Simulates physical state (closed loops, active components, power delivery)
    to generate visual tasks that test physical law comprehension in VLMs.
    """

    @staticmethod
    def simulate_circuit_power(G: nx.Graph, switch_closed: bool = True) -> Dict[str, Any]:
        """
        Calculates whether power flows from Battery (BAT) to Ground (GND),
        and identifies which active components (Diodes, Resistors) are energized.
        """
        # Find Battery and Ground nodes
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
