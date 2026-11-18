import random
import networkx as nx
from typing import Dict, Any, List, Tuple
from src.schema import TaskDomain, QuestionType, SyntheticTask

ELECTRONIC_COMPONENTS = ["Resistor", "Capacitor", "Inductor", "Diode", "Switch", "Ground", "Battery"]
COMPONENT_PREFIXES = {
    "Resistor": "R",
    "Capacitor": "C",
    "Inductor": "L",
    "Diode": "D",
    "Switch": "SW",
    "Ground": "GND",
    "Battery": "BAT"
}

class GroundTruthEngine:
    """
    Deterministically constructs topological graphs and computes mathematically
    verifiable reference answers before images are rendered.
    """

    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)

    def generate_circuit_topology(self, num_nodes: int = 7) -> nx.Graph:
        """Generates a connected planar-like circuit topology on a 2D integer grid."""
        G = nx.Graph()
        
        # Place components on an orderly grid with orthographic coordinates
        grid_positions = [(x, y) for x in range(1, 5) for y in range(1, 4)]
        random.shuffle(grid_positions)
        chosen_positions = grid_positions[:num_nodes]
        
        comp_counters = {k: 1 for k in ELECTRONIC_COMPONENTS}
        
        for i, pos in enumerate(chosen_positions):
            comp_type = random.choice(ELECTRONIC_COMPONENTS)
            label = f"{COMPONENT_PREFIXES[comp_type]}{comp_counters[comp_type]}"
            comp_counters[comp_type] += 1
            G.add_node(i, label=label, type=comp_type, pos=pos)

        # Build spanning tree to ensure strong connectivity
        nodes = list(G.nodes())
        connected = [nodes[0]]
        unconnected = nodes[1:]
        
        while unconnected:
            u = random.choice(connected)
            v = unconnected.pop(random.randrange(len(unconnected)))
            G.add_edge(u, v)
            connected.append(v)
            
        # Add 1-3 extra edges for loops/branches
        extra_edges = random.randint(1, min(3, len(nodes) - 1))
        for _ in range(extra_edges):
            u, v = random.sample(nodes, 2)
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)

        return G

    def formulate_circuit_tasks(self, G: nx.Graph, task_id_prefix: str) -> List[SyntheticTask]:
        """Creates diverse verifiable questions based on the graph topology."""
        tasks = []
        nodes = list(G.nodes())
        
        # 1. Connectivity Verification (Yes/No)
        u, v = random.sample(nodes, 2)
        label_u = G.nodes[u]['label']
        label_v = G.nodes[v]['label']
        
        # Test direct connection
        is_direct = G.has_edge(u, v)
        q1 = SyntheticTask(
            task_id=f"{task_id_prefix}_conn_direct",
            domain=TaskDomain.CIRCUIT_SCHEMATIC,
            question_type=QuestionType.CONNECTIVITY_EXISTS,
            question_text=f"Is component '{label_u}' directly wired to component '{label_v}' without passing through any intermediate component? Answer strictly with 'Yes' or 'No'.",
            reference_answer="Yes" if is_direct else "No",
            answer_format="boolean",
            image_filename="",
            visual_evidence_summary=f"Direct wiring check between {label_u} at {G.nodes[u]['pos']} and {label_v} at {G.nodes[v]['pos']}. Edge exists: {is_direct}.",
            graph_metadata={"node_u": label_u, "node_v": label_v, "is_direct": is_direct}
        )
        tasks.append(q1)

        # 2. Component Counting Question (Integer)
        target_type = random.choice(ELECTRONIC_COMPONENTS)
        actual_count = sum(1 for _, data in G.nodes(data=True) if data['type'] == target_type)
        q2 = SyntheticTask(
            task_id=f"{task_id_prefix}_count",
            domain=TaskDomain.CIRCUIT_SCHEMATIC,
            question_type=QuestionType.COMPONENT_COUNT,
            question_text=f"How many '{target_type}' components (labeled with prefix '{COMPONENT_PREFIXES[target_type]}') are present in this schematic? Provide only the integer count.",
            reference_answer=actual_count,
            answer_format="integer",
            image_filename="",
            visual_evidence_summary=f"Count of {target_type} components in graph: {actual_count}.",
            graph_metadata={"target_type": target_type, "count": actual_count}
        )
        tasks.append(q2)

        # 3. Shortest Path Hop Count (Integer)
        if nx.has_path(G, u, v):
            path = nx.shortest_path(G, source=u, target=v)
            hop_count = len(path) - 1
            intermediate_components = [G.nodes[n]['label'] for n in path[1:-1]]
            q3 = SyntheticTask(
                task_id=f"{task_id_prefix}_path_hops",
                domain=TaskDomain.CIRCUIT_SCHEMATIC,
                question_type=QuestionType.SHORTEST_PATH_LENGTH,
                question_text=f"What is the minimum number of wire hops (branches) required to trace a signal from component '{label_u}' to component '{label_v}'? Provide only the integer number.",
                reference_answer=hop_count,
                answer_format="integer",
                image_filename="",
                visual_evidence_summary=f"Shortest path from {label_u} to {label_v}: {' -> '.join([G.nodes[n]['label'] for n in path])}. Total hops: {hop_count}.",
                graph_metadata={"source": label_u, "target": label_v, "hops": hop_count, "path": [G.nodes[n]['label'] for n in path]}
            )
            tasks.append(q3)

        # 4. Immediate Neighbor Identification (String identifier)
        neighbors = list(G.neighbors(u))
        if neighbors:
            chosen_neighbor = random.choice(neighbors)
            neighbor_label = G.nodes[chosen_neighbor]['label']
            q4 = SyntheticTask(
                task_id=f"{task_id_prefix}_neighbor",
                domain=TaskDomain.CIRCUIT_SCHEMATIC,
                question_type=QuestionType.NEIGHBOR_IDENTIFICATION,
                question_text=f"Name any one component directly connected by a single wire to '{label_u}'. Provide only the component label (e.g. R1, C1).",
                reference_answer=[G.nodes[n]['label'] for n in neighbors],  # Set of valid answers
                answer_format="string",
                image_filename="",
                visual_evidence_summary=f"Neighbors of {label_u}: {[G.nodes[n]['label'] for n in neighbors]}.",
                graph_metadata={"source": label_u, "valid_neighbors": [G.nodes[n]['label'] for n in neighbors]}
            )
            tasks.append(q4)

        return tasks
