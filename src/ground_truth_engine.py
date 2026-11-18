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

ARCHITECTURAL_ROOM_TYPES = ["Office", "Server_Room", "Conference", "Cafeteria", "Lobby", "Emergency_Exit"]
ROOM_PREFIXES = {
    "Office": "OFF",
    "Server_Room": "SRV",
    "Conference": "CONF",
    "Cafeteria": "CAF",
    "Lobby": "LOB",
    "Emergency_Exit": "EXIT"
}

class GroundTruthEngine:
    """
    Deterministically constructs topological graphs and computes mathematically
    verifiable reference answers before images are rendered.
    Supports both Circuit Schematics and Architectural Floor Plans.
    """

    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)

    def generate_circuit_topology(self, num_nodes: int = 7) -> nx.Graph:
        """Generates a connected planar circuit topology on a 2D integer grid."""
        G = nx.Graph()
        G.graph["domain"] = TaskDomain.CIRCUIT_SCHEMATIC.value
        
        grid_positions = [(x, y) for x in range(1, 5) for y in range(1, 4)]
        random.shuffle(grid_positions)
        chosen_positions = grid_positions[:num_nodes]
        
        comp_counters = {k: 1 for k in ELECTRONIC_COMPONENTS}
        
        for i, pos in enumerate(chosen_positions):
            comp_type = random.choice(ELECTRONIC_COMPONENTS)
            label = f"{COMPONENT_PREFIXES[comp_type]}{comp_counters[comp_type]}"
            comp_counters[comp_type] += 1
            G.add_node(i, label=label, type=comp_type, pos=pos)

        # Spanning tree
        nodes = list(G.nodes())
        connected = [nodes[0]]
        unconnected = nodes[1:]
        
        while unconnected:
            u = random.choice(connected)
            v = unconnected.pop(random.randrange(len(unconnected)))
            G.add_edge(u, v)
            connected.append(v)
            
        extra_edges = random.randint(1, min(3, len(nodes) - 1))
        for _ in range(extra_edges):
            u, v = random.sample(nodes, 2)
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)

        return G

    def generate_floor_plan_topology(self, num_rooms: int = 6) -> nx.Graph:
        """Generates an architectural floor plan with rooms, hallways, and emergency exits."""
        G = nx.Graph()
        G.graph["domain"] = TaskDomain.FLOOR_PLAN.value
        
        grid_positions = [(x, y) for x in range(1, 4) for y in range(1, 4)]
        random.shuffle(grid_positions)
        chosen_positions = grid_positions[:num_rooms]
        
        room_counters = {k: 1 for k in ARCHITECTURAL_ROOM_TYPES}
        
        # Ensure at least one Emergency Exit and one Lobby
        room_types = ["Lobby", "Emergency_Exit"] + [random.choice(ARCHITECTURAL_ROOM_TYPES) for _ in range(num_rooms - 2)]
        random.shuffle(room_types)
        
        for i, pos in enumerate(chosen_positions):
            rtype = room_types[i]
            label = f"{ROOM_PREFIXES[rtype]}-{room_counters[rtype]}"
            room_counters[rtype] += 1
            G.add_node(i, label=label, type=rtype, pos=pos)

        # Spanning tree connections (corridors / doorways)
        nodes = list(G.nodes())
        connected = [nodes[0]]
        unconnected = nodes[1:]
        
        while unconnected:
            u = random.choice(connected)
            v = unconnected.pop(random.randrange(len(unconnected)))
            G.add_edge(u, v)
            connected.append(v)

        # Add 1-2 extra doorways
        for _ in range(random.randint(1, 2)):
            u, v = random.sample(nodes, 2)
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)

        return G

    def formulate_circuit_tasks(self, G: nx.Graph, task_id_prefix: str) -> List[SyntheticTask]:
        """Creates diverse verifiable questions based on circuit graph topology."""
        tasks = []
        nodes = list(G.nodes())
        
        # 1. Connectivity Verification (Yes/No)
        u, v = random.sample(nodes, 2)
        label_u = G.nodes[u]['label']
        label_v = G.nodes[v]['label']
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

        # 4. Immediate Neighbor Identification
        neighbors = list(G.neighbors(u))
        if neighbors:
            q4 = SyntheticTask(
                task_id=f"{task_id_prefix}_neighbor",
                domain=TaskDomain.CIRCUIT_SCHEMATIC,
                question_type=QuestionType.NEIGHBOR_IDENTIFICATION,
                question_text=f"Name any one component directly connected by a single wire to '{label_u}'. Provide only the component label (e.g. R1, C1).",
                reference_answer=[G.nodes[n]['label'] for n in neighbors],
                answer_format="string",
                image_filename="",
                visual_evidence_summary=f"Neighbors of {label_u}: {[G.nodes[n]['label'] for n in neighbors]}.",
                graph_metadata={"source": label_u, "valid_neighbors": [G.nodes[n]['label'] for n in neighbors]}
            )
            tasks.append(q4)

        return tasks

    def formulate_floor_plan_tasks(self, G: nx.Graph, task_id_prefix: str) -> List[SyntheticTask]:
        """Creates verifiable spatial and egress questions based on architectural floor plans."""
        tasks = []
        nodes = list(G.nodes())
        u, v = random.sample(nodes, 2)
        label_u = G.nodes[u]['label']
        label_v = G.nodes[v]['label']
        
        # 1. Direct Doorway Connectivity
        is_adjacent = G.has_edge(u, v)
        q1 = SyntheticTask(
            task_id=f"{task_id_prefix}_doorway",
            domain=TaskDomain.FLOOR_PLAN,
            question_type=QuestionType.CONNECTIVITY_EXISTS,
            question_text=f"Can an occupant walk directly from room '{label_u}' into room '{label_v}' through a single direct doorway without traversing another room? Answer 'Yes' or 'No'.",
            reference_answer="Yes" if is_adjacent else "No",
            answer_format="boolean",
            image_filename="",
            visual_evidence_summary=f"Direct doorway between {label_u} and {label_v}. Edge exists: {is_adjacent}.",
            graph_metadata={"room_u": label_u, "room_v": label_v, "direct_door": is_adjacent}
        )
        tasks.append(q1)

        # 2. Emergency Exit Count
        exit_count = sum(1 for _, data in G.nodes(data=True) if data['type'] == "Emergency_Exit")
        q2 = SyntheticTask(
            task_id=f"{task_id_prefix}_exit_count",
            domain=TaskDomain.FLOOR_PLAN,
            question_type=QuestionType.COMPONENT_COUNT,
            question_text="How many designated Emergency Exits (marked with prefix 'EXIT') are present on this architectural floor plan? Provide only the integer count.",
            reference_answer=exit_count,
            answer_format="integer",
            image_filename="",
            visual_evidence_summary=f"Total designated emergency exits on floor plan: {exit_count}.",
            graph_metadata={"exit_count": exit_count}
        )
        tasks.append(q2)

        # 3. Egress Path Length
        if nx.has_path(G, u, v):
            path = nx.shortest_path(G, source=u, target=v)
            room_transitions = len(path) - 1
            q3 = SyntheticTask(
                task_id=f"{task_id_prefix}_egress_hops",
                domain=TaskDomain.FLOOR_PLAN,
                question_type=QuestionType.SHORTEST_PATH_LENGTH,
                question_text=f"What is the minimum number of room-to-room transitions required to travel from room '{label_u}' to room '{label_v}'? Provide only the integer count.",
                reference_answer=room_transitions,
                answer_format="integer",
                image_filename="",
                visual_evidence_summary=f"Shortest route from {label_u} to {label_v}: {' -> '.join([G.nodes[n]['label'] for n in path])}. Transitions: {room_transitions}.",
                graph_metadata={"source": label_u, "destination": label_v, "transitions": room_transitions}
            )
            tasks.append(q3)

        return tasks
