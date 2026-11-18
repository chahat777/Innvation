import os
import json
import random
from typing import List
from src.schema import SyntheticTask, RenderStyle, TaskDomain
from src.ground_truth_engine import GroundTruthEngine
from src.visual_renderer import VisualRenderer

class PipelineGenerator:
    """
    Automated factory that generates batches of synthetic technical images,
    formulates topological QA tasks, and exports full verifiable ground truth.
    Supports both Circuit Schematics and Architectural Floor Plans.
    """

    def __init__(self, base_output_dir: str = "dataset"):
        self.base_output_dir = base_output_dir
        self.images_dir = os.path.join(base_output_dir, "images")
        self.tasks_file = os.path.join(base_output_dir, "tasks.json")
        
        os.makedirs(self.images_dir, exist_ok=True)
        self.renderer = VisualRenderer(self.images_dir)

    def generate_batch(self, num_graphs: int = 16, seed_offset: int = 100) -> List[SyntheticTask]:
        all_tasks: List[SyntheticTask] = []
        styles = [RenderStyle.BLUEPRINT, RenderStyle.CAD_WHITE, RenderStyle.AGED_PARCHMENT]
        
        print(f"🏭 Starting Task Factory: Generating {num_graphs} multi-domain synthetic blueprints...")

        for i in range(num_graphs):
            seed = seed_offset + i
            engine = GroundTruthEngine(seed=seed)
            chosen_style = random.choice(styles)
            add_distortion = (i % 3 == 0)
            
            # Alternate between Circuit Schematics and Architectural Floor Plans
            domain = TaskDomain.CIRCUIT_SCHEMATIC if (i % 2 == 0) else TaskDomain.FLOOR_PLAN

            if domain == TaskDomain.CIRCUIT_SCHEMATIC:
                num_nodes = random.randint(6, 9)
                G = engine.generate_circuit_topology(num_nodes=num_nodes)
                image_filename = f"circuit_schematic_{i+1:03d}_{seed}.png"
                img_path = self.renderer.render_schematic(
                    G, filename=image_filename, style=chosen_style, add_noise=add_distortion
                )
                prefix = f"circuit_{i+1:03d}"
                graph_tasks = engine.formulate_circuit_tasks(G, task_id_prefix=prefix)
            else:
                num_rooms = random.randint(5, 8)
                G = engine.generate_floor_plan_topology(num_rooms=num_rooms)
                image_filename = f"floor_plan_{i+1:03d}_{seed}.png"
                img_path = self.renderer.render_floor_plan(
                    G, filename=image_filename, style=chosen_style, add_noise=add_distortion
                )
                prefix = f"arch_{i+1:03d}"
                graph_tasks = engine.formulate_floor_plan_tasks(G, task_id_prefix=prefix)

            for task in graph_tasks:
                task.image_filename = os.path.join("images", image_filename)
                all_tasks.append(task)

        # Save metadata to JSON
        tasks_data = [t.to_dict() for t in all_tasks]
        with open(self.tasks_file, "w", encoding="utf-8") as f:
            json.dump(tasks_data, f, indent=2)

        print(f"✅ Generated {len(all_tasks)} verifiable tasks across {num_graphs} schematics & floor plans.")
        print(f"💾 Tasks catalog saved to: {self.tasks_file}")
        print(f"🖼️ Images saved in: {self.images_dir}")
        return all_tasks
