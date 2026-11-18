import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
from PIL import Image, ImageFilter
from src.schema import RenderStyle

THEME_CONFIGS = {
    RenderStyle.BLUEPRINT: {
        "bg_color": "#0B1D3A",
        "grid_color": "#18325B",
        "wire_color": "#58A6FF",
        "node_fill": "#1F4273",
        "text_color": "#FFFFFF",
        "accent_color": "#38D430",
        "border_color": "#58A6FF"
    },
    RenderStyle.CAD_WHITE: {
        "bg_color": "#FFFFFF",
        "grid_color": "#E5E7EB",
        "wire_color": "#111827",
        "node_fill": "#F3F4F6",
        "text_color": "#111827",
        "accent_color": "#DC2626",
        "border_color": "#374151"
    },
    RenderStyle.AGED_PARCHMENT: {
        "bg_color": "#FAF6EE",
        "grid_color": "#EFE6D5",
        "wire_color": "#2A2421",
        "node_fill": "#EDE4D4",
        "text_color": "#2A2421",
        "accent_color": "#9A3412",
        "border_color": "#443831"
    }
}

class VisualRenderer:
    """
    Renders topological circuit graphs into realistic, high-resolution engineering schematics.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def draw_component_symbol(self, ax, comp_type: str, pos: tuple[float, float], colors: dict):
        x, y = pos
        w, h = 0.5, 0.35
        
        # Component bounding box / backing
        rect = patches.Rectangle(
            (x - w / 2, y - h / 2), w, h,
            linewidth=1.5,
            edgecolor=colors["wire_color"],
            facecolor=colors["node_fill"],
            zorder=3
        )
        ax.add_patch(rect)

        # Internal glyph representation
        if comp_type == "Resistor":
            # Zig-zag trace
            xs = np.linspace(x - 0.18, x + 0.18, 5)
            ys = [y, y + 0.08, y - 0.08, y + 0.08, y]
            ax.plot(xs, ys, color=colors["wire_color"], lw=1.8, zorder=4)
        elif comp_type == "Capacitor":
            # Parallel plates
            ax.plot([x - 0.06, x - 0.06], [y - 0.1, y + 0.1], color=colors["wire_color"], lw=2.2, zorder=4)
            ax.plot([x + 0.06, x + 0.06], [y - 0.1, y + 0.1], color=colors["wire_color"], lw=2.2, zorder=4)
        elif comp_type == "Inductor":
            # Arcs
            t = np.linspace(0, np.pi, 20)
            ax.plot(x - 0.08 + 0.06 * np.cos(t), y + 0.06 * np.sin(t), color=colors["wire_color"], lw=1.8, zorder=4)
            ax.plot(x + 0.04 + 0.06 * np.cos(t), y + 0.06 * np.sin(t), color=colors["wire_color"], lw=1.8, zorder=4)
        elif comp_type == "Diode":
            # Triangle + Bar
            tri = patches.Polygon([[x - 0.08, y - 0.08], [x - 0.08, y + 0.08], [x + 0.05, y]], closed=True, color=colors["wire_color"], zorder=4)
            ax.add_patch(tri)
            ax.plot([x + 0.06, x + 0.06], [y - 0.09, y + 0.09], color=colors["wire_color"], lw=2, zorder=4)
        elif comp_type == "Ground":
            # Tapered horizontal bars
            ax.plot([x - 0.12, x + 0.12], [y + 0.04, y + 0.04], color=colors["wire_color"], lw=2, zorder=4)
            ax.plot([x - 0.07, x + 0.07], [y, y], color=colors["wire_color"], lw=1.8, zorder=4)
            ax.plot([x - 0.03, x + 0.03], [y - 0.04, y - 0.04], color=colors["wire_color"], lw=1.5, zorder=4)
        elif comp_type == "Switch":
            # Break with angled arm
            ax.plot([x - 0.12, x - 0.04], [y, y], color=colors["wire_color"], lw=1.8, zorder=4)
            ax.plot([x - 0.04, x + 0.08], [y, y + 0.08], color=colors["accent_color"], lw=2.0, zorder=4)
            ax.plot([x + 0.08, x + 0.14], [y, y], color=colors["wire_color"], lw=1.8, zorder=4)
        else:
            # Default Circle/Plate
            circle = patches.Circle((x, y), 0.08, color=colors["accent_color"], zorder=4)
            ax.add_patch(circle)

    def draw_manhattan_wire(self, ax, p1: tuple[float, float], p2: tuple[float, float], colors: dict):
        """Draws orthogonal (right-angled) electrical wiring between two components."""
        x1, y1 = p1
        x2, y2 = p2
        
        # Orthogonal bend at midpoint
        mid_x = (x1 + x2) / 2
        ax.plot([x1, mid_x, mid_x, x2], [y1, y1, y2, y2], color=colors["wire_color"], lw=2.0, zorder=2)
        
        # Solder junction dots
        ax.scatter([x1, x2], [y1, y2], color=colors["wire_color"], s=20, zorder=2)

    def render_schematic(self, G: nx.Graph, filename: str, style: RenderStyle = RenderStyle.BLUEPRINT, add_noise: bool = False) -> str:
        """Renders the graph into an image file and returns the full saved file path."""
        colors = THEME_CONFIGS[style]
        
        fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
        fig.patch.set_facecolor(colors["bg_color"])
        ax.set_facecolor(colors["bg_color"])
        
        # Canvas bounds
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 4)
        
        # Technical engineering grid
        ax.grid(True, which='both', color=colors["grid_color"], linestyle='--', linewidth=0.7, alpha=0.8)
        ax.set_xticks(np.arange(0, 5, 0.5))
        ax.set_yticks(np.arange(0, 4, 0.5))
        ax.tick_params(colors=colors["grid_color"], labelsize=7)
        
        # Outer border
        border = patches.Rectangle((0.15, 0.15), 4.7, 3.7, fill=False, edgecolor=colors["border_color"], lw=2, zorder=5)
        ax.add_patch(border)

        # Title Block (Bottom-right industrial blueprint stamp)
        title_box = patches.Rectangle((3.0, 0.15), 1.85, 0.6, fill=True, facecolor=colors["node_fill"], edgecolor=colors["border_color"], lw=1.2, zorder=5)
        ax.add_patch(title_box)
        ax.text(3.05, 0.58, "INNODATA TASK FACTORY: SCHEM-AI", fontsize=7, fontweight='bold', color=colors["text_color"], zorder=6)
        ax.text(3.05, 0.43, f"REF-HASH: {abs(hash(filename)) % 1000000:06d} | REV: 2.4A", fontsize=6, color=colors["text_color"], zorder=6)
        ax.text(3.05, 0.28, "VALIDATION: MATHEMATICALLY VERIFIED", fontsize=6, fontweight='bold', color=colors["accent_color"], zorder=6)

        # Draw all wires (edges)
        for u, v in G.edges():
            pos_u = G.nodes[u]['pos']
            pos_v = G.nodes[v]['pos']
            self.draw_manhattan_wire(ax, pos_u, pos_v, colors)

        # Draw components and labels (nodes)
        for node_id, data in G.nodes(data=True):
            pos = data['pos']
            comp_type = data['type']
            label = data['label']
            
            self.draw_component_symbol(ax, comp_type, pos, colors)
            
            # Label placement above component
            ax.text(
                pos[0], pos[1] + 0.26,
                label,
                fontsize=9,
                fontweight='bold',
                color=colors["text_color"],
                ha='center',
                va='bottom',
                zorder=7,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=colors["bg_color"], edgecolor=colors["grid_color"], lw=0.5, alpha=0.9)
            )

        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        # Apply realistic environmental visual distortion if requested
        if add_noise:
            with Image.open(filepath) as img:
                # Add slight camera lens blur or subtle grain
                noisy_img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
                noisy_img.save(filepath)

        return filepath
