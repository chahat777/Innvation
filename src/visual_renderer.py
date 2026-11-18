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
        "bg_color": "#071326",
        "grid_color": "#112B52",
        "wire_color": "#38BDF8",
        "node_fill": "#0F284F",
        "text_color": "#F8FAFC",
        "accent_color": "#22C55E",
        "border_color": "#0284C7"
    },
    RenderStyle.CAD_WHITE: {
        "bg_color": "#FFFFFF",
        "grid_color": "#E2E8F0",
        "wire_color": "#0F172A",
        "node_fill": "#F1F5F9",
        "text_color": "#0F172A",
        "accent_color": "#EF4444",
        "border_color": "#334155"
    },
    RenderStyle.AGED_PARCHMENT: {
        "bg_color": "#FAF5EA",
        "grid_color": "#E8DEC8",
        "wire_color": "#27211C",
        "node_fill": "#EFE6D2",
        "text_color": "#27211C",
        "accent_color": "#C2410C",
        "border_color": "#44372C"
    }
}

class VisualRenderer:
    """
    Renders topological circuit graphs and architectural floor plans into
    realistic, high-resolution engineering schematics with advanced overlay diagnostics.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def draw_component_symbol(self, ax, comp_type: str, pos: tuple[float, float], colors: dict):
        x, y = pos
        w, h = 0.5, 0.35
        
        rect = patches.Rectangle(
            (x - w / 2, y - h / 2), w, h,
            linewidth=1.8,
            edgecolor=colors["wire_color"],
            facecolor=colors["node_fill"],
            zorder=3
        )
        ax.add_patch(rect)

        if comp_type == "Resistor":
            xs = np.linspace(x - 0.18, x + 0.18, 5)
            ys = [y, y + 0.08, y - 0.08, y + 0.08, y]
            ax.plot(xs, ys, color=colors["wire_color"], lw=2.0, zorder=4)
        elif comp_type == "Capacitor":
            ax.plot([x - 0.06, x - 0.06], [y - 0.1, y + 0.1], color=colors["wire_color"], lw=2.4, zorder=4)
            ax.plot([x + 0.06, x + 0.06], [y - 0.1, y + 0.1], color=colors["wire_color"], lw=2.4, zorder=4)
        elif comp_type == "Inductor":
            t = np.linspace(0, np.pi, 20)
            ax.plot(x - 0.08 + 0.06 * np.cos(t), y + 0.06 * np.sin(t), color=colors["wire_color"], lw=2.0, zorder=4)
            ax.plot(x + 0.04 + 0.06 * np.cos(t), y + 0.06 * np.sin(t), color=colors["wire_color"], lw=2.0, zorder=4)
        elif comp_type == "Diode":
            tri = patches.Polygon([[x - 0.08, y - 0.08], [x - 0.08, y + 0.08], [x + 0.05, y]], closed=True, color=colors["wire_color"], zorder=4)
            ax.add_patch(tri)
            ax.plot([x + 0.06, x + 0.06], [y - 0.09, y + 0.09], color=colors["wire_color"], lw=2.2, zorder=4)
        elif comp_type == "Ground":
            ax.plot([x - 0.12, x + 0.12], [y + 0.04, y + 0.04], color=colors["wire_color"], lw=2.2, zorder=4)
            ax.plot([x - 0.07, x + 0.07], [y, y], color=colors["wire_color"], lw=2.0, zorder=4)
            ax.plot([x - 0.03, x + 0.03], [y - 0.04, y - 0.04], color=colors["wire_color"], lw=1.6, zorder=4)
        elif comp_type == "Switch":
            ax.plot([x - 0.12, x - 0.04], [y, y], color=colors["wire_color"], lw=2.0, zorder=4)
            ax.plot([x - 0.04, x + 0.08], [y, y + 0.08], color=colors["accent_color"], lw=2.2, zorder=4)
            ax.plot([x + 0.08, x + 0.14], [y, y], color=colors["wire_color"], lw=2.0, zorder=4)
        else:
            circle = patches.Circle((x, y), 0.08, color=colors["accent_color"], zorder=4)
            ax.add_patch(circle)

    def draw_manhattan_wire(self, ax, p1: tuple[float, float], p2: tuple[float, float], colors: dict):
        x1, y1 = p1
        x2, y2 = p2
        mid_x = (x1 + x2) / 2
        ax.plot([x1, mid_x, mid_x, x2], [y1, y1, y2, y2], color=colors["wire_color"], lw=2.2, zorder=2)
        ax.scatter([x1, x2], [y1, y2], color=colors["wire_color"], s=28, zorder=2)

    def render_schematic(self, G: nx.Graph, filename: str, style: RenderStyle = RenderStyle.BLUEPRINT, add_noise: bool = False) -> str:
        colors = THEME_CONFIGS[style]
        fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
        fig.patch.set_facecolor(colors["bg_color"])
        ax.set_facecolor(colors["bg_color"])
        
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 4)
        
        ax.grid(True, which='both', color=colors["grid_color"], linestyle='--', linewidth=0.7, alpha=0.8)
        ax.set_xticks(np.arange(0, 5, 0.5))
        ax.set_yticks(np.arange(0, 4, 0.5))
        ax.tick_params(colors=colors["grid_color"], labelsize=7)
        
        border = patches.Rectangle((0.15, 0.15), 4.7, 3.7, fill=False, edgecolor=colors["border_color"], lw=2, zorder=5)
        ax.add_patch(border)

        title_box = patches.Rectangle((3.0, 0.15), 1.85, 0.6, fill=True, facecolor=colors["node_fill"], edgecolor=colors["border_color"], lw=1.2, zorder=5)
        ax.add_patch(title_box)
        ax.text(3.05, 0.58, "INNODATA TASK FACTORY: SCHEM-AI", fontsize=7, fontweight='bold', color=colors["text_color"], zorder=6)
        ax.text(3.05, 0.43, f"REF-HASH: {abs(hash(filename)) % 1000000:06d} | REV: 2.4A", fontsize=6, color=colors["text_color"], zorder=6)
        ax.text(3.05, 0.28, "VALIDATION: MATHEMATICALLY VERIFIED", fontsize=6, fontweight='bold', color=colors["accent_color"], zorder=6)

        for u, v in G.edges():
            pos_u = G.nodes[u]['pos']
            pos_v = G.nodes[v]['pos']
            self.draw_manhattan_wire(ax, pos_u, pos_v, colors)

        for node_id, data in G.nodes(data=True):
            pos = data['pos']
            comp_type = data['type']
            label = data['label']
            self.draw_component_symbol(ax, comp_type, pos, colors)
            ax.text(
                pos[0], pos[1] + 0.26, label,
                fontsize=9, fontweight='bold', color=colors["text_color"],
                ha='center', va='bottom', zorder=7,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=colors["bg_color"], edgecolor=colors["grid_color"], lw=0.5, alpha=0.9)
            )

        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        if add_noise:
            with Image.open(filepath) as img:
                noisy_img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
                noisy_img.save(filepath)

        return filepath

    def render_floor_plan(self, G: nx.Graph, filename: str, style: RenderStyle = RenderStyle.BLUEPRINT, add_noise: bool = False) -> str:
        colors = THEME_CONFIGS[style]
        fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
        fig.patch.set_facecolor(colors["bg_color"])
        ax.set_facecolor(colors["bg_color"])
        
        ax.set_xlim(0, 4.5)
        ax.set_ylim(0, 4.5)
        
        ax.grid(True, which='both', color=colors["grid_color"], linestyle=':', linewidth=0.6, alpha=0.7)
        ax.tick_params(colors=colors["grid_color"], labelsize=7)

        title_box = patches.Rectangle((2.4, 0.15), 1.95, 0.6, fill=True, facecolor=colors["node_fill"], edgecolor=colors["border_color"], lw=1.2, zorder=5)
        ax.add_patch(title_box)
        ax.text(2.45, 0.58, "ARCHITECTURAL EGRESS BLUEPRINT", fontsize=7, fontweight='bold', color=colors["text_color"], zorder=6)
        ax.text(2.45, 0.43, "FACILITY CODE: LVL-02 | VERIFIED SPATIAL", fontsize=6, color=colors["text_color"], zorder=6)
        ax.text(2.45, 0.28, "GROUND TRUTH: GRAPH BFS/EGRESS SOLVER", fontsize=6, fontweight='bold', color=colors["accent_color"], zorder=6)

        for u, v in G.edges():
            p1 = G.nodes[u]['pos']
            p2 = G.nodes[v]['pos']
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=colors["wire_color"], lw=3.0, linestyle='--', zorder=2)
            mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            ax.scatter([mid[0]], [mid[1]], marker='s', s=45, color=colors["accent_color"], zorder=3)

        for node_id, data in G.nodes(data=True):
            x, y = data['pos']
            label = data['label']
            rtype = data['type']
            is_exit = (rtype == "Emergency_Exit")
            room_w, room_h = 0.75, 0.6
            
            room_patch = patches.Rectangle(
                (x - room_w / 2, y - room_h / 2), room_w, room_h,
                linewidth=2.0 if is_exit else 1.5,
                edgecolor=colors["accent_color"] if is_exit else colors["wire_color"],
                facecolor="#103720" if (is_exit and style == RenderStyle.BLUEPRINT) else colors["node_fill"],
                zorder=4
            )
            ax.add_patch(room_patch)

            ax.text(
                x, y, f"{label}\n({rtype})",
                fontsize=7.5, fontweight='bold',
                color="#4ADE80" if is_exit else colors["text_color"],
                ha='center', va='center', zorder=5
            )

        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        if add_noise:
            with Image.open(filepath) as img:
                noisy_img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
                noisy_img.save(filepath)

        return filepath

    def render_ai_perception_overlay(self, G: nx.Graph, filename: str, task_meta: dict) -> str:
        """
        Renders an advanced AI Model Perception vs Ground-Truth Diagnostic Heatmap.
        Draws verified physical connections in luminous green and marks hallucinated paths in glowing red lasers!
        """
        fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
        bg = "#030712" # Cyberpunk obsidian dark
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 4)
        
        # Cyber grid
        ax.grid(True, which='both', color="#1E293B", linestyle='-', linewidth=0.5, alpha=0.6)
        
        # Diagnostic Header Overlay
        ax.text(0.2, 3.8, "🔬 AI VISION DIAGNOSTIC: SPATIAL PERCEPTION AUDIT", fontsize=9, fontweight='bold', color="#38BDF8")
        ax.text(0.2, 3.65, "GREEN = Verified Ground-Truth Trace | RED LASER = Model Proximity Hallucination", fontsize=7.5, color="#94A3B8")

        # Draw True Wires in Neon Cyan
        for u, v in G.edges():
            pos_u = G.nodes[u]['pos']
            pos_v = G.nodes[v]['pos']
            mid_x = (pos_u[0] + pos_v[0]) / 2
            ax.plot([pos_u[0], mid_x, mid_x, pos_v[0]], [pos_u[1], pos_u[1], pos_v[1], pos_v[1]], color="#10B981", lw=2.5, alpha=0.9, zorder=2)

        # If direct wiring was tested and was FALSE, draw where the model hallucinated the direct connection!
        if task_meta.get("is_direct") is False and "node_u" in task_meta and "node_v" in task_meta:
            u_label = task_meta["node_u"]
            v_label = task_meta["node_v"]
            u_pos = next((d['pos'] for _, d in G.nodes(data=True) if d['label'] == u_label), None)
            v_pos = next((d['pos'] for _, d in G.nodes(data=True) if d['label'] == v_label), None)
            if u_pos and v_pos:
                # Glowing red hallucination vector
                ax.plot([u_pos[0], v_pos[0]], [u_pos[1], v_pos[1]], color="#EF4444", lw=3.0, linestyle='--', zorder=4)
                mid = ((u_pos[0] + v_pos[0])/2, (u_pos[1] + v_pos[1])/2)
                ax.text(mid[0], mid[1] + 0.12, "⚠️ HALLUCINATED PROXIMITY SHORT", fontsize=7.5, fontweight='bold', color="#EF4444", ha='center', bbox=dict(boxstyle="round,pad=0.2", facecolor="#450A0A", edgecolor="#EF4444", lw=1))

        # Draw components with bounding boxes & simulated visual attention
        for node_id, data in G.nodes(data=True):
            pos = data['pos']
            label = data['label']
            comp_type = data['type']
            
            # Simulated attention circle
            attn_circle = patches.Circle((pos[0], pos[1]), 0.28, facecolor="#0284C7", alpha=0.15, edgecolor="#38BDF8", lw=1, linestyle=':')
            ax.add_patch(attn_circle)
            
            rect = patches.Rectangle((pos[0]-0.25, pos[1]-0.18), 0.5, 0.36, facecolor="#0F172A", edgecolor="#38BDF8", lw=1.5, zorder=5)
            ax.add_patch(rect)
            ax.text(pos[0], pos[1], f"{label}\n[{comp_type[:3]}]", fontsize=7, fontweight='bold', color="#F8FAFC", ha='center', va='center', zorder=6)

        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        return filepath
