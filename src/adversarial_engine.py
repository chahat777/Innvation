import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from typing import Dict, Any, List

class AdversarialStressEngine:
    """
    Adversarial Visual Red-Teaming & Stress Engine.
    Simulates real-world field conditions (tablet scans, paper folds, optical noise)
    and adversarial topological traps (unconnected overpass bypass bridges)
    to compute a model's Robustness Under Degradation (RUD) score.
    """

    @staticmethod
    def apply_mobile_camera_scan(image_path: str, output_path: str) -> str:
        """Simulates camera lens distortion, perspective warp, and ambient field lighting."""
        with Image.open(image_path).convert("RGB") as img:
            w, h = img.size
            
            # Subtle perspective warp (simulating phone photo angled at schematic on workbench)
            coeffs = [
                1.0, 0.04, -10,
                0.02, 1.0, -10,
                0.0001, 0.0001
            ]
            warped = img.transform((w, h), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)
            
            # Ambient illumination vignette
            vignette = Image.new('L', (w, h), 0)
            draw = ImageDraw.Draw(vignette)
            draw.ellipse([(-w*0.2, -h*0.2), (w*1.2, h*1.2)], fill=255)
            vignette = vignette.filter(ImageFilter.GaussianBlur(radius=w*0.15))
            
            # Blend
            final_img = Image.composite(warped, Image.new('RGB', (w, h), (15, 23, 42)), vignette)
            final_img.save(output_path)
            return output_path

    @staticmethod
    def apply_field_creases_and_wear(image_path: str, output_path: str) -> str:
        """Simulates physical workshop handling: paper fold creases across traces and coffee/smudge stains."""
        with Image.open(image_path).convert("RGB") as img:
            draw = ImageDraw.Draw(img)
            w, h = img.size
            
            # Diagonal fold crease across central wires
            draw.line([(0, h*0.35), (w, h*0.65)], fill=(200, 200, 200), width=2)
            draw.line([(0, h*0.35 + 1), (w, h*0.65 + 1)], fill=(40, 40, 40), width=1)
            
            # Vertical blueprint fold
            draw.line([(w*0.5, 0), (w*0.5, h)], fill=(180, 180, 180), width=1)
            
            # Subtle workshop fluid ring smudge
            smudge_mask = Image.new('L', (w, h), 0)
            s_draw = ImageDraw.Draw(smudge_mask)
            s_draw.ellipse([(w*0.6, h*0.4), (w*0.75, h*0.6)], outline=60, width=8)
            smudge_mask = smudge_mask.filter(ImageFilter.GaussianBlur(radius=5))
            
            img.paste(Image.new('RGB', (w, h), (110, 80, 50)), (0, 0), smudge_mask)
            
            # Add slight sensor grain
            grain = np.random.normal(0, 10, (h, w, 3)).astype(np.uint8)
            img_arr = np.clip(np.array(img).astype(np.int16) + grain, 0, 255).astype(np.uint8)
            
            res_img = Image.fromarray(img_arr)
            res_img.save(output_path)
            return output_path

    @classmethod
    def apply_overpass_bridge_trap(cls, image_path: str, output_path: str) -> str:
        """
        Adversarial Level 4 Transformation:
        Draws an electronic bridge jumper 'overpass' across two orthogonal conductive lines
        in the center of the diagram. The wires cross without contact.
        Tests if the VLM hallucinates a 4-way solder junction.
        Preserves original image and saves modified artifact to output_path.
        """
        with Image.open(image_path).convert("RGB") as img:
            draw = ImageDraw.Draw(img)
            w, h = img.size
            cx, cy = w // 2, h // 2
            
            line_color = (56, 189, 248) # Cyan blueprint wire
            # Vertical underlying trace
            draw.line([(cx, cy - 80), (cx, cy + 80)], fill=line_color, width=3)
            
            # Horizontal jumping trace over vertical
            cls.draw_overpass_bridge_trap(
                draw=draw,
                p1=(cx - 80, cy),
                p2=(cx + 80, cy),
                cross_pos=(cx, cy),
                color=line_color
            )
            
            img.save(output_path)
            return output_path

    @staticmethod
    def draw_overpass_bridge_trap(draw, p1: tuple[float, float], p2: tuple[float, float], cross_pos: tuple[float, float], color: tuple[int, int, int]):
        """
        Adversarial Optical Trap:
        Draws an electronic bridge 'hop' where wire 1 jumps OVER wire 2 in an arc.
        Humans know they are NOT connected. 2D VLMs invariably hallucinate an intersection.
        """
        cx, cy = cross_pos
        # Gap break on underlying wire
        draw.line([p1, (cx - 12, cy)], fill=color, width=3)
        draw.line([(cx + 12, cy), p2], fill=color, width=3)
        # Semicircular jumper overpass
        draw.arc([cx - 12, cy - 12, cx + 12, cy + 12], start=180, end=360, fill=color, width=3)
