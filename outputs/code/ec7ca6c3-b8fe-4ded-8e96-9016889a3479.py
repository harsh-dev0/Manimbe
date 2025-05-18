# Force Error Scenario Test Visualization
from manim import *
import numpy as np

class ForceErrorScenario(Scene):
    def construct(self):
        config.pixel_height = 720
        config.pixel_width = 1280
        config.frame_width = 14
        config.frame_height = 8

        # Title
        title = Text("Force Error Scenario Analysis", font_size=36, color=BLUE).to_edge(UP)
        
        # Create error representation
        error_box = Rectangle(
            width=6, 
            height=3, 
            stroke_color=RED, 
            fill_color=DARK_BLUE, 
            fill_opacity=0.2
        ).center()

        # Error description
        error_text = Text(
            "Potential Error Zones\n- Input Validation\n- Boundary Conditions\n- Unexpected Inputs", 
            font_size=24, 
            color=YELLOW
        ).move_to(error_box.get_center())

        # Arrows indicating error paths
        arrow_left = Arrow(start=error_box.get_left(), end=error_box.get_left() + LEFT*2, color=RED)
        arrow_right = Arrow(start=error_box.get_right(), end=error_box.get_right() + RIGHT*2, color=RED)

        # Animation sequence
        self.play(Write(title))
        self.wait(0.5)
        self.play(Create(error_box))
        self.play(Write(error_text))
        self.play(Create(arrow_left), Create(arrow_right))
        self.wait(1)