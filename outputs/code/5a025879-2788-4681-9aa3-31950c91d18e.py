# Black Hole Formation Visualization
from manim import *
import numpy as np

class BlackHoleFormation(Scene):
    def construct(self):
        config.pixel_height = 720
        config.pixel_width = 1280
        config.frame_width = 14
        config.frame_height = 8

        # Star initial state
        star = Sphere(radius=1.5, color=YELLOW)
        star.set_opacity(0.7)
        star.set_color(color=[YELLOW, ORANGE, RED])

        # Plasma waves simulation
        plasma_points = VGroup()
        for _ in range(50):
            point = Dot(radius=0.05, color=ORANGE)
            point.move_to(star.get_center() + np.random.uniform(-0.5, 0.5, 3))
            plasma_points.add(point)

        # Gravitational well grid
        grid = NumberPlane(
            x_range=[-5, 5, 1], 
            y_range=[-5, 5, 1],
            background_line_style={"stroke_opacity": 0.4}
        )

        # Temperature graph
        temp_graph = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 1000, 100],
            x_axis_config={"include_tip": False},
            y_axis_config={"include_tip": False}
        )
        temp_label = Text("Temperature (K)").next_to(temp_graph, DOWN)

        # Gravitational force graph
        force_graph = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 100, 10],
            x_axis_config={"include_tip": False},
            y_axis_config={"include_tip": False}
        )
        force_label = Text("Gravitational Force").next_to(force_graph, DOWN)

        # Animation sequence
        self.play(Create(star), run_time=2)
        self.play(Create(plasma_points), run_time=1.5)
        
        # Star collapse
        self.play(
            star.animate.scale(0.3),
            plasma_points.animate.scale(0.3),
            run_time=3
        )

        # Gravitational well formation
        self.play(Create(grid), run_time=2)
        self.play(
            grid.animate.rotate(PI/4),
            grid.animate.set_color(BLUE_D),
            run_time=2
        )

        # Black hole formation
        black_hole = Sphere(radius=0.5, color=BLACK, stroke_color=WHITE)
        accretion_disk = Annulus(
            inner_radius=0.6, 
            outer_radius=1.2, 
            color=RED_D
        )

        self.play(
            ReplacementTransform(star, black_hole),
            Create(accretion_disk),
            run_time=3
        )

        # Final dramatic zoom
        self.play(
            self.camera.frame.animate.move_to(black_hole).set_height(3),
            run_time=2
        )

        self.wait(1)