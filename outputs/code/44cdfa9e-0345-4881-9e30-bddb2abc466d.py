# Sine Wave Animation
from manim import *
import numpy as np

class SineWave(Scene):
    def construct(self):
        # Set camera resolution and frame dimensions
        config.pixel_height = 720
        config.pixel_width = 1280
        config.frame_width = 14
        config.frame_height = 8

        # Create axes
        axes = Axes(
            x_range=[-2 * PI, 2 * PI, PI / 2],
            y_range=[-1.5, 1.5, 0.5],
            x_axis_config={"numbers_to_include": [-PI, 0, PI]},
            y_axis_config={"numbers_to_include": [-1, 0, 1]},
            axis_config={"color": BLUE},
        ).center() # Center the axes on the screen

        # Add axis labels manually
        x_label = MathTex("x").next_to(axes.x_axis.get_end(), DOWN)
        y_label = MathTex("y").next_to(axes.y_axis.get_end(), LEFT)

        # Define the function for sine wave
        def func_sin(x):
            return np.sin(x)

        # Plot the sine wave
        graph = axes.plot(func_sin, color=YELLOW)

        # Create the equation text
        eq_text = MathTex("y = \\sin(x)", color=GREEN).to_edge(UP)

        # Animation sequence
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.wait(0.5)
        self.play(Write(eq_text))
        self.wait(0.5)
        self.play(Create(graph))
        self.wait(1)