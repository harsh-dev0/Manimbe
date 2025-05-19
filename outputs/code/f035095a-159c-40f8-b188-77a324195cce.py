# Sine Function Visualization
from manim import *
import numpy as np

class SineFunctionDemo(Scene):
    def construct(self):
        config.pixel_height = 720
        config.pixel_width = 1280
        config.frame_width = 14
        config.frame_height = 8

        # Create axes
        axes = Axes(
            x_range=[-2*PI, 2*PI, PI/2],
            y_range=[-1.5, 1.5, 0.5],
            x_length=10,
            y_length=4
        ).center()

        # Sine function graph
        sine_graph = axes.plot(np.sin, color=BLUE)

        # Labels
        x_label = MathTex("x").next_to(axes.x_axis.get_end(), RIGHT)
        y_label = MathTex("y").next_to(axes.y_axis.get_end(), UP)

        # Equation
        equation = MathTex("y = \sin(x)", color=GREEN).to_edge(UP)

        # Animation sequence
        self.play(Create(axes))
        self.play(Write(x_label), Write(y_label))
        self.play(Write(equation))
        self.play(Create(sine_graph))
        self.wait(1)