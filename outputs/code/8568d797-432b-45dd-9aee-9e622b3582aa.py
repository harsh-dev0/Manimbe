# Sine Wave Animation
from manim import *
import numpy as np

class SineWave(Scene):
    def construct(self):
        # Set camera resolution and frame size
        config.pixel_height = 720
        config.pixel_width = 1280
        config.frame_width = 14
        config.frame_height = 8

        # Create axes
        axes = Axes(
            x_range=[-2 * PI, 2 * PI, PI / 2],
            y_range=[-1.5, 1.5, 0.5],
            x_length=10,
            y_length=6,
            axis_config={"color": BLUE},
            tips=True
        ).center() # Center the axes on the screen

        # Add axis labels manually
        x_label = MathTex("x").next_to(axes.x_axis.get_end(), DOWN, buff=0.1)
        y_label = MathTex("y").next_to(axes.y_axis.get_end(), LEFT, buff=0.1)
        axes_labels = VGroup(x_label, y_label)

        # Define the function for sine wave
        def func(x):
            return np.sin(x)

        # Plot the sine wave
        graph = axes.plot(func, color=YELLOW)

        # Create the equation text
        eq_text = MathTex("y = \sin(x)", color=GREEN).to_edge(UP).shift(LEFT * 2)

        # Create an introductory text
        intro_text = Text("Visualizing y = sin(x)", font_size=48).to_edge(UP).shift(RIGHT * 2)

        # Animations
        self.play(Write(intro_text))
        self.wait(0.5)
        self.play(
            FadeOut(intro_text),
            Create(axes),
            Write(axes_labels),
            run_time=2
        )
        self.play(Write(eq_text))
        self.wait(0.5)
        self.play(Create(graph), run_time=3)
        self.wait(1)