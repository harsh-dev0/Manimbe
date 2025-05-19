# Derivative of Exponential Function Visualization
from manim import *
import numpy as np

class ExponentialDerivative(Scene):
    def construct(self):
        config.pixel_height = 720
        config.pixel_width = 1280
        config.frame_width = 14
        config.frame_height = 8

        # Create axes
        axes = Axes(
            x_range=[-2, 2, 0.5],
            y_range=[-1, 5, 1],
            x_length=10,
            y_length=6
        ).center()

        # Define exponential and derivative functions
        def exp_func(x):
            return np.exp(x)
        
        def derivative_func(x):
            return np.exp(x)

        # Plot original exponential function
        exp_graph = axes.plot(exp_func, color=BLUE, stroke_width=3)
        
        # Plot derivative function
        derivative_graph = axes.plot(derivative_func, color=RED, stroke_width=3)

        # Function labels
        exp_label = MathTex(r"f(x) = e^x", color=BLUE).to_edge(UP+RIGHT)
        derivative_label = MathTex(r"f'(x) = e^x", color=RED).next_to(exp_label, DOWN)

        # Title
        title = Text("Derivative of Exponential Function", font_size=30).to_edge(UP)

        # Animation sequence
        self.play(Create(axes))
        self.play(Create(title))
        self.wait(0.5)
        
        # Show original function
        self.play(Create(exp_graph), Write(exp_label))
        self.wait(1)
        
        # Show derivative function
        self.play(Create(derivative_graph), Write(derivative_label))
        self.wait(1)

        # Highlight key observation
        observation = Text("Derivative of e^x is itself!", 
                           font_size=24, 
                           color=GREEN).to_edge(DOWN)
        self.play(Write(observation))
        self.wait(1)