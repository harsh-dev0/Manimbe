# Binary Search Algorithm Visualization
from manim import *
import numpy as np

class BinarySearch(Scene):
    def construct(self):
        config.pixel_height = 720
        config.pixel_width = 1280
        config.frame_width = 14
        config.frame_height = 8

        # Sorted array for binary search
        array = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        target = 60

        # Create array elements
        array_squares = VGroup()
        for i, num in enumerate(array):
            square = Square(side_length=1).set_fill(BLUE, opacity=0.5)
            text = Text(str(num)).move_to(square.get_center())
            group = VGroup(square, text)
            group.move_to(RIGHT * (i - len(array)/2 + 0.5) * 1.2)
            array_squares.add(group)

        # Title and explanation
        title = Text("Binary Search Algorithm", font_size=36).to_edge(UP)
        target_text = Text(f"Target: {target}", font_size=24).next_to(title, DOWN)

        # Add initial array
        self.play(Write(title), Write(target_text))
        self.play(Create(array_squares))
        self.wait(1)

        # Binary search visualization
        left = 0
        right = len(array) - 1

        while left <= right:
            mid = (left + right) // 2
            mid_square = array_squares[mid][0]

            # Highlight middle element
            self.play(mid_square.animate.set_fill(GREEN, opacity=0.7))
            self.wait(0.5)

            mid_value = array[mid]
            
            if mid_value == target:
                # Target found
                self.play(mid_square.animate.set_fill(RED, opacity=0.8))
                found_text = Text(f"Target {target} found at index {mid}", font_size=24).next_to(target_text, DOWN)
                self.play(Write(found_text))
                break
            elif mid_value < target:
                # Search right half
                left = mid + 1
                self.play(mid_square.animate.set_fill(BLUE, opacity=0.5))
                for i in range(left):
                    array_squares[i][0].set_fill(GRAY, opacity=0.3)
            else:
                # Search left half
                right = mid - 1
                self.play(mid_square.animate.set_fill(BLUE, opacity=0.5))
                for i in range(mid + 1, len(array)):
                    array_squares[i][0].set_fill(GRAY, opacity=0.3)

            self.wait(0.5)

        self.wait(1)