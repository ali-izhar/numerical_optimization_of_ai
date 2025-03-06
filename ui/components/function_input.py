"""Function input component for the numerical optimization UI."""

from PyQt6.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QDoubleSpinBox,
    QCheckBox,
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPixmap, QImage
import numpy as np
import re
import traceback
from io import BytesIO

# Matplotlib imports needed for LaTeX rendering
from matplotlib.figure import Figure
from matplotlib import patheffects
import matplotlib.pyplot as plt


class FunctionInput(QGroupBox):
    """Custom widget for function input with validation."""

    def __init__(self, parent=None):
        super().__init__("Function Definition", parent)
        self._latex_update_timer = QTimer()
        self._latex_update_timer.setSingleShot(True)
        self._latex_update_timer.timeout.connect(self._update_latex_display)
        self.setup_ui()
        self.update_derivative_requirements("Newton's Method")  # Default method

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)  # Reduced spacing between elements

        # Function input section with updated styling
        input_layout = QHBoxLayout()
        func_label = QLabel("f(x) =")
        func_label.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 11px;")
        input_layout.addWidget(func_label)

        self.func_input = QLineEdit()
        # Himmelblau function as default
        self.func_input.setText("(x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2")
        self.func_input.setPlaceholderText(
            "Python-compatible function, e.g., x1**2 + x2**2"
        )
        self.func_input.textChanged.connect(self._schedule_latex_update)
        self.func_input.setStyleSheet(
            """
            QLineEdit {
                background-color: #222222;
                color: #ffffff;
                border: 1px solid #333333;
                border-radius: 3px;
                padding: 5px;
                selection-background-color: #4a90e2;
            }
            QLineEdit:focus {
                border: 1px solid #4a90e2;
            }
        """
        )
        input_layout.addWidget(self.func_input)
        layout.addLayout(input_layout)

        # LaTeX display with updated styling
        self.latex_display = QLabel()
        self.latex_display.setStyleSheet(
            """
            QLabel {
                background-color: #111111;
                border: 1px solid #333333;
                border-radius: 3px;
                padding: 10px;
                min-height: 80px;
                margin: 5px 0;
            }
        """
        )
        self.latex_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.latex_display)

        # Variable inputs and initial guess with updated styling
        var_layout = QHBoxLayout()
        var_layout.setSpacing(10)  # Reduced spacing between variable sections
        self.var_inputs = []
        self.initial_guess_inputs = []

        for i in range(2):
            var_group = QGroupBox(f"x{i+1} settings")
            var_group.setStyleSheet(
                """
                QGroupBox {
                    font-weight: bold;
                    color: #ffffff;
                    font-size: 10px;
                }
            """
            )
            var_layout_inner = QVBoxLayout()
            var_layout_inner.setSpacing(5)  # Reduced spacing between elements

            # Range inputs with updated styling
            range_layout = QHBoxLayout()
            min_spin = QDoubleSpinBox()
            min_spin.setRange(-1000, 1000)
            min_spin.setValue(-5)
            min_spin.setStyleSheet(
                """
                QDoubleSpinBox {
                    padding-right: 10px;
                    background-color: #222222;
                    color: #ffffff;
                    border: 1px solid #333333;
                    border-radius: 3px;
                }
            """
            )

            max_spin = QDoubleSpinBox()
            max_spin.setRange(-1000, 1000)
            max_spin.setValue(5)
            max_spin.setStyleSheet(
                """
                QDoubleSpinBox {
                    padding-right: 10px;
                    background-color: #222222;
                    color: #ffffff;
                    border: 1px solid #333333;
                    border-radius: 3px;
                }
            """
            )

            # Updated labels
            min_label = QLabel("Min:")
            min_label.setStyleSheet("color: #b3b3b3; font-size: 10px;")
            max_label = QLabel("Max:")
            max_label.setStyleSheet("color: #b3b3b3; font-size: 10px;")

            range_layout.addWidget(min_label)
            range_layout.addWidget(min_spin)
            range_layout.addWidget(max_label)
            range_layout.addWidget(max_spin)
            var_layout_inner.addLayout(range_layout)

            # Initial guess input with updated styling
            guess_layout = QHBoxLayout()
            guess_spin = QDoubleSpinBox()
            guess_spin.setRange(-1000, 1000)
            guess_spin.setValue(0)  # Default to 0
            guess_spin.setStyleSheet(
                """
                QDoubleSpinBox {
                    padding-right: 10px;
                    background-color: #222222;
                    color: #ffffff;
                    border: 1px solid #333333;
                    border-radius: 3px;
                }
            """
            )

            # Updated label
            initial_label = QLabel("Initial:")
            initial_label.setStyleSheet("color: #b3b3b3; font-size: 10px;")

            guess_layout.addWidget(initial_label)
            guess_layout.addWidget(guess_spin)
            var_layout_inner.addLayout(guess_layout)

            var_group.setLayout(var_layout_inner)
            var_layout.addWidget(var_group)
            self.var_inputs.append((min_spin, max_spin))
            self.initial_guess_inputs.append(guess_spin)

        layout.addLayout(var_layout)

        # Derivatives with updated styling
        deriv_group = QGroupBox("Derivatives")
        deriv_group.setStyleSheet(
            """
            QGroupBox {
                font-weight: bold;
                color: #ffffff;
                font-size: 10px;
            }
        """
        )
        deriv_layout = QVBoxLayout()
        deriv_layout.setSpacing(5)  # Reduced spacing

        # Create more compact derivatives section
        self.show_deriv = QCheckBox("Show first derivative")
        self.show_second_deriv = QCheckBox("Show second derivative")
        self.show_second_deriv.setChecked(
            True
        )  # Enable second derivative for Newton's method

        # Add updated styling to checkboxes
        checkbox_style = """
            QCheckBox {
                color: #b3b3b3;
                font-size: 10px;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 12px;
                height: 12px;
                border-radius: 2px;
                border: 1px solid #333333;
            }
            QCheckBox::indicator:checked {
                background-color: #4a90e2;
            }
        """
        self.show_deriv.setStyleSheet(checkbox_style)
        self.show_second_deriv.setStyleSheet(checkbox_style)

        deriv_layout.addWidget(self.show_deriv)
        deriv_layout.addWidget(self.show_second_deriv)

        deriv_group.setLayout(deriv_layout)
        layout.addWidget(deriv_group)

        self.setLayout(layout)

        # Initial LaTeX render
        self._update_latex_display()

    def _schedule_latex_update(self):
        """Schedule a LaTeX update after a short delay to prevent excessive updates."""
        self._latex_update_timer.start(500)  # 500ms delay

    def _update_latex_display(self):
        """Update the LaTeX display with the current function."""
        try:
            # Get the function text and convert Python notation to LaTeX
            func_text = self.func_input.text()

            # Basic replacements for common mathematical notation
            latex_text = func_text

            # Handle parentheses groups first
            def replace_in_parentheses(match):
                content = match.group(1)
                # Handle powers inside parentheses
                content = re.sub(r"\*\*(\d+)", r"^{\1}", content)
                return f"({content})"

            # Replace contents in parentheses
            latex_text = re.sub(r"\(([^()]+)\)", replace_in_parentheses, latex_text)

            # Handle remaining powers
            latex_text = re.sub(r"\*\*(\d+)", r"^{\1}", latex_text)

            # Handle multiplication
            latex_text = re.sub(r"(?<=\d)\*", r" \\cdot ", latex_text)  # Number * ...
            latex_text = re.sub(r"\*(?=\d)", r" \\cdot ", latex_text)  # ... * Number
            latex_text = re.sub(r"(?<=[x])\*", r" \\cdot ", latex_text)  # x * ...
            latex_text = re.sub(r"\*(?=[x])", r" \\cdot ", latex_text)  # ... * x
            latex_text = re.sub(r"\*", r" \\cdot ", latex_text)  # Remaining *

            # Handle variables with subscripts
            latex_text = re.sub(r"x1", r"x_1", latex_text)
            latex_text = re.sub(r"x2", r"x_2", latex_text)

            # Remove extra spaces around operators
            latex_text = re.sub(r"\s*([+\-])\s*", r" \1 ", latex_text)
            latex_text = re.sub(r"\s*\\cdot\s*", r" \\cdot ", latex_text)

            # Create a figure with fixed size
            width = self.latex_display.width() / 80  # Adjusted for better scaling
            height = 1.0  # Reduced height from 2.0 to 1.0 for more compact display
            fig = Figure(figsize=(width, height))
            fig.patch.set_facecolor("#242935")

            # Create axes that fill the figure
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_axis_off()
            ax.set_facecolor("#242935")

            # Render the LaTeX equation with sharper font
            eq = ax.text(
                0.5,
                0.5,
                f"$f(x) = {latex_text}$",
                color="#00ffff",
                fontsize=22,  # Slightly reduced font size for compactness
                horizontalalignment="center",
                verticalalignment="center",
                fontweight="bold",
                family="serif",  # Use serif font for sharper appearance
            )

            # Add a stronger glow effect
            eq.set_path_effects(
                [
                    patheffects.withStroke(linewidth=4, foreground="#242935"),
                    patheffects.Normal(),
                    patheffects.withStroke(
                        linewidth=2, foreground="#00ffff", alpha=0.3
                    ),  # Added subtle cyan glow
                ]
            )

            # Convert matplotlib figure to QPixmap with higher DPI for sharper rendering
            buf = BytesIO()
            fig.savefig(
                buf,
                format="png",
                facecolor="#242935",
                transparent=True,
                bbox_inches="tight",
                pad_inches=0.2,  # Reduced padding
                dpi=300,  # Increased DPI from 200 to 300 for sharper text
            )
            buf.seek(0)

            # Create QImage from buffer
            image = QImage.fromData(buf.getvalue())
            pixmap = QPixmap.fromImage(image)

            # Set a fixed height for the label while maintaining aspect ratio
            fixed_height = 80  # Reduced height from 120 to 80
            scaled_pixmap = pixmap.scaledToHeight(
                fixed_height, Qt.TransformationMode.SmoothTransformation
            )

            # Update the label
            self.latex_display.setPixmap(scaled_pixmap)

            # Clean up
            plt.close(fig)
            buf.close()

        except Exception as e:
            # If there's an error, show a simple message
            self.latex_display.setText("Invalid expression")
            print(f"LaTeX rendering error: {str(e)}")
            traceback.print_exc()

    def resizeEvent(self, event):
        """Handle resize events to update the LaTeX display size."""
        super().resizeEvent(event)
        # Only update if we have a valid pixmap
        if not self.latex_display.pixmap().isNull():
            self._update_latex_display()

    def update_derivative_requirements(self, method_name):
        """Update derivative checkboxes based on method requirements."""
        # Reset styles
        self.show_deriv.setStyleSheet("")
        self.show_second_deriv.setStyleSheet("")

        # Define requirements for each method
        requirements = {
            "Newton's Method": {"first": True, "second": True},
            "BFGS": {"first": True, "second": False},
            "Steepest Descent": {"first": True, "second": False},
            "Nelder-Mead": {"first": False, "second": False},
            "Powell's Method": {"first": False, "second": False},
        }

        # Base checkbox style
        checkbox_style = """
            QCheckBox {
                color: #b3b3b3;
                font-size: 10px;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 12px;
                height: 12px;
                border-radius: 2px;
                border: 1px solid #333333;
            }
            QCheckBox::indicator:checked {
                background-color: #4a90e2;
            }
        """

        if method_name in requirements:
            req = requirements[method_name]

            # First derivative
            if req["first"]:
                self.show_deriv.setChecked(True)
                self.show_deriv.setStyleSheet(
                    checkbox_style
                    + """
                    QCheckBox {
                        color: #4a90e2;
                        font-weight: bold;
                    }
                """
                )
            else:
                # Apply the base style
                self.show_deriv.setStyleSheet(checkbox_style)

            # Second derivative
            if req["second"]:
                self.show_second_deriv.setChecked(True)
                self.show_second_deriv.setStyleSheet(
                    checkbox_style
                    + """
                    QCheckBox {
                        color: #4a90e2;
                        font-weight: bold;
                    }
                """
                )
            else:
                # Apply the base style
                self.show_second_deriv.setStyleSheet(checkbox_style)

    def get_initial_guess(self):
        """Get the initial guess values."""
        return np.array([spin.value() for spin in self.initial_guess_inputs])
