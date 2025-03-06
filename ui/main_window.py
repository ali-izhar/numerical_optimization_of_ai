# ui/main_window.py

"""Main window for the numerical methods UI."""

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QTabWidget,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QGroupBox,
    QTextEdit,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPalette, QColor, QPixmap, QImage
import numpy as np
from sympy import sympify, symbols, diff, lambdify
import traceback
import re
from io import BytesIO
import time

# Plotly imports for interactive visualizations
import plotly.graph_objects as go
from plotly.offline import plot
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings

# Matplotlib imports needed for LaTeX rendering
from matplotlib.figure import Figure
from matplotlib import patheffects
import matplotlib.pyplot as plt

from .methods import METHOD_MAP, run_optimization
from .components import FunctionInput, MethodSelector
from .components.surface_plot import SurfacePlot
from .components.convergence_plot import ConvergencePlot
from .components.results_display import ResultsDisplay
from .components.tab_view import TabView


class MainWindow(QMainWindow):
    """Main window for the numerical methods UI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Numerical Methods Visualizer (use full screen - recommended)"
        )
        self.optimization_result = None

        # Initialize plot_colors before setup_ui to avoid errors
        self.plot_colors = {
            "background": "#000000",  # Pure black for consistent background
            "surface": "#000000",  # Pure black for surface
            "input_bg": "#222222",  # Input background
            "panel_header": "#111111",  # Panel header background
            "panel_highlight": "#333333",  # Panel highlight
            "text": "#ffffff",  # White text
            "text_secondary": "#b3e5fc",  # Light blue secondary text
            "border": "#2f3646",  # Dark border color
            "primary": "#4a90e2",  # Professional blue
            "secondary": "#5c6bc0",  # Indigo accent
            "accent": "#00ffff",  # Cyan accent
            "gradient_start": "#3a7bd5",  # Gradient start
            "gradient_end": "#00d2ff",  # Gradient end
            "success": "#66bb6a",  # Green for success states
            "warning": "#ffa726",  # Orange for warnings
            "error": "#ef5350",  # Red for errors
        }

        self.setup_ui()

    def setup_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Left panel for inputs - Enhanced styling
        left_panel = QWidget()
        left_panel.setObjectName("leftPanel")  # Add object name for specific styling
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)  # Reduced spacing between major sections
        left_layout.setContentsMargins(
            10, 10, 10, 10
        )  # Reduced margins for more compact look

        # Add function input widget
        self.func_input = FunctionInput()
        left_layout.addWidget(self.func_input)

        # Add method selector widget
        self.method_selector = MethodSelector()
        left_layout.addWidget(self.method_selector)

        # Connect method change to derivative requirements update
        self.method_selector.method_combo.currentTextChanged.connect(
            self.func_input.update_derivative_requirements
        )

        # Add solve button with updated styling
        button_layout = QHBoxLayout()
        button_layout.addSpacing(5)
        self.solve_btn = QPushButton("SOLVE")
        self.solve_btn.setFixedHeight(35)  # Make button more compact
        self.solve_btn.setObjectName(
            "solveButton"
        )  # Add object name for specific styling
        self.solve_btn.setStyleSheet(
            """
            #solveButton {
                background-color: #4a90e2;
                color: #ffffff;
                border: none;
                border-radius: 3px;
                font-weight: bold;
                font-size: 12px;
                letter-spacing: 1px;
            }
            #solveButton:hover {
                background-color: #2c7be5;
            }
            #solveButton:pressed {
                background-color: #1a5eb8;
                padding-top: 2px;
            }
        """
        )
        button_layout.addWidget(self.solve_btn)
        button_layout.addSpacing(5)
        left_layout.addLayout(button_layout)

        # Add results display
        self.results_display = ResultsDisplay(self.plot_colors)
        left_layout.addWidget(self.results_display)

        # Connect solve button to solve method
        self.solve_btn.clicked.connect(self.solve)

        # Right panel for visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins for more space

        # Create tab view with plots
        self.tab_view = TabView(self.plot_colors)
        right_layout.addWidget(self.tab_view)

        # Add panels to main layout with appropriate sizing
        layout.addWidget(left_panel, 1)  # 1/3 of the width
        layout.addWidget(right_panel, 2)  # 2/3 of the width

        # Apply global styling
        self.apply_styling()

        # Initial plot
        self.update_plots()

    def apply_styling(self):
        """Apply global styling to the application."""
        # Set the application style
        self.setStyleSheet(
            f"""
            QMainWindow {{
                background-color: {self.plot_colors['background']};
                color: {self.plot_colors['text']};
            }}
            QWidget {{
                background-color: {self.plot_colors['background']};
                color: {self.plot_colors['text']};
            }}
            QGroupBox {{
                border: 1px solid {self.plot_colors['border']};
                border-radius: 3px;
                margin-top: 0.5em;
                padding-top: 0.5em;
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
                color: {self.plot_colors['text_secondary']};
            }}
            #leftPanel {{
                background-color: {self.plot_colors['background']};
                border-right: 1px solid {self.plot_colors['border']};
            }}
        """
        )

    def parse_function(self):
        """Parse the function and its derivatives from the input."""
        try:
            # Get the function expression
            func_expr = self.func_input.func_input.text()
            if not func_expr:
                raise ValueError("Function expression is empty")

            # Create symbolic variables
            x1, x2 = symbols("x1 x2")

            # Parse the expression
            expr = sympify(func_expr)

            # Create a function that can be evaluated
            def func(x):
                return float(expr.subs([(x1, x[0]), (x2, x[1])]))

            # Check if we need derivatives
            if self.func_input.show_deriv.isChecked():
                # Compute first derivatives
                dx1 = diff(expr, x1)
                dx2 = diff(expr, x2)

                # Create gradient function
                def grad(x):
                    return np.array(
                        [
                            float(dx1.subs([(x1, x[0]), (x2, x[1])])),
                            float(dx2.subs([(x1, x[0]), (x2, x[1])])),
                        ]
                    )

            else:
                grad = None

            # Check if we need second derivatives
            if self.func_input.show_second_deriv.isChecked():
                # Compute second derivatives
                dx1x1 = diff(expr, x1, 2)
                dx1x2 = diff(expr, x1, x2)
                dx2x1 = diff(expr, x2, x1)
                dx2x2 = diff(expr, x2, 2)

                # Create Hessian function
                def hessian(x):
                    return np.array(
                        [
                            [
                                float(dx1x1.subs([(x1, x[0]), (x2, x[1])])),
                                float(dx1x2.subs([(x1, x[0]), (x2, x[1])])),
                            ],
                            [
                                float(dx2x1.subs([(x1, x[0]), (x2, x[1])])),
                                float(dx2x2.subs([(x1, x[0]), (x2, x[1])])),
                            ],
                        ]
                    )

            else:
                hessian = None

            return func, grad, hessian, expr

        except Exception as e:
            self.results_display.display_error(str(e))
            traceback.print_exc()
            return None, None, None, None

    def get_bounds(self):
        """Get the bounds for the variables."""
        bounds = []
        for i, (min_spin, max_spin) in enumerate(self.func_input.var_inputs):
            min_val = min_spin.value()
            max_val = max_spin.value()
            bounds.append((min_val, max_val))
        return bounds

    def update_plots(self, func=None):
        """Update all visualization plots."""
        if func is None:
            func, _, _, _ = self.parse_function()
            if func is None:
                return

        bounds = self.get_bounds()
        self.tab_view.update_plots(func, bounds, self.optimization_result)

    def solve(self):
        """Handle solve button click."""
        try:
            # Show loading message in results area before starting computation
            self.results_display.set_loading()

            # Process events to update the UI before continuing
            from PyQt6.QtCore import QCoreApplication

            QCoreApplication.processEvents()

            # Get function and parameters
            func, grad, hessian, expr = self.parse_function()
            if func is None:
                return

            method = self.method_selector.method_combo.currentText()
            tol = self.method_selector.tol_spin.value()
            max_iter = self.method_selector.max_iter_spin.value()
            bounds = self.get_bounds()

            # Get advanced optimization parameters if advanced options are shown
            step_length_method = None
            descent_direction_method = None
            initial_step_size = 1.0

            if self.method_selector.show_advanced.isChecked():
                # Only use these params if advanced options are enabled by the user
                step_length_method = (
                    self.method_selector.step_length_combo.currentText()
                )
                descent_direction_method = (
                    self.method_selector.descent_combo.currentText()
                )
                initial_step_size = self.method_selector.step_size_spin.value()

            # Get initial guess from user input instead of using center of bounds
            x0 = self.func_input.get_initial_guess()

            # Run optimization
            start_time = time.time()
            # Get the method class from the METHOD_MAP using the method name
            method_class = METHOD_MAP[method]
            self.optimization_result = run_optimization(
                method_class,  # Pass the method class, not the method name
                func,
                grad,
                hessian,
                x0,
                tol,
                max_iter,
                bounds,
                step_length_method,
                None,  # step_length_params
                descent_direction_method,
                None,  # descent_direction_params
                initial_step_size,
            )
            execution_time = time.time() - start_time

            # Update result display
            self.results_display.display_result(
                self.optimization_result, execution_time
            )

            # Update plots
            self.update_plots()

        except Exception as e:
            self.results_display.display_error(str(e))
            traceback.print_exc()

    def visualize(self):
        """Deprecated - functionality merged into solve()."""
        pass
