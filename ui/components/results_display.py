"""Results display component for optimization outputs."""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QTextEdit,
)
from PyQt6.QtCore import Qt


class ResultsDisplay(QWidget):
    """Widget for displaying optimization results."""

    def __init__(self, plot_colors, parent=None):
        super().__init__(parent)
        self.plot_colors = plot_colors
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Results header
        self.header = QLabel("RESULTS")
        self.header.setObjectName("resultsHeader")
        self.header.setStyleSheet(
            f"""
            #resultsHeader {{
                color: {self.plot_colors['text']};
                font-weight: bold;
                font-size: 12px;
                letter-spacing: 1px;
                padding: 5px;
                background-color: {self.plot_colors['panel_header']};
                border-radius: 3px;
            }}
            """
        )
        self.header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.header)

        # Results text area
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet(
            f"""
            QTextEdit {{
                background-color: {self.plot_colors['input_bg']};
                color: {self.plot_colors['text']};
                border: 1px solid {self.plot_colors['border']};
                border-radius: 3px;
                padding: 5px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10px;
            }}
            """
        )
        layout.addWidget(self.results_text)

    def display_result(self, optimization_result=None, execution_time=None):
        """Display the optimization results."""
        if optimization_result is None:
            self.results_text.setText("No results to display")
            return

        # Format the results
        result_text = "=== Optimization Results ===\n\n"

        if "success" in optimization_result:
            status = "Success" if optimization_result["success"] else "Failed"
            result_text += f"Status: {status}\n"

        if "message" in optimization_result:
            result_text += f"Message: {optimization_result['message']}\n"

        if "x" in optimization_result:
            result_text += f"\nSolution:\n"
            for i, val in enumerate(optimization_result["x"]):
                result_text += f"  x{i+1} = {val:.8f}\n"

        if "fun" in optimization_result:
            result_text += f"\nFunction Value: {optimization_result['fun']:.8f}\n"

        if "nit" in optimization_result:
            result_text += f"\nIterations: {optimization_result['nit']}\n"

        if "nfev" in optimization_result:
            result_text += f"Function Evaluations: {optimization_result['nfev']}\n"

        if execution_time is not None:
            result_text += f"\nExecution Time: {execution_time:.4f} seconds\n"

        self.results_text.setText(result_text)

    def set_loading(self):
        """Show loading message while optimization is running."""
        self.results_text.setText("Optimizing! Please wait...")

    def display_error(self, error_message):
        """Display an error message."""
        self.results_text.setText(f"Error: {error_message}")
