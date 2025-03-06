"""Tab view component for organizing visualizations."""

from PyQt6.QtWidgets import QTabWidget
from PyQt6.QtWebEngineCore import QWebEngineSettings

from .surface_plot import SurfacePlot
from .contour_plot import ContourPlot
from .convergence_plot import ConvergencePlot


class TabView(QTabWidget):
    """Widget with tabs for different visualizations."""

    def __init__(self, plot_colors, parent=None):
        super().__init__(parent)
        self.plot_colors = plot_colors
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI components."""
        # Style the tab widget
        self.setStyleSheet(
            f"""
            QTabWidget::pane {{
                border: 1px solid {self.plot_colors['border']};
                background-color: {self.plot_colors['background']};
                border-radius: 3px;
            }}
            QTabBar::tab {{
                background-color: {self.plot_colors['panel_header']};
                color: {self.plot_colors['text']};
                border: 1px solid {self.plot_colors['border']};
                border-bottom-color: {self.plot_colors['border']};
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
                padding: 5px 10px;
                margin-right: 2px;
                font-size: 10px;
            }}
            QTabBar::tab:selected {{
                background-color: {self.plot_colors['panel_highlight']};
                border-bottom-color: {self.plot_colors['panel_highlight']};
            }}
            QTabBar::tab:hover {{
                background-color: {self.plot_colors['panel_highlight']};
            }}
            """
        )

        # Create tabs
        self.surface_tab = SurfacePlot()
        self.contour_tab = ContourPlot()  # Using our new ContourPlot class
        self.convergence_tab = ConvergencePlot()

        # Add tabs to the widget
        self.addTab(self.surface_tab, "3D Surface")
        self.addTab(self.contour_tab, "Contour Plot")
        self.addTab(self.convergence_tab, "Convergence")

    def update_plots(self, func=None, bounds=None, optimization_result=None):
        """Update all plots with the given data."""
        # Update surface plot
        self.surface_tab.update_plot(func, bounds, optimization_result)

        # Update contour plot
        self.contour_tab.update_plot(func, bounds, optimization_result)

        # Update convergence plot
        self.convergence_tab.update_plot(optimization_result)
