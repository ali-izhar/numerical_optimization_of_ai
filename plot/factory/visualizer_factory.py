# plot/factory/visualizer_factory.py

"""
This module provides factory classes for creating appropriate visualizers based on the
numerical method type (root finding or optimization).
"""

from typing import Dict, List, Optional, Any
from enum import Enum, auto

from algorithms.convex.protocols import (
    BaseNumericalMethod,
    NumericalMethodConfig,
    MethodType,
)
from plot.components.config import VisualizationConfig
from plot.components.function_space import FunctionSpace
from plot.components.convergence_plot import ConvergencePlot
from plot.components.error_plot import ErrorPlot
from plot.components.animation import MethodAnimation


class VisualizerType(Enum):
    """Enum for different types of visualizers."""

    ROOT_FINDING = auto()
    OPTIMIZATION = auto()


class VisualizerFactory:
    """
    Factory class for creating appropriate visualizers based on method type.

    This class encapsulates the logic for determining which type of visualizer to create
    based on the numerical method being visualized (root-finding or optimization).
    """

    @staticmethod
    def create_visualizer(
        method_config: NumericalMethodConfig,
        methods: List[BaseNumericalMethod],
        vis_config: Optional[VisualizationConfig] = None,
    ) -> Any:
        """
        Create an appropriate visualizer based on method type.

        Args:
            method_config: Configuration for the numerical method
            methods: List of numerical methods to visualize
            vis_config: Visualization configuration

        Returns:
            An appropriate visualizer object
        """
        # Create default visualization config if not provided
        if vis_config is None:
            vis_config = VisualizationConfig()

        # Determine type of visualizer to create
        visualizer_type = VisualizerFactory._determine_visualizer_type(method_config)

        # Import appropriate visualizer classes
        if visualizer_type == VisualizerType.ROOT_FINDING:
            from plot.root_finder_viz import RootFindingVisualizer

            return RootFindingVisualizer(method_config, methods, vis_config)
        else:  # OPTIMIZATION
            from plot.optimizer_viz import OptimizationVisualizer

            return OptimizationVisualizer(method_config, methods, vis_config)

    @staticmethod
    def _determine_visualizer_type(
        method_config: NumericalMethodConfig,
    ) -> VisualizerType:
        """
        Determine the appropriate visualizer type based on method configuration.

        Args:
            method_config: Configuration for the numerical method

        Returns:
            VisualizerType: The appropriate visualizer type
        """
        if method_config.method_type == MethodType.ROOT:
            return VisualizerType.ROOT_FINDING
        else:
            return VisualizerType.OPTIMIZATION

    @staticmethod
    def create_components(
        method_config: NumericalMethodConfig,
        methods: List[BaseNumericalMethod],
        vis_config: Optional[VisualizationConfig] = None,
    ) -> Dict[str, Any]:
        """
        Create individual visualization components for the methods.

        This method provides more fine-grained control over the visualization
        by returning individual component objects instead of a complete visualizer.

        Args:
            method_config: Configuration for the numerical method
            methods: List of numerical methods to visualize
            vis_config: Visualization configuration

        Returns:
            Dict[str, Any]: Dictionary of visualization components
        """
        # Create default visualization config if not provided
        if vis_config is None:
            vis_config = VisualizationConfig()

        # Choose titles based on method type
        if method_config.method_type == MethodType.ROOT:
            function_title = "Root Finding Visualization"
            convergence_title = "Root Approximation Progress"
            error_title = "Root Finding Error Reduction"
            animation_title = "Root Finding Method Animation"
            ylabel = "Root Approximation"
        else:
            function_title = "Optimization Visualization"
            convergence_title = "Optimization Progress"
            error_title = "Optimization Error Reduction"
            animation_title = "Optimization Method Animation"
            ylabel = "Parameter Value"

        # Create function space component with enhanced styling
        function_space = FunctionSpace(
            func=method_config.func,
            x_range=method_config.x_range,
            title=function_title,
            is_2d=method_config.is_2d,
            # Set colormap based on config if available
            colormap=(
                vis_config.plotly_colorscales.get("surface", "Viridis")
                if hasattr(vis_config, "plotly_colorscales")
                and vis_config.plotly_colorscales
                else "Viridis"
            ),
        )

        # Create convergence plot component with enhanced styling
        convergence_plot = ConvergencePlot(
            title=convergence_title,
            xlabel="Iteration",
            ylabel=ylabel,
            color_palette=vis_config.palette,
        )

        # Create error plot component with enhanced styling
        error_plot = ErrorPlot(
            title=error_title,
            xlabel="Iteration",
            ylabel="Error",
            color_palette=vis_config.palette,
            log_scale=True,
        )

        # Create animation component with enhanced styling
        animation = MethodAnimation(
            function_space=function_space,
            title=animation_title,
            color_palette=vis_config.palette,
        )

        # Return all components in a dictionary
        return {
            "function_space": function_space,
            "convergence_plot": convergence_plot,
            "error_plot": error_plot,
            "animation": animation,
        }
