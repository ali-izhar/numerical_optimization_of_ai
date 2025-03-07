# plot/components/__init__.py

from plot.components.config import VisualizationConfig
from plot.components.function_space import FunctionSpace
from plot.components.convergence_plot import ConvergencePlot
from plot.components.error_plot import ErrorPlot
from plot.components.animation import MethodAnimation

__all__ = [
    "VisualizationConfig",
    "FunctionSpace",
    "ConvergencePlot",
    "ErrorPlot",
    "MethodAnimation",
]
