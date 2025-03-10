# plot/components/__init__.py

from .config import VisualizationConfig
from .function_space import FunctionSpace
from .convergence_plot import ConvergencePlot
from .error_plot import ErrorPlot
from .animation import MethodAnimation

__all__ = [
    "VisualizationConfig",
    "FunctionSpace",
    "ConvergencePlot",
    "ErrorPlot",
    "MethodAnimation",
]
