# utils/funcs.py

"""Common test functions for numerical optimization and root-finding algorithms."""

import os
import numpy as np
import torch  # type: ignore
from typing import Tuple, Callable, Union, List, Optional, Dict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Type aliases for function and derivative pairs
FuncPair = Tuple[Callable[[float], float], Callable[[float], float]]
FuncTriple = Tuple[
    Callable[[float], float], Callable[[float], float], Callable[[float], float]
]


class Function:
    """Base class for mathematical functions used in numerical methods."""

    def __init__(
        self,
        name: str,
        description: str,
        x_range: Tuple[float, float] = (-10, 10),
        known_roots: Optional[List[float]] = None,
    ):
        """
        Initialize function.

        Args:
            name: Name of the function
            description: Description of the function
            x_range: Default x-range for visualization
            known_roots: Known roots of the function (if any)
        """
        self.name = name
        self.description = description
        self.x_range = x_range
        self.known_roots = known_roots or []

    def f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate the function at x."""
        raise NotImplementedError("Function must implement f(x)")

    def df(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate the derivative at x."""
        raise NotImplementedError("Function must implement df(x)")

    def d2f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate the second derivative at x."""
        raise NotImplementedError("Function must implement d2f(x) for optimization")

    def get_for_root_finding(self) -> FuncPair:
        """Get function and derivative for root finding."""
        return (self.f, self.df)

    def get_for_optimization(
        self, with_second_derivative: bool = False
    ) -> Union[FuncPair, FuncTriple]:
        """Get function and derivatives for optimization."""
        if with_second_derivative:
            return (self.f, self.df, self.d2f)
        return (self.f, self.df)

    def find_root(
        self, x0: float, method: str = "newton", tol: float = 1e-6, max_iter: int = 100
    ) -> float:
        """
        Find a root of the function using the specified method.

        Args:
            x0: Initial guess
            method: Root-finding method ('newton', 'bisection', etc.)
            tol: Tolerance for convergence
            max_iter: Maximum number of iterations

        Returns:
            Root of the function
        """
        from solve import run_methods

        methods, results = run_methods(
            function_name=self.name,
            method_names=[method],
            x0_values=[x0],
            method_type="root",
            tol=tol,
            max_iter=max_iter,
            visualize=False,
        )

        if len(methods) > 0 and methods[0].has_converged():
            return methods[0].get_current_x()
        else:
            raise ValueError(f"Failed to find root using {method} method")

    def find_minimum(
        self,
        x0: float,
        method: str = "newton_opt",
        tol: float = 1e-6,
        max_iter: int = 100,
    ) -> float:
        """
        Find a minimum of the function using the specified method.

        Args:
            x0: Initial guess
            method: Optimization method ('newton_opt', 'steepest_descent', etc.)
            tol: Tolerance for convergence
            max_iter: Maximum number of iterations

        Returns:
            Point at which function attains a minimum
        """
        from solve import run_methods

        methods, results = run_methods(
            function_name=self.name,
            method_names=[method],
            x0_values=[x0] if not isinstance(x0, list) else x0,
            method_type="optimize",
            tol=tol,
            max_iter=max_iter,
            visualize=False,
        )

        if len(methods) > 0 and methods[0].has_converged():
            return methods[0].get_current_x()
        else:
            raise ValueError(f"Failed to find minimum using {method} method")


class PolynomialFunction(Function):
    """Class for polynomial functions."""

    pass


class QuadraticFunction(PolynomialFunction):
    """f(x) = x² - 2, root at ±√2."""

    def __init__(self):
        super().__init__(
            name="quadratic",
            description="f(x) = x² - 2, roots at ±√2",
            x_range=(-3, 3),
            known_roots=[-np.sqrt(2), np.sqrt(2)],
        )

    def f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return x**2 - 2

    def df(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 2 * x

    def d2f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(x, np.ndarray):
            return np.full_like(x, 2.0)
        return 2.0


class QuadraticMinFunction(PolynomialFunction):
    """f(x) = x², minimum at x = 0."""

    def __init__(self):
        super().__init__(
            name="quadratic_min",
            description="f(x) = x², minimum at x = 0",
            x_range=(-3, 3),
            known_roots=[0],  # Min point, not a root
        )

    def f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(x, np.ndarray) and x.size > 1:
            return np.sum(x**2)
        return float(x**2)

    def df(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(x, np.ndarray) and x.size > 1:
            return 2 * x
        return float(2 * x)

    def d2f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(x, np.ndarray) and x.size > 1:
            return np.full_like(x, 2.0)
        return 2.0


class CubicFunction(PolynomialFunction):
    """f(x) = x³ - x - 2, one real root near x ≈ 1.7693."""

    def __init__(self):
        super().__init__(
            name="cubic",
            description="f(x) = x³ - x - 2, one real root near x ≈ 1.7693",
            x_range=(-2, 4),
            known_roots=[1.7693],
        )

    def f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return x**3 - x - 2

    def df(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 3 * x**2 - 1

    def d2f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 6 * x


class CubicMinFunction(PolynomialFunction):
    """f(x) = x³ + x, minimum at x ≈ -0.577."""

    def __init__(self):
        super().__init__(
            name="cubic_min",
            description="f(x) = x³ + x, minimum at x ≈ -0.577",
            x_range=(-2, 2),
            known_roots=[-0.577],  # Min point, not a root
        )

    def f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return x**3 + x

    def df(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 3 * x**2 + 1

    def d2f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 6 * x


class QuarticFunction(PolynomialFunction):
    """f(x) = x⁴ - 5x² + 4, roots at ±1, ±2."""

    def __init__(self):
        super().__init__(
            name="quartic",
            description="f(x) = x⁴ - 5x² + 4, roots at ±1, ±2",
            x_range=(-3, 3),
            known_roots=[-2, -1, 1, 2],
        )

    def f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return x**4 - 5 * x**2 + 4

    def df(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 4 * x**3 - 10 * x

    def d2f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 12 * x**2 - 10


class QuarticMinFunction(PolynomialFunction):
    """f(x) = x⁴ - 2x² + 1, minima at x = ±1."""

    def __init__(self):
        super().__init__(
            name="quartic_min",
            description="f(x) = x⁴ - 2x² + 1, minima at x = ±1",
            x_range=(-2, 2),
            known_roots=[-1, 1],  # Min points, not roots
        )

    def f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return x**4 - 2 * x**2 + 1

    def df(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 4 * x**3 - 4 * x

    def d2f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 12 * x**2 - 4


class TranscendentalFunction(Function):
    """Class for transcendental functions."""

    pass


class ExponentialFunction(TranscendentalFunction):
    """f(x) = e^x - 4, root at ln(4)."""

    def __init__(self):
        super().__init__(
            name="exponential",
            description="f(x) = e^x - 4, root at ln(4)",
            x_range=(0, 2),
            known_roots=[np.log(4)],
        )

    def f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.exp(x) - 4

    def df(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.exp(x)

    def d2f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.exp(x)


class LogarithmicFunction(TranscendentalFunction):
    """f(x) = ln(x) - 1, root at e."""

    def __init__(self):
        super().__init__(
            name="logarithmic",
            description="f(x) = ln(x) - 1, root at e",
            x_range=(0.1, 5),
            known_roots=[np.e],
        )

    def f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.log(x) - 1

    def df(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 1 / x

    def d2f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return -1 / (x**2)


class ExpLinearFunction(TranscendentalFunction):
    """f(x) = e^x - 2x - 1, root near x ≈ 0.5671."""

    def __init__(self):
        super().__init__(
            name="exp_linear",
            description="f(x) = e^x - 2x - 1, root near x ≈ 0.5671",
            x_range=(-1, 3),
            known_roots=[0.5671],
        )

    def f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.exp(x) - 2 * x - 1

    def df(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.exp(x) - 2

    def d2f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.exp(x)


class TrigonometricFunction(TranscendentalFunction):
    """Class for trigonometric functions."""

    pass


class SinusoidalFunction(TrigonometricFunction):
    """f(x) = sin(x) - 0.5, roots near x ≈ 0.5236, 2.6180."""

    def __init__(self):
        super().__init__(
            name="sinusoidal",
            description="f(x) = sin(x) - 0.5, roots near x ≈ 0.5236, 2.6180",
            x_range=(0, 2 * np.pi),
            known_roots=[0.5236, 2.6180],
        )

    def f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.sin(x) - 0.5

    def df(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.cos(x)

    def d2f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return -np.sin(x)


class CosineFunction(TrigonometricFunction):
    """f(x) = cos(x) - x, root near x ≈ 0.7390."""

    def __init__(self):
        super().__init__(
            name="cosine",
            description="f(x) = cos(x) - x, root near x ≈ 0.7390",
            x_range=(-2, 2),
            known_roots=[0.7390],
        )

    def f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.cos(x) - x

    def df(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return -np.sin(x) - 1

    def d2f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return -np.cos(x)


class TangentFunction(TrigonometricFunction):
    """f(x) = tan(x) - x, multiple roots."""

    def __init__(self):
        super().__init__(
            name="tangent",
            description="f(x) = tan(x) - x, multiple roots",
            x_range=(-1.5, 1.5),
            known_roots=[0],  # And many others
        )

    def f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.tan(x) - x

    def df(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 1 / np.cos(x) ** 2 - 1

    def d2f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 2 * np.tan(x) / np.cos(x) ** 2


# Multivariable functions for optimization


class RosenbrockFunction(Function):
    """Rosenbrock function (banana function): f(x,y) = (1-x)² + 100(y-x²)², minimum at (1,1)."""

    def __init__(self):
        super().__init__(
            name="rosenbrock",
            description="Rosenbrock function (banana function): f(x,y) = (1-x)² + 100(y-x²)², minimum at (1,1)",
            x_range=(-2, 2),
            known_roots=[1, 1],  # Min point (x,y), not a root
        )

    def f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if not isinstance(x, np.ndarray) or x.size == 1:
            # 1D case - simplified version
            return (1 - x) ** 2
        # 2D case
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    def df(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if not isinstance(x, np.ndarray) or x.size == 1:
            # 1D case
            return -2 * (1 - x)
        # 2D case - return gradient
        return np.array(
            [
                -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
                200 * (x[1] - x[0] ** 2),
            ]
        )

    def d2f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if not isinstance(x, np.ndarray) or x.size == 1:
            # 1D case
            return 2.0
        # 2D case - return Hessian
        return np.array(
            [
                [2 + 1200 * x[0] ** 2 - 400 * x[1], -400 * x[0]],
                [-400 * x[0], 200],
            ]
        )


class HimmelblauFunction(Function):
    """Himmelblau's function: f(x,y) = (x²+y-11)² + (x+y²-7)², 4 local minima."""

    def __init__(self):
        super().__init__(
            name="himmelblau",
            description="Himmelblau's function: f(x,y) = (x²+y-11)² + (x+y²-7)², 4 local minima",
            x_range=(-5, 5),
            # Min points (x,y), not roots
            known_roots=[
                [3.0, 2.0],
                [-2.805118, 3.131312],
                [-3.779310, -3.283186],
                [3.584428, -1.848126],
            ],
        )

    def f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if not isinstance(x, np.ndarray) or x.size == 1:
            # 1D case (not typical for Himmelblau)
            return (x**2 + 2 - 11) ** 2 + (x + 4 - 7) ** 2
        # 2D case
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

    def df(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if not isinstance(x, np.ndarray) or x.size == 1:
            # 1D case
            return 4 * x * (x**2 + 2 - 11) + 2 * (x + 4 - 7)
        # 2D case - return gradient
        return np.array(
            [
                4 * x[0] * (x[0] ** 2 + x[1] - 11) + 2 * (x[0] + x[1] ** 2 - 7),
                2 * (x[0] ** 2 + x[1] - 11) + 4 * x[1] * (x[0] + x[1] ** 2 - 7),
            ]
        )

    def d2f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if not isinstance(x, np.ndarray) or x.size == 1:
            # 1D case
            return 12 * x**2 + 4 * 2 - 44 + 2
        # 2D case - return Hessian
        return np.array(
            [
                [12 * x[0] ** 2 + 4 * x[1] - 42, 4 * x[0] + 4 * x[1]],
                [4 * x[0] + 4 * x[1], 4 * x[0] + 12 * x[1] ** 2 - 26],
            ]
        )


# New function for Problem 5
class DiagonalQuadraticFunction(Function):
    """Diagonal Quadratic function: f(x) = sum_{i=1}^20 (1/i) * x_i^2, minimum at origin."""

    def __init__(self):
        self.n_dim = 20
        super().__init__(
            name="diagonal_quadratic",
            description=f"f(x) = sum_{{i=1}}^{self.n_dim} (1/i) * x_i^2, minimum at x=0",
            x_range=(-5, 5),  # Default range per dimension
            known_roots=[np.zeros(self.n_dim)],  # Min point (x vector)
        )

    def f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # Handle scalar case (single dimension)
        if isinstance(x, (int, float)):
            return x**2  # Just the first dimension

        # Convert to numpy array if it's a list
        if isinstance(x, list):
            x = np.array(x)

        # Handle incomplete arrays (pad with zeros)
        if isinstance(x, np.ndarray):
            if x.size < self.n_dim:
                temp = np.zeros(self.n_dim)
                temp[: x.size] = x
                x = temp
            elif x.size > self.n_dim:
                x = x[: self.n_dim]

        # Now compute the actual function
        coeffs = 1.0 / np.arange(1, self.n_dim + 1)
        return np.sum(coeffs * (x**2))

    def df(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # Handle scalar case
        if isinstance(x, (int, float)):
            return 2.0 * x  # Just derivative for first dimension

        # Convert to numpy array if it's a list
        if isinstance(x, list):
            x = np.array(x)

        # Handle incomplete arrays
        if isinstance(x, np.ndarray):
            if x.size < self.n_dim:
                temp = np.zeros(self.n_dim)
                temp[: x.size] = x
                x = temp
            elif x.size > self.n_dim:
                x = x[: self.n_dim]

        # Now compute the gradient
        coeffs = 2.0 / np.arange(1, self.n_dim + 1)
        return coeffs * x

    def d2f(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # Handle scalar case
        if isinstance(x, (int, float)):
            return 2.0  # Second derivative for first dimension

        # Handle different array sizes
        if isinstance(x, np.ndarray) and x.size != self.n_dim:
            if x.size < self.n_dim:
                # Just return the Hessian for the dimensions we have
                n = x.size
                coeffs = 2.0 / np.arange(1, n + 1)
                return np.diag(coeffs)
            else:
                # Truncate to our dimensions
                x = x[: self.n_dim]

        # Hessian is constant and diagonal
        coeffs = 2.0 / np.arange(1, self.n_dim + 1)
        return np.diag(coeffs)


# Registry of functions
FUNCTION_REGISTRY = {
    # Root-finding functions
    "quadratic": QuadraticFunction(),
    "cubic": CubicFunction(),
    "quartic": QuarticFunction(),
    "exponential": ExponentialFunction(),
    "logarithmic": LogarithmicFunction(),
    "exp_linear": ExpLinearFunction(),
    "sinusoidal": SinusoidalFunction(),
    "cosine": CosineFunction(),
    "tangent": TangentFunction(),
    # Legacy names compatibility
    "simple_quadratic": QuadraticFunction(),
    "trigonometric": SinusoidalFunction(),
    # Optimization functions
    "quadratic_min": QuadraticMinFunction(),
    "cubic_min": CubicMinFunction(),
    "quartic_min": QuarticMinFunction(),
    "rosenbrock": RosenbrockFunction(),
    "himmelblau": HimmelblauFunction(),
    "diagonal_quadratic": DiagonalQuadraticFunction(),  # Add the new function here
}

# Create maps for backward compatibility
FUNCTION_MAP = {
    name: func.get_for_root_finding() for name, func in FUNCTION_REGISTRY.items()
}

MINIMIZATION_MAP = {
    name: func.get_for_optimization()
    for name, func in FUNCTION_REGISTRY.items()
    if hasattr(func, "get_for_optimization")
}

# Default ranges for visualization
FUNCTION_RANGES = {name: func.x_range for name, func in FUNCTION_REGISTRY.items()}

MINIMIZATION_RANGES = FUNCTION_RANGES.copy()

# List of all available test functions
AVAILABLE_FUNCTIONS = list(FUNCTION_REGISTRY.keys())

# PyTorch-based functions
TORCH_FUNCTIONS = {
    "quadratic": lambda x: x**2 - 2,
    "cubic": lambda x: x**3 - x - 2,
    "exp_linear": lambda x: torch.exp(x) - 2 * x - 1,
    "sinusoidal": lambda x: torch.sin(x) - x / 2,
    "cosine": lambda x: torch.cos(x) - x,
}


def get_function(name: str) -> Function:
    """Get a Function object by name.

    Args:
        name: Name of the function

    Returns:
        Function: The function object

    Raises:
        ValueError: If the function is not found
    """
    if name not in FUNCTION_REGISTRY:
        raise ValueError(f"Unknown function: {name}")
    return FUNCTION_REGISTRY[name]


def get_test_function(
    name: str, with_second_derivative: bool = False
) -> Union[FuncPair, FuncTriple]:
    """Get a test function and its derivatives by name for root-finding.

    Args:
        name: Name of the test function
        with_second_derivative: If True, also returns second derivative

    Returns:
        Tuple of (function, first derivative) or (function, first derivative, second derivative)
    """
    func = get_function(name)
    if with_second_derivative:
        return func.f, func.df, func.d2f
    return func.f, func.df


def get_torch_function(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Get a PyTorch-based test function by name."""
    return TORCH_FUNCTIONS[name]


def get_minimization_function(
    name: str, with_second_derivative: bool = False
) -> Union[FuncPair, FuncTriple]:
    """Get a minimization test function and its derivatives by name.

    Args:
        name: Name of the minimization function
        with_second_derivative: If True, also returns second derivative

    Returns:
        Tuple of (function, gradient) or (function, gradient, hessian)
    """
    func = get_function(name)
    if with_second_derivative:
        return func.f, func.df, func.d2f
    return func.f, func.df


def determine_x_range(
    function_name: str,
    x0_values: List[float],
    method_type: str,
    specified_range: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """
    Determine an appropriate x-range for visualization based on the function and initial points.

    Args:
        function_name: Name of the function
        x0_values: Initial points
        method_type: Type of method (root-finding or optimization)
        specified_range: User-specified range (if provided)

    Returns:
        Tuple[float, float]: Appropriate x-range for visualization
    """
    if specified_range is not None:
        return specified_range

    # Get the function object
    try:
        func = get_function(function_name)
        return func.x_range
    except ValueError:
        pass

    # Choose the appropriate ranges dictionary based on method type
    ranges_dict = MINIMIZATION_RANGES if method_type == "optimize" else FUNCTION_RANGES

    # If the function is in the ranges dictionary, return its range
    if function_name in ranges_dict:
        # Special case for drug_effectiveness which has a dictionary of ranges
        if isinstance(ranges_dict[function_name], dict):
            # For 3D functions with separate parameter ranges, return default range
            return (-5, 5)  # Default range for complex functions
        return ranges_dict[function_name]

    # If no default range exists, use the initial points to determine a range
    min_x0 = min(x0_values)
    max_x0 = max(x0_values)

    # Ensure the range has a minimum width
    width = max(max_x0 - min_x0, 1.0)

    # Add padding
    padding = width * 0.5

    return (min_x0 - padding, max_x0 + padding)


def register_custom_function(func: Function) -> None:
    """
    Register a custom function to be used with the numerical methods.

    Args:
        func: Custom Function object to register
    """
    FUNCTION_REGISTRY[func.name] = func
    FUNCTION_RANGES[func.name] = func.x_range
    FUNCTION_MAP[func.name] = func.get_for_root_finding()
    MINIMIZATION_MAP[func.name] = func.get_for_optimization()
    AVAILABLE_FUNCTIONS.append(func.name)


def list_function_categories() -> Dict[str, List[str]]:
    """
    List all available functions by category.

    Returns:
        Dictionary of function categories and their function names
    """
    categories = {}

    # Categorize functions
    for name, func in FUNCTION_REGISTRY.items():
        category = func.__class__.__bases__[0].__name__
        if category not in categories:
            categories[category] = []
        if name not in categories[category]:
            categories[category].append(name)

    return categories
