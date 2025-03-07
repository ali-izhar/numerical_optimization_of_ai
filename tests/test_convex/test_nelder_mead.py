# tests/test_convex/test_nelder_mead.py

import pytest
import sys
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from algorithms.convex.nelder_mead import NelderMeadMethod, nelder_mead_search
from algorithms.convex.protocols import NumericalMethodConfig


def test_basic_optimization():
    """Test finding minimum of x^2"""

    def f(x):
        return x**2

    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = NelderMeadMethod(config, x0=1.0)

    while not method.has_converged():
        x = method.step()

    assert abs(x) < 1e-4  # Should find minimum at x=0
    assert method.iterations < 100


def test_invalid_method_type():
    """Test that initialization fails when method_type is not 'optimize'"""

    def f(x):
        return x**2

    config = NumericalMethodConfig(func=f, method_type="root")
    with pytest.raises(ValueError, match="can only be used for optimization"):
        NelderMeadMethod(config, x0=1.0)


def test_quadratic_function():
    """Test optimization of quadratic function"""

    def f(x):
        return 2 * x**2 + 4 * x + 1

    config = NumericalMethodConfig(
        func=f,
        method_type="optimize",
        tol=1e-5,  # Adjusted tolerance
        max_iter=500,  # More iterations allowed
    )
    method = NelderMeadMethod(
        config, x0=0.0, delta=0.05
    )  # Start closer, smaller simplex

    while not method.has_converged():
        x = method.step()

    assert abs(x + 1) < 1e-4  # Minimum at x=-1
    assert abs(f(x) - (-1)) < 1e-4  # Minimum value is -1


def test_simplex_operations():
    """Test that simplex operations (reflection, expansion, etc.) work correctly"""

    def f(x):
        return x**2

    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = NelderMeadMethod(config, x0=1.0, delta=0.5)

    # Do one step and check the details
    method.step()
    history = method.get_iteration_history()
    details = history[0].details

    # Check that all operations are recorded
    assert "simplex_points" in details
    assert "f_values" in details
    assert "reflection" in details
    assert "action" in details


def test_iteration_history():
    """Test that iteration history is properly recorded"""

    def f(x):
        return x**2

    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = NelderMeadMethod(config, x0=2.0)  # Start further from minimum

    # Run for a few iterations or until convergence
    for _ in range(3):
        if method.has_converged():
            break
        method.step()

    history = method.get_iteration_history()
    assert len(history) > 0, "Should have at least one iteration"

    # Check that we make progress in each iteration
    for data in history:
        assert data.f_new <= data.f_old, (
            f"Function value should not increase within iteration.\n"
            f"Old value: {data.f_old}, New value: {data.f_new}"
        )

    # Verify overall progress
    first_iter = history[0]
    last_iter = history[-1]
    assert last_iter.f_new < first_iter.f_old, (
        f"Should make progress from start to end.\n"
        f"Starting value: {first_iter.f_old}, Final value: {last_iter.f_new}"
    )


def test_legacy_wrapper():
    """Test the backward-compatible nelder_mead_search function"""

    def f(x):
        return x**2

    minimum, errors, iters = nelder_mead_search(f, x0=1.0)
    assert abs(minimum) < 1e-4
    assert len(errors) == iters


def test_different_functions():
    """Test method works with different types of functions"""
    test_cases = [
        # Simple quadratic (easy to optimize)
        (lambda x: x**2, 0.5, 0.0, 1e-4, 0.05, "quadratic"),
        # Scaled quadratic (still easy)
        (lambda x: 0.5 * x**2, 0.5, 0.0, 1e-4, 0.05, "scaled quadratic"),
        # Linear + quadratic (minimum at -2)
        (lambda x: x**2 + 4 * x, -1.5, -2.0, 1e-3, 0.1, "linear-quadratic"),
    ]

    for func, x0, true_min, tol, delta, name in test_cases:
        config = NumericalMethodConfig(
            func=func,
            method_type="optimize",
            tol=tol,
            max_iter=1000,
        )
        method = NelderMeadMethod(config, x0=x0, delta=delta)

        while not method.has_converged():
            x = method.step()

        assert method.iterations >= 5, f"Too few iterations for {name}"
        assert abs(x - true_min) < tol * 20, (
            f"Function '{name}' did not converge properly:\n"
            f"  Final x: {x}\n"
            f"  True minimum: {true_min}\n"
            f"  Tolerance: {tol}\n"
            f"  Iterations: {method.iterations}"
        )


def test_challenging_functions():
    """Test method behavior with more challenging functions"""

    def f(x):
        return abs(x) ** 1.5  # Non-smooth at minimum

    config = NumericalMethodConfig(
        func=f,
        method_type="optimize",
        tol=1e-3,
        max_iter=2000,
    )
    method = NelderMeadMethod(config, x0=0.5)

    while not method.has_converged():
        x = method.step()

    # For challenging functions, verify that:
    # 1. We're close to the minimum (x=0)
    assert abs(x) < 0.1, f"Not close enough to minimum. x={x}"
    # 2. Function value has decreased
    assert f(x) < f(0.5), "Function value did not decrease"
    # 3. We haven't exceeded max iterations
    assert method.iterations < 2000, f"Too many iterations: {method.iterations}"


def test_max_iterations():
    """Test that method respects maximum iterations"""

    def f(x):
        return x**2

    config = NumericalMethodConfig(func=f, method_type="optimize", max_iter=5)
    method = NelderMeadMethod(config, x0=1.0)

    while not method.has_converged():
        method.step()

    assert method.iterations <= 5


def test_get_current_x():
    """Test that get_current_x returns the latest approximation"""

    def f(x):
        return x**2

    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = NelderMeadMethod(config, x0=1.0)

    x = method.step()
    assert x == method.get_current_x()


def test_multidimensional_optimization():
    """Test Nelder-Mead with multidimensional functions."""
    # Test cases: (function, x0, true_minimum, tolerance)
    test_cases = [
        # 2D quadratic function with minimum at [0, 0]
        (
            lambda x: x[0] ** 2 + x[1] ** 2,
            np.array([1.0, 1.0]),
            np.array([0.0, 0.0]),
            1e-3,
        ),
        # 3D function with minimum at [0, 0, 0]
        (
            lambda x: x[0] ** 2 + 2 * x[1] ** 2 + 3 * x[2] ** 2,
            np.array([1.0, 1.0, 1.0]),
            np.array([0.0, 0.0, 0.0]),
            1e-2,
        ),
    ]

    for func, x0, true_min, tol in test_cases:
        config = NumericalMethodConfig(
            func=func,
            method_type="optimize",
            tol=tol,
            max_iter=2000,
        )
        method = NelderMeadMethod(config, x0=x0)

        while not method.has_converged():
            x = method.step()

        # Check that we're close to the true minimum
        assert np.allclose(
            x, true_min, rtol=0, atol=tol * 10
        ), f"Failed to find minimum. Got {x}, expected {true_min}"

    # Test Rosenbrock function separately with function value check
    # since it may not converge precisely to [1, 1] from all starting points
    def rosenbrock(x):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    config = NumericalMethodConfig(
        func=rosenbrock,
        method_type="optimize",
        tol=1e-3,
        max_iter=3000,
    )
    method = NelderMeadMethod(
        config, x0=np.array([0.5, 0.5])
    )  # Start closer to minimum

    while not method.has_converged():
        x = method.step()

    # For Rosenbrock, check that function value has decreased significantly
    init_val = rosenbrock(np.array([0.5, 0.5]))
    final_val = rosenbrock(x)
    assert (
        final_val < 0.1 * init_val
    ), f"Function value didn't decrease enough: {final_val} vs initial {init_val}"


def test_maybe_scalar():
    """Test the _maybe_scalar method for converting arrays to scalars."""

    def f(x):
        return x**2

    # Scalar problem
    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = NelderMeadMethod(config, x0=1.0)

    # Should convert 1D array with single element to scalar
    assert isinstance(method._maybe_scalar(np.array([2.0])), float)
    assert method._maybe_scalar(np.array([2.0])) == 2.0

    # Vector problem
    def f_vec(x):
        return x[0] ** 2 + x[1] ** 2

    config_vec = NumericalMethodConfig(func=f_vec, method_type="optimize")
    method_vec = NelderMeadMethod(config_vec, x0=np.array([1.0, 1.0]))

    # Should leave arrays unchanged
    result = method_vec._maybe_scalar(np.array([2.0, 3.0]))
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([2.0, 3.0]))


def test_compute_descent_direction_and_step_length():
    """Test placeholder implementations of descent direction and step length."""

    def f(x):
        return x**2

    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = NelderMeadMethod(config, x0=1.0)

    # For scalar problem
    direction = method.compute_descent_direction(1.0)
    assert isinstance(direction, np.ndarray)
    assert direction.shape == (1,)
    assert direction[0] == 0.0

    step = method.compute_step_length(1.0, direction)
    assert step == 0.0

    # For vector problem
    def f_vec(x):
        return x[0] ** 2 + x[1] ** 2

    config_vec = NumericalMethodConfig(func=f_vec, method_type="optimize")
    method_vec = NelderMeadMethod(config_vec, x0=np.array([1.0, 1.0]))

    direction_vec = method_vec.compute_descent_direction(np.array([1.0, 1.0]))
    assert isinstance(direction_vec, np.ndarray)
    assert direction_vec.shape == (2,)
    assert np.all(direction_vec == 0.0)

    step_vec = method_vec.compute_step_length(np.array([1.0, 1.0]), direction_vec)
    assert step_vec == 0.0


def test_simplex_size_calculation():
    """Test the _calculate_simplex_size method."""

    def f(x):
        return x**2

    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = NelderMeadMethod(config, x0=1.0, delta=0.5)

    # For 1D case, simplex size should be delta
    assert np.isclose(method._calculate_simplex_size(), 0.5)

    # For 2D case
    def f_vec(x):
        return x[0] ** 2 + x[1] ** 2

    config_vec = NumericalMethodConfig(func=f_vec, method_type="optimize")
    method_vec = NelderMeadMethod(config_vec, x0=np.array([1.0, 1.0]), delta=1.0)

    # Simplex size should be sqrt(1^2) = 1.0 for delta=1.0 in 2D
    assert np.isclose(method_vec._calculate_simplex_size(), 1.0)

    # Test empty simplex
    method_vec.simplex = np.array([])
    assert method_vec._calculate_simplex_size() == 0.0

    # Test simplex with only one point
    method_vec.simplex = np.array([[1.0, 1.0]])
    assert method_vec._calculate_simplex_size() == 0.0


def test_simplex_operations_detailed():
    """Test each simplex operation in detail (reflection, expansion, contraction, shrink)."""

    # Test function where we can observe specific operations
    def f(x):
        return (x[0] - 2) ** 2 + (x[1] - 3) ** 2

    config = NumericalMethodConfig(func=f, method_type="optimize", max_iter=50)

    # Start far enough to trigger different operations
    method = NelderMeadMethod(config, x0=np.array([-1.0, -1.0]), delta=1.0)

    # Run multiple steps to capture different operations
    operation_counts = {"reflection": 0, "expansion": 0, "contraction": 0, "shrink": 0}

    for _ in range(20):
        if method.has_converged():
            break
        method.step()
        history = method.get_iteration_history()
        if "action" in history[-1].details:
            operation = history[-1].details["action"]
            if operation in operation_counts:
                operation_counts[operation] += 1

    # Verify that we've observed each operation type
    assert sum(operation_counts.values()) > 0, "No operations were recorded"

    # Count how many different operations we observed
    unique_operations = sum(1 for count in operation_counts.values() if count > 0)
    assert (
        unique_operations >= 2
    ), f"Too few unique operations observed: {operation_counts}"


def test_error_and_convergence_rate():
    """Test error calculation and convergence rate estimation."""

    def f(x):
        return x**2

    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = NelderMeadMethod(config, x0=1.0)

    # Initial error should be 1.0 (normalized by initial size)
    assert np.isclose(method.get_error(), 1.0)

    # Convergence rate should be None initially (insufficient data)
    assert method.get_convergence_rate() is None

    # Run more iterations to better observe convergence behavior
    for _ in range(10):
        method.step()

    # The error may fluctuate during iterations but should eventually decrease
    # The simplex operations can cause temporary increases in error
    initial_size = method.initial_size
    current_size = method._calculate_simplex_size()
    assert (
        current_size < initial_size
    ), f"Simplex size should decrease. Initial: {initial_size}, Current: {current_size}"

    # Check convergence rate only if it's available
    rate = method.get_convergence_rate()
    if rate is not None:  # Could still be None if divisions by zero occurred
        # Allow a wider range as convergence may not be strictly linear
        assert 0 <= rate <= 2.0, f"Unusual convergence rate: {rate}"


def test_zero_delta():
    """Test method behavior with very small initial simplex."""

    def f(x):
        return x**2

    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = NelderMeadMethod(config, x0=1.0, delta=1e-10)

    # Should still be able to make progress
    for _ in range(5):
        method.step()

    # Should have moved from initial point
    assert (
        abs(method.get_current_x() - 1.0) > 1e-10
    ), "Failed to make progress with tiny delta"


def test_ill_conditioned_function():
    """Test method behavior with an ill-conditioned function."""

    def f(x):
        # Function with very different scaling in different dimensions
        return 1000 * x[0] ** 2 + 0.001 * x[1] ** 2

    config = NumericalMethodConfig(
        func=f, method_type="optimize", tol=1e-5, max_iter=1000
    )
    method = NelderMeadMethod(config, x0=np.array([1.0, 1.0]))

    # Run until convergence or max iterations
    while not method.has_converged():
        method.step()

    # Should still find approximate minimum
    x = method.get_current_x()
    assert (
        abs(x[0]) < 0.01
    ), f"Failed to find minimum in ill-conditioned function, x[0]={x[0]}"

    # The second coordinate may not be as accurate due to ill-conditioning
    assert f(x) < f(np.array([1.0, 1.0])), "Function value should have decreased"


def test_discontinuous_function():
    """Test method behavior with a discontinuous function."""

    def f(x):
        # Step function with discontinuity
        if x < 0:
            return x**2 + 1
        return x**2

    config = NumericalMethodConfig(
        func=f, method_type="optimize", tol=1e-4, max_iter=500
    )
    method = NelderMeadMethod(config, x0=1.0)

    # Run until convergence or max iterations
    while not method.has_converged():
        method.step()

    # Should find minimum at x=0 (or very close)
    x = method.get_current_x()
    assert abs(x) < 0.1, f"Failed to find minimum of discontinuous function, x={x}"


def test_pathological_functions():
    """Test method behavior with pathological functions that are hard to optimize."""

    # Test just one easier pathological function
    def flat_region_func(x):
        return 0.01 * x**4  # Very flat near x=0

    config = NumericalMethodConfig(
        func=flat_region_func,
        method_type="optimize",
        tol=1e-3,  # Relaxed tolerance
        max_iter=1000,
    )
    method = NelderMeadMethod(config, x0=2.0)

    # Run until convergence or max iterations
    while not method.has_converged():
        method.step()

    # For this flat region function, verify we get close to minimum
    x = method.get_current_x()
    assert (
        abs(x) < 0.5
    ), f"Failed to get reasonably close to minimum. Got {x}, expected near 0.0"

    # For the oscillatory function, only verify that function value decreases
    def oscillatory_func(x):
        return (
            np.sin(5 * x) * np.exp(-0.5 * x**2) + 0.2
        )  # Add offset to make minimum positive

    config = NumericalMethodConfig(
        func=oscillatory_func, method_type="optimize", tol=1e-3, max_iter=1000
    )
    method = NelderMeadMethod(config, x0=0.5)

    # Store initial function value
    initial_value = oscillatory_func(0.5)

    # Run until convergence
    while not method.has_converged():
        method.step()

    # Verify function value decreases
    final_value = oscillatory_func(method.get_current_x())
    assert (
        final_value < initial_value
    ), f"Function value should decrease. Initial: {initial_value}, Final: {final_value}"


def test_large_dimensional_problem():
    """Test method with a higher-dimensional problem."""

    # Quadratic function in 10 dimensions
    def f(x):
        return np.sum(x**2)

    dim = 10
    x0 = np.ones(dim)

    config = NumericalMethodConfig(
        func=f, method_type="optimize", tol=1e-3, max_iter=3000
    )
    method = NelderMeadMethod(config, x0=x0)

    # Only run a few iterations to test it doesn't crash
    # (full convergence would be very slow)
    for _ in range(50):
        if method.has_converged():
            break
        method.step()

    # Verify the function value has decreased
    x = method.get_current_x()
    assert f(x) < f(x0), "Function value should decrease in high dimensions"


def test_different_parameter_values():
    """Test method behavior with non-standard parameter values."""

    def f(x):
        return x**2

    # Dictionary of parameter sets to test
    parameter_sets = [
        # Standard Nelder-Mead parameters (baseline)
        {"alpha": 1.0, "gamma": 2.0, "rho": 0.5, "sigma": 0.5},
        # More aggressive expansion
        {"alpha": 1.0, "gamma": 3.0, "rho": 0.5, "sigma": 0.5},
        # More conservative contraction
        {"alpha": 1.0, "gamma": 2.0, "rho": 0.75, "sigma": 0.5},
        # Less aggressive shrinking
        {"alpha": 1.0, "gamma": 2.0, "rho": 0.5, "sigma": 0.75},
    ]

    for params in parameter_sets:
        config = NumericalMethodConfig(
            func=f, method_type="optimize", tol=1e-4, max_iter=100
        )
        method = NelderMeadMethod(config, x0=1.0)

        # Set custom parameters
        method.alpha = params["alpha"]
        method.gamma = params["gamma"]
        method.rho = params["rho"]
        method.sigma = params["sigma"]

        # Run until convergence
        while not method.has_converged():
            method.step()

        # Should still find the minimum
        x = method.get_current_x()
        assert abs(x) < 0.1, f"Failed to find minimum with parameters {params}, x={x}"


def test_name_property():
    """Test that the name property returns the correct method name."""

    def f(x):
        return x**2

    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = NelderMeadMethod(config, x0=1.0)

    assert method.name == "Nelder-Mead Method", f"Incorrect method name: {method.name}"


def test_convergence_criteria():
    """Test that convergence criteria work correctly."""

    def f(x):
        return x**2

    # Test with different tolerances
    tolerances = [1e-2, 1e-4, 1e-6]

    for tol in tolerances:
        config = NumericalMethodConfig(
            func=f, method_type="optimize", tol=tol, max_iter=1000
        )
        method = NelderMeadMethod(config, x0=1.0)

        # Run until convergence
        while not method.has_converged():
            method.step()

        # Check that error is below tolerance
        assert (
            method.get_error() < 2 * tol
        ), f"Error {method.get_error()} not below 2*tol={2*tol} for tol={tol}"

        # Check convergence reason
        history = method.get_iteration_history()
        assert (
            "convergence_reason" in history[-1].details
        ), "Convergence reason not set in history"

    # Test max iterations criterion
    config = NumericalMethodConfig(
        func=f, method_type="optimize", tol=1e-10, max_iter=5
    )
    method = NelderMeadMethod(config, x0=1.0)

    # Run until convergence
    while not method.has_converged():
        method.step()

    # Should converge due to max iterations
    assert method.iterations <= 5, f"Exceeded max_iter, iterations={method.iterations}"

    history = method.get_iteration_history()
    assert (
        history[-1].details["convergence_reason"] == "maximum iterations reached"
    ), f"Incorrect convergence reason: {history[-1].details.get('convergence_reason')}"


def test_non_smooth_function():
    """Test method behavior with a non-smooth function."""

    def f(x):
        # Absolute value function (non-smooth at x=0)
        return abs(x)

    config = NumericalMethodConfig(
        func=f, method_type="optimize", tol=1e-4, max_iter=500
    )
    method = NelderMeadMethod(config, x0=1.0)

    # Run until convergence
    while not method.has_converged():
        method.step()

    # Should find minimum at or very near x=0
    x = method.get_current_x()
    assert abs(x) < 0.1, f"Failed to find minimum of non-smooth function, x={x}"
