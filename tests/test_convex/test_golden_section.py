# tests/test_convex/test_golden_section.py

import math
import sys
import numpy as np
import pytest
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from algorithms.convex.golden_section import GoldenSectionMethod, golden_section_search
from algorithms.convex.protocols import NumericalMethodConfig


def test_basic_root_finding():
    """Test finding sqrt(2) using x^2 - 2 = 0"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root")
    method = GoldenSectionMethod(config, 1, 2)

    while not method.has_converged():
        x = method.step()

    assert abs(x - math.sqrt(2)) < 1e-6
    assert method.iterations < 100


def test_golden_ratio_constants():
    """Test that golden ratio constants are correctly initialized"""

    def f(x):
        return x - 1

    config = NumericalMethodConfig(func=f, method_type="root")
    method = GoldenSectionMethod(config, 0, 2)

    # Check phi (golden ratio)
    assert abs(method.phi - (1 + math.sqrt(5)) / 2) < 1e-10
    # Check tau (inverse golden ratio)
    assert abs(method.tau - 1 / method.phi) < 1e-10
    # Verify relationship
    assert abs(method.phi * method.tau - 1) < 1e-10


def test_test_points_placement():
    """Test that test points are properly placed using golden ratio"""

    def f(x):
        return x**2 - 4

    config = NumericalMethodConfig(func=f, method_type="root")
    method = GoldenSectionMethod(config, 0, 3)

    # Check initial points
    assert 0 < method.x1 < method.x2 < 3
    # Verify golden ratio relationships
    ratio1 = (method.x1 - method.a) / (method.b - method.a)
    ratio2 = (method.x2 - method.a) / (method.b - method.a)
    assert abs(ratio1 - (1 - method.tau)) < 1e-10
    assert abs(ratio2 - method.tau) < 1e-10


def test_convergence_criteria():
    """Test that method converges within tolerance"""

    def f(x):
        return x**3 - x - 2

    config = NumericalMethodConfig(func=f, method_type="root", tol=1e-6)
    method = GoldenSectionMethod(config, 1, 2)

    while not method.has_converged():
        x = method.step()

    # Check convergence to root
    assert abs(f(x)) < 1e-6


def test_iteration_history():
    """Test that iteration history is properly recorded"""

    def f(x):
        return x**2 - 4

    config = NumericalMethodConfig(func=f, method_type="root")
    method = GoldenSectionMethod(config, 0, 3)

    # Perform a few steps
    for _ in range(3):
        method.step()

    history = method.get_iteration_history()
    assert len(history) == 3

    # Verify history contains golden ratio details
    for data in history:
        assert "tau" in data.details
        assert abs(data.details["tau"] - method.tau) < 1e-10


def test_legacy_wrapper():
    """Test the backward-compatible golden_section_search function"""

    def f(x):
        return x**2 - 2

    root, errors, iters = golden_section_search(f, 1, 2, tol=1e-6, method_type="root")

    assert abs(root - math.sqrt(2)) < 1e-6
    assert len(errors) == iters


def test_numerical_stability():
    """Test handling of nearly equal function values"""

    def f(x):
        return x**2  # Function with minimum at x=0

    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = GoldenSectionMethod(config, -1, 1)

    # Force nearly equal function values
    method.f1 = method.f2 = 1e-10

    # Should not raise any errors
    x = method.step()
    assert -1 <= x <= 1


def test_different_functions():
    """Test method works with different types of functions"""
    test_cases = [
        (lambda x: x**3 - x - 2, 1, 2),  # Cubic
        (lambda x: math.exp(x) - 4, 1.3, 1.4),  # Exponential
        (lambda x: math.sin(x), 3.1, 3.2),  # Trigonometric
    ]

    for func, a, b in test_cases:
        config = NumericalMethodConfig(func=func, method_type="root", tol=1e-4)
        method = GoldenSectionMethod(config, a, b)

        while not method.has_converged():
            x = method.step()

        assert abs(func(x)) < 1e-4


def test_get_current_x():
    """Test that get_current_x returns the latest approximation"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root")
    method = GoldenSectionMethod(config, 1, 2)

    x = method.step()
    assert x == method.get_current_x()


def test_interval_reduction():
    """Test that interval is properly reduced using golden ratio"""

    def f(x):
        return x - 1

    config = NumericalMethodConfig(func=f, method_type="root")
    method = GoldenSectionMethod(config, 0, 2)

    initial_width = method.b - method.a
    x = method.step()

    # Check interval reduction
    assert method.b - method.a < initial_width
    # Check new point is within bounds
    assert method.a <= x <= method.b
    # Verify reduction ratio approximately follows golden ratio
    reduction_ratio = (method.b - method.a) / initial_width
    assert abs(reduction_ratio - method.tau) < 0.1


def test_max_iterations():
    """Test that method respects maximum iterations"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root", max_iter=5)
    method = GoldenSectionMethod(config, 1, 2)

    while not method.has_converged():
        method.step()

    assert method.iterations <= 5


def test_basic_optimization():
    """Test finding the minimum of a simple quadratic function"""

    def f(x):
        # Simple quadratic function with minimum at x=2
        return (x - 2) ** 2

    config = NumericalMethodConfig(func=f, method_type="optimize", tol=1e-5)
    method = GoldenSectionMethod(config, 0, 4)

    while not method.has_converged():
        x = method.step()

    # The minimum should be close to x=2
    assert abs(x - 2) < 1e-4

    # Function value should be close to zero at minimum
    assert f(x) < 1e-8


def test_optimization_vs_root_finding():
    """Test that the method behaves differently in optimization vs root-finding modes"""

    def f(x):
        # Function with a root at x=2 and minimum at x=1.5
        return (x - 2) * (x - 1)

    # Test in optimization mode (should find minimum at x=1.5)
    config_opt = NumericalMethodConfig(func=f, method_type="optimize", tol=1e-5)
    method_opt = GoldenSectionMethod(config_opt, 0, 3)

    while not method_opt.has_converged():
        x_opt = method_opt.step()

    # Test in root-finding mode (should find root at x=2)
    config_root = NumericalMethodConfig(func=f, method_type="root", tol=1e-5)
    method_root = GoldenSectionMethod(config_root, 1.5, 3)

    while not method_root.has_converged():
        x_root = method_root.step()

    # Check optimization mode found the minimum close to x=1.5
    assert abs(x_opt - 1.5) < 1e-4

    # Check root-finding mode found the root close to x=2
    assert abs(x_root - 2) < 1e-4


def test_convergence_rate():
    """Test that convergence rate estimation works"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root")
    method = GoldenSectionMethod(config, 1, 2)

    # Run enough iterations to get convergence rate data
    for _ in range(5):
        method.step()

    rate = method.get_convergence_rate()

    # Rate should be close to the inverse golden ratio (0.618)
    if rate is not None:  # Will be None if not enough iterations
        assert 0.5 < rate < 0.7  # Allow some flexibility


def test_name_property():
    """Test that the name property returns the correct name based on method type"""

    def f(x):
        return x**2 - 2

    # Test root-finding name
    config_root = NumericalMethodConfig(func=f, method_type="root")
    method_root = GoldenSectionMethod(config_root, 1, 2)
    assert method_root.name == "Golden Section Root-Finding Method"

    # Test optimization name
    config_opt = NumericalMethodConfig(func=f, method_type="optimize")
    method_opt = GoldenSectionMethod(config_opt, 1, 2)
    assert method_opt.name == "Golden Section Optimization Method"


def test_error_calculation():
    """Test that error is calculated correctly for both methods"""

    # Root finding: error = |f(x)|
    def f1(x):
        return x**2 - 2

    config_root = NumericalMethodConfig(func=f1, method_type="root")
    method_root = GoldenSectionMethod(config_root, 1, 2)
    method_root.step()
    x = method_root.get_current_x()
    assert abs(method_root.get_error() - abs(f1(x))) < 1e-10

    # Optimization without derivative: error calculated using numerical estimation
    def f2(x):
        return x**2

    config_opt = NumericalMethodConfig(func=f2, method_type="optimize")
    method_opt = GoldenSectionMethod(config_opt, -1, 1)
    method_opt.step()
    x = method_opt.get_current_x()
    # Error should be reasonable for an optimization step
    assert method_opt.get_error() >= 0

    # Optimization with derivative: error = |f'(x)|
    def f3(x):
        return x**2

    def df3(x):
        return 2 * x

    config_with_deriv = NumericalMethodConfig(
        func=f3, method_type="optimize", derivative=df3
    )
    method_with_deriv = GoldenSectionMethod(config_with_deriv, -1, 1)
    method_with_deriv.step()
    x = method_with_deriv.get_current_x()
    assert abs(method_with_deriv.get_error() - abs(df3(x))) < 1e-10


def test_pathological_function_steep():
    """Test with a very steep function that could cause numerical issues"""

    def f(x):
        # Function with very large derivative near x=1
        return 1e6 * (x - 1) ** 3

    config = NumericalMethodConfig(func=f, method_type="root")
    method = GoldenSectionMethod(
        config, 0.99, 1.01
    )  # Tight interval around root at x=1

    while not method.has_converged():
        x = method.step()

    # Golden Section method might be less accurate with very steep functions
    assert abs(x - 1) < 1e-4  # Use a relaxed tolerance
    assert abs(f(x)) < 1e-3


def test_pathological_function_flat():
    """Test with a very flat function that could make convergence difficult"""

    def f(x):
        # Function with very small derivative near x=1
        return 1e-6 * (x - 1) ** 3

    config = NumericalMethodConfig(func=f, method_type="root")
    method = GoldenSectionMethod(config, 0, 2)

    while not method.has_converged():
        x = method.step()

    # Golden Section may struggle with very flat functions
    assert abs(x - 1) < 0.5  # Use a relaxed tolerance
    assert abs(f(x)) < 1e-5


def test_multiple_roots():
    """Test behavior with a function that has multiple roots in the interval"""

    def f(x):
        # Function with roots at x=0, x=1, and x=2
        return x * (x - 1) * (x - 2)

    # Test each subinterval to find all three roots
    roots = []
    for interval in [(-0.5, 0.5), (0.5, 1.5), (1.5, 2.5)]:
        config = NumericalMethodConfig(func=f, method_type="root")
        method = GoldenSectionMethod(config, interval[0], interval[1])

        while not method.has_converged():
            x = method.step()

        roots.append(x)

    # Check that we found all three roots
    expected_roots = [0, 1, 2]
    for expected, actual in zip(expected_roots, roots):
        assert abs(actual - expected) < 1e-4


def test_multiple_extrema():
    """Test behavior with a function that has multiple extrema"""

    def f(x):
        # Function with minimum at x=1 and maximum at x=3
        return (x - 1) ** 2 * (x - 3) ** 2

    # Test finding the minimum at x=1
    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = GoldenSectionMethod(config, 0, 2)

    while not method.has_converged():
        x = method.step()

    # Should find the minimum at x=1
    assert abs(x - 1) < 1e-4

    # Now test finding the minimum at x=3
    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = GoldenSectionMethod(config, 2, 4)

    while not method.has_converged():
        x = method.step()

    # Should find the minimum at x=3
    assert abs(x - 3) < 1e-4


def test_extreme_tolerance():
    """Test the method with extremely small tolerance values"""

    def f(x):
        return x**2 - 2  # Root at sqrt(2)

    # Test with extremely small tolerance
    config = NumericalMethodConfig(func=f, method_type="root", tol=1e-12)
    method = GoldenSectionMethod(config, 1, 2)

    while not method.has_converged():
        x = method.step()

    assert abs(x - math.sqrt(2)) < 1e-6
    assert abs(f(x)) < 1e-6


def test_very_small_intervals():
    """Test with extremely small intervals"""

    def f(x):
        return x**2 - 2  # Root at sqrt(2)

    # Calculate sqrt(2) with high precision
    actual_root = math.sqrt(2)

    # Use a small interval around the root
    epsilon = 1e-4
    left = actual_root - epsilon
    right = actual_root + epsilon

    config = NumericalMethodConfig(func=f, method_type="root")
    method = GoldenSectionMethod(config, left, right)

    while not method.has_converged():
        x = method.step()

    assert abs(x - actual_root) < 2 * epsilon
    assert abs(f(x)) < 1e-5


def test_overflow_function():
    """Test with functions that could cause overflow"""

    def f(x):
        # Function with controlled growth
        return math.exp(x) - 1000  # Root around x=6.91

    if sys.float_info.max > 1e100:  # Only run if platform supports large floats
        config = NumericalMethodConfig(func=f, method_type="root")
        method = GoldenSectionMethod(config, 0, 10)

        while not method.has_converged():
            x = method.step()

        expected_root = math.log(1000)  # Should be approximately 6.91
        assert abs(x - expected_root) < 1e-2
        # Relaxed tolerance due to numerical issues with exponentials
        assert abs(f(x)) < 5e-3


def test_underflow_function():
    """Test with functions that could cause underflow"""

    def f(x):
        # This function approaches zero very rapidly
        return math.exp(-(x**2)) - 0.5  # Roots around x=±0.83

    config = NumericalMethodConfig(func=f, method_type="root")
    method = GoldenSectionMethod(config, 0, 2)

    while not method.has_converged():
        x = method.step()

    expected_root = math.sqrt(-math.log(0.5))  # Should be approximately 0.83
    assert abs(x - expected_root) < 1e-2
    assert abs(f(x)) < 1e-3


def test_weird_function():
    """Test with a function that has unusual behavior"""

    def f(x):
        # A function that's zero at integer values and oscillates in between
        return math.sin(math.pi * x)

    # Test finding multiple zeros
    for expected_root in range(-2, 3):
        config = NumericalMethodConfig(func=f, method_type="root")
        method = GoldenSectionMethod(config, expected_root - 0.5, expected_root + 0.5)

        while not method.has_converged():
            x = method.step()

        assert abs(x - expected_root) < 1e-3
        assert abs(f(x)) < 1e-3


def test_compute_descent_direction():
    """Test that compute_descent_direction returns reasonable values"""

    def f(x):
        return x**2 - 2

    # Root-finding mode
    config_root = NumericalMethodConfig(func=f, method_type="root")
    method_root = GoldenSectionMethod(config_root, 1, 2)

    # For x < sqrt(2), f(x) < 0, so direction should be positive
    direction_low = method_root.compute_descent_direction(1.0)
    assert direction_low > 0

    # For x > sqrt(2), f(x) > 0, so direction should be negative
    direction_high = method_root.compute_descent_direction(1.5)
    assert direction_high < 0

    # Optimization mode
    config_opt = NumericalMethodConfig(func=f, method_type="optimize")
    method_opt = GoldenSectionMethod(config_opt, 1, 2)

    # Should return a placeholder value
    direction_opt = method_opt.compute_step_length(1.5, 0.0)
    assert isinstance(direction_opt, float)


def test_compute_step_length():
    """Test that compute_step_length returns a reasonable value"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root")
    method = GoldenSectionMethod(config, 1, 2)

    # Should return a fraction of the interval width
    step = method.compute_step_length(1.5, 1.0)
    assert 0 < step < 1
    assert abs(step - method.tau * (method.b - method.a)) < 1e-10


def test_record_initial_state():
    """Test that initial state is properly recorded when requested"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root")
    method = GoldenSectionMethod(config, 1, 2, record_initial_state=True)

    # Get history without doing any steps
    history = method.get_iteration_history()
    assert len(history) == 1  # Should have one record

    # Initial record should contain expected data
    initial_data = history[0]
    assert "a" in initial_data.details
    assert "b" in initial_data.details
    assert "x1" in initial_data.details
    assert "x2" in initial_data.details
    assert "f(x1)" in initial_data.details
    assert "f(x2)" in initial_data.details
    assert "phi" in initial_data.details
    assert "tau" in initial_data.details
    assert "method_type" in initial_data.details


def test_callback_function():
    """Test with a function that includes a callback counter to verify function calls"""

    call_counts = []

    def f(x):
        nonlocal call_counts
        call_counts.append(x)  # Record function calls
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root", max_iter=10)
    method = GoldenSectionMethod(config, 1, 2)

    # Reset call tracking
    call_counts = []

    # Do one step and count calls
    method.step()
    first_step_calls = len(call_counts)

    # The Golden Section method should make function calls during the step
    assert first_step_calls > 0

    # Clear and do another step
    call_counts = []
    method.step()
    second_step_calls = len(call_counts)

    # Verify consistent behavior across steps
    assert second_step_calls > 0


def test_random_functions():
    """Test with random polynomials to ensure robustness"""

    # Set random seed for reproducibility
    np.random.seed(42)

    for _ in range(5):  # Test with 5 different random functions
        # Generate random polynomial coefficients
        degree = np.random.randint(2, 6)  # degree between 2 and 5
        coeffs = np.random.uniform(-10, 10, size=degree + 1)

        def polynomial(x):
            return np.polyval(coeffs, x)

        # Find interval with sign change using random search
        found_interval = False
        for _ in range(20):  # Try 20 times to find interval
            a = np.random.uniform(-10, 10)
            b = np.random.uniform(a + 0.1, a + 5)  # Ensure b > a

            if polynomial(a) * polynomial(b) < 0:
                found_interval = True
                break

        if found_interval:
            config = NumericalMethodConfig(func=polynomial, method_type="root")
            method = GoldenSectionMethod(config, a, b)

            while not method.has_converged():
                x = method.step()

            # Verify the root
            assert abs(polynomial(x)) < 1e-3


def test_invalid_method_type():
    """Test that initialization fails when method_type is not valid"""

    def f(x):
        return x**2 - 2

    # Test with invalid method type
    config = NumericalMethodConfig(func=f, method_type="invalid")
    with pytest.raises(ValueError):
        GoldenSectionMethod(config, 1, 2)


def test_invalid_interval():
    """Test that initialization fails with invalid interval"""

    def f(x):
        return x**2 - 2

    # Test with invalid interval (a >= b)
    config = NumericalMethodConfig(func=f, method_type="root")
    with pytest.raises(ValueError):
        GoldenSectionMethod(config, 2, 1)  # a > b


def test_convergence_reasons():
    """Test that different convergence reasons are properly recorded"""

    def f(x):
        return x**2 - 2

    # Test convergence due to function value within tolerance
    config1 = NumericalMethodConfig(func=f, method_type="root", tol=1e-2)
    method1 = GoldenSectionMethod(config1, 1.4, 1.5)  # Close to sqrt(2)

    while not method1.has_converged():
        method1.step()

    last_iteration1 = method1.get_iteration_history()[-1]
    assert "convergence_reason" in last_iteration1.details

    # Test convergence due to interval width
    config2 = NumericalMethodConfig(func=f, method_type="root", tol=1e-10)
    method2 = GoldenSectionMethod(config2, 1, 2)

    while not method2.has_converged():
        method2.step()

    last_iteration2 = method2.get_iteration_history()[-1]
    assert "convergence_reason" in last_iteration2.details

    # Test convergence due to max iterations
    config3 = NumericalMethodConfig(func=f, method_type="root", tol=1e-15, max_iter=3)
    method3 = GoldenSectionMethod(config3, 1, 2)

    while not method3.has_converged():
        method3.step()

    assert method3.iterations <= 3
    last_iteration3 = method3.get_iteration_history()[-1]
    assert "convergence_reason" in last_iteration3.details
    assert "maximum iterations" in last_iteration3.details["convergence_reason"]


def test_optimization_with_derivative():
    """Test optimization when derivative is provided"""

    def f(x):
        return (x - 3) ** 2 + 1  # Minimum at x=3

    def df(x):
        return 2 * (x - 3)  # Derivative is zero at x=3

    config = NumericalMethodConfig(func=f, method_type="optimize", derivative=df)
    method = GoldenSectionMethod(config, 2, 4)

    while not method.has_converged():
        x = method.step()

    assert abs(x - 3) < 1e-3  # Should find minimum at x=3
    assert method.iterations < 100
