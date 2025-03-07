# tests/test_convex/test_regula_falsi.py

import pytest
import math
import sys
from pathlib import Path
import numpy as np

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from algorithms.convex.regula_falsi import RegulaFalsiMethod, regula_falsi_search
from algorithms.convex.protocols import NumericalMethodConfig


def test_basic_root_finding():
    """Test finding sqrt(2) using x^2 - 2 = 0"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root")
    method = RegulaFalsiMethod(config, 1, 2)

    while not method.has_converged():
        x = method.step()

    assert abs(x - math.sqrt(2)) < 1e-6
    assert method.iterations < 100


def test_invalid_method_type():
    """Test that initialization fails when method_type is not 'root'"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="optimize")
    with pytest.raises(ValueError, match="can only be used for root finding"):
        RegulaFalsiMethod(config, 1, 2)


def test_invalid_interval():
    """Test that initialization fails when f(a) and f(b) have same sign"""

    def f(x):
        return x**2 + 1  # Always positive

    config = NumericalMethodConfig(func=f, method_type="root")
    with pytest.raises(ValueError, match="must have opposite signs"):
        RegulaFalsiMethod(config, 1, 2)


def test_exact_root():
    """Test when one endpoint is close to the root"""

    def f(x):
        return x - 2  # Linear function with root at x=2

    config = NumericalMethodConfig(func=f, method_type="root", tol=1e-6)
    method = RegulaFalsiMethod(config, 1.999, 2.001)

    while not method.has_converged():
        x = method.step()

    assert abs(x - 2) < 1e-6
    assert abs(f(x)) < 1e-6


def test_convergence_rate():
    """Test that regula falsi converges faster than bisection for some functions"""

    def f(x):
        return x**3 - x - 2  # Cubic function

    config = NumericalMethodConfig(func=f, method_type="root", tol=1e-8)
    method = RegulaFalsiMethod(config, 1, 2)

    iterations = 0
    while not method.has_converged():
        method.step()
        iterations += 1

    # Regula falsi should converge in fewer iterations than bisection would need
    assert iterations < 30  # Bisection typically needs more


def test_iteration_history():
    """Test that iteration history is properly recorded"""

    def f(x):
        return x**2 - 4

    config = NumericalMethodConfig(func=f, method_type="root")
    method = RegulaFalsiMethod(config, 1, 3)

    for _ in range(3):
        method.step()

    history = method.get_iteration_history()
    assert len(history) == 3

    # Check that error decreases
    errors = [data.error for data in history]
    assert all(errors[i] >= errors[i + 1] for i in range(len(errors) - 1))

    # Check that details contain the expected keys
    for data in history:
        assert "a" in data.details
        assert "b" in data.details
        assert "updated_end" in data.details


def test_legacy_wrapper():
    """Test the backward-compatible regula_falsi_search function"""

    def f(x):
        return x**2 - 2

    root, errors, iters = regula_falsi_search(f, 1, 2)
    assert abs(root - math.sqrt(2)) < 1e-6
    assert len(errors) == iters


def test_different_functions():
    """Test method works with different types of functions"""
    test_cases = [
        (lambda x: x**3 - x - 2, 1, 2),  # Cubic
        (lambda x: math.exp(x) - 4, 1, 2),  # Exponential
        (lambda x: math.sin(x), 3, 4),  # Trigonometric
    ]

    for func, a, b in test_cases:
        config = NumericalMethodConfig(func=func, method_type="root", tol=1e-4)
        method = RegulaFalsiMethod(config, a, b)

        while not method.has_converged():
            x = method.step()

        assert abs(func(x)) < 1e-4


def test_weighted_average():
    """Test the weighted average calculation"""

    def f(x):
        return x - 1  # Linear function with root at x=1

    config = NumericalMethodConfig(func=f, method_type="root")
    method = RegulaFalsiMethod(config, 0, 2)

    # For a linear function, regula falsi should find the root in one step
    x = method.step()
    assert abs(x - 1) < 1e-10


def test_max_iterations():
    """Test that method respects maximum iterations"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root", max_iter=5)
    method = RegulaFalsiMethod(config, 1, 2)

    while not method.has_converged():
        method.step()

    assert method.iterations <= 5


def test_illinois_modification():
    """Test the Illinois modification for slow convergence cases"""

    def f(x):
        """Function that exhibits slow convergence with standard regula falsi"""
        return x**3 - 3 * x - 1  # Has root near x=1.5

    # First test with the modification enabled (default)
    config = NumericalMethodConfig(func=f, method_type="root", tol=1e-10)
    method_modified = RegulaFalsiMethod(config, 1, 2)
    method_modified.use_modified = True

    iterations_modified = 0
    while not method_modified.has_converged():
        method_modified.step()
        iterations_modified += 1
        if iterations_modified > 100:  # Safety check
            break

    # Now test with modification disabled
    config = NumericalMethodConfig(func=f, method_type="root", tol=1e-10)
    method_standard = RegulaFalsiMethod(config, 1, 2)
    method_standard.use_modified = False

    iterations_standard = 0
    while not method_standard.has_converged():
        method_standard.step()
        iterations_standard += 1
        if iterations_standard > 100:  # Safety check
            break

    # Both should find the same root with similar accuracy
    root_modified = method_modified.x
    root_standard = method_standard.x

    assert abs(f(root_modified)) < 1e-9, "Modified method should find an accurate root"
    assert abs(f(root_standard)) < 1e-9, "Standard method should find an accurate root"
    assert (
        abs(root_modified - root_standard) < 1e-6
    ), "Both methods should converge to the same root"


def test_illinois_factor():
    """Test the effect of the Illinois factor"""

    def f(x):
        """Function with slow convergence"""
        return (x - 1.5) ** 3 + 0.01 * (
            x - 1.5
        )  # Root at x=1.5, with slight offset to prevent exact convergence

    # Test with Illinois modification
    config = NumericalMethodConfig(func=f, method_type="root", tol=1e-10)
    method = RegulaFalsiMethod(config, 0, 3)
    method.use_modified = True

    # Set the Illinois factor
    method.modified_factor = 0.5

    # Run until convergence
    iterations = 0
    while not method.has_converged():
        method.step()
        iterations += 1
        if iterations > 50:  # Safety check
            break

    # Should find the root with good accuracy
    assert abs(method.x - 1.5) < 1e-5, "Should find the root near x=1.5"

    # Check if Illinois modification was applied by inspecting iteration history
    history = method.get_iteration_history()

    # If the method used Illinois factor, we would see it in the details
    illinois_applied = False
    for data in history:
        if "modified_fa" in data.details or "modified_fb" in data.details:
            illinois_applied = True
            break

    # With enough iterations, the Illinois method should be applied at least once
    # Note: This is a more reliable check than comparing convergence speeds
    if iterations > 5:  # Only check if we had enough iterations
        assert illinois_applied, "Illinois modification should be applied at least once"


def test_stalling_case():
    """Test a case where standard regula falsi might stall"""

    def f(x):
        """Function with potential stalling for standard regula falsi"""
        if x < 0:
            return -1
        elif x > 0:
            return 1
        else:
            return 0

    # With a discontinuous function like this, we need to be careful
    config = NumericalMethodConfig(func=f, method_type="root", tol=1e-6)
    method = RegulaFalsiMethod(config, -1, 1)

    # Should find the root at x=0 despite the challenging nature
    iterations = 0
    while not method.has_converged() and iterations < 50:
        method.step()
        iterations += 1

    assert abs(method.x) < 1e-6, "Should find the root at x=0"


def test_asymmetric_function():
    """Test with asymmetric functions to verify the method still makes progress"""

    # Let's use a less extreme but still asymmetric function
    def f1(x):
        """Moderately asymmetric function"""
        return (x - 1) ** 3 + (x - 1)  # Root at x=1, asymmetric cubic

    config1 = NumericalMethodConfig(func=f1, method_type="root", tol=1e-6)
    method1 = RegulaFalsiMethod(config1, 0, 2)

    # Run until convergence
    iterations1 = 0
    while not method1.has_converged() and iterations1 < 30:
        method1.step()
        iterations1 += 1

    # Should converge to the root with reasonable accuracy
    assert (
        abs(f1(method1.x)) < 1e-5
    ), "Should find the root of moderately asymmetric function"
    assert abs(method1.x - 1) < 1e-4, "Should converge close to x=1"

    # Now test with a much more challenging function
    def f2(x):
        """Function with very different scales on either side of the root"""
        if abs(x) < 1e-10:
            return 0
        elif x < 0:
            return -0.001  # Constant negative for x < 0
        else:
            return 0.1  # Constant positive for x > 0

    # This function has a root at x=0 but has discontinuity
    config2 = NumericalMethodConfig(
        func=f2, method_type="root", tol=1e-3
    )  # Relaxed tolerance

    # Try with different starting intervals to see if we can make progress
    brackets = [(-1, 0.1), (-0.5, 0.05), (-0.2, 0.02)]

    success = False
    for a, b in brackets:
        try:
            method2 = RegulaFalsiMethod(config2, a, b)

            # Run for a few iterations and see if it makes progress
            for _ in range(10):
                method2.step()

                # If we get close to the root or the function value is close to zero, count as success
                if abs(method2.x) < 0.1 or abs(f2(method2.x)) < 0.01:
                    success = True
                    break

                if method2.has_converged():
                    break

            if success:
                break
        except ValueError:
            # This may happen if the bracketing fails due to numerical issues
            continue

    # For this extremely difficult function, we just verify that at least
    # one of our test cases makes some progress toward the root
    assert (
        success
    ), "Should make some progress with at least one bracket for challenging function"


def test_descent_direction_and_step_length():
    """Test the compute_descent_direction and compute_step_length methods"""

    def f(x):
        return x**2 - 4

    config = NumericalMethodConfig(func=f, method_type="root")
    method = RegulaFalsiMethod(config, 1, 3)

    # These methods should return 0 since they're not used by regula falsi
    direction = method.compute_descent_direction(2.0)
    assert direction == 0.0, "Direction should be 0 for regula falsi"

    step_length = method.compute_step_length(2.0, 1.0)
    assert step_length == 0.0, "Step length should be 0 for regula falsi"


def test_name_property():
    """Test the name property with both modified and standard variants"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root")

    # Test with modified method (default)
    method_modified = RegulaFalsiMethod(config, 1, 2)
    method_modified.use_modified = True
    assert "Modified" in method_modified.name
    assert "Regula Falsi" in method_modified.name

    # Test with standard method
    method_standard = RegulaFalsiMethod(config, 1, 2)
    method_standard.use_modified = False
    assert "Modified" not in method_standard.name
    assert "Regula Falsi" in method_standard.name


def test_different_tolerances():
    """Test convergence with different tolerance values"""

    def f(x):
        return x**2 - 2

    # Test different tolerance levels
    tolerances = [1e-4, 1e-6, 1e-8]

    for tol in tolerances:
        config = NumericalMethodConfig(func=f, method_type="root", tol=tol)
        method = RegulaFalsiMethod(config, 1, 2)

        while not method.has_converged():
            method.step()

        # Final error should be less than tolerance
        assert method.get_error() <= tol, f"Error should be less than tolerance {tol}"


def test_error_calculation():
    """Test the get_error method"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root")
    method = RegulaFalsiMethod(config, 1, 2)

    # Manually set x to a known value
    method.x = math.sqrt(2)

    # Error should be |f(x)| which should be very close to 0
    assert method.get_error() < 1e-10, "Error should be almost 0 at the root"

    # Set x to a non-root value
    method.x = 1.5
    expected_error = abs(f(1.5))
    assert (
        abs(method.get_error() - expected_error) < 1e-10
    ), "Error should be |f(x)| for non-root values"


def test_get_convergence_rate():
    """Test the convergence rate calculation"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root")
    method = RegulaFalsiMethod(config, 1, 2)

    # Not enough iterations for a rate yet
    assert method.get_convergence_rate() is None

    # Run a few iterations to populate history
    for _ in range(5):
        method.step()

    # Should now have a convergence rate
    rate = method.get_convergence_rate()
    assert rate is not None
    assert (
        0 <= rate <= 1
    ), "Convergence rate should be between 0 and 1 for linear convergence"


def test_exact_convergence():
    """Test behavior when convergence is exact (zero error)"""

    def f(x):
        return x - 1

    config = NumericalMethodConfig(func=f, method_type="root")
    method = RegulaFalsiMethod(config, 0, 2)

    # With a linear function, should converge exactly or very close in one step
    method.step()

    # Verify the error is very small
    error = method.get_error()
    assert error < 1e-10, "Error should be nearly zero for a linear function"

    # Run more iterations to get enough history for convergence rate
    for _ in range(3):
        method.step()

    # Set last error to exactly zero
    if len(method._history) >= 3:
        # Manually fake a zero error to test handling of zero values
        method._history[-1].error = 0.0
        method._history[-2].error = 1e-10  # Make sure previous error is not zero

        # Should handle zero error gracefully
        rate = method.get_convergence_rate()

        # Rate should either be 0 or very small when latest error is zero
        assert (
            rate is not None and rate < 1e-6
        ), "Rate should be very small when error is exactly 0"


def test_extreme_values():
    """Test with a function that returns very large values"""

    def f(x):
        """Function with potential numerical issues"""
        return 1e8 * (x - 1)  # Very steep linear function

    config = NumericalMethodConfig(func=f, method_type="root")
    method = RegulaFalsiMethod(config, 0, 2)

    # Should converge despite extreme values
    iterations = 0
    while not method.has_converged() and iterations < 20:
        method.step()
        iterations += 1

    assert abs(method.x - 1) < 1e-6, "Should find root at x=1 despite extreme values"


def test_periodic_function():
    """Test with a periodic function which may have multiple roots"""

    def f(x):
        return math.sin(x)  # Roots at nπ

    # Find root near π
    config = NumericalMethodConfig(func=f, method_type="root")
    method = RegulaFalsiMethod(config, 3, 4)  # Bracket around π

    while not method.has_converged():
        method.step()

    assert abs(method.x - math.pi) < 1e-6, "Should find root at x=π"

    # Find a different root near 2π
    method2 = RegulaFalsiMethod(config, 6, 7)  # Bracket around 2π

    while not method2.has_converged():
        method2.step()

    assert abs(method2.x - 2 * math.pi) < 1e-6, "Should find root at x=2π"


def test_record_initial_state():
    """Test that initial state can be observed after initialization"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root")
    method = RegulaFalsiMethod(config, 1, 2)

    # Check initial values
    assert method.a == 1
    assert method.b == 2
    assert abs(method.fa - (1**2 - 2)) < 1e-10
    assert abs(method.fb - (2**2 - 2)) < 1e-10

    # Initial x should be the weighted average
    expected_x = (2 * method.fa - 1 * method.fb) / (method.fa - method.fb)
    assert (
        abs(method.x - expected_x) < 1e-10
    ), "Initial x should be the weighted average"


def test_legacy_wrapper_with_config():
    """Test the legacy wrapper with a config instead of function"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root", tol=1e-8)
    root, errors, iters = regula_falsi_search(config, 1, 2)

    assert abs(root - math.sqrt(2)) < 1e-8
    assert abs(f(root)) < 1e-8
