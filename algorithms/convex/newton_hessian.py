# algorithms/convex/newton_hessian.py

"""Newton-Hessian method for root finding."""

from typing import List, Tuple
import torch  # Used for automatic differentiation

from .protocols import BaseRootFinder, RootFinderConfig


class NewtonHessianMethod(BaseRootFinder):
    """Implementation of Newton-Hessian method."""

    def __init__(self, config: RootFinderConfig, x0: float):
        """
        Initialize Newton-Hessian method.

        Args:
            config: Configuration including function, derivative, and tolerances.
            x0: Initial guess for the root.

        Raises:
            ValueError: If derivative function is not provided in config.
        """
        # Ensure that a derivative is provided, as this method requires it.
        if config.derivative is None:
            raise ValueError("Newton-Hessian method requires derivative function")

        # Initialize common attributes from the base class.
        super().__init__(config)
        self.x = x0  # Current approximation of the root.

        # Convert the initial guess to a torch tensor for automatic differentiation.
        self.x_tensor = torch.tensor(x0, requires_grad=True, dtype=torch.float64)

    def get_current_x(self) -> float:
        """Get current x value."""
        return self.x

    def _compute_hessian(self) -> float:
        """
        Compute the Hessian (second derivative) of f(x) using automatic differentiation.

        Returns:
            float: The Hessian value at the current point.
        """
        # Convert the current approximation to a tensor with gradient tracking
        x = torch.tensor(self.x, requires_grad=True, dtype=torch.float64)

        # Compute the function value and its first derivative
        fx = self.func(float(x))
        fx_tensor = torch.tensor(fx, requires_grad=True)
        fx_tensor.backward()

        # Get first derivative
        if x.grad is None:
            return 1.0

        grad = x.grad.clone()
        x.grad.zero_()  # Clear the gradients for second derivative computation

        # Compute second derivative
        grad.backward()
        hess = x.grad

        if hess is None:
            return 1.0

        return float(hess)

    def step(self) -> float:
        """
        Perform one iteration of Newton-Hessian method.
        """
        if self._converged:
            return self.x

        # Store old x value
        x_old = self.x

        # Compute function, derivative, and Hessian
        fx = self.func(self.x)
        dfx = self.derivative(self.x)  # type: ignore
        hess = self._compute_hessian()

        # Store iteration details
        details = {
            "f(x)": fx,
            "f'(x)": dfx,
            "hessian": hess,
            "step": -fx / dfx if abs(dfx) > 1e-10 else 0,
        }

        # Check for small derivative
        if abs(dfx) < 1e-10:
            self._converged = True
            return self.x

        # Update x
        try:
            self.x = self.x - fx / dfx
            # Store iteration data
            self.add_iteration(x_old, self.x, details)
            self.iterations += 1
        except RuntimeError:
            self._converged = True
            return self.x

        # Check convergence
        if abs(fx) <= self.tol or self.iterations >= self.max_iter:
            self._converged = True

        return self.x

    @property
    def name(self) -> str:
        return "Newton-Hessian Method"


def newton_hessian_search(
    f: RootFinderConfig,
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[float, List[float], int]:
    """
    Legacy wrapper for backward compatibility.

    Args:
        f: Function configuration (or callable) for root finding.
        x0: Initial guess.
        tol: Error tolerance.
        max_iter: Maximum number of iterations.

    Returns:
        Tuple of (root, errors, iterations) where:
         - root is the final approximation,
         - errors is a list of error values per iteration,
         - iterations is the number of iterations performed.
    """
    # If f is a callable (old style), create a derivative function using finite differences
    if callable(f):

        def derivative(x: float, h: float = 1e-7) -> float:
            return (f(x + h) - f(x)) / h

        config = RootFinderConfig(
            func=f, derivative=derivative, tol=tol, max_iter=max_iter
        )
    else:
        # f is already a RootFinderConfig
        config = f

    # Instantiate the Newton-Hessian method with the configuration and initial guess
    method = NewtonHessianMethod(config, x0)

    errors = []  # List to record error values at each iteration
    # Continue iterating until convergence
    while not method.has_converged():
        method.step()
        errors.append(method.get_error())

    # Return the final approximation, error history, and iteration count
    return method.x, errors, method.iterations


# if __name__ == "__main__":
#     # Define function f(x) = x^2 - 2, aiming to find sqrt(2)
#     def f(x):
#         return x**2 - 2

#     # Define its derivative f'(x) = 2x
#     def df(x):
#         return 2 * x

#     # Use the new protocol-based implementation.
#     config = RootFinderConfig(func=f, derivative=df, tol=1e-6)
#     method = NewtonHessianMethod(config, x0=1.5)

#     # Iterate until convergence, printing progress.
#     while not method.has_converged():
#         x = method.step()
#         print(f"x = {x:.6f}, error = {method.get_error():.6f}")

#     print(f"\nFound root: {x}")
#     print(f"Iterations: {method.iterations}")
#     print(f"Final error: {method.get_error():.6e}")
