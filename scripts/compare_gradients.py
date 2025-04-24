"""Compare performance between analytical and finite difference gradients"""

import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from pathlib import Path

# Set the style for plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("viridis")
sns.set_context("notebook", font_scale=1.2)

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)


from algorithms.convex.bfgs import bfgs
from algorithms.differentiation.central_difference import central_difference


def rosenbrock(x, alpha=100):
    """Extended Rosenbrock function"""
    n = len(x)
    f = 0.0
    for i in range(n // 2):
        f += alpha * (x[2 * i + 1] - x[2 * i] ** 2) ** 2 + (1.0 - x[2 * i]) ** 2
    return f


def rosenbrock_grad_analytical(x, alpha=100):
    """Analytical gradient of extended Rosenbrock function"""
    n = len(x)
    grad = np.zeros_like(x)
    for i in range(n // 2):
        grad[2 * i] = -4.0 * alpha * (x[2 * i + 1] - x[2 * i] ** 2) * x[2 * i] - 2.0 * (
            1.0 - x[2 * i]
        )
        grad[2 * i + 1] = 2.0 * alpha * (x[2 * i + 1] - x[2 * i] ** 2)
    return grad


def rosenbrock_grad_finite_diff(x, alpha=100, h=1e-6):
    """Finite difference approximation of the Rosenbrock gradient"""
    n = len(x)
    grad = np.zeros_like(x)

    # Create a wrapper function for a specific dimension
    def f_wrapper(xi, i):
        x_temp = x.copy()
        x_temp[i] = xi
        return rosenbrock(x_temp, alpha)

    # Compute gradient for each dimension using central difference
    for i in range(n):
        grad[i] = central_difference(lambda xi: f_wrapper(xi, i), x[i], h)

    return grad


def evaluate_performance(dim_sizes, alpha=100, tol=1e-6):
    """Compare performance between analytical and finite difference gradients"""
    results = []

    for n in dim_sizes:
        print(f"\nTesting with dimension n={n}")

        # Starting point
        x0 = np.full(n, -1.0)

        # 1. Run with analytical gradient
        print("Running with analytical gradient...")

        def f(x):
            return rosenbrock(x, alpha)

        def grad_analytical(x):
            return rosenbrock_grad_analytical(x, alpha)

        start_time = time.time()
        func_count_analytical = [0]  # Use list to allow modification in closure

        def f_counted(x):
            func_count_analytical[0] += 1
            return f(x)

        x_opt_analytical, f_opt_analytical, iters_analytical, success_analytical = bfgs(
            f_counted, grad_analytical, x0, tol=tol, max_iter=2000
        )
        time_analytical = time.time() - start_time

        # 2. Run with finite difference gradient
        print("Running with finite difference gradient...")

        def grad_finite_diff(x):
            return rosenbrock_grad_finite_diff(x, alpha)

        start_time = time.time()
        func_count_finite = [0]

        def f_counted_finite(x):
            func_count_finite[0] += 1
            return f(x)

        # Track actual function evaluations from finite differences
        actual_func_evals = [0]

        def grad_finite_diff_counted(x):
            # Each gradient approximation requires 2n function evaluations for central difference
            actual_func_evals[0] += 2 * len(x)
            return grad_finite_diff(x)

        x_opt_finite, f_opt_finite, iters_finite, success_finite = bfgs(
            f_counted_finite, grad_finite_diff_counted, x0, tol=tol, max_iter=2000
        )
        time_finite = time.time() - start_time

        # Calculate error from known solution (all ones)
        x_star = np.ones_like(x0)
        error_analytical = np.linalg.norm(x_opt_analytical - x_star)
        error_finite = np.linalg.norm(x_opt_finite - x_star)

        # Store results
        result = {
            "dimension": n,
            "analytical": {
                "iterations": iters_analytical,
                "time": time_analytical,
                "func_evals": func_count_analytical[0],
                "error": error_analytical,
                "success": success_analytical,
            },
            "finite_diff": {
                "iterations": iters_finite,
                "time": time_finite,
                "func_evals": func_count_finite[0],
                "total_func_evals": func_count_finite[0] + actual_func_evals[0],
                "error": error_finite,
                "success": success_finite,
            },
        }

        results.append(result)

        # Print summary
        print(f"\nResults for n={n}:")
        print("\nAnalytical Gradient:")
        print(f"  Iterations: {iters_analytical}")
        print(f"  Time: {time_analytical:.4f} seconds")
        print(f"  Function evaluations: {func_count_analytical[0]}")
        print(f"  Error: {error_analytical:.8e}")
        print(f"  Success: {success_analytical}")

        print("\nFinite Difference Gradient:")
        print(f"  Iterations: {iters_finite}")
        print(f"  Time: {time_finite:.4f} seconds")
        print(f"  Function evaluations from BFGS: {func_count_finite[0]}")
        print(
            f"  Additional function evaluations from finite diff: {actual_func_evals[0]}"
        )
        print(
            f"  Total function evaluations: {func_count_finite[0] + actual_func_evals[0]}"
        )
        print(f"  Error: {error_finite:.8e}")
        print(f"  Success: {success_finite}")

    return results


def plot_results(results):
    """Create enhanced plots for performance comparison metrics"""
    # Create a directory for plots
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Extract data from results
    dimensions = [r["dimension"] for r in results]

    # Prepare data for plotting
    iters_analytical = [r["analytical"]["iterations"] for r in results]
    iters_finite = [r["finite_diff"]["iterations"] for r in results]

    time_analytical = [r["analytical"]["time"] for r in results]
    time_finite = [r["finite_diff"]["time"] for r in results]

    evals_analytical = [r["analytical"]["func_evals"] for r in results]
    evals_finite = [r["finite_diff"]["total_func_evals"] for r in results]

    # Prepare the data frame
    data = pd.DataFrame(
        {
            "Dimension": dimensions,
            "Analytical Iterations": iters_analytical,
            "Finite Difference Iterations": iters_finite,
            "Analytical Time (s)": time_analytical,
            "Finite Difference Time (s)": time_finite,
            "Analytical Function Evaluations": evals_analytical,
            "Finite Difference Function Evaluations": evals_finite,
        }
    )

    # Calculate ratios
    data["Time Ratio (FD/Analytical)"] = data["Finite Difference Time (s)"] / data[
        "Analytical Time (s)"
    ].replace(0, 0.00001)
    data["Function Evaluation Ratio (FD/Analytical)"] = (
        data["Finite Difference Function Evaluations"]
        / data["Analytical Function Evaluations"]
    )

    # Bar width for grouped bar charts
    bar_width = 0.35
    x = np.arange(len(dimensions))

    # 1. Enhanced Iteration Comparison
    plt.figure(figsize=(10, 6))

    bars1 = plt.bar(
        x - bar_width / 2,
        iters_analytical,
        bar_width,
        label="Analytical",
        color="#3498db",
        alpha=0.8,
    )
    bars2 = plt.bar(
        x + bar_width / 2,
        iters_finite,
        bar_width,
        label="Finite Difference",
        color="#e74c3c",
        alpha=0.8,
    )

    plt.xlabel("Dimension Size", fontweight="bold")
    plt.ylabel("Number of Iterations", fontweight="bold")
    plt.title(
        "BFGS Iterations Comparison: Analytical vs. Finite Difference Gradients",
        fontsize=14,
        fontweight="bold",
    )
    plt.xticks(x, dimensions)
    plt.legend()

    # Add data labels on top of the bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 5,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(plots_dir / "iterations_comparison.png", dpi=300, bbox_inches="tight")

    # 2. Enhanced Function Evaluations (Log Scale)
    plt.figure(figsize=(10, 6))

    bars1 = plt.bar(
        x - bar_width / 2,
        evals_analytical,
        bar_width,
        label="Analytical",
        color="#3498db",
        alpha=0.8,
    )
    bars2 = plt.bar(
        x + bar_width / 2,
        evals_finite,
        bar_width,
        label="Finite Difference",
        color="#e74c3c",
        alpha=0.8,
    )

    plt.xlabel("Dimension Size", fontweight="bold")
    plt.ylabel("Function Evaluations (log scale)", fontweight="bold")
    plt.title(
        "Function Evaluations Comparison (Log Scale)", fontsize=14, fontweight="bold"
    )
    plt.xticks(x, dimensions)
    plt.yscale("log")
    plt.legend()

    # Add data labels on top of the bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height * 1.1,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(
        plots_dir / "function_evaluations_log.png", dpi=300, bbox_inches="tight"
    )

    # 3. Enhanced Time Comparison (Dual Scale)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Analytical time (left y-axis)
    ax1.set_xlabel("Dimension Size", fontweight="bold")
    ax1.set_ylabel("Analytical Time (s)", fontweight="bold", color="#3498db")
    analytical_line = ax1.plot(
        dimensions,
        time_analytical,
        "o-",
        linewidth=2,
        markersize=8,
        label="Analytical",
        color="#3498db",
    )
    ax1.tick_params(axis="y", labelcolor="#3498db")

    # Finite difference time (right y-axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Finite Difference Time (s)", fontweight="bold", color="#e74c3c")
    finite_line = ax2.plot(
        dimensions,
        time_finite,
        "s-",
        linewidth=2,
        markersize=8,
        label="Finite Difference",
        color="#e74c3c",
    )
    ax2.tick_params(axis="y", labelcolor="#e74c3c")

    # Add title and legend
    plt.title(
        "Computation Time Comparison with Different Scales",
        fontsize=14,
        fontweight="bold",
    )
    lines = analytical_line + finite_line
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels, loc="upper left")

    # Add data values as annotations
    for i, (dim, time_a, time_f) in enumerate(
        zip(dimensions, time_analytical, time_finite)
    ):
        ax1.annotate(
            f"{time_a:.4f}s",
            (dim, time_a),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color="#3498db",
        )
        ax2.annotate(
            f"{time_f:.4f}s",
            (dim, time_f),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
            color="#e74c3c",
        )

    plt.tight_layout()
    plt.savefig(plots_dir / "time_comparison_dual.png", dpi=300, bbox_inches="tight")

    # 4. Efficiency Ratios
    plt.figure(figsize=(10, 6))

    time_ratio = data["Time Ratio (FD/Analytical)"].replace([np.inf, np.nan], 0)
    eval_ratio = data["Function Evaluation Ratio (FD/Analytical)"]

    plt.plot(
        dimensions,
        time_ratio,
        "o-",
        linewidth=2,
        markersize=8,
        label="Time Ratio (FD/Analytical)",
        color="#9b59b6",
    )
    plt.plot(
        dimensions,
        eval_ratio,
        "s-",
        linewidth=2,
        markersize=8,
        label="Function Evaluations Ratio (FD/Analytical)",
        color="#2ecc71",
    )

    # Add a reference line showing N=dimension
    plt.plot(
        dimensions,
        dimensions,
        "--",
        linewidth=1.5,
        label="N (dimension)",
        color="grey",
        alpha=0.7,
    )
    plt.plot(
        dimensions,
        [2 * n for n in dimensions],
        "--",
        linewidth=1.5,
        label="2N",
        color="black",
        alpha=0.7,
    )

    plt.xlabel("Dimension Size", fontweight="bold")
    plt.ylabel("Ratio (Finite Difference / Analytical)", fontweight="bold")
    plt.title(
        "Efficiency Ratios: How Much More Expensive is Finite Difference?",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Annotate points with actual ratio values
    for i, (dim, t_ratio, e_ratio) in enumerate(
        zip(dimensions, time_ratio, eval_ratio)
    ):
        if i > 0:  # Skip dimension 2 where time ratio might be undefined
            plt.annotate(
                f"{t_ratio:.1f}x",
                (dim, t_ratio),
                xytext=(5, 5),
                textcoords="offset points",
                color="#9b59b6",
            )
        plt.annotate(
            f"{e_ratio:.1f}x",
            (dim, e_ratio),
            xytext=(5, 5),
            textcoords="offset points",
            color="#2ecc71",
        )

    plt.tight_layout()
    plt.savefig(plots_dir / "efficiency_ratios.png", dpi=300, bbox_inches="tight")

    # 5. Comprehensive Dashboard
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Iterations (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(
        x - bar_width / 2,
        iters_analytical,
        bar_width,
        label="Analytical",
        color="#3498db",
        alpha=0.8,
    )
    ax1.bar(
        x + bar_width / 2,
        iters_finite,
        bar_width,
        label="Finite Difference",
        color="#e74c3c",
        alpha=0.8,
    )
    ax1.set_xlabel("Dimension Size")
    ax1.set_ylabel("Iterations")
    ax1.set_title("BFGS Iterations")
    ax1.set_xticks(x)
    ax1.set_xticklabels(dimensions)
    ax1.legend(loc="upper left")

    # Function Evaluations (top right, log scale)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(
        x - bar_width / 2,
        evals_analytical,
        bar_width,
        label="Analytical",
        color="#3498db",
        alpha=0.8,
    )
    ax2.bar(
        x + bar_width / 2,
        evals_finite,
        bar_width,
        label="Finite Difference",
        color="#e74c3c",
        alpha=0.8,
    )
    ax2.set_xlabel("Dimension Size")
    ax2.set_ylabel("Function Evaluations")
    ax2.set_title("Function Evaluations (Log Scale)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(dimensions)
    ax2.set_yscale("log")
    ax2.legend(loc="upper left")

    # Time (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(
        dimensions,
        time_analytical,
        "o-",
        linewidth=2,
        markersize=8,
        label="Analytical",
        color="#3498db",
    )
    ax3.plot(
        dimensions,
        time_finite,
        "s-",
        linewidth=2,
        markersize=8,
        label="Finite Difference",
        color="#e74c3c",
    )
    ax3.set_xlabel("Dimension Size")
    ax3.set_ylabel("Time (seconds)")
    ax3.set_title("Computation Time")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    # Efficiency Ratio (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(
        dimensions,
        eval_ratio,
        "s-",
        linewidth=2,
        markersize=8,
        label="Function Evaluations Ratio",
        color="#2ecc71",
    )
    ax4.plot(
        dimensions,
        [2 * n for n in dimensions],
        "--",
        linewidth=1.5,
        label="2N",
        color="black",
        alpha=0.7,
    )
    ax4.set_xlabel("Dimension Size")
    ax4.set_ylabel("FD/Analytical Ratio")
    ax4.set_title("Cost Ratio vs. Problem Size")
    ax4.legend(loc="upper left")
    ax4.grid(True, alpha=0.3)

    plt.suptitle(
        "BFGS Optimization: Analytical vs. Finite Difference Gradients",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    plt.savefig(plots_dir / "comprehensive_dashboard.png", dpi=300, bbox_inches="tight")

    print(
        f"Enhanced plots have been generated and saved to the '{plots_dir}' directory."
    )
    plt.show()


if __name__ == "__main__":
    # Test with different problem dimensions
    dimension_sizes = [2, 4, 10, 20, 50]

    # Run performance evaluation
    results = evaluate_performance(dimension_sizes)

    # Plot results
    plot_results(results)
