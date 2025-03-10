# utils/file_manager.py

"""File management utilities for numerical optimization."""

import json
import sys
import pandas as pd
from pathlib import Path
from typing import List

from algorithms.convex.protocols import BaseNumericalMethod


def load_config_file(config_path: Path) -> dict:
    """
    Load configuration from a JSON file.

    Args:
        config_path: Path to the configuration file

    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)


def save_iteration_history(
    methods: List[BaseNumericalMethod], function_name: str, save_dir: Path
):
    """
    Save iteration history data to CSV files.

    Args:
        methods: List of method instances
        function_name: Name of the function
        save_dir: Directory to save data
    """
    # Create directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)

    for method in methods:
        history = method.get_iteration_history()
        if not history:
            continue

        # Create DataFrame
        data = []
        for h in history:
            row = {
                "iteration": h.iteration,
                "x_old": h.x_old,
                "x_new": h.x_new,
                "f_old": h.f_old,
                "f_new": h.f_new,
                "error": h.error,
            }

            # Add details
            for k, v in h.details.items():
                row[f"detail_{k}"] = v

            data.append(row)

        df = pd.DataFrame(data)

        # Save to CSV
        filename = f"{function_name}_{method.name}.csv"
        filepath = save_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Saved iteration history to {filepath}")


def save_visualization(fig, save_path: Path, viz_format: str = "html"):
    """
    Save a visualization figure to a file.

    Args:
        fig: The figure object to save (Plotly figure)
        save_path: Path to save the visualization
        viz_format: Format for saved visualization ("html", "png", etc.)
    """
    # Create directory if it doesn't exist
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the figure based on format
    if viz_format == "html":
        fig.write_html(
            save_path.with_suffix(".html"),
            full_html=True,
            include_plotlyjs="cdn",
            include_mathjax="cdn",
            config={"responsive": True},  # Make responsive in HTML output
        )
        print(f"Saved visualization to {save_path.with_suffix('.html')}")
    elif viz_format in ["png", "jpg", "jpeg", "webp", "svg", "pdf"]:
        # For static images, set a reasonable size with higher resolution
        fig.write_image(
            save_path.with_suffix(f".{viz_format}"), width=1200, height=800, scale=2
        )
        print(f"Saved visualization to {save_path.with_suffix(f'.{viz_format}')}")


def save_animation(anim_fig, save_path: Path, viz_format: str = "html"):
    """
    Save an animation figure to a file.

    Args:
        anim_fig: The animation figure object to save (Plotly figure)
        save_path: Path to save the animation
        viz_format: Format for saved animation ("html", "mp4")
    """
    # Create directory if it doesn't exist
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the animation based on format
    if viz_format == "html":
        anim_fig.write_html(
            save_path.with_suffix("_animation.html"),
            full_html=True,
            include_plotlyjs="cdn",
            config={"responsive": True},  # Make responsive in HTML output
        )
        print(f"Saved animation to {save_path.with_suffix('_animation.html')}")
    elif viz_format == "mp4" and hasattr(anim_fig, "write_video"):
        try:
            # For video, set a reasonable size and framerate
            anim_fig.write_video(
                save_path.with_suffix(".mp4"),
                width=1200,
                height=800,
                fps=15,  # Smoother framerate
            )
            print(f"Saved animation to {save_path.with_suffix('.mp4')}")
        except Exception as e:
            print(f"Could not save animation as MP4: {e}")
            print("Falling back to HTML format for animation.")
            anim_fig.write_html(
                save_path.with_suffix("_animation.html"),
                config={"responsive": True},
            )
            print(f"Saved animation to {save_path.with_suffix('_animation.html')}")
