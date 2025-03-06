"""Surface plot component for 3D visualization."""

from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings
import plotly.graph_objects as go
from plotly.offline import plot
import numpy as np
import traceback


class SurfacePlot(QWebEngineView):
    """Widget for interactive 3D surface plots."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.page().settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True
        )
        self.page().settings().setAttribute(
            QWebEngineSettings.WebAttribute.WebGLEnabled, True
        )

    def update_plot(self, func=None, bounds=None, optimization_result=None):
        """Update the 3D surface plot."""
        try:
            if func is None or bounds is None:
                return

            # Extract bounds
            x_min, x_max = bounds[0]
            y_min, y_max = bounds[1]

            # Create meshgrid
            resolution = 50
            x = np.linspace(x_min, x_max, resolution)
            y = np.linspace(y_min, y_max, resolution)
            X, Y = np.meshgrid(x, y)

            # Evaluate the function over the meshgrid
            Z = np.zeros((resolution, resolution))
            for i in range(resolution):
                for j in range(resolution):
                    try:
                        if callable(func):
                            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
                        else:
                            # Handle non-callable func during animation
                            Z[i, j] = np.nan
                    except Exception:
                        Z[i, j] = np.nan

            # Replace any NaN or infinite values with finite values to prevent rendering errors
            Z = np.nan_to_num(
                Z, nan=np.nanmean(Z[np.isfinite(Z)]) if np.any(np.isfinite(Z)) else 0
            )

            # Create plot
            fig = go.Figure()

            # Add surface
            fig.add_trace(
                go.Surface(
                    z=Z,
                    x=X,
                    y=Y,
                    colorscale="Viridis",
                    lighting=dict(
                        ambient=0.5,
                        diffuse=0.5,
                        roughness=0.5,
                        specular=0.8,
                        fresnel=0.8,
                    ),
                    contours=dict(
                        z=dict(
                            show=True,
                            usecolormap=True,
                            highlightcolor="white",
                            project=dict(z=True),
                        )
                    ),
                    opacity=0.95,
                )
            )

            # Add optimization path if available
            if optimization_result is not None and "path" in optimization_result:
                path = np.array(optimization_result["path"])
                # If plotting specific frame, handle that
                if isinstance(func, np.ndarray):
                    frame_index = np.where((path == func).all(axis=1))[0]
                    if len(frame_index) > 0:
                        path = path[: frame_index[0] + 1]

                # Calculate z-values for path
                if isinstance(func, np.ndarray):
                    # In animation mode, func is actually a point from the path
                    # Let's use function values from the optimization result if available
                    if "function_values" in optimization_result and len(
                        optimization_result["function_values"]
                    ) == len(path):
                        z_path = np.array(optimization_result["function_values"])
                    else:
                        # If no function values, use z value at each path point from our calculated Z grid
                        z_path = np.zeros(len(path))
                        for i, point in enumerate(path):
                            # Find closest grid point
                            x_idx = np.abs(x - point[0]).argmin()
                            y_idx = np.abs(y - point[1]).argmin()
                            z_path[i] = Z[y_idx, x_idx]
                else:
                    # Normal case - func is a callable function
                    try:
                        z_path = np.array([func(p) for p in path])
                        # Replace any NaN values
                        z_path = np.nan_to_num(
                            z_path,
                            nan=(
                                np.nanmean(z_path[np.isfinite(z_path)])
                                if np.any(np.isfinite(z_path))
                                else 0
                            ),
                        )
                    except Exception:
                        # Fallback if function evaluation fails
                        z_path = np.zeros(len(path))

                # Plot the path
                fig.add_trace(
                    go.Scatter3d(
                        x=path[:, 0],
                        y=path[:, 1],
                        z=z_path,
                        mode="lines+markers",
                        marker=dict(
                            size=4,
                            color=z_path,
                            colorscale="Viridis",
                            opacity=0.9,
                        ),
                        line=dict(
                            color="#00ffff",
                            width=5,
                        ),
                        name="Optimization Path",
                    )
                )

                # Add start and end points
                fig.add_trace(
                    go.Scatter3d(
                        x=[path[0, 0]],
                        y=[path[0, 1]],
                        z=[z_path[0]],
                        mode="markers",
                        marker=dict(
                            size=8,
                            color="#ff3333",
                            opacity=0.9,
                            symbol="diamond",
                        ),
                        name="Starting Point",
                    )
                )

                fig.add_trace(
                    go.Scatter3d(
                        x=[path[-1, 0]],
                        y=[path[-1, 1]],
                        z=[z_path[-1]],
                        mode="markers",
                        marker=dict(
                            size=8,
                            color="#33ff33",
                            opacity=0.9,
                            symbol="circle",
                        ),
                        name="Final Point",
                    )
                )

            # Layout settings
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                scene=dict(
                    xaxis=dict(
                        title=dict(text="x₁", font=dict(size=12, color="#ffffff")),
                        showgrid=True,
                        gridcolor="#222222",
                        showline=True,
                        linecolor="#444444",
                    ),
                    yaxis=dict(
                        title=dict(text="x₂", font=dict(size=12, color="#ffffff")),
                        showgrid=True,
                        gridcolor="#222222",
                        showline=True,
                        linecolor="#444444",
                    ),
                    zaxis=dict(
                        title=dict(text="f(x)", font=dict(size=12, color="#ffffff")),
                        showgrid=True,
                        gridcolor="#222222",
                        showline=True,
                        linecolor="#444444",
                    ),
                    aspectmode="cube",
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1),
                    ),
                    annotations=[],
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                title=dict(
                    text="3D Function Surface",
                    font=dict(size=14, color="#ffffff"),
                    x=0.5,
                    y=0.95,
                ),
                legend=dict(
                    x=0.75,
                    y=0.95,
                    bgcolor="rgba(0,0,0,0.3)",
                    bordercolor="rgba(255,255,255,0.2)",
                ),
            )

            # Export to HTML
            html_str = plot(
                fig,
                output_type="div",
                include_plotlyjs="cdn",
                config=dict(
                    displayModeBar=True,
                    displaylogo=False,
                    modeBarButtonsToRemove=[
                        "sendDataToCloud",
                        "autoScale2d",
                        "hoverClosestCartesian",
                        "toggleSpikelines",
                    ],
                    toImageButtonOptions=dict(
                        format="png",
                        filename="surface_plot",
                        width=1200,
                        height=800,
                        scale=2,
                    ),
                ),
            )

            # Style the HTML
            html_str = f"""
            <html>
            <head>
                <style>
                    body, html {{
                        margin: 0;
                        padding: 0;
                        height: 100%;
                        width: 100%;
                        overflow: hidden;
                        background-color: #000000;
                    }}
                    .plotly-graph-div {{
                        height: 100vh;
                        width: 100%;
                    }}
                </style>
            </head>
            <body>
                {html_str}
            </body>
            </html>
            """

            # Set the HTML
            self.setHtml(html_str)

        except Exception as e:
            print(f"Surface plot error: {str(e)}")
            traceback.print_exc()
