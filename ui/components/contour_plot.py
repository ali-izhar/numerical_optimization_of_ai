"""Contour plot component for 2D visualization."""

from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings
import plotly.graph_objects as go
from plotly.offline import plot
import numpy as np
import traceback


class ContourPlot(QWebEngineView):
    """Widget for interactive 2D contour plots."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.page().settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True
        )
        self.page().settings().setAttribute(
            QWebEngineSettings.WebAttribute.WebGLEnabled, True
        )

        # Store state for updates
        self.current_func = None
        self.current_bounds = None
        self.current_optimization_result = None
        self.fig = None
        self.x = None
        self.y = None
        self.Z = None
        self.path = None

    def update_plot(self, func=None, bounds=None, optimization_result=None):
        """Update the 2D contour plot."""
        try:
            # If we don't have a function or bounds, there's nothing to plot
            if func is None and self.current_func is None:
                return
            if bounds is None and self.current_bounds is None:
                return

            # Get the actual function to use (current or new)
            actual_func = func if func is not None else self.current_func
            actual_bounds = bounds if bounds is not None else self.current_bounds

            # Determine if we need a full replot
            needs_full_replot = (
                self.fig is None
                or actual_func != self.current_func
                or actual_bounds != self.current_bounds
                or optimization_result != self.current_optimization_result
            )

            # Create the contour plot if needed
            if needs_full_replot:
                self._create_contour_plot(
                    actual_func, actual_bounds, optimization_result
                )

        except Exception as e:
            print(f"Contour plot error: {str(e)}")
            traceback.print_exc()

    def _create_contour_plot(self, func, bounds, optimization_result=None):
        """Create the contour plot from scratch."""
        # Extract bounds
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]

        # Create meshgrid with higher resolution for smoother contours
        resolution = 100
        self.x = np.linspace(x_min, x_max, resolution)
        self.y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(self.x, self.y)

        # Evaluate function over the meshgrid
        Z = np.zeros((resolution, resolution))
        for i in range(resolution):
            for j in range(resolution):
                try:
                    if callable(func):
                        Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
                    else:
                        Z[i, j] = np.nan
                except Exception:
                    Z[i, j] = np.nan

        # Replace any NaN or infinite values
        Z = np.nan_to_num(
            Z, nan=np.nanmean(Z[np.isfinite(Z)]) if np.any(np.isfinite(Z)) else 0
        )
        self.Z = Z

        # Create fresh figure
        self.fig = go.Figure()

        # Add contour with improved styling
        self.fig.add_trace(
            go.Contour(
                z=Z,
                x=self.x,
                y=self.y,
                colorscale="Viridis",
                contours=dict(
                    showlabels=True,
                    labelfont=dict(size=10, color="white"),
                ),
                colorbar=dict(
                    title=dict(text="Function Value", font=dict(color="white")),
                    tickfont=dict(color="white"),
                ),
                name="Contour Plot",
            )
        )

        # Store state
        self.current_func = func
        self.current_bounds = bounds
        self.current_optimization_result = optimization_result

        # If we have optimization data, store it and show the path
        if optimization_result is not None and "path" in optimization_result:
            self.path = np.array(optimization_result["path"])
            self._show_complete_path()
        else:
            self.path = None

        # Apply layout styling
        self._apply_layout_styling()

        # Render HTML
        self._render_html()

    def _show_complete_path(self):
        """Show the complete optimization path."""
        if self.fig is None or self.path is None or len(self.path) == 0:
            return

        # Add the full path
        self.fig.add_trace(
            go.Scatter(
                x=self.path[:, 0],
                y=self.path[:, 1],
                mode="lines+markers",
                marker=dict(
                    size=6,
                    color="cyan",
                    line=dict(width=1, color="white"),
                ),
                line=dict(
                    color="cyan",
                    width=2,
                ),
                name="Optimization Path",
                showlegend=True,
            )
        )

        # Add starting point
        self.fig.add_trace(
            go.Scatter(
                x=[self.path[0, 0]],
                y=[self.path[0, 1]],
                mode="markers",
                marker=dict(
                    size=10,
                    color="red",
                    symbol="diamond",
                ),
                name="Starting Point",
                showlegend=True,
            )
        )

        # Add ending point
        self.fig.add_trace(
            go.Scatter(
                x=[self.path[-1, 0]],
                y=[self.path[-1, 1]],
                mode="markers",
                marker=dict(
                    size=10,
                    color="green",
                    symbol="circle",
                ),
                name="Final Point",
                showlegend=True,
            )
        )

    def _apply_layout_styling(self):
        """Apply layout styling to the figure."""
        if self.fig is None:
            return

        self.fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title=dict(text="x₁", font=dict(size=12, color="#ffffff")),
                showgrid=True,
                gridcolor="#222222",
                zeroline=True,
                zerolinecolor="#444444",
            ),
            yaxis=dict(
                title=dict(text="x₂", font=dict(size=12, color="#ffffff")),
                showgrid=True,
                gridcolor="#222222",
                zeroline=True,
                zerolinecolor="#444444",
                scaleanchor="x",
                scaleratio=1,
            ),
            title=dict(
                text="Contour Plot",
                font=dict(size=14, color="#ffffff"),
                x=0.5,
                y=0.95,
            ),
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="rgba(255,255,255,0.2)",
            ),
            margin=dict(l=50, r=20, t=50, b=50),
        )

    def _render_html(self):
        """Generate and set the HTML content."""
        if self.fig is None:
            return

        # Export to HTML
        html_str = plot(
            self.fig,
            output_type="div",
            include_plotlyjs="cdn",
            config=dict(
                displayModeBar=True,
                displaylogo=False,
                toImageButtonOptions=dict(
                    format="png",
                    filename="contour_plot",
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
