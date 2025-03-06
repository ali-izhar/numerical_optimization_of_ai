"""Convergence plot component for optimization visualization."""

from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings
import plotly.graph_objects as go
from plotly.offline import plot
import numpy as np
import traceback


class ConvergencePlot(QWebEngineView):
    """Widget for interactive convergence plots."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.page().settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True
        )

    def update_plot(self, optimization_result=None):
        """Update the convergence plot with optimization history."""
        try:
            if optimization_result is None:
                return

            # Extract data from optimization result
            if (
                optimization_result is not None
                and "function_values" in optimization_result
            ):
                function_values = optimization_result["function_values"]
                iterations = list(range(len(function_values)))

                # Create figure
                fig = go.Figure()

                # Function value trace
                fig.add_trace(
                    go.Scatter(
                        x=iterations,
                        y=function_values,
                        mode="lines+markers",
                        name="Function Value",
                        line=dict(color="#4a90e2", width=2),
                        marker=dict(size=6, color="#4a90e2"),
                    )
                )

                # Layout setup
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    title=dict(
                        text="Convergence History",
                        font=dict(size=14, color="#ffffff"),
                        x=0.5,
                        y=0.95,
                    ),
                    xaxis=dict(
                        title=dict(
                            text="Iteration", font=dict(size=12, color="#ffffff")
                        ),
                        showgrid=True,
                        gridcolor="#222222",
                        zeroline=False,
                    ),
                    yaxis=dict(
                        title=dict(
                            text="Function Value", font=dict(size=12, color="#ffffff")
                        ),
                        showgrid=True,
                        gridcolor="#222222",
                        zeroline=False,
                        type="log" if min(function_values) > 0 else "linear",
                    ),
                    margin=dict(l=50, r=20, t=50, b=50),
                    legend=dict(
                        orientation="h",
                        y=1.05,
                        xanchor="right",
                        x=1,
                        bgcolor="rgba(0,0,0,0.3)",
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
                        toImageButtonOptions=dict(
                            format="png",
                            filename="convergence_plot",
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
            print(f"Convergence plot error: {str(e)}")
            traceback.print_exc()
