#!/usr/bin/env python
"""
Automatic Differentiation Solver

This script helps analyze functions using computational graphs and automatic differentiation.
It can:
1. Define intermediate variables in the computation
2. Draw the computational graph
3. Compute gradients using forward or reverse mode automatic differentiation
4. Show the step-by-step process of gradient computation
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy import symbols, diff, lambdify
import re
from typing import Dict, List, Tuple, Callable, Union, Optional

# Try to import graphviz
HAS_GRAPHVIZ = False
try:
    import graphviz

    HAS_GRAPHVIZ = True
except ImportError:
    pass


class Node:
    """Represents a node in the computational graph."""

    def __init__(
        self,
        name: str,
        operation: str,
        inputs: List[str],
        expr: Optional[sp.Expr] = None,
        value: Optional[float] = None,
    ):
        self.name = name
        self.operation = operation
        self.inputs = inputs
        self.expr = expr
        self.value = value
        self.grad = None
        self.forward_grad = {}  # For forward mode: maps var_name -> gradient
        self.reverse_grad = None  # For reverse mode

    def __repr__(self) -> str:
        return f"{self.name}: {self.operation}"


class ComputationalGraph:
    """Represents a computational graph for automatic differentiation."""

    def __init__(self):
        self.nodes = {}  # Map from node name to Node object
        self.input_nodes = []  # Names of input nodes
        self.output_node = None  # Name of output node
        self.var_symbols = {}  # Map from variable name to sympy symbol
        self.evaluation_order = []  # Nodes in evaluation order

    def add_input(self, name: str, value: float = None):
        """Add an input node to the graph."""
        self.nodes[name] = Node(name, "input", [], value=value)
        self.input_nodes.append(name)
        self.var_symbols[name] = sp.Symbol(name)
        if name not in self.evaluation_order:
            self.evaluation_order.append(name)

    def add_node(
        self, name: str, operation: str, inputs: List[str], expr: sp.Expr = None
    ):
        """Add an intermediate or output node to the graph."""
        self.nodes[name] = Node(name, operation, inputs, expr=expr)
        # Add symbol for this node too, to enable proper substitution
        self.var_symbols[name] = sp.Symbol(name)

        # Update evaluation order
        if name not in self.evaluation_order:
            for input_name in inputs:
                if input_name not in self.evaluation_order:
                    raise ValueError(
                        f"Input node {input_name} should be added before node {name}"
                    )
            self.evaluation_order.append(name)

    def set_output(self, name: str):
        """Set the output node of the graph."""
        if name not in self.nodes:
            raise ValueError(f"Node {name} not found in the graph")
        self.output_node = name

    def evaluate(self, input_values: Dict[str, float] = None):
        """Evaluate the computational graph with the given input values."""
        if input_values is None:
            input_values = {}

        # Initialize input nodes with provided values
        for name in self.input_nodes:
            if name in input_values:
                self.nodes[name].value = input_values[name]
            elif self.nodes[name].value is None:
                raise ValueError(f"No value provided for input node {name}")

        # Evaluate nodes in topological order
        for name in self.evaluation_order:
            node = self.nodes[name]
            if name in self.input_nodes:
                continue  # Already set the value for input nodes

            # Get input values
            input_vals = [self.nodes[input_name].value for input_name in node.inputs]

            # Evaluate the node
            if node.operation == "+":
                node.value = sum(input_vals)
            elif node.operation == "*":
                node.value = np.prod(input_vals)
            elif node.operation == "sin":
                node.value = np.sin(input_vals[0])
            elif node.operation == "cos":
                node.value = np.cos(input_vals[0])
            elif node.operation == "exp":
                node.value = np.exp(input_vals[0])
            elif node.operation == "log":
                node.value = np.log(input_vals[0])
            elif node.operation == "^":
                node.value = (
                    input_vals[0] ** input_vals[1]
                    if len(input_vals) > 1
                    else input_vals[0] ** 2
                )
            elif node.operation == "/":
                node.value = input_vals[0] / input_vals[1]
            elif node.operation == "-":
                node.value = (
                    input_vals[0] - input_vals[1]
                    if len(input_vals) > 1
                    else -input_vals[0]
                )
            else:
                # For complex operations, try numerical evaluation
                try:
                    # Create a substitution dictionary
                    subs_dict = {
                        self.var_symbols[input_name]: self.nodes[input_name].value
                        for input_name in node.inputs
                    }
                    # Evaluate with evalf for numerical precision
                    result = node.expr.evalf(subs=subs_dict)
                    node.value = float(result)
                except (TypeError, ValueError):
                    # If that fails, manually compute based on the operation string
                    print(f"Warning: Falling back to manual evaluation for node {name}")
                    node.value = 0.0  # Default value

        # Return the output value
        return self.nodes[self.output_node].value if self.output_node else None

    def forward_mode_ad(self, seed_var: str):
        """
        Perform forward mode automatic differentiation with respect to seed_var.
        """
        if seed_var not in self.input_nodes:
            raise ValueError(f"Seed variable {seed_var} not found in input nodes")

        # Initialize gradients for input nodes
        for name in self.input_nodes:
            self.nodes[name].forward_grad[seed_var] = 1.0 if name == seed_var else 0.0

        # Propagate gradients forward
        steps = [f"Initialize d{seed_var} = 1.0"]
        for name in self.evaluation_order:
            if name in self.input_nodes:
                continue  # Skip input nodes - already initialized

            node = self.nodes[name]
            input_nodes = [self.nodes[input_name] for input_name in node.inputs]

            # Initialize the gradient for this variable with respect to seed_var
            node.forward_grad[seed_var] = 0.0

            if node.operation == "+":
                # d(u+v)/dx = du/dx + dv/dx
                node.forward_grad[seed_var] = sum(
                    input_node.forward_grad[seed_var] for input_node in input_nodes
                )
                steps.append(
                    f"d{node.name}/d{seed_var} = {' + '.join(f'd{input_node.name}/d{seed_var}' for input_node in input_nodes)} = {node.forward_grad[seed_var]:.6f}"
                )

            elif node.operation == "*":
                # d(u*v)/dx = v*du/dx + u*dv/dx
                terms = []
                for i, input_node in enumerate(input_nodes):
                    term = input_node.forward_grad[seed_var]
                    term_expr = f"d{input_node.name}/d{seed_var}"

                    for j, other_input in enumerate(input_nodes):
                        if i != j:
                            term *= other_input.value
                            term_expr = f"{other_input.name} * {term_expr}"

                    node.forward_grad[seed_var] += term
                    terms.append(term_expr)

                steps.append(
                    f"d{node.name}/d{seed_var} = {' + '.join(terms)} = {node.forward_grad[seed_var]:.6f}"
                )

            elif node.operation == "sin":
                # d(sin(u))/dx = cos(u) * du/dx
                input_node = input_nodes[0]
                node.forward_grad[seed_var] = (
                    np.cos(input_node.value) * input_node.forward_grad[seed_var]
                )
                steps.append(
                    f"d{node.name}/d{seed_var} = cos({input_node.name}) * d{input_node.name}/d{seed_var} = {np.cos(input_node.value):.6f} * {input_node.forward_grad[seed_var]:.6f} = {node.forward_grad[seed_var]:.6f}"
                )

            elif node.operation == "cos":
                # d(cos(u))/dx = -sin(u) * du/dx
                input_node = input_nodes[0]
                node.forward_grad[seed_var] = (
                    -np.sin(input_node.value) * input_node.forward_grad[seed_var]
                )
                steps.append(
                    f"d{node.name}/d{seed_var} = -sin({input_node.name}) * d{input_node.name}/d{seed_var} = {-np.sin(input_node.value):.6f} * {input_node.forward_grad[seed_var]:.6f} = {node.forward_grad[seed_var]:.6f}"
                )

            elif node.operation == "exp":
                # d(exp(u))/dx = exp(u) * du/dx
                input_node = input_nodes[0]
                node.forward_grad[seed_var] = (
                    np.exp(input_node.value) * input_node.forward_grad[seed_var]
                )
                steps.append(
                    f"d{node.name}/d{seed_var} = exp({input_node.name}) * d{input_node.name}/d{seed_var} = {np.exp(input_node.value):.6f} * {input_node.forward_grad[seed_var]:.6f} = {node.forward_grad[seed_var]:.6f}"
                )

            elif node.operation == "log":
                # d(log(u))/dx = (1/u) * du/dx
                input_node = input_nodes[0]
                node.forward_grad[seed_var] = (
                    1.0 / input_node.value
                ) * input_node.forward_grad[seed_var]
                steps.append(
                    f"d{node.name}/d{seed_var} = (1/{input_node.name}) * d{input_node.name}/d{seed_var} = {1.0 / input_node.value:.6f} * {input_node.forward_grad[seed_var]:.6f} = {node.forward_grad[seed_var]:.6f}"
                )

            elif node.operation == "^":
                # d(u^v)/dx = v * u^(v-1) * du/dx + u^v * ln(u) * dv/dx
                u, v = input_nodes[0], input_nodes[1]

                if (
                    v.value == 2.0 and len(input_nodes) == 1
                ):  # Special case for squaring
                    node.forward_grad[seed_var] = 2 * u.value * u.forward_grad[seed_var]
                    steps.append(
                        f"d{node.name}/d{seed_var} = 2 * {u.name} * d{u.name}/d{seed_var} = 2 * {u.value:.6f} * {u.forward_grad[seed_var]:.6f} = {node.forward_grad[seed_var]:.6f}"
                    )
                else:
                    # First term: v * u^(v-1) * du/dx
                    term1 = (
                        v.value * (u.value ** (v.value - 1)) * u.forward_grad[seed_var]
                    )

                    # Second term: u^v * ln(u) * dv/dx
                    term2 = (
                        (u.value**v.value) * np.log(u.value) * v.forward_grad[seed_var]
                        if u.value > 0
                        else 0
                    )

                    node.forward_grad[seed_var] = term1 + term2
                    steps.append(
                        f"d{node.name}/d{seed_var} = {v.name} * {u.name}^({v.name}-1) * d{u.name}/d{seed_var} + {u.name}^{v.name} * ln({u.name}) * d{v.name}/d{seed_var} = {node.forward_grad[seed_var]:.6f}"
                    )

            elif node.operation == "/":
                # d(u/v)/dx = (du/dx * v - u * dv/dx) / v^2
                u, v = input_nodes[0], input_nodes[1]
                node.forward_grad[seed_var] = (
                    u.forward_grad[seed_var] * v.value
                    - u.value * v.forward_grad[seed_var]
                ) / (v.value**2)
                steps.append(
                    f"d{node.name}/d{seed_var} = (d{u.name}/d{seed_var} * {v.name} - {u.name} * d{v.name}/d{seed_var}) / {v.name}^2 = {node.forward_grad[seed_var]:.6f}"
                )

            elif node.operation == "-":
                if len(input_nodes) == 1:  # Unary minus
                    input_node = input_nodes[0]
                    node.forward_grad[seed_var] = -input_node.forward_grad[seed_var]
                    steps.append(
                        f"d{node.name}/d{seed_var} = -d{input_node.name}/d{seed_var} = -{input_node.forward_grad[seed_var]:.6f} = {node.forward_grad[seed_var]:.6f}"
                    )
                else:  # Binary minus
                    u, v = input_nodes[0], input_nodes[1]
                    node.forward_grad[seed_var] = (
                        u.forward_grad[seed_var] - v.forward_grad[seed_var]
                    )
                    steps.append(
                        f"d{node.name}/d{seed_var} = d{u.name}/d{seed_var} - d{v.name}/d{seed_var} = {u.forward_grad[seed_var]:.6f} - {v.forward_grad[seed_var]:.6f} = {node.forward_grad[seed_var]:.6f}"
                    )

            else:
                # For complex operations, try to use the expression derivative
                if node.expr is not None:
                    deriv_expr = 0
                    for i, input_name in enumerate(node.inputs):
                        input_sym = self.var_symbols[input_name]
                        partial_deriv = sp.diff(node.expr, input_sym)

                        # Substitute values
                        subs_dict = {
                            self.var_symbols[input_name]: self.nodes[input_name].value
                            for input_name in node.inputs
                        }
                        partial_value = float(partial_deriv.subs(subs_dict))

                        # Chain rule
                        deriv_expr += (
                            partial_value
                            * self.nodes[input_name].forward_grad[seed_var]
                        )

                    node.forward_grad[seed_var] = deriv_expr
                    steps.append(
                        f"d{node.name}/d{seed_var} = (complex expression, see symbolic differentiation) = {node.forward_grad[seed_var]:.6f}"
                    )
                else:
                    raise ValueError(
                        f"Unsupported operation for forward mode: {node.operation}"
                    )

        # Return the gradient of the output node with respect to the seed variable
        return self.nodes[self.output_node].forward_grad[seed_var], steps

    def reverse_mode_ad(self):
        """
        Perform reverse mode automatic differentiation.
        """
        if not self.output_node:
            raise ValueError("Output node not set")

        # Initialize gradients to zero
        for name in self.nodes:
            self.nodes[name].reverse_grad = 0.0

        # Set the gradient of the output node to 1.0
        self.nodes[self.output_node].reverse_grad = 1.0

        # Go through nodes in reverse order
        steps = [f"Initialize d{self.output_node} = 1.0"]
        for name in reversed(self.evaluation_order):
            node = self.nodes[name]
            if name in self.input_nodes:
                continue  # Skip further processing of input nodes

            # For each input to this node, propagate the gradient
            for input_name in node.inputs:
                input_node = self.nodes[input_name]

                # Calculate partial derivative
                if node.operation == "+":
                    # ∂(u+v)/∂u = 1
                    input_node.reverse_grad += node.reverse_grad
                    steps.append(
                        f"d{self.output_node}/d{input_name} += d{self.output_node}/d{name} * ∂{name}/∂{input_name} = {node.reverse_grad:.6f} * 1 = {input_node.reverse_grad:.6f}"
                    )

                elif node.operation == "*":
                    # ∂(u*v)/∂u = v
                    other_inputs = [
                        self.nodes[other]
                        for other in node.inputs
                        if other != input_name
                    ]
                    derivative = np.prod([other.value for other in other_inputs])
                    input_node.reverse_grad += node.reverse_grad * derivative
                    steps.append(
                        f"d{self.output_node}/d{input_name} += d{self.output_node}/d{name} * ∂{name}/∂{input_name} = {node.reverse_grad:.6f} * {derivative:.6f} = {input_node.reverse_grad:.6f}"
                    )

                elif node.operation == "sin" and len(node.inputs) == 1:
                    # ∂(sin(u))/∂u = cos(u)
                    derivative = np.cos(input_node.value)
                    input_node.reverse_grad += node.reverse_grad * derivative
                    steps.append(
                        f"d{self.output_node}/d{input_name} += d{self.output_node}/d{name} * ∂{name}/∂{input_name} = {node.reverse_grad:.6f} * cos({input_name}) = {node.reverse_grad:.6f} * {derivative:.6f} = {input_node.reverse_grad:.6f}"
                    )

                elif node.operation == "cos" and len(node.inputs) == 1:
                    # ∂(cos(u))/∂u = -sin(u)
                    derivative = -np.sin(input_node.value)
                    input_node.reverse_grad += node.reverse_grad * derivative
                    steps.append(
                        f"d{self.output_node}/d{input_name} += d{self.output_node}/d{name} * ∂{name}/∂{input_name} = {node.reverse_grad:.6f} * (-sin({input_name})) = {node.reverse_grad:.6f} * {derivative:.6f} = {input_node.reverse_grad:.6f}"
                    )

                elif node.operation == "exp" and len(node.inputs) == 1:
                    # ∂(exp(u))/∂u = exp(u)
                    derivative = np.exp(input_node.value)
                    input_node.reverse_grad += node.reverse_grad * derivative
                    steps.append(
                        f"d{self.output_node}/d{input_name} += d{self.output_node}/d{name} * ∂{name}/∂{input_name} = {node.reverse_grad:.6f} * exp({input_name}) = {node.reverse_grad:.6f} * {derivative:.6f} = {input_node.reverse_grad:.6f}"
                    )

                elif node.operation == "log" and len(node.inputs) == 1:
                    # ∂(log(u))/∂u = 1/u
                    derivative = 1.0 / input_node.value
                    input_node.reverse_grad += node.reverse_grad * derivative
                    steps.append(
                        f"d{self.output_node}/d{input_name} += d{self.output_node}/d{name} * ∂{name}/∂{input_name} = {node.reverse_grad:.6f} * (1/{input_name}) = {node.reverse_grad:.6f} * {derivative:.6f} = {input_node.reverse_grad:.6f}"
                    )

                elif node.operation == "^":
                    if (
                        input_name == node.inputs[0]
                    ):  # Differentiate with respect to base
                        # ∂(u^v)/∂u = v * u^(v-1)
                        power = self.nodes[node.inputs[1]].value
                        derivative = power * (input_node.value ** (power - 1))
                        input_node.reverse_grad += node.reverse_grad * derivative
                        steps.append(
                            f"d{self.output_node}/d{input_name} += d{self.output_node}/d{name} * ∂{name}/∂{input_name} = {node.reverse_grad:.6f} * {derivative:.6f} = {input_node.reverse_grad:.6f}"
                        )
                    else:  # Differentiate with respect to exponent
                        # ∂(u^v)/∂v = u^v * ln(u)
                        base = self.nodes[node.inputs[0]].value
                        if base > 0:
                            derivative = (
                                base ** self.nodes[node.inputs[1]].value
                            ) * np.log(base)
                            input_node.reverse_grad += node.reverse_grad * derivative
                            steps.append(
                                f"d{self.output_node}/d{input_name} += d{self.output_node}/d{name} * ∂{name}/∂{input_name} = {node.reverse_grad:.6f} * {derivative:.6f} = {input_node.reverse_grad:.6f}"
                            )

                elif node.operation == "/":
                    if (
                        input_name == node.inputs[0]
                    ):  # Differentiate with respect to numerator
                        # ∂(u/v)/∂u = 1/v
                        derivative = 1.0 / self.nodes[node.inputs[1]].value
                        input_node.reverse_grad += node.reverse_grad * derivative
                        steps.append(
                            f"d{self.output_node}/d{input_name} += d{self.output_node}/d{name} * ∂{name}/∂{input_name} = {node.reverse_grad:.6f} * {derivative:.6f} = {input_node.reverse_grad:.6f}"
                        )
                    else:  # Differentiate with respect to denominator
                        # ∂(u/v)/∂v = -u/v^2
                        derivative = -self.nodes[node.inputs[0]].value / (
                            self.nodes[node.inputs[1]].value ** 2
                        )
                        input_node.reverse_grad += node.reverse_grad * derivative
                        steps.append(
                            f"d{self.output_node}/d{input_name} += d{self.output_node}/d{name} * ∂{name}/∂{input_name} = {node.reverse_grad:.6f} * {derivative:.6f} = {input_node.reverse_grad:.6f}"
                        )

                elif node.operation == "-":
                    if len(node.inputs) == 1:  # Unary minus
                        # ∂(-u)/∂u = -1
                        input_node.reverse_grad += node.reverse_grad * (-1)
                        steps.append(
                            f"d{self.output_node}/d{input_name} += d{self.output_node}/d{name} * ∂{name}/∂{input_name} = {node.reverse_grad:.6f} * (-1) = {input_node.reverse_grad:.6f}"
                        )
                    else:  # Binary minus
                        if (
                            input_name == node.inputs[0]
                        ):  # Differentiate with respect to first operand
                            # ∂(u-v)/∂u = 1
                            input_node.reverse_grad += node.reverse_grad
                            steps.append(
                                f"d{self.output_node}/d{input_name} += d{self.output_node}/d{name} * ∂{name}/∂{input_name} = {node.reverse_grad:.6f} * 1 = {input_node.reverse_grad:.6f}"
                            )
                        else:  # Differentiate with respect to second operand
                            # ∂(u-v)/∂v = -1
                            input_node.reverse_grad += node.reverse_grad * (-1)
                            steps.append(
                                f"d{self.output_node}/d{input_name} += d{self.output_node}/d{name} * ∂{name}/∂{input_name} = {node.reverse_grad:.6f} * (-1) = {input_node.reverse_grad:.6f}"
                            )

                else:
                    # For complex operations, try to use the expression derivative
                    if node.expr is not None:
                        input_sym = self.var_symbols[input_name]
                        partial_deriv = sp.diff(node.expr, input_sym)

                        # Substitute values
                        subs_dict = {
                            self.var_symbols[input_name]: self.nodes[input_name].value
                            for input_name in node.inputs
                        }
                        partial_value = float(partial_deriv.subs(subs_dict))

                        input_node.reverse_grad += node.reverse_grad * partial_value
                        steps.append(
                            f"d{self.output_node}/d{input_name} += d{self.output_node}/d{name} * ∂{name}/∂{input_name} = {node.reverse_grad:.6f} * {partial_value:.6f} = {input_node.reverse_grad:.6f}"
                        )
                    else:
                        raise ValueError(
                            f"Unsupported operation for reverse mode: {node.operation}"
                        )

        # Return the gradients for the input nodes
        gradients = {name: self.nodes[name].reverse_grad for name in self.input_nodes}
        return gradients, steps

    def draw_graph(self, ax=None, figsize=(12, 10)):
        """Draw the computational graph in a professional style with rectangular boxes for values."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Create a directed graph
        G = nx.DiGraph()

        # Define node types
        input_nodes = self.input_nodes
        output_node = self.output_node
        operation_nodes = []
        value_nodes = []

        # Classify nodes into operations and values
        for name, node in self.nodes.items():
            if node.operation not in ["input"]:
                operation_nodes.append(name)
            else:
                value_nodes.append(name)

        # Add output node to value nodes if not already there
        if output_node not in value_nodes:
            value_nodes.append(output_node)

        # Add nodes to the graph with appropriate attributes
        for name in operation_nodes:
            node = self.nodes[name]
            # Operations are circular
            G.add_node(
                name,
                type="operation",
                label=node.operation,
                width=0.6,
                height=0.6,
                shape="circle",
            )

        for name in value_nodes:
            node = self.nodes[name]
            # Values are rectangular
            label = f"Value({node.value:.2f})" if node.value is not None else name
            G.add_node(
                name,
                type="value",
                label=label,
                width=1.2,
                height=0.8,
                shape="rectangle",
            )

        # Add edges
        for name, node in self.nodes.items():
            for input_name in node.inputs:
                G.add_edge(input_name, name)

        # Compute a hierarchical layout
        # Group nodes by their level in the graph
        node_levels = {}
        for node_name in input_nodes:
            node_levels[node_name] = 0

        # Assign levels to nodes based on their dependencies
        for name in self.evaluation_order:
            if name in input_nodes:
                continue

            # Level is one more than the maximum level of inputs
            max_input_level = (
                max(
                    [
                        node_levels.get(input_name, 0)
                        for input_name in self.nodes[name].inputs
                    ]
                )
                if self.nodes[name].inputs
                else 0
            )
            node_levels[name] = max_input_level + 1

        # Group nodes by level
        levels = {}
        for name, level in node_levels.items():
            if level not in levels:
                levels[level] = []
            levels[level].append(name)

        # Assign positions based on levels
        pos = {}
        max_level = max(levels.keys()) if levels else 0
        for level, nodes in levels.items():
            # Sort nodes within each level
            nodes.sort()

            # Assign y-coordinate based on level (from top to bottom)
            y = 1.0 - level / (max_level + 1) if max_level > 0 else 0.5

            # Distribute nodes horizontally
            num_nodes = len(nodes)
            for i, node in enumerate(nodes):
                # Calculate x-coordinate to spread nodes evenly
                x = (i + 1) / (num_nodes + 1) if num_nodes > 1 else 0.5
                pos[node] = (x, y)

        # Draw different node types with appropriate shapes
        # Draw value nodes (rectangles)
        value_nodes_list = [n for n in G.nodes() if G.nodes[n].get("type") == "value"]
        value_node_colors = [
            (
                "lightblue"
                if n in input_nodes
                else "lightgreen" if n == output_node else "lightgray"
            )
            for n in value_nodes_list
        ]

        # Draw rectangles for value nodes
        for i, node in enumerate(value_nodes_list):
            rect = plt.Rectangle(
                (
                    pos[node][0] - G.nodes[node].get("width", 0.5) / 2,
                    pos[node][1] - G.nodes[node].get("height", 0.3) / 2,
                ),
                G.nodes[node].get("width", 1.0),
                G.nodes[node].get("height", 0.6),
                facecolor=value_node_colors[i],
                edgecolor="black",
                alpha=0.8,
                zorder=1,
            )
            ax.add_patch(rect)

        # Draw operation nodes (circles)
        op_nodes_list = [n for n in G.nodes() if G.nodes[n].get("type") == "operation"]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=op_nodes_list,
            node_shape="o",
            node_color="white",
            edgecolors="black",
            node_size=1800,
            alpha=0.8,
            ax=ax,
        )

        # Draw edges
        nx.draw_networkx_edges(
            G, pos, width=1.5, edge_color="gray", arrowsize=20, arrowstyle="-|>", ax=ax
        )

        # Draw labels for value nodes
        value_labels = {n: G.nodes[n].get("label", n) for n in value_nodes_list}
        nx.draw_networkx_labels(
            G, pos, labels=value_labels, font_size=10, font_weight="bold", ax=ax
        )

        # Draw labels for operation nodes
        op_labels = {n: G.nodes[n].get("label", n) for n in op_nodes_list}
        nx.draw_networkx_labels(
            G, pos, labels=op_labels, font_size=12, font_weight="bold", ax=ax
        )

        ax.set_title("Computational Graph", fontsize=16, fontweight="bold")
        ax.axis("off")

        # Set margins to make graph fit better
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        return ax

    def get_gradient(self, input_values=None, mode="forward"):
        """
        Compute the gradient of the function with respect to all input variables.

        Args:
            input_values: Dictionary mapping input variable names to values
            mode: Mode of automatic differentiation ('forward' or 'reverse')

        Returns:
            Dictionary mapping input variable names to their gradients
        """
        # Evaluate the graph first to ensure all values are computed
        self.evaluate(input_values)

        if mode == "forward":
            gradients = {}
            all_steps = []
            for var_name in self.input_nodes:
                grad, steps = self.forward_mode_ad(var_name)
                gradients[var_name] = grad
                all_steps.append(f"\nComputing ∂{self.output_node}/∂{var_name}:")
                all_steps.extend([f"  {step}" for step in steps])
            return gradients, all_steps
        elif mode == "reverse":
            gradients, steps = self.reverse_mode_ad()
            all_steps = [f"\nComputing gradients using reverse mode:"]
            all_steps.extend([f"  {step}" for step in steps])
            return gradients, all_steps
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def draw_graph_micrograd(
        self, filename="computational_graph", format="png", rankdir="TB", view=False
    ):
        """
        Draw the computational graph using graphviz in a style similar to micrograd.

        Args:
            filename: Name of the output file
            format: Format of the output file (png, svg, pdf)
            rankdir: Direction of the graph layout (TB = top-bottom, LR = left-right)
            view: Whether to open the rendered graph
        """
        if not HAS_GRAPHVIZ:
            raise ImportError(
                "Graphviz is not available. Install with 'pip install graphviz'"
            )

        # Create a new directed graph
        dot = graphviz.Digraph(format=format, graph_attr={"rankdir": rankdir})

        # First, determine which mode was used based on available data
        mode = "unknown"
        if any(
            hasattr(node, "reverse_grad") and node.reverse_grad is not None
            for _, node in self.nodes.items()
        ):
            mode = "reverse"
        elif any(
            hasattr(node, "forward_grad") and node.forward_grad
            for _, node in self.nodes.items()
        ):
            mode = "forward"

        # For forward mode, collect gradients with respect to the output
        output_gradients = {}
        if mode == "forward" and self.output_node:
            # Find gradients of output with respect to each input
            for var_name in self.input_nodes:
                for node_name, node in self.nodes.items():
                    if hasattr(node, "forward_grad") and var_name in node.forward_grad:
                        if node_name == self.output_node:
                            output_gradients[var_name] = node.forward_grad[var_name]

        # Add nodes to the graph
        for name, node in self.nodes.items():
            # Format the value to a reasonable precision
            if isinstance(node.value, (int, float)):
                value_str = f"{node.value:.4f}"
            else:
                value_str = str(node.value)

            # Get gradient value based on the AD mode that was used
            grad_str = "?"

            # For reverse mode, use reverse_grad
            if (
                mode == "reverse"
                and hasattr(node, "reverse_grad")
                and node.reverse_grad is not None
            ):
                grad_str = f"{node.reverse_grad:.4f}"

            # For forward mode
            elif mode == "forward":
                # Output node has gradient 1.0 by definition
                if name == self.output_node:
                    grad_str = "1.0000"
                # For input nodes, show the output gradient with respect to this input
                elif name in self.input_nodes and name in output_gradients:
                    grad_str = f"{output_gradients[name]:.4f}"
                # For intermediate nodes in forward mode, we need to show last accumulated gradient
                elif hasattr(node, "forward_grad") and node.forward_grad:
                    # Try to get the most relevant gradient for this node
                    # (the gradient of output with respect to this node's contribution)
                    if self.output_node and name in node.forward_grad:
                        grad_str = f"{node.forward_grad[name]:.4f}"
                    else:
                        # Show the appropriate derivative based on context
                        # For operations involving input variables, show those gradients
                        relevant_grads = []
                        for var_name in self.input_nodes:
                            if (
                                var_name in node.forward_grad
                                and abs(node.forward_grad[var_name]) > 1e-10
                            ):
                                relevant_grads.append(
                                    f"{node.forward_grad[var_name]:.4f}"
                                )

                        if relevant_grads:
                            grad_str = relevant_grads[
                                0
                            ]  # Just show the first relevant gradient

            # Create the node label
            if name in self.input_nodes:
                label = f"{{{name}|data: {value_str}|grad: {grad_str}}}"
                node_color = "lightblue"
            elif name == self.output_node:
                label = f"{{{name}|data: {value_str}|grad: {grad_str}}}"
                node_color = "lightgreen"
            else:
                label = f"{{{name}|op: {node.operation}|data: {value_str}|grad: {grad_str}}}"
                node_color = "lightgrey"

            # Add the node to the graph
            dot.node(
                name, label=label, shape="record", style="filled", fillcolor=node_color
            )

        # Add edges to the graph
        for name, node in self.nodes.items():
            if not hasattr(node, "inputs"):
                continue

            for input_name in node.inputs:
                dot.edge(input_name, name)

        # Render the graph
        try:
            dot.render(filename, view=view, cleanup=True)
            print(f"\nComputational graph saved as '{filename}.{format}'")
            if view:
                print(f"The graph should open automatically in your default viewer")
        except Exception as e:
            print(f"Error rendering graph: {e}")
            print("Make sure the graphviz executable is installed on your system")


def parse_function_and_build_graph(func_str, var_names=None):
    """
    Parse a function string and build a computational graph.

    Args:
        func_str: String representation of the function
        var_names: List of input variable names (if None, will try to infer from func_str)

    Returns:
        ComputationalGraph object
    """
    # If variable names are not provided, try to infer them from the function string
    if var_names is None:
        # Simple heuristic: look for letters that might be variables
        var_pattern = re.compile(r"\b([a-zA-Z][a-zA-Z0-9]*)\b")
        matches = var_pattern.findall(func_str)
        var_names = [m for m in matches if m not in ["sin", "cos", "exp", "log", "ln"]]
        var_names = list(set(var_names))  # Remove duplicates

    # Create sympy symbols and expression
    sympy_vars = {var: sp.Symbol(var) for var in var_names}

    # Replace ln with log for sympy and ^ with ** for power operation
    func_str = func_str.replace("ln", "log")  # Replace ln with log for sympy
    func_str = func_str.replace("^", "**")  # Replace ^ with ** for power operation

    # Parse the expression
    try:
        expr = parse_expr(func_str, local_dict=sympy_vars)
    except Exception as e:
        raise ValueError(f"Failed to parse expression: {e}")

    print(f"Parsed expression: {expr}")

    # Build computational graph using sympy expression tree
    graph = ComputationalGraph()

    # Add input nodes
    for var in var_names:
        graph.add_input(var)

    # Function to recursively process the expression tree
    node_counter = 0
    node_map = {}  # Maps sympy expression to graph node name

    def process_expr(expr):
        nonlocal node_counter

        # Check if we've already processed this subexpression
        if expr in node_map:
            return node_map[expr]

        # Handle constants
        if expr.is_number:
            node_name = f"const_{node_counter}"
            node_counter += 1
            graph.add_input(node_name, float(expr))
            node_map[expr] = node_name
            return node_name

        # Handle symbols (variables)
        if expr.is_symbol:
            return str(expr)

        # Handle operations
        op = expr.func
        args = expr.args

        # Process arguments first
        arg_nodes = [process_expr(arg) for arg in args]

        # Create the operation node
        node_name = f"v_{node_counter}"
        node_counter += 1

        # Determine the operation type
        if op == sp.Add:
            op_name = "+"
        elif op == sp.Mul:
            op_name = "*"
        elif op == sp.sin:
            op_name = "sin"
        elif op == sp.cos:
            op_name = "cos"
        elif op == sp.exp:
            op_name = "exp"
        elif op == sp.log:
            op_name = "log"
        elif op == sp.Pow:
            op_name = "^"
        elif op == sp.div or op == sp.Rational:
            op_name = "/"
        elif op == sp.Neg:
            op_name = "-"
        elif op == sp.Sub:
            op_name = "-"
        else:
            op_name = str(op)

        # Add the node to the graph
        graph.add_node(node_name, op_name, arg_nodes, expr=expr)
        node_map[expr] = node_name

        return node_name

    # Process the entire expression
    output_node = process_expr(expr)
    graph.set_output(output_node)

    return graph, expr


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze functions using automatic differentiation."
    )
    parser.add_argument("--function", "-f", type=str, help="Function to analyze")
    parser.add_argument(
        "--variables", "-v", type=str, nargs="+", help="Input variable names"
    )
    parser.add_argument(
        "--values", "-val", type=float, nargs="+", help="Values for input variables"
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="both",
        choices=["forward", "reverse", "both"],
        help="Mode of automatic differentiation",
    )
    parser.add_argument(
        "--viz",
        type=str,
        default="micrograd",
        choices=["micrograd", "matplotlib"],
        help="Visualization method to use",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "svg", "pdf"],
        help="Output format for the graph visualization",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open the visualization after generating",
    )

    args = parser.parse_args()

    # Interactive mode if no function is provided
    if args.function is None:
        print("Welcome to the Automatic Differentiation Solver!")
        print(
            "This tool helps analyze functions using computational graphs and automatic differentiation.\n"
        )

        func_str = input(
            "Enter the function (e.g., 'sin(x*y) + exp(z)*cos(x) + x^2 + y^2 + z^2'): "
        )
        var_names = input(
            "Enter the input variable names separated by spaces (e.g., 'x y z'): "
        ).split()

        values_str = input("Enter the values for each variable (e.g., '1.0 2.0 0.5'): ")
        if values_str.strip():
            values = list(map(float, values_str.split()))
            input_values = {var: val for var, val in zip(var_names, values)}
        else:
            input_values = None

        mode = input(
            "Enter the mode of automatic differentiation (forward/reverse/both): "
        ).lower()
        if mode not in ["forward", "reverse", "both"]:
            mode = "both"

        viz = input("Enter the visualization method (micrograd/matplotlib): ").lower()
        if viz not in ["micrograd", "matplotlib"]:
            viz = "micrograd"

        format = input("Enter the output format (png/svg/pdf): ").lower()
        if format not in ["png", "svg", "pdf"]:
            format = "png"

        view = input("Open the visualization after generating? (y/n): ").lower() == "y"
    else:
        func_str = args.function
        var_names = args.variables

        if args.values:
            input_values = {var: val for var, val in zip(var_names, args.values)}
        else:
            input_values = None

        mode = args.mode
        viz = args.viz
        format = args.format
        view = args.view

    # Parse the function and build the computational graph
    graph, expr = parse_function_and_build_graph(func_str, var_names)

    # Evaluate the graph
    if input_values:
        output_value = graph.evaluate(input_values)
        print(f"\nFunction value at {input_values}: {output_value}")
    else:
        print("\nNo input values provided, using default values.")
        input_values = {var: 1.0 for var in var_names}  # Default values
        output_value = graph.evaluate(input_values)
        print(f"Using default values {input_values}")
        print(f"Function value: {output_value}")

    # Print intermediate variables
    print("\nIntermediate variables:")
    for name, node in graph.nodes.items():
        if name in graph.input_nodes:
            print(f"{name} (input): {node.value}")
        elif name == graph.output_node:
            print(f"{name} (output): {node.value}")
        else:
            print(f"{name} = {node.operation}({', '.join(node.inputs)}): {node.value}")

    # Compute gradients before visualization so they appear in the graph
    if mode in ["forward", "both"]:
        gradients, steps = graph.get_gradient(input_values, mode="forward")
        print("\nForward mode automatic differentiation:")
        print("Gradients:", gradients)
        print("Steps:", "\n".join(steps))

    if mode in ["reverse", "both"]:
        gradients, steps = graph.get_gradient(input_values, mode="reverse")
        print("\nReverse mode automatic differentiation:")
        print("Gradients:", gradients)
        print("Steps:", "\n".join(steps))

    # Draw the computational graph
    filename = "computational_graph"
    if viz == "micrograd" and HAS_GRAPHVIZ:
        # Try to use graphviz visualization
        graph.draw_graph_micrograd(filename=filename, format=format, view=view)
    else:
        # Fall back to matplotlib visualization
        if viz == "micrograd" and not HAS_GRAPHVIZ:
            print(
                "\nWarning: graphviz not available, falling back to matplotlib visualization."
            )
            print("To use graphviz visualization, install graphviz:")
            print("  pip install graphviz")
            print(
                "  And also the system package: brew install graphviz (Mac) or apt-get install graphviz (Linux)"
            )

        plt.figure(figsize=(12, 10))
        graph.draw_graph()
        plt.savefig(f"{filename}.{format}", bbox_inches="tight")
        print(f"\nComputational graph saved as '{filename}.{format}'")

        if view:
            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    main()
