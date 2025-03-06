"""Method selector component for the numerical optimization UI."""

from PyQt6.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
)


class MethodSelector(QGroupBox):
    """Widget for selecting and configuring numerical methods."""

    def __init__(self, parent=None):
        super().__init__("Method Selection", parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)  # Reduced spacing

        # Method selection with updated styling
        method_label = QLabel("Method:")
        method_label.setStyleSheet("color: #b3b3b3; font-size: 10px;")
        layout.addWidget(method_label)

        self.method_combo = QComboBox()
        methods = [
            "Newton's Method",
            "BFGS",
            "Steepest Descent",
            "Nelder-Mead",
            "Powell's Method",
        ]
        self.method_combo.addItems(methods)
        self.method_combo.setCurrentText(
            "Newton's Method"
        )  # Set Newton's method as default

        # Connect method selection to update advanced options
        self.method_combo.currentTextChanged.connect(self.update_advanced_options)

        # Updated combobox styling
        self.method_combo.setStyleSheet(
            """
            QComboBox {
                background-color: #222222;
                color: #ffffff;
                border: 1px solid #333333;
                border-radius: 3px;
                padding: 5px;
                padding-right: 15px;
                font-size: 10px;
                min-height: 20px;
            }
            QComboBox:focus {
                border: 1px solid #4a90e2;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 15px;
                border-left: none;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid #ffffff;
                width: 0;
                height: 0;
                margin-right: 6px;
            }
            QComboBox QAbstractItemView {
                background-color: #222222;
                border: 1px solid #333333;
                border-radius: 3px;
                selection-background-color: #333333;
                selection-color: #ffffff;
                padding: 2px;
            }
        """
        )
        layout.addWidget(self.method_combo)

        # Advanced options group
        self.advanced_group = QGroupBox("Advanced Options")
        self.advanced_group.setStyleSheet(
            """
            QGroupBox {
                font-weight: bold;
                color: #ffffff;
                font-size: 10px;
            }
        """
        )
        advanced_layout = QVBoxLayout()

        # Step Length Method
        step_length_layout = QHBoxLayout()
        step_length_label = QLabel("Step Length:")
        step_length_label.setStyleSheet("color: #b3b3b3; font-size: 10px;")

        self.step_length_combo = QComboBox()
        step_length_methods = [
            "fixed",
            "backtracking",
            "wolfe",
            "strong_wolfe",
            "goldstein",
        ]
        self.step_length_combo.addItems(step_length_methods)
        self.step_length_combo.setCurrentText("backtracking")  # Default
        self.step_length_combo.setStyleSheet(
            """
            QComboBox {
                background-color: #222222;
                color: #ffffff;
                border: 1px solid #333333;
                border-radius: 3px;
                padding: 5px;
                padding-right: 15px;
                font-size: 10px;
                min-height: 20px;
            }
            """
        )

        step_length_layout.addWidget(step_length_label)
        step_length_layout.addWidget(self.step_length_combo)
        advanced_layout.addLayout(step_length_layout)

        # Descent Direction Method
        descent_layout = QHBoxLayout()
        descent_label = QLabel("Descent Direction:")
        descent_label.setStyleSheet("color: #b3b3b3; font-size: 10px;")

        self.descent_combo = QComboBox()
        descent_methods = [
            "steepest_descent",
            "newton",
            "bfgs",
        ]
        self.descent_combo.addItems(descent_methods)
        self.descent_combo.setCurrentText("newton")  # Default
        self.descent_combo.setStyleSheet(
            """
            QComboBox {
                background-color: #222222;
                color: #ffffff;
                border: 1px solid #333333;
                border-radius: 3px;
                padding: 5px;
                padding-right: 15px;
                font-size: 10px;
                min-height: 20px;
            }
            """
        )

        descent_layout.addWidget(descent_label)
        descent_layout.addWidget(self.descent_combo)
        advanced_layout.addLayout(descent_layout)

        # Initial step size
        step_size_layout = QHBoxLayout()
        step_size_label = QLabel("Initial Step Size:")
        step_size_label.setStyleSheet("color: #b3b3b3; font-size: 10px;")

        self.step_size_spin = QDoubleSpinBox()
        self.step_size_spin.setRange(0.001, 10.0)
        self.step_size_spin.setValue(1.0)
        self.step_size_spin.setDecimals(3)
        self.step_size_spin.setSingleStep(0.1)
        self.step_size_spin.setStyleSheet(
            """
            QDoubleSpinBox {
                background-color: #222222;
                color: #ffffff;
                border: 1px solid #333333;
                border-radius: 3px;
                padding: 3px;
                padding-right: 12px;
                min-width: 100px;
                font-size: 10px;
            }
            """
        )

        step_size_layout.addWidget(step_size_label)
        step_size_layout.addWidget(self.step_size_spin)
        advanced_layout.addLayout(step_size_layout)

        # Show advanced options checkbox
        self.show_advanced = QCheckBox("Show Advanced Options")
        self.show_advanced.setStyleSheet(
            """
            QCheckBox {
                color: #b3b3b3;
                font-size: 10px;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 12px;
                height: 12px;
                border-radius: 2px;
                border: 1px solid #333333;
            }
            QCheckBox::indicator:checked {
                background-color: #4a90e2;
            }
            """
        )
        self.show_advanced.setChecked(False)
        self.show_advanced.toggled.connect(self.toggle_advanced_options)
        layout.addWidget(self.show_advanced)

        self.advanced_group.setLayout(advanced_layout)
        layout.addWidget(self.advanced_group)
        self.advanced_group.setVisible(False)  # Hide by default

        # Parameters with updated styling
        param_group = QGroupBox("Parameters")
        param_group.setStyleSheet(
            """
            QGroupBox {
                font-weight: bold;
                color: #ffffff;
                font-size: 10px;
            }
        """
        )
        param_layout = QVBoxLayout()
        param_layout.setSpacing(10)  # Reduced spacing

        # Tolerance with updated styling
        tol_layout = QHBoxLayout()
        tol_label = QLabel("Tolerance:")
        tol_label.setStyleSheet("color: #b3b3b3; font-size: 10px;")

        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setRange(1e-6, 1.0)  # Changed minimum to 1e-6
        self.tol_spin.setValue(1e-4)  # Default tolerance of 1e-4
        self.tol_spin.setDecimals(6)  # Show 6 decimal places
        self.tol_spin.setSingleStep(1e-4)  # Step by 1e-4

        # Updated spinbox styling
        self.tol_spin.setStyleSheet(
            """
            QDoubleSpinBox {
                background-color: #222222;
                color: #ffffff;
                border: 1px solid #333333;
                border-radius: 3px;
                padding: 3px;
                padding-right: 12px;
                min-width: 100px;
                font-size: 10px;
            }
            QDoubleSpinBox:focus {
                border: 1px solid #4a90e2;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                background-color: #333333;
                width: 12px;
                border-radius: 2px;
                margin: 1px;
            }
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #4a90e2;
            }
        """
        )

        tol_layout.addWidget(tol_label)
        tol_layout.addWidget(self.tol_spin)
        param_layout.addLayout(tol_layout)

        # Max iterations with updated styling
        iter_layout = QHBoxLayout()
        iter_label = QLabel("Max iterations:")
        iter_label.setStyleSheet("color: #b3b3b3; font-size: 10px;")

        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1, 10000)
        self.max_iter_spin.setValue(100)

        # Updated spinbox styling
        self.max_iter_spin.setStyleSheet(
            """
            QSpinBox {
                background-color: #222222;
                color: #ffffff;
                border: 1px solid #333333;
                border-radius: 3px;
                padding: 3px;
                padding-right: 12px;
                min-width: 100px;
                font-size: 10px;
            }
            QSpinBox:focus {
                border: 1px solid #4a90e2;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #333333;
                width: 12px;
                border-radius: 2px;
                margin: 1px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #4a90e2;
            }
        """
        )

        iter_layout.addWidget(iter_label)
        iter_layout.addWidget(self.max_iter_spin)
        param_layout.addLayout(iter_layout)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        self.setLayout(layout)

        # Initialize advanced options
        self.update_advanced_options(self.method_combo.currentText())

    def toggle_advanced_options(self, checked):
        """Toggle visibility of advanced options."""
        self.advanced_group.setVisible(checked)

    def update_advanced_options(self, method_name):
        """Update available advanced options based on selected method."""
        # Set default values based on the selected method
        if method_name == "Newton's Method":
            self.descent_combo.setCurrentText("newton")
            self.step_length_combo.setCurrentText("backtracking")
        elif method_name == "BFGS":
            self.descent_combo.setCurrentText("bfgs")
            self.step_length_combo.setCurrentText("wolfe")
        elif method_name == "Steepest Descent":
            self.descent_combo.setCurrentText("steepest_descent")
            self.step_length_combo.setCurrentText("backtracking")

        # Disable/enable options based on compatibility
        # Some methods may have fixed descent direction or step length methods
        if method_name in ["Nelder-Mead", "Powell's Method"]:
            # Derivative-free methods don't use step length or descent direction
            self.descent_combo.setEnabled(False)
            self.step_length_combo.setEnabled(False)
            self.step_size_spin.setEnabled(False)
        else:
            self.descent_combo.setEnabled(True)
            self.step_length_combo.setEnabled(True)
            self.step_size_spin.setEnabled(True)
