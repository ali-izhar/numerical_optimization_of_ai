"""Entry point for the numerical optimization UI."""

import sys
from PyQt6.QtWidgets import QApplication
from .main_window import MainWindow


def main():
    """Run the UI application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
