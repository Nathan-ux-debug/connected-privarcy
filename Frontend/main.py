"""PrivARcy desktop app entry point.

Run with the project's interpreter, from anywhere:
    "Backend\\venv\\Scripts\\python.exe" Frontend\\main.py
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
for p in (str(PROJECT_ROOT), str(BASE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Import torch BEFORE PyQt5 on Windows to avoid c10.dll initialisation errors.
try:
    import torch  # noqa: F401
except Exception as exc:  # the loader screen will show the real error if the backend needs it
    print(f"[warn] torch import failed: {exc}")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from ui import theme
from ui.shell import MainWindow

LOGO = BASE_DIR / "assets" / "privarcy-main-logo.png"


def main() -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("PrivARcy")
    app.setStyleSheet(theme.stylesheet())
    window = MainWindow(str(LOGO) if LOGO.exists() else "")
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
