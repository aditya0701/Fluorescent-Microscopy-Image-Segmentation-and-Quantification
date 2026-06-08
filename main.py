"""
BoutonViewer — local desktop application for 3D bouton segmentation
and quantification using MicroSAM on Drosophila MB calyx confocal data.
"""

import os
import sys

# Must be set before any PyQt6 or pyvista import so pyvistaqt
# picks up the correct Qt binding.
os.environ.setdefault("QT_API", "pyqt6")

# Must be set before torch initialises CUDA (it reads this on first use).
# Reduces allocator fragmentation — without it, small-VRAM GPUs (e.g. 4 GB)
# can hit "CUDA out of memory" while gigabytes are technically free but
# split into chunks too small for the next allocation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont
from app.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("BoutonViewer")
    app.setOrganizationName("NeuroBio Lab")
    app.setStyle("Fusion")

    # Set a clean base font for the entire application.
    font = QFont("Segoe UI", 9)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
