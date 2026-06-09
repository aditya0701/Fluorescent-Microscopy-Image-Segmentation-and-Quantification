"""
BoutonViewer — 3D bouton segmentation and quantification.
Napari handles all display; this file just wires the pieces together.
"""

import os

# Must be set before torch/CUDA initialises.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import napari
from app.dock_widget import BoutonDockWidget
from app.controller import BoutonController

# Module-level reference keeps BoutonController alive for the entire process.
# PyQt5 stores signal connections to non-QObject receivers via weak references —
# if the controller were GC'd, every connection would silently die and no slots
# would fire. napari.Viewer is a pydantic model so arbitrary attribute assignment
# on it is blocked; a module-level variable is the simplest safe alternative.
_controller = None


def main():
    global _controller
    viewer = napari.Viewer(title="BoutonViewer")

    dock = BoutonDockWidget()
    viewer.window.add_dock_widget(dock, area="right", name="Bouton Analysis")

    _controller = BoutonController(viewer, dock)

    napari.run()


if __name__ == "__main__":
    main()
