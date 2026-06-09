"""
controller.py
Orchestrates the BoutonViewer application around a napari Viewer.

Responsibilities
----------------
- Loads TIFF images and adds them to napari as Image layers (two
  independent channels so napari can give each its own blending mode,
  colourmap, contrast, and gamma independently).
- Runs the MicroSAM prediction pipeline in a background QThread and
  adds the resulting label array as a napari Labels layer.
- Registers mouse-move and mouse-press callbacks on the Labels layer
  so that hovering and clicking a bouton updates the dock panel.
- Handles deletion: zeroes the label in the numpy array and refreshes
  the Labels layer so both 2D and 3D views update instantly.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
from qtpy.QtWidgets import QMessageBox, QFileDialog

from app.data.image_loader import load_and_standardize
from app.data.bouton_store import BoutonStore


class BoutonController:

    def __init__(self, viewer, dock):
        self._viewer = viewer
        self._dock   = dock
        self._store  = BoutonStore()
        self._worker = None

        self._image_layers  = []   # list of napari Image layers (one per channel)
        self._labels_layer  = None

        # Wire dock-widget signals to controller methods.
        dock.load_requested.connect(self._on_load)
        dock.predict_requested.connect(self._on_predict)
        dock.delete_requested.connect(self._on_delete)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _on_load(self, path: str, image_type: str):
        try:
            raw = load_and_standardize(path)
        except Exception as exc:
            QMessageBox.critical(None, "Load Error", str(exc))
            self._dock.set_status("Load failed.")
            return

        self._store.clear()
        self._store.image      = raw
        self._store.image_type = image_type
        self._store.image_path = path

        # Remove any existing layers so reloading a new image is clean.
        for layer in list(self._image_layers):
            self._viewer.layers.remove(layer)
        self._image_layers = []
        if self._labels_layer is not None:
            self._viewer.layers.remove(self._labels_layer)
            self._labels_layer = None

        vz, vy, vx = self._store.voxel_size_um

        # channel_axis=1 splits (Z, C, Y, X) → two (Z, Y, X) Image layers.
        # Each layer gets its own colourmap and blending mode, replicating the
        # napari per-layer settings the user had: green=translucent, red=additive.
        try:
            self._image_layers = self._viewer.add_image(
                raw,
                channel_axis=1,
                colormap=["green", "red"],
                blending=["translucent", "additive"],
                name=["channel-0 (green)", "channel-1 (red)"],
                scale=(vz, vy, vx),
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(None, "Display Error", str(exc))
            self._dock.set_status("Failed to display image.")
            return

        self._viewer.reset_view()

        self._dock.set_status(
            f"Loaded: {Path(path).name}  |  shape: {raw.shape}  |  type: {image_type}"
        )
        self._dock.set_predict_enabled(bool(self._dock.checkpoint_path))
        self._dock.clear_stats()

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _on_predict(self):
        if self._store.image is None:
            self._dock.set_status("Load a TIFF image first.")
            return
        checkpoint = self._dock.checkpoint_path
        if not checkpoint:
            QMessageBox.warning(None, "No Checkpoint", "Please select a MicroSAM checkpoint first.")
            return

        from app.model.worker import PredictionWorker

        self._dock.set_predict_enabled(False)
        self._dock.set_progress_visible(True)
        self._dock.set_status("Starting prediction…")

        self._worker = PredictionWorker(
            image=self._store.image,
            image_type=self._dock.image_type,
            checkpoint_path=checkpoint,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, step: str, pct: int):
        self._dock.set_progress(step, pct)

    def _on_finished(self, labels: np.ndarray):
        self._store.set_labels(labels)

        # Remove a stale Labels layer if there is one.
        if self._labels_layer is not None:
            self._viewer.layers.remove(self._labels_layer)

        vz, vy, vx = self._store.voxel_size_um
        self._labels_layer = self._viewer.add_labels(
            labels,
            name="boutons",
            scale=(vz, vy, vx),
        )
        self._register_label_callbacks()

        self._dock.update_stats(self._store.get_stats_list())
        self._dock.set_progress_visible(False)
        self._dock.set_predict_enabled(True)
        self._dock.set_status(
            f"Prediction complete — {self._store.total_count} boutons detected."
        )

    def _on_error(self, message: str):
        QMessageBox.critical(None, "Prediction Error", message)
        self._dock.set_progress_visible(False)
        self._dock.set_predict_enabled(True)
        self._dock.set_status("Prediction failed.")

    # ------------------------------------------------------------------
    # Hover and click callbacks on the Labels layer
    # ------------------------------------------------------------------

    def _register_label_callbacks(self):
        layer = self._labels_layer

        @layer.mouse_move_callbacks.append
        def _on_hover(layer, event):
            try:
                label_id = layer.get_value(
                    event.position,
                    view_direction=getattr(event, "view_direction", None),
                    dims_displayed=getattr(event, "dims_displayed", None),
                    world=True,
                )
            except Exception:
                label_id = None

            if label_id and int(label_id) > 0:
                lid = int(label_id)
                stats = self._store.stats.get(lid)
                if stats:
                    self._dock.set_hover_info(lid, stats.volume_um3, stats.surface_area_um2)
                    return
            self._dock.set_hover_info(-1, 0.0, 0.0)

        @layer.mouse_press_callbacks.append
        def _on_click(layer, event):
            if event.button != 1:
                return
            try:
                label_id = layer.get_value(
                    event.position,
                    view_direction=getattr(event, "view_direction", None),
                    dims_displayed=getattr(event, "dims_displayed", None),
                    world=True,
                )
            except Exception:
                label_id = None

            if label_id and int(label_id) > 0:
                self._dock.highlight_row(int(label_id))

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def _on_delete(self, label_id: int):
        reply = QMessageBox.question(
            None,
            "Delete Bouton",
            f"Remove bouton {label_id} from the analysis?\n\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._store.delete_bouton(label_id)

        if self._labels_layer is not None:
            self._labels_layer.data[self._labels_layer.data == label_id] = 0
            self._labels_layer.refresh()

        self._dock.remove_row(label_id)
        self._dock.set_status(
            f"Bouton {label_id} removed.  {self._store.total_count} boutons remaining."
        )
