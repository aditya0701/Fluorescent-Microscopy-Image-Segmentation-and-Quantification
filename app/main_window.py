"""
main_window.py
Main application window for BoutonViewer.  Orchestrates all components:
the 3D PyVista viewer, the 2D slice viewer, the right-hand statistics
sidebar, the toolbar, the progress bar, and all dialogs.

Component communication follows a strict hub-and-spoke pattern: every
signal from a viewer or sidebar widget routes through the main window,
which updates the BoutonStore and then instructs each component to
refresh.  Components do not communicate directly with each other.

Toolbar actions
---------------
  Load Image       — opens a TIFF file, asks LSM vs Airyscan, renders image
  Set Checkpoint   — sets the path to the MicroSAM .pt checkpoint file
  Model            — selects the MicroSAM model variant (large/base) used
                      the next time Predict is clicked
  Predict          — runs the full preprocessing + inference pipeline
  View toggle      — switches between 3D volume and 2D slice modes

Keyboard shortcuts
------------------
  Delete           — deletes the currently selected bouton (if any)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QPushButton, QLabel, QSlider, QFileDialog, QMessageBox,
    QProgressBar, QComboBox, QToolBar, QStatusBar, QDialog,
    QDialogButtonBox, QRadioButton, QLineEdit, QFormLayout,
    QGroupBox, QStackedWidget, QDoubleSpinBox,
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QAction, QKeySequence, QShortcut

from app.data.image_loader import load_and_standardize
from app.data.bouton_store import BoutonStore
from app.viewer.volume_viewer import VolumeViewer
from app.viewer.slice_viewer import SliceViewer
from app.sidebar.bouton_panel import BoutonPanel


# ------------------------------------------------------------------
# Helper dialogs
# ------------------------------------------------------------------

class ImageTypeDialog(QDialog):
    """
    Modal dialog asking the user which acquisition modality produced the
    loaded image.  The answer determines both the preprocessing pipeline
    that is applied and the physical voxel calibration used for all
    downstream measurements (see BoutonStore.VOXEL_SIZE_BY_TYPE).

    The voxel size (z, y, x in µm) is shown alongside the modality choice
    and may be edited manually.  Each modality remembers its own values
    independently: switching the radio button swaps in that modality's
    last-used values (the built-in default the first time, or whatever the
    user previously typed for it), so toggling back and forth between
    options never discards an edit made earlier in the same dialog session.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Acquisition Type")
        self.setModal(True)
        self.setFixedSize(440, 320)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel("Select the acquisition modality of the loaded image:")
        )

        self._lsm_radio      = QRadioButton(
            "LSM  (rolling ball + Richardson-Lucy deconvolution will be applied)"
        )
        self._airyscan_radio = QRadioButton(
            "Airyscan  (image will be downscaled to 1100 px if necessary)"
        )
        self._airyscan_1100_radio = QRadioButton(
            "Airyscan, already downscaled to 1100 px\n"
            "(no LSM preprocessing — basic normalisation only)"
        )
        self._lsm_radio.setChecked(True)

        layout.addWidget(self._lsm_radio)
        layout.addWidget(self._airyscan_radio)
        layout.addWidget(self._airyscan_1100_radio)

        # Per-modality voxel size memory, seeded from the built-in defaults.
        self._voxel_by_type = dict(BoutonStore.VOXEL_SIZE_BY_TYPE)
        self._current_type  = "LSM"

        voxel_box = QGroupBox("Voxel size (µm)")
        form      = QFormLayout(voxel_box)
        self._z_spin = QDoubleSpinBox()
        self._y_spin = QDoubleSpinBox()
        self._x_spin = QDoubleSpinBox()
        for spin in (self._z_spin, self._y_spin, self._x_spin):
            spin.setDecimals(4)
            spin.setRange(0.0001, 100.0)
            spin.setSingleStep(0.001)
        form.addRow("Z:", self._z_spin)
        form.addRow("Y:", self._y_spin)
        form.addRow("X:", self._x_spin)
        layout.addWidget(voxel_box)

        self._load_voxel_spins("LSM")

        self._lsm_radio.toggled.connect(
            lambda checked: checked and self._on_type_changed("LSM")
        )
        self._airyscan_radio.toggled.connect(
            lambda checked: checked and self._on_type_changed("Airyscan")
        )
        self._airyscan_1100_radio.toggled.connect(
            lambda checked: checked and self._on_type_changed("Airyscan_1100")
        )

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    def _on_type_changed(self, new_type: str):
        # Stash the values currently shown under the outgoing modality
        # before loading the incoming one's, so repeated toggling preserves
        # whatever the user typed for each modality.
        self._voxel_by_type[self._current_type] = (
            self._z_spin.value(), self._y_spin.value(), self._x_spin.value()
        )
        self._current_type = new_type
        self._load_voxel_spins(new_type)

    def _load_voxel_spins(self, image_type: str):
        z, y, x = self._voxel_by_type.get(
            image_type, BoutonStore.VOXEL_SIZE_BY_TYPE["LSM"]
        )
        self._z_spin.setValue(z)
        self._y_spin.setValue(y)
        self._x_spin.setValue(x)

    @property
    def image_type(self) -> str:
        if self._lsm_radio.isChecked():
            return "LSM"
        if self._airyscan_1100_radio.isChecked():
            return "Airyscan_1100"
        return "Airyscan"

    @property
    def voxel_size_um(self) -> tuple:
        """Physical voxel size (z, y, x) in µm as currently shown in the spin boxes."""
        return (self._z_spin.value(), self._y_spin.value(), self._x_spin.value())


class CheckpointDialog(QDialog):
    """
    Modal dialog for selecting the MicroSAM .pt checkpoint file.
    """

    def __init__(self, last_path: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select MicroSAM Checkpoint")
        self.setModal(True)
        self.setFixedSize(520, 110)

        layout = QVBoxLayout(self)
        form   = QFormLayout()

        self._path_edit = QLineEdit(last_path)
        self._path_edit.setPlaceholderText("Path to best.pt checkpoint file…")
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse)

        row = QHBoxLayout()
        row.addWidget(self._path_edit)
        row.addWidget(browse_btn)
        form.addRow("Checkpoint:", row)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select MicroSAM Checkpoint", "", "PyTorch checkpoint (*.pt)"
        )
        if path:
            self._path_edit.setText(path)

    @property
    def checkpoint_path(self) -> str:
        return self._path_edit.text().strip()


# ------------------------------------------------------------------
# Main window
# ------------------------------------------------------------------

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("BoutonViewer")
        self.setMinimumSize(1280, 800)

        self._store:           BoutonStore = BoutonStore()
        self._worker           = None
        self._checkpoint_path: str = ""
        self._view_mode:       str = "3d"   # "3d" or "2d"

        self._build_ui()
        self._connect_signals()
        self._register_shortcuts()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ---- Central widget ----
        central      = QWidget()
        root_layout  = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(central)

        # ---- Toolbar ----
        self.addToolBar(self._build_toolbar())

        # ---- Main horizontal split: viewer area | sidebar ----
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # -- Left side: stacked viewer + controls --
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(4)

        # Stacked widget holding the 3D and 2D viewers.
        self._stack   = QStackedWidget()
        self._viewer3 = VolumeViewer()
        self._viewer2 = SliceViewer()
        self._stack.addWidget(self._viewer3)   # index 0 = 3D
        self._stack.addWidget(self._viewer2)   # index 1 = 2D
        left_layout.addWidget(self._stack, stretch=1)

        # Z slice slider (visible only in 2D mode).
        self._slice_group = QGroupBox("Z Slice")
        slice_row         = QHBoxLayout(self._slice_group)
        self._slice_label  = QLabel("Z: 0")
        self._slice_label.setFixedWidth(50)
        self._slice_slider = QSlider(Qt.Orientation.Horizontal)
        self._slice_slider.setMinimum(0)
        self._slice_slider.setMaximum(0)
        self._slice_slider.valueChanged.connect(self._on_slice_changed)
        slice_row.addWidget(self._slice_label)
        slice_row.addWidget(self._slice_slider)
        self._slice_group.setVisible(False)
        left_layout.addWidget(self._slice_group)

        splitter.addWidget(left_widget)

        # -- Right side: statistics sidebar --
        self._sidebar = BoutonPanel()
        self._sidebar.setMinimumWidth(250)
        self._sidebar.setMaximumWidth(320)
        splitter.addWidget(self._sidebar)

        splitter.setSizes([960, 280])
        root_layout.addWidget(splitter)

        # ---- Status bar ----
        # The progress label and bar live here so they are always visible
        # regardless of which viewer or layout is active.
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready.  Load a TIFF image to begin.")

        self._prog_label = QLabel("")
        self._prog_label.setMinimumWidth(260)
        self._prog_label.setVisible(False)
        self._status.addPermanentWidget(self._prog_label)

        self._prog_bar = QProgressBar()
        self._prog_bar.setRange(0, 100)
        self._prog_bar.setFixedWidth(180)
        self._prog_bar.setVisible(False)
        self._status.addPermanentWidget(self._prog_bar)

    def _build_toolbar(self) -> QToolBar:
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)

        load_act = QAction("Load Image", self)
        load_act.setToolTip("Open a TIFF image stack")
        load_act.triggered.connect(self._on_load_image)
        toolbar.addAction(load_act)

        toolbar.addSeparator()

        ckpt_act = QAction("Set Checkpoint", self)
        ckpt_act.setToolTip("Select the MicroSAM .pt checkpoint file")
        ckpt_act.triggered.connect(self._on_set_checkpoint)
        toolbar.addAction(ckpt_act)

        self._predict_act = QAction("▶  Predict", self)
        self._predict_act.setToolTip("Run MicroSAM segmentation on the loaded image")
        self._predict_act.triggered.connect(self._on_predict)
        self._predict_act.setEnabled(False)
        toolbar.addAction(self._predict_act)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel("  Model: "))
        self._model_combo = QComboBox()
        self._model_combo.addItem("Large (vit_l_lm)", "vit_l_lm")
        self._model_combo.addItem("Base (vit_b_lm)", "vit_b_lm")
        toolbar.addWidget(self._model_combo)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel("  View: "))
        self._view_combo = QComboBox()
        self._view_combo.addItems(["3D Volume", "2D Slices"])
        self._view_combo.currentIndexChanged.connect(self._on_view_mode_changed)
        toolbar.addWidget(self._view_combo)

        return toolbar

    def _connect_signals(self):
        # Viewer signals -> main window
        self._viewer3.bouton_hovered.connect(self._on_bouton_hovered)
        self._viewer3.bouton_selected.connect(self._on_bouton_selected)
        self._viewer2.bouton_hovered.connect(self._on_bouton_hovered)
        self._viewer2.bouton_selected.connect(self._on_bouton_selected)

        # Sidebar signals -> main window
        self._sidebar.delete_requested.connect(self._on_delete_bouton)
        self._sidebar.row_selected.connect(self._viewer3.highlight_bouton)

    def _register_shortcuts(self):
        # Allow the Delete key to remove the currently selected bouton.
        del_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), self)
        del_shortcut.activated.connect(self._on_delete_shortcut)

    # ------------------------------------------------------------------
    # Toolbar and dialog slots
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open TIFF Image", "", "TIFF files (*.tif *.tiff)"
        )
        if not path:
            return

        dlg = ImageTypeDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        image_type    = dlg.image_type
        voxel_size_um = dlg.voxel_size_um

        self._status.showMessage(f"Loading {Path(path).name}…")
        QApplication_processEvents()

        try:
            raw = load_and_standardize(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            self._status.showMessage("Load failed.")
            return

        # Update the store.
        self._store.clear()
        self._store.image         = raw
        self._store.image_type    = image_type
        self._store.image_path    = path
        self._store.voxel_size_um = voxel_size_um

        # Configure the Z slider.
        Z = raw.shape[0]
        self._slice_slider.setMaximum(Z - 1)
        self._slice_slider.setValue(Z // 2)

        # Send raw image to both viewers.
        self._viewer3.set_image(raw, self._store.voxel_size_um)
        self._viewer2.set_image(raw)
        self._viewer2.set_z(Z // 2)

        # Reset sidebar.
        self._sidebar.update_total_count(0)
        self._sidebar.set_hover_info(-1, 0.0, 0.0)

        self._predict_act.setEnabled(bool(self._checkpoint_path))
        self._status.showMessage(
            f"Loaded: {Path(path).name}   |   Shape: {raw.shape}   |   "
            f"Type: {image_type}"
        )

    @pyqtSlot()
    def _on_set_checkpoint(self):
        dlg = CheckpointDialog(last_path=self._checkpoint_path, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.checkpoint_path:
            self._checkpoint_path = dlg.checkpoint_path
            self._predict_act.setEnabled(self._store.image is not None)
            self._status.showMessage(
                f"Checkpoint: {Path(self._checkpoint_path).name}"
            )

    @pyqtSlot()
    def _on_predict(self):
        if self._store.image is None or not self._checkpoint_path:
            return

        self._set_controls_enabled(False)
        self._viewer3.pause_rendering()
        self._prog_label.setText("Starting up…")
        self._prog_label.setVisible(True)
        self._prog_bar.setValue(0)
        self._prog_bar.setVisible(True)

        from app.model.worker import PredictionWorker

        self._worker = PredictionWorker(
            image=self._store.image,
            image_type=self._store.image_type,
            checkpoint_path=self._checkpoint_path,
            model_type=self._model_combo.currentData(),
            parent=self,
        )
        self._worker.progress.connect(self._on_prediction_progress)
        self._worker.finished.connect(self._on_prediction_finished)
        self._worker.error.connect(self._on_prediction_error)
        self._worker.start()

    # ------------------------------------------------------------------
    # Prediction pipeline slots
    # ------------------------------------------------------------------

    _STEP_DESCRIPTIONS = [
        ("background",   "Removing background noise (rolling ball filter)"),
        ("deconvolution","Sharpening structures (Richardson-Lucy deconvolution)"),
        ("Downsampling", "Rescaling image to 1100 × 1100 px"),
        ("Normalising",  "Normalising pixel intensities"),
        ("Normalizing",  "Normalising pixel intensities"),
        ("Finalising",   "Finalising image preparation"),
        ("Preprocessing","Preparing image for segmentation"),
        ("Converting",   "Converting to RGB format for MicroSAM"),
        ("Loading model","Loading neural network weights"),
        ("Segmenting",   "Running AI segmentation — slice by slice"),
        ("connected",    "Linking detections across Z-slices (3D)"),
        ("relabelling",  "Assigning final bouton labels"),
        ("Done",         "Counting detected boutons"),
        ("Upscaling",    "Restoring original image resolution"),
    ]

    @pyqtSlot(str, int)
    def _on_prediction_progress(self, step: str, pct: int):
        display = next(
            (d for k, d in self._STEP_DESCRIPTIONS if k.lower() in step.lower()),
            step,
        )
        self._prog_label.setText(display)
        self._prog_bar.setValue(pct)
        self._status.showMessage(f"[{pct}%]  {step}")

    @pyqtSlot(object)
    def _on_prediction_finished(self, labels: np.ndarray):
        self._store.set_labels(labels)

        self._viewer3.set_labels(labels, self._store.label_colors)
        self._viewer2.set_labels(labels, self._store.label_colors)
        self._sidebar.update_stats(self._store.get_stats_list())

        self._viewer3.resume_rendering()
        self._set_controls_enabled(True)
        self._prog_label.setVisible(False)
        self._prog_bar.setVisible(False)
        self._status.showMessage(
            f"Prediction complete — {self._store.total_count} boutons detected."
        )

    @pyqtSlot(str)
    def _on_prediction_error(self, message: str):
        QMessageBox.critical(self, "Prediction Error", message)
        self._viewer3.resume_rendering()
        self._set_controls_enabled(True)
        self._prog_label.setVisible(False)
        self._prog_bar.setVisible(False)
        self._status.showMessage("Prediction failed.")

    # ------------------------------------------------------------------
    # View mode
    # ------------------------------------------------------------------

    @pyqtSlot(int)
    def _on_view_mode_changed(self, index: int):
        self._view_mode = "3d" if index == 0 else "2d"
        self._stack.setCurrentIndex(index)
        self._slice_group.setVisible(self._view_mode == "2d")

    @pyqtSlot(int)
    def _on_slice_changed(self, value: int):
        self._slice_label.setText(f"Z: {value}")
        self._viewer2.set_z(value)

    # ------------------------------------------------------------------
    # Bouton interaction slots
    # ------------------------------------------------------------------

    @pyqtSlot(int)
    def _on_bouton_hovered(self, label_id: int):
        if label_id > 0 and label_id in self._store.stats:
            stats = self._store.stats[label_id]
            self._sidebar.set_hover_info(
                label_id, stats.volume_um3, stats.surface_area_um2
            )
        else:
            self._sidebar.set_hover_info(-1, 0.0, 0.0)

    @pyqtSlot(int)
    def _on_bouton_selected(self, label_id: int):
        if label_id > 0:
            self._sidebar.highlight_row(label_id)
            # Sync highlight to 3D viewer if selection came from 2D viewer.
            if self._view_mode == "2d":
                self._viewer3.highlight_bouton(label_id)

    @pyqtSlot(int)
    def _on_delete_bouton(self, label_id: int):
        reply = QMessageBox.question(
            self,
            "Delete Bouton",
            f"Remove bouton {label_id} from the analysis?\n\n"
            f"This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._store.delete_bouton(label_id)
        self._viewer3.remove_bouton(label_id)
        self._viewer2.remove_label(label_id)
        self._sidebar.remove_row(label_id)

        self._status.showMessage(
            f"Bouton {label_id} removed.  "
            f"{self._store.total_count} boutons remaining."
        )

    @pyqtSlot()
    def _on_delete_shortcut(self):
        """Triggers deletion of the currently selected bouton via the Delete key."""
        # Ask the sidebar for the currently selected label, then route through
        # the normal deletion flow.
        if self._sidebar._selected_label is not None:
            self._on_delete_bouton(self._sidebar._selected_label)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _set_controls_enabled(self, enabled: bool):
        """Enables or disables interactive controls during prediction."""
        self._predict_act.setEnabled(enabled and bool(self._checkpoint_path))
        self._view_combo.setEnabled(enabled)


# Avoid a circular import from calling QApplication.processEvents inside
# a method before the application is fully constructed.
def QApplication_processEvents():
    from PyQt6.QtWidgets import QApplication
    QApplication.processEvents()
