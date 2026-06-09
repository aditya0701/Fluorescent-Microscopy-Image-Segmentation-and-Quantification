"""
dock_widget.py
Right-hand dock panel for BoutonViewer inside a napari window.

Contains all user-facing controls: file loading, image-type selection,
checkpoint selection, prediction trigger, progress display, per-bouton
statistics table, hover info box, and deletion.

Uses qtpy so it works under whichever Qt binding napari has loaded
(PyQt5 in this environment).
"""

from __future__ import annotations
from typing import List, Optional

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QFileDialog, QProgressBar, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QFrame, QGroupBox,
    QSizePolicy,
)
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QFont

from app.data.bouton_store import BoutonStats


class BoutonDockWidget(QWidget):
    """
    Dock panel that lives on the right side of the napari window.

    Signals
    -------
    load_requested(path, image_type)
        User clicked Load and confirmed an image type.
    predict_requested()
        User clicked Predict.
    delete_requested(label_id)
        User clicked Delete (after confirmation in the controller).
    """

    load_requested    = Signal(str, str)
    predict_requested = Signal()
    delete_requested  = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(260)
        self.setMaximumWidth(340)
        self._selected_label: Optional[int] = None
        self._build_ui()

    # ------------------------------------------------------------------
    # Public API called by the controller
    # ------------------------------------------------------------------

    @property
    def checkpoint_path(self) -> str:
        return self._ckpt_edit.text().strip()

    def set_status(self, text: str):
        self._status_label.setText(text)

    def set_predict_enabled(self, enabled: bool):
        self._predict_btn.setEnabled(enabled)

    def set_progress_visible(self, visible: bool):
        self._prog_label.setVisible(visible)
        self._prog_bar.setVisible(visible)
        if not visible:
            self._prog_label.setText("")
            self._prog_bar.setValue(0)

    def set_progress(self, step: str, pct: int):
        self._prog_label.setText(step)
        self._prog_bar.setValue(pct)

    def set_hover_info(self, label_id: int, volume: float, surface_area: float):
        if label_id <= 0:
            self._hover_box.setText("Hover over a bouton to see its details.")
        else:
            self._hover_box.setText(
                f"ID           :  {label_id}\n"
                f"Volume       :  {volume:.3f} µm³\n"
                f"Surface area :  {surface_area:.3f} µm²"
            )

    def update_stats(self, stats_list: List[BoutonStats]):
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)
        for s in stats_list:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, _NumericItem(str(s.label_id)))
            self._table.setItem(row, 1, _NumericItem(f"{s.volume_um3:.3f}"))
            self._table.setItem(row, 2, _NumericItem(f"{s.surface_area_um2:.3f}"))
        self._count_label.setText(f"Total boutons: {len(stats_list)}")
        self._table.setSortingEnabled(True)

    def clear_stats(self):
        self._table.setRowCount(0)
        self._count_label.setText("Total boutons: —")
        self._hover_box.setText("Hover over a bouton to see its details.")
        self._selected_label = None
        self._delete_btn.setEnabled(False)

    def highlight_row(self, label_id: int):
        self._table.itemSelectionChanged.disconnect(self._on_row_selected)
        try:
            for row in range(self._table.rowCount()):
                item = self._table.item(row, 0)
                if item and int(item.text()) == label_id:
                    self._table.selectRow(row)
                    self._table.scrollToItem(item)
                    self._selected_label = label_id
                    self._delete_btn.setEnabled(True)
                    break
        finally:
            self._table.itemSelectionChanged.connect(self._on_row_selected)

    def remove_row(self, label_id: int):
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item and int(item.text()) == label_id:
                self._table.removeRow(row)
                break
        self._count_label.setText(f"Total boutons: {self._table.rowCount()}")
        if self._selected_label == label_id:
            self._selected_label = None
            self._delete_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title = QLabel("Bouton Analysis")
        title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        layout.addWidget(title)

        layout.addWidget(_separator())

        # ---- Load section ----
        load_group = QGroupBox("Image")
        load_layout = QVBoxLayout(load_group)
        load_layout.setSpacing(4)

        self._path_label = QLabel("No image loaded.")
        self._path_label.setWordWrap(True)
        self._path_label.setStyleSheet("color: #888888; font-size: 8pt;")
        load_layout.addWidget(self._path_label)

        self._type_combo = QComboBox()
        self._type_combo.addItems([
            "LSM (rolling ball + deconvolution)",
            "Airyscan (downscale to 1100 px if needed)",
            "Airyscan, already 1100 px (normalisation only)",
        ])
        load_layout.addWidget(self._type_combo)

        load_btn = QPushButton("Load TIFF…")
        load_btn.clicked.connect(self._on_load_clicked)
        load_layout.addWidget(load_btn)

        layout.addWidget(load_group)

        # ---- Checkpoint section ----
        ckpt_group = QGroupBox("MicroSAM Checkpoint")
        ckpt_layout = QVBoxLayout(ckpt_group)
        ckpt_layout.setSpacing(4)

        ckpt_row = QHBoxLayout()
        self._ckpt_edit = QLineEdit()
        self._ckpt_edit.setPlaceholderText("Path to best.pt…")
        ckpt_row.addWidget(self._ckpt_edit)
        browse_btn = QPushButton("…")
        browse_btn.setFixedWidth(28)
        browse_btn.clicked.connect(self._on_browse_checkpoint)
        ckpt_row.addWidget(browse_btn)
        ckpt_layout.addLayout(ckpt_row)

        layout.addWidget(ckpt_group)

        # ---- Predict section ----
        self._predict_btn = QPushButton("▶  Run Prediction")
        self._predict_btn.setEnabled(False)
        self._predict_btn.setFixedHeight(34)
        self._predict_btn.clicked.connect(self.predict_requested.emit)
        layout.addWidget(self._predict_btn)

        self._prog_label = QLabel("")
        self._prog_label.setVisible(False)
        self._prog_label.setWordWrap(True)
        self._prog_label.setStyleSheet("font-size: 8pt; color: #aaaaaa;")
        layout.addWidget(self._prog_label)

        self._prog_bar = QProgressBar()
        self._prog_bar.setRange(0, 100)
        self._prog_bar.setFixedHeight(14)
        self._prog_bar.setVisible(False)
        layout.addWidget(self._prog_bar)

        # ---- Status ----
        self._status_label = QLabel("Load a TIFF image to begin.")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("font-size: 8pt; color: #888888;")
        layout.addWidget(self._status_label)

        layout.addWidget(_separator())

        # ---- Stats section ----
        self._count_label = QLabel("Total boutons: —")
        self._count_label.setFont(QFont("Segoe UI", 9))
        layout.addWidget(self._count_label)

        self._hover_box = QLabel("Hover over a bouton to see its details.")
        self._hover_box.setWordWrap(True)
        self._hover_box.setFixedHeight(76)
        self._hover_box.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._hover_box.setStyleSheet(
            "background-color: #2b2b2b;"
            "color: #d4d4d4;"
            "padding: 7px;"
            "border: 1px solid #3a3a3a;"
            "border-radius: 4px;"
            "font-size: 9pt;"
            "font-family: 'Consolas', monospace;"
        )
        layout.addWidget(self._hover_box)

        table_hint = QLabel("All detected boutons")
        table_hint.setStyleSheet("color: #888888; font-size: 8pt;")
        layout.addWidget(table_hint)

        self._table = QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["ID", "Vol (µm³)", "SA (µm²)"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        self._table.setSortingEnabled(True)
        self._table.itemSelectionChanged.connect(self._on_row_selected)
        layout.addWidget(self._table, stretch=1)

        self._delete_btn = QPushButton("Delete Selected Bouton")
        self._delete_btn.setEnabled(False)
        self._delete_btn.setFixedHeight(32)
        self._delete_btn.clicked.connect(self._on_delete_clicked)
        layout.addWidget(self._delete_btn)

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _on_load_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open TIFF Image", "", "TIFF files (*.tif *.tiff)"
        )
        if not path:
            return
        image_type = self._selected_image_type()
        from pathlib import Path
        self._path_label.setText(Path(path).name)
        self.load_requested.emit(path, image_type)

    def _on_browse_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select MicroSAM Checkpoint", "", "PyTorch checkpoint (*.pt)"
        )
        if path:
            self._ckpt_edit.setText(path)
            self._predict_btn.setEnabled(True)

    def _selected_image_type(self) -> str:
        idx = self._type_combo.currentIndex()
        return ["LSM", "Airyscan", "Airyscan_1100"][idx]

    @property
    def image_type(self) -> str:
        return self._selected_image_type()

    def _on_row_selected(self):
        rows = self._table.selectedItems()
        if not rows:
            self._selected_label = None
            self._delete_btn.setEnabled(False)
            return
        item = self._table.item(self._table.currentRow(), 0)
        if item is None:
            return
        self._selected_label = int(item.text())
        self._delete_btn.setEnabled(True)

    def _on_delete_clicked(self):
        if self._selected_label is not None:
            self.delete_requested.emit(self._selected_label)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _separator() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setFrameShadow(QFrame.Sunken)
    return sep


class _NumericItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically."""
    def __lt__(self, other: QTableWidgetItem) -> bool:
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return super().__lt__(other)
