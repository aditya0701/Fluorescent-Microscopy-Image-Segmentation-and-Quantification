"""
bouton_panel.py
Right-hand sidebar panel for BoutonViewer.  Provides:
  - A real-time hover information box showing the label ID, volume,
    and surface area of the bouton currently under the cursor.
  - A scrollable table listing every detected bouton with its physical
    measurements (volume in µm³, surface area in µm²).
  - A total bouton count label that updates whenever the table changes.
  - A 'Delete Selected Bouton' button that emits a signal to the main
    window so the deletion can be coordinated across the store, the
    viewer, and this panel.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QPushButton, QHeaderView, QFrame, QSizePolicy, QAbstractItemView,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from app.data.bouton_store import BoutonStats


class BoutonPanel(QWidget):
    """
    Sidebar widget displaying per-bouton statistics and deletion controls.

    Signals
    -------
    delete_requested(int)
        Emitted when the user clicks 'Delete Selected Bouton', carrying
        the label_id of the selected row.
    row_selected(int)
        Emitted when the user clicks a row in the table, carrying the
        corresponding label_id so the main window can synchronise the
        3D viewer highlight.
    """

    delete_requested = pyqtSignal(int)
    row_selected     = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._selected_label: Optional[int] = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Panel title
        title = QLabel("Bouton Analysis")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout.addWidget(title)

        # Horizontal separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        # Total count label
        self._count_label = QLabel("Total boutons: —")
        self._count_label.setFont(QFont("Segoe UI", 9))
        layout.addWidget(self._count_label)

        # Hover information box
        self._hover_box = QLabel("Hover over a bouton to see its details.")
        self._hover_box.setWordWrap(True)
        self._hover_box.setFixedHeight(76)
        self._hover_box.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
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

        # Bouton statistics table
        table_label = QLabel("All detected boutons")
        table_label.setFont(QFont("Segoe UI", 8))
        table_label.setStyleSheet("color: #888888;")
        layout.addWidget(table_label)

        self._table = QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["ID", "Vol (µm³)", "SA (µm²)"])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        self._table.setSortingEnabled(True)
        self._table.itemSelectionChanged.connect(self._on_row_selected)
        layout.addWidget(self._table, stretch=1)

        # Delete button
        self._delete_btn = QPushButton("Delete Selected Bouton")
        self._delete_btn.setEnabled(False)
        self._delete_btn.setFixedHeight(32)
        self._delete_btn.clicked.connect(self._on_delete_clicked)
        self._delete_btn.setStyleSheet(
            "QPushButton {"
            "  background-color: #7b1e1e;"
            "  color: #ffffff;"
            "  border-radius: 4px;"
            "  font-weight: bold;"
            "  padding: 4px 8px;"
            "}"
            "QPushButton:hover { background-color: #a32020; }"
            "QPushButton:pressed { background-color: #5c1616; }"
            "QPushButton:disabled { background-color: #3a3a3a; color: #666666; }"
        )
        layout.addWidget(self._delete_btn)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update_stats(self, stats_list: List[BoutonStats]):
        """
        Populates the table with the given list of BoutonStats objects.
        Sorting is temporarily disabled during the bulk insert to avoid
        the overhead of re-sorting on every row addition.
        """
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)

        for stats in stats_list:
            row = self._table.rowCount()
            self._table.insertRow(row)

            id_item  = QTableWidgetItem(str(stats.label_id))
            vol_item = _NumericItem(f"{stats.volume_um3:.3f}")
            sa_item  = _NumericItem(f"{stats.surface_area_um2:.3f}")

            for item in (id_item, vol_item, sa_item):
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )

            self._table.setItem(row, 0, id_item)
            self._table.setItem(row, 1, vol_item)
            self._table.setItem(row, 2, sa_item)

        self._table.setSortingEnabled(True)
        self._count_label.setText(f"Total boutons: {len(stats_list)}")

    def update_total_count(self, count: int):
        self._count_label.setText(f"Total boutons: {count}")

    def set_hover_info(self, label_id: int, volume: float, surface_area: float):
        """
        Updates the hover information box with measurements for the bouton
        currently under the cursor.  Pass label_id = -1 to clear the box.
        """
        if label_id <= 0:
            self._hover_box.setText("Hover over a bouton to see its details.")
        else:
            self._hover_box.setText(
                f"ID           :  {label_id}\n"
                f"Volume       :  {volume:.3f} µm³\n"
                f"Surface area :  {surface_area:.3f} µm²"
            )

    def highlight_row(self, label_id: int):
        """
        Selects and scrolls to the table row corresponding to label_id.
        Called from the main window when a bouton is clicked in the 3D view.
        Temporarily disconnects the selection signal to avoid a feedback loop.
        """
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
        """Removes the table row for the given label_id."""
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item and int(item.text()) == label_id:
                self._table.removeRow(row)
                break
        self._count_label.setText(
            f"Total boutons: {self._table.rowCount()}"
        )
        if self._selected_label == label_id:
            self._selected_label = None
            self._delete_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _on_row_selected(self):
        rows = self._table.selectedItems()
        if not rows:
            self._selected_label = None
            self._delete_btn.setEnabled(False)
            return

        current_row  = self._table.currentRow()
        id_item      = self._table.item(current_row, 0)
        if id_item is None:
            return

        self._selected_label = int(id_item.text())
        self._delete_btn.setEnabled(True)
        self.row_selected.emit(self._selected_label)

    def _on_delete_clicked(self):
        if self._selected_label is not None:
            self.delete_requested.emit(self._selected_label)


# ------------------------------------------------------------------
# Helper: QTableWidgetItem with numeric sorting
# ------------------------------------------------------------------

class _NumericItem(QTableWidgetItem):
    """
    QTableWidgetItem subclass that sorts by numeric value rather than
    alphabetically so that the volume and surface area columns sort correctly.
    """
    def __lt__(self, other: QTableWidgetItem) -> bool:
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return super().__lt__(other)
