"""
slice_viewer.py
2D slice-by-slice viewer for BoutonViewer.  Renders one Z plane at a time
inside a QGraphicsView/QGraphicsScene, which provides mouse-wheel zoom and
middle-mouse-button drag panning for free.  Label masks are displayed as
semi-transparent colour overlays matching the palette used in the 3D view.

Hover/click picking is a direct label-array lookup at the scene coordinates
under the cursor.  Scene coordinates map 1:1 onto image pixel coordinates
(the pixmap sits at the scene origin with no extra scaling — all zoom and
pan live in the view's transform), so picking stays instantaneous and
correct at any zoom level, with no manual coordinate bookkeeping needed.
"""

from __future__ import annotations
from typing import Dict, Optional, Set
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPainter


class _ZoomPanView(QGraphicsView):
    """
    QGraphicsView with mouse-wheel zoom (anchored under the cursor, so the
    point you're pointing at stays put) and middle-mouse-button drag
    panning.  Left-clicks and cursor moves are forwarded to the parent
    SliceViewer as scene-space positions for bouton picking; view_adjusted
    fires whenever the user actively zooms or pans, so the parent can stop
    auto-fitting the image to the viewport once the user has taken control.
    """

    hovered       = pyqtSignal(QPointF)
    clicked       = pyqtSignal(QPointF)
    view_adjusted = pyqtSignal()

    _ZOOM_IN_FACTOR  = 1.25
    _ZOOM_OUT_FACTOR = 1.0 / 1.25

    def __init__(self, scene: QGraphicsScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("background-color: #000000; border: none;")
        self.setMouseTracking(True)

        self._panning:   bool = False
        self._pan_start: QPoint = QPoint()

    def wheelEvent(self, event):
        factor = self._ZOOM_IN_FACTOR if event.angleDelta().y() > 0 else self._ZOOM_OUT_FACTOR
        self.scale(factor, factor)
        self.view_adjusted.emit()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning   = True
            self._pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.mapToScene(event.pos()))

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta           = event.pos() - self._pan_start
            self._pan_start = event.pos()
            h_bar = self.horizontalScrollBar()
            v_bar = self.verticalScrollBar()
            h_bar.setValue(h_bar.value() - delta.x())
            v_bar.setValue(v_bar.value() - delta.y())
            self.view_adjusted.emit()
            event.accept()
            return

        self.hovered.emit(self.mapToScene(event.pos()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return

        super().mouseReleaseEvent(event)


class SliceViewer(QWidget):
    """
    QWidget that displays one Z slice of a (Z, C, Y, X) image together with
    semi-transparent coloured overlays for each bouton label, with
    mouse-wheel zoom and middle-button drag panning.

    Signals
    -------
    bouton_hovered(int)
        Emitted when the cursor moves over or off a bouton.
        Value is the label_id, or -1 when the cursor is over background.
    bouton_selected(int)
        Emitted on left-click with the label_id under the cursor,
        or -1 when the click hits background.
    """

    bouton_hovered  = pyqtSignal(int)
    bouton_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._image:        Optional[np.ndarray] = None   # (Z, C, Y, X) float32
        self._labels:       Optional[np.ndarray] = None   # (Z, Y, X) uint32
        self._label_colors: Dict[int, tuple]      = {}    # label_id -> (R,G,B)
        self._hidden:       Set[int]               = set()
        self._current_z:    int                    = 0
        self._last_hovered: int                    = -1
        self._img_h:        int                    = 0
        self._img_w:        int                    = 0

        # Tracks whether the displayed slice should keep auto-fitting to
        # the viewport (new image / before the user has touched zoom or
        # pan) or whether the user has taken control of the view.
        self._user_adjusted_view: bool = False

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._scene       = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        self._view = _ZoomPanView(self._scene)
        self._view.hovered.connect(self._on_hover)
        self._view.clicked.connect(self._on_click)
        self._view.view_adjusted.connect(self._on_view_adjusted)

        layout.addWidget(self._view)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_image(self, image: np.ndarray):
        """Stores the image array and triggers a display refresh."""
        new_size = (image.shape[2], image.shape[3]) != (self._img_h, self._img_w)

        self._image = image
        self._img_h = image.shape[2]
        self._img_w = image.shape[3]

        if new_size:
            # Stale labels from a different-sized image would cause a shape
            # mismatch in _update_display, so clear them on any size change.
            self._labels       = None
            self._label_colors = {}
            self._hidden.clear()

        self._update_display()

        if new_size:
            # A genuinely new image shape gets a fresh fitted view.
            self._user_adjusted_view = False
            self._fit_to_view()

    def set_labels(
        self,
        labels:       np.ndarray,
        label_colors: Dict[int, tuple],
    ):
        """Stores the label array and colour map, then refreshes."""
        self._labels       = labels
        self._label_colors = label_colors
        self._hidden.clear()
        self._update_display()

    def clear_labels(self):
        """Removes all label overlays."""
        self._labels       = None
        self._label_colors = {}
        self._hidden.clear()
        self._update_display()

    def set_z(self, z: int):
        """Switches to the given Z slice and refreshes the display."""
        if self._image is not None:
            self._current_z = int(np.clip(z, 0, self._image.shape[0] - 1))
            self._update_display()

    def hide_label(self, label_id: int):
        self._hidden.add(label_id)
        self._update_display()

    def show_label(self, label_id: int):
        self._hidden.discard(label_id)
        self._update_display()

    def remove_label(self, label_id: int):
        """Removes a single bouton from the overlay without clearing others."""
        self._label_colors.pop(label_id, None)
        self._hidden.discard(label_id)
        self._update_display()

    def fit_to_view(self):
        """Resets zoom/pan so the whole slice fills the viewport again."""
        self._user_adjusted_view = False
        self._fit_to_view()

    # ------------------------------------------------------------------
    # Display rendering
    # ------------------------------------------------------------------

    def _fit_to_view(self):
        if self._pixmap_item.pixmap().isNull():
            return
        self._view.resetTransform()
        self._view.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def _on_view_adjusted(self):
        self._user_adjusted_view = True

    def _update_display(self):
        """
        Builds a QPixmap from the current Z slice and label overlay and
        pushes it into the scene.  Zoom/pan are entirely the QGraphicsView's
        responsibility — this method never rescales anything.
        """
        if self._image is None:
            self._pixmap_item.setPixmap(QPixmap())
            return

        z = self._current_z

        def _stretch(channel: np.ndarray) -> np.ndarray:
            """
            Percentile-based contrast stretch using the same 0.0001/99.9999
            bounds as the preprocessing pipeline (see _normalise_percentile
            and to_microsam_rgb in preprocessing.py), so the displayed
            contrast matches what the model actually sees. Plain min/max
            scaling lets a handful of bright outlier pixels compress the
            real signal into a narrow, washed-out grey band.
            """
            lo, hi = np.percentile(channel, (0.0001, 99.9999))
            if hi > lo:
                stretched = np.clip((channel - lo) / (hi - lo), 0.0, 1.0)
                return (stretched * 255).astype(np.uint8)
            return np.zeros_like(channel, dtype=np.uint8)

        # Two-colour composite matching the standard fluorescence convention:
        # channel 0 -> green, channel 1 -> red.
        H, W = self._image.shape[2], self._image.shape[3]
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[:, :, 1] = _stretch(self._image[z, 0, :, :])
        if self._image.shape[1] > 1:
            rgba[:, :, 0] = _stretch(self._image[z, 1, :, :])
        rgba[:, :, 3] = 255

        # Paint label overlays by blending the label colour into the
        # underlying slice (rather than overwriting it), so the image stays
        # visible beneath every mask. The blended pixel is written back at
        # full alpha — relying on the widget's background for transparency
        # would hide the image data entirely instead of showing it through.
        OVERLAY_OPACITY = 0.5
        if self._labels is not None:
            label_slice = self._labels[z]
            for label_id, rgb in self._label_colors.items():
                if label_id in self._hidden:
                    continue
                mask = label_slice == label_id
                if not mask.any():
                    continue
                base    = rgba[mask, :3].astype(np.float32)
                color   = np.asarray(rgb, dtype=np.float32)
                blended = OVERLAY_OPACITY * color + (1 - OVERLAY_OPACITY) * base
                rgba[mask, :3] = blended.astype(np.uint8)

        qimage = QImage(
            rgba.tobytes(), W, H, W * 4, QImage.Format.Format_RGBA8888
        )
        self._pixmap_item.setPixmap(QPixmap.fromImage(qimage))
        self._scene.setSceneRect(0, 0, W, H)

        if not self._user_adjusted_view:
            self._fit_to_view()

    def resizeEvent(self, event):
        """Keeps the slice fitted to the viewport until the user zooms/pans."""
        super().resizeEvent(event)
        if not self._user_adjusted_view:
            self._fit_to_view()

    # ------------------------------------------------------------------
    # Picking — scene coordinates map 1:1 onto image pixel coordinates,
    # since the pixmap item sits at the scene origin at its native scale.
    # ------------------------------------------------------------------

    def _label_at(self, scene_pos: QPointF) -> int:
        """
        Returns the label_id at the given scene-space coordinates,
        or -1 if the position is outside the image area or on background.
        """
        if self._labels is None:
            return -1

        ix, iy = int(scene_pos.x()), int(scene_pos.y())
        H, W   = self._labels.shape[1], self._labels.shape[2]

        if not (0 <= iy < H and 0 <= ix < W):
            return -1

        return int(self._labels[self._current_z, iy, ix])

    # ------------------------------------------------------------------
    # View signal handlers
    # ------------------------------------------------------------------

    def _on_hover(self, scene_pos: QPointF):
        label_id = self._label_at(scene_pos)
        if label_id != self._last_hovered:
            self._last_hovered = label_id
            self.bouton_hovered.emit(label_id)

    def _on_click(self, scene_pos: QPointF):
        label_id = self._label_at(scene_pos)
        self.bouton_selected.emit(label_id)
