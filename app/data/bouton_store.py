"""
bouton_store.py
Central in-memory state for one BoutonViewer session.  Holds the raw image,
the preprocessed image (cached to avoid repeating RL deconvolution), the
3D label array produced by prediction, and per-bouton statistics derived
from that label array.

All physical unit calculations use the voxel_size_um tuple (z, y, x) in
micrometres.  The value is selected automatically (see VOXEL_SIZE_BY_TYPE)
from the acquisition type chosen at load time: Confocal/LSM and Airyscan
share the same 0.3 µm Z-step but differ in XY pixel pitch — Airyscan's
computational super-resolution achieves finer lateral sampling.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from app.model.preprocessing import TARGET_XY


# 20 visually distinct colours as (R, G, B) tuples in [0, 255].
# The same list is used for PyVista hex colours in the 3D view and for
# QColor construction in the 2D slice view, ensuring visual consistency.
LABEL_COLORS_RGB: List[tuple] = [
    (230,  25,  75),   # red
    ( 60, 180,  75),   # green
    (255, 225,  25),   # yellow
    ( 67,  99, 216),   # blue
    (245, 130,  49),   # orange
    (145,  30, 180),   # purple
    ( 66, 212, 244),   # cyan
    (240,  50, 230),   # magenta
    (191, 239,  69),   # lime
    (250, 190, 212),   # pink
    ( 70, 153, 144),   # teal
    (220, 190, 255),   # lavender
    (154,  99,  36),   # brown
    (  0, 117, 220),   # cobalt
    (128,   0,   0),   # maroon
    (170, 255, 195),   # mint
    (128, 128,   0),   # olive
    (255, 216, 177),   # apricot
    (  0,   0, 117),   # navy
    (169, 169, 169),   # grey
]


@dataclass
class BoutonStats:
    """Physical measurements for a single segmented bouton instance."""
    label_id:        int
    voxel_count:     int
    volume_um3:      float
    surface_area_um2: float


class BoutonStore:
    """
    Holds all session state: images, label array, per-bouton statistics,
    colour assignments, and voxel calibration.

    The preprocessed_image field caches the result of the (slow) RL
    deconvolution pipeline so that re-running prediction on the same file
    does not repeat deconvolution.  The cache is invalidated whenever a new
    image is loaded.
    """

    # Physical voxel size (z_um, y_um, x_um) per acquisition type. All
    # share the same 0.3 µm Z-step. Native-resolution Airyscan's computational
    # super-resolution achieves finer lateral sampling (0.0425 µm) than
    # standard confocal/LSM scanning (0.0709 µm) — but downscaling Airyscan
    # to TARGET_XY px brings its field of view back in line with confocal's
    # (1834px * 0.0425µm ≈ 1100px * 0.0709µm ≈ 78µm), so an Airyscan image
    # already at or below TARGET_XY effectively has the LSM pitch. See
    # detect_airyscan_voxel_size(), which picks between these two values
    # from the image's actual pixel dimensions.
    VOXEL_SIZE_BY_TYPE = {
        "LSM":      (0.3, 0.0709, 0.0709),
        "Airyscan": (0.3, 0.0425, 0.0425),
    }

    @classmethod
    def detect_airyscan_voxel_size(cls, y: int, x: int) -> tuple:
        """
        Picks the Airyscan voxel pitch from the image's actual Y/X pixel
        dimensions rather than a separate "already downscaled" flag, using
        the same TARGET_XY threshold preprocess_airyscan uses to decide
        whether to downscale: e.g. 1834x1834 (native) gets the finer
        0.0425 µm pitch, 1100x1100 or 1101x1101 (already downscaled) gets
        the LSM-matching 0.0709 µm pitch.
        """
        if max(y, x) > TARGET_XY:
            return cls.VOXEL_SIZE_BY_TYPE["Airyscan"]
        return cls.VOXEL_SIZE_BY_TYPE["LSM"]

    @classmethod
    def detect_voxel_size(cls, image_type: str, y: int, x: int) -> tuple:
        """
        Single entry point for deriving a voxel size from (image_type, Y, X) —
        used at load time and whenever the image type selector changes
        afterward, so both stay in sync without duplicating this branch.
        """
        if image_type == "LSM":
            return cls.VOXEL_SIZE_BY_TYPE["LSM"]
        return cls.detect_airyscan_voxel_size(y, x)

    def __init__(self):
        self.image:              Optional[np.ndarray] = None  # (Z, C, Y, X) float32 raw
        self.preprocessed_image: Optional[np.ndarray] = None  # (Z, C, Y, X) float32
        # (image_type, model_type) the cached preprocessed_image was computed
        # for — LSM preprocessing depends on model_type (see PredictionWorker),
        # so a cache hit requires both to still match the current selection.
        self.preprocessed_cache_key: Optional[tuple] = None
        self.labels:             Optional[np.ndarray] = None  # (Z, Y, X) uint32
        self.stats:              Dict[int, BoutonStats] = {}
        self.label_colors:       Dict[int, tuple] = {}        # label_id -> (R,G,B)
        self.image_type:         str = "LSM"
        self.image_path:         str = ""
        self._color_counter:     int = 0
        self._voxel_size_override: Optional[tuple] = None

    @property
    def voxel_size_um(self) -> tuple:
        """
        Physical voxel size (z, y, x) in µm.  Returns the manually-entered
        value if one was set (see the setter, used by the load-image dialog
        to apply a user override), otherwise the per-acquisition-type
        default.
        """
        if self._voxel_size_override is not None:
            return self._voxel_size_override
        return self.VOXEL_SIZE_BY_TYPE.get(
            self.image_type, self.VOXEL_SIZE_BY_TYPE["LSM"]
        )

    @voxel_size_um.setter
    def voxel_size_um(self, value: Optional[tuple]):
        self._voxel_size_override = tuple(value) if value is not None else None

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def clear(self):
        """Resets all session state except voxel size and image type."""
        self.image                  = None
        self.preprocessed_image     = None
        self.preprocessed_cache_key = None
        self.labels             = None
        self.stats              = {}
        self.label_colors       = {}
        self._color_counter     = 0

    def set_labels(self, labels: np.ndarray):
        """
        Stores the prediction result, assigns colours, and computes
        per-bouton volume and surface area statistics.
        """
        self.labels = labels.astype(np.uint32)
        self._assign_colors()
        self._compute_stats()

    def recompute_stats(self):
        """
        Re-runs volume/surface-area statistics against the current label
        array and voxel size.  Used when the user edits the voxel size
        after a prediction has already run.
        """
        if self.labels is not None:
            self._compute_stats()

    # ------------------------------------------------------------------
    # Per-bouton operations
    # ------------------------------------------------------------------

    def delete_bouton(self, label_id: int):
        """
        Removes a bouton from all data structures.  The label array is
        updated in place so that the 2D slice view reflects the deletion
        without requiring a full re-render.
        """
        self.stats.pop(label_id, None)
        self.label_colors.pop(label_id, None)
        if self.labels is not None:
            self.labels[self.labels == label_id] = 0

    @property
    def total_count(self) -> int:
        return len(self.stats)

    def get_stats_list(self) -> List[BoutonStats]:
        return sorted(self.stats.values(), key=lambda s: s.label_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assign_colors(self):
        """Assigns a colour from the palette to every unique label."""
        self.label_colors = {}
        unique = np.unique(self.labels)
        unique = unique[unique > 0]
        for i, lbl in enumerate(unique):
            self.label_colors[int(lbl)] = LABEL_COLORS_RGB[i % len(LABEL_COLORS_RGB)]

    def _compute_stats(self):
        """
        Computes voxel count, physical volume, and surface area for every
        label in the current label array.  Surface area is estimated via
        marching cubes on a binary mask for each label, using the physical
        voxel spacing as the mesh spacing so the result is in µm².
        """
        from skimage.measure import marching_cubes, mesh_surface_area

        self.stats = {}
        vz, vy, vx = self.voxel_size_um
        voxel_volume = float(vz * vy * vx)

        unique = np.unique(self.labels)
        unique = unique[unique > 0]

        for lbl in unique:
            mask = self.labels == lbl
            voxel_count = int(mask.sum())
            volume_um3  = voxel_count * voxel_volume

            try:
                verts, faces, _, _ = marching_cubes(
                    mask.astype(np.float32),
                    level=0.5,
                    spacing=self.voxel_size_um,
                )
                surface_area_um2 = float(mesh_surface_area(verts, faces))
            except Exception:
                surface_area_um2 = 0.0

            self.stats[int(lbl)] = BoutonStats(
                label_id=int(lbl),
                voxel_count=voxel_count,
                volume_um3=volume_um3,
                surface_area_um2=surface_area_um2,
            )
