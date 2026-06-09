"""
worker.py
Background QThread worker that runs the full pipeline (preprocessing +
MicroSAM inference + post-processing) without blocking the main UI thread.

The worker emits granular progress signals so the main window can display
a meaningful progress bar and status message throughout the pipeline.
All exceptions are caught and forwarded as an error signal rather than
crashing the application.
"""

from __future__ import annotations
import numpy as np
from qtpy.QtCore import QThread, Signal as pyqtSignal


class PredictionWorker(QThread):
    """
    Runs image preprocessing and MicroSAM inference in a background thread.

    Signals
    -------
    progress(step_name: str, pct: int)
        Emitted at each meaningful step so the UI can update its progress bar.
        pct is in [0, 100].
    finished(labels: np.ndarray)
        Emitted on success with the final (Z, Y, X) uint32 label array.
    error(message: str)
        Emitted on any exception with the human-readable error message.
    """

    progress = pyqtSignal(str, int)       # (step_name, percent)
    finished = pyqtSignal(object)         # np.ndarray passed as object to avoid
                                          # issues with PyQt6 type registration
    error    = pyqtSignal(str)

    def __init__(
        self,
        image:           np.ndarray,
        image_type:      str,
        checkpoint_path: str,
        device:          str | None = None,
        parent=None,
    ):
        """
        Parameters
        ----------
        image : np.ndarray
            Raw (Z, C, Y, X) float32 array from the image loader.
        image_type : str
            'LSM', 'Airyscan', or 'Airyscan_1100'.  'LSM' runs the full
            background-subtraction + Richardson-Lucy deconvolution
            pipeline; both Airyscan variants run the lighter
            normalisation-only path (preprocess_airyscan), which
            downscales to 1100 px only if the image is larger — a no-op
            for images already at that size, such as 'Airyscan_1100'.
        checkpoint_path : str
            Absolute path to the MicroSAM .pt checkpoint file.
        device : str, optional
            PyTorch device string.  Auto-detected from CUDA availability
            if not provided.
        """
        super().__init__(parent)
        self.image           = image
        self.image_type      = image_type
        self.checkpoint_path = checkpoint_path
        self.device          = device

    def run(self):
        try:
            self._run_pipeline()
        except Exception as exc:
            self.error.emit(str(exc))

    def _run_pipeline(self):
        from app.model.preprocessing import (
            preprocess_lsm,
            preprocess_airyscan,
            to_microsam_rgb,
        )
        from app.model.predictor import run_inference

        def _progress(step: str, pct: int):
            self.progress.emit(step, pct)

        # ----------------------------------------------------------
        # Step 1: preprocessing
        # ----------------------------------------------------------
        self.progress.emit("Preprocessing image…", 0)

        if self.image_type == "LSM":
            preprocessed = preprocess_lsm(self.image, progress=_progress)
        else:
            preprocessed = preprocess_airyscan(self.image, progress=_progress)

        # ----------------------------------------------------------
        # Step 2: convert to MicroSAM RGB format
        # ----------------------------------------------------------
        self.progress.emit("Converting to RGB for MicroSAM…", 0)
        rgb_stack = to_microsam_rgb(preprocessed)
        del preprocessed  # free before inference to reduce peak RAM

        # ----------------------------------------------------------
        # Step 3: inference + post-processing
        # ----------------------------------------------------------
        labels = run_inference(
            rgb_stack=rgb_stack,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            progress=_progress,
        )

        # ----------------------------------------------------------
        # Step 4: upscale labels to original XY dimensions if the
        # Airyscan path downscaled the image to 1100×1100.
        # ----------------------------------------------------------
        orig_Y = self.image.shape[2]
        orig_X = self.image.shape[3]

        if labels.shape[1] != orig_Y or labels.shape[2] != orig_X:
            from skimage.transform import resize as sk_resize

            self.progress.emit("Upscaling labels to original resolution…", 0)
            labels = sk_resize(
                labels,
                (labels.shape[0], orig_Y, orig_X),
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            ).astype(np.uint32)

        self.finished.emit(labels)
