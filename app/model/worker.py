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
from typing import Optional
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
    preprocessed_ready(preprocessed: np.ndarray)
        Emitted once preprocessing has actually run (i.e. cached_preprocessed
        was not supplied), so the caller can cache the result for the next
        prediction on the same image.  Not emitted when the cache was used.
    finished(labels: np.ndarray)
        Emitted on success with the final (Z, Y, X) uint32 label array.
    error(message: str)
        Emitted on any exception with the human-readable error message.
    """

    progress           = pyqtSignal(str, int)   # (step_name, percent)
    preprocessed_ready = pyqtSignal(object)     # np.ndarray, passed as object
    finished            = pyqtSignal(object)    # np.ndarray passed as object to avoid
                                                 # issues with PyQt6 type registration
    error               = pyqtSignal(str)

    def __init__(
        self,
        image:               np.ndarray,
        image_type:          str,
        checkpoint_path:     str,
        model_type:          str = "vit_l_lm",
        device:              str | None = None,
        cached_preprocessed: Optional[np.ndarray] = None,
        parent=None,
    ):
        """
        Parameters
        ----------
        image : np.ndarray
            Raw (Z, C, Y, X) float32 array from the image loader.
        image_type : str
            'LSM' or 'Airyscan'.  'LSM' runs the full background-subtraction
            + Richardson-Lucy deconvolution pipeline; 'Airyscan' runs the
            lighter normalisation-only path (preprocess_airyscan), which
            downscales to 1100 px only if the image is larger — a no-op for
            Airyscan images already at or below that size.
        checkpoint_path : str
            Absolute path to the MicroSAM .pt checkpoint file.
        model_type : str
            MicroSAM model variant to run inference with, e.g. 'vit_l_lm'
            (large) or 'vit_b_lm' (base).
        device : str, optional
            PyTorch device string.  Auto-detected from CUDA availability
            if not provided.
        cached_preprocessed : np.ndarray, optional
            Result of a previous preprocessing run on this same image and
            image_type.  When supplied, preprocessing is skipped entirely —
            the caller (BoutonStore.preprocessed_image) is responsible for
            invalidating this whenever a new image is loaded or the image
            type changes.
        """
        super().__init__(parent)
        self.image               = image
        self.image_type          = image_type
        self.checkpoint_path     = checkpoint_path
        self.model_type          = model_type
        self.device              = device
        self.cached_preprocessed = cached_preprocessed

    def run(self):
        try:
            self._run_pipeline()
        except Exception as exc:
            # The UI only ever shows str(exc) in a one-line message box —
            # print the full traceback to the console too, otherwise the
            # actual failure point (e.g. a CUDA OOM deep in micro_sam) is
            # unrecoverable after the fact.
            import traceback
            traceback.print_exc()
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
        # Step 1: preprocessing (skipped if a cached result was supplied)
        # ----------------------------------------------------------
        # The 'vit_b_lm' (Base) model is run without the LSM deconvolution
        # pipeline — it uses the same lighter normalisation-only path as
        # Airyscan (downscale-if-needed -> percentile normalisation to
        # [0, 1]), skipping rolling-ball background subtraction and
        # Richardson-Lucy deconvolution entirely.
        use_lsm_deconv = self.image_type == "LSM" and self.model_type != "vit_b_lm"

        if self.cached_preprocessed is not None:
            self.progress.emit("Using cached preprocessed image…", 0)
            preprocessed = self.cached_preprocessed
        else:
            self.progress.emit("Preprocessing image…", 0)

            if use_lsm_deconv:
                preprocessed = preprocess_lsm(self.image, progress=_progress)
            else:
                preprocessed = preprocess_airyscan(self.image, progress=_progress)

            self.preprocessed_ready.emit(preprocessed)

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
            model_type=self.model_type,
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
