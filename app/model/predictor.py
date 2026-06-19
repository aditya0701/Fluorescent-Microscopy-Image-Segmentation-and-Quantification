"""
predictor.py
Single-file MicroSAM inference wrapper for BoutonViewer.  Adapted from the
batch inference script (vit_l_lm_neo_augmented runs) for interactive use.

The model is cached after the first load so that subsequent predictions on
the same checkpoint do not reload weights from disk.  Calling load_model()
with a different checkpoint path clears the cache automatically.

Post-processing mirrors the batch script:
  • 3-D connected components with full 26-connectivity structure.
  • Small object removal (min_size = 862 voxels).
  • Z-span filter: objects present in three or fewer Z slices are removed.
  • Final relabelling so label IDs are contiguous starting from 1.
"""

from __future__ import annotations
from typing import Callable, Optional
import sys
import numpy as np
import torch


# best.pt was saved by torch_em with its full training-time state, which
# includes a custom intensity-augmentation class that only ever existed in
# the original training script's __main__ module. MicroSAM's checkpoint
# loader replaces classes it cannot resolve with None, but pickle's NEWOBJ
# opcode then raises "NEWOBJ class argument must be a type, not NoneType"
# because it still needs a real type to instantiate (and discard) that
# fragment of state. Registering a stub under the same name lets unpickling
# finish; the stub instance itself is never used.
class ChannelSafeIntensityAug:
    def __setstate__(self, state):
        self.__dict__.update(state)


sys.modules["__main__"].ChannelSafeIntensityAug = ChannelSafeIntensityAug

TILE_SHAPE             = (896, 896)
HALO                   = (64, 64)
DEFAULT_MODEL_TYPE     = "vit_l_lm"
BLOB_REMOVAL_THRESHOLD = 862   # minimum voxel count to retain an object
MIN_Z_SPAN             = 3     # objects spanning this many Z slices or fewer
                                # are removed as acquisition artefacts

# Module-level cache so the weights are not reloaded on every prediction.
_cached_predictor  = None
_cached_segmenter  = None
_cached_checkpoint = None
_cached_model_type = None


def load_model(
    checkpoint_path: str,
    model_type:      str = DEFAULT_MODEL_TYPE,
    device:          Optional[str] = None,
) -> tuple:
    """
    Loads the MicroSAM predictor and segmenter from a .pt checkpoint.
    Returns the cached instances if the same checkpoint and model variant
    were previously loaded.

    Parameters
    ----------
    checkpoint_path : str
        Path to the best.pt checkpoint file.
    model_type : str
        MicroSAM model variant, e.g. 'vit_l_lm' (large) or 'vit_b_lm' (base).
    device : str, optional
        PyTorch device string.  Defaults to 'cuda' if a GPU is available,
        otherwise 'cpu'.

    Returns
    -------
    tuple
        (predictor, segmenter) ready for use with automatic_instance_segmentation.
    """
    global _cached_predictor, _cached_segmenter, _cached_checkpoint, _cached_model_type

    if _cached_checkpoint == checkpoint_path and _cached_model_type == model_type:
        return _cached_predictor, _cached_segmenter

    from micro_sam.automatic_segmentation import get_predictor_and_segmenter

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,
        checkpoint=checkpoint_path,
        device=device,
        is_tiled=True,
    )

    _cached_predictor  = predictor
    _cached_segmenter  = segmenter
    _cached_checkpoint = checkpoint_path
    _cached_model_type = model_type

    return predictor, segmenter


def run_inference(
    rgb_stack:       np.ndarray,
    checkpoint_path: str,
    model_type:      str = DEFAULT_MODEL_TYPE,
    device:          Optional[str] = None,
    progress:        Optional[Callable] = None,
) -> np.ndarray:
    """
    Runs slice-wise 2D MicroSAM inference on a (Z, Y, X, 3) float32 RGB
    stack and returns a post-processed (Z, Y, X) uint32 label array.

    The function follows the same inference logic as the batch script:
    each Z slice is segmented independently using tiled automatic instance
    segmentation, the 2D per-slice label maps are stacked into a 3D volume,
    and post-processing filters are applied to produce the final result.

    Parameters
    ----------
    rgb_stack : np.ndarray
        (Z, Y, X, 3) float32 array in [0, 255] as produced by to_microsam_rgb.
    checkpoint_path : str
        Path to the MicroSAM .pt checkpoint.
    model_type : str
        MicroSAM model variant, e.g. 'vit_l_lm' (large) or 'vit_b_lm' (base).
    device : str, optional
        PyTorch device.  Auto-detected if not provided.
    progress : callable, optional
        Callback with signature (step: str, pct: int) for UI progress updates.

    Returns
    -------
    np.ndarray
        (Z, Y, X) uint32 label array.  Background is 0; each bouton instance
        has a unique positive integer label.
    """
    import gc

    from micro_sam.automatic_segmentation import automatic_instance_segmentation
    from scipy.ndimage import label as ndimage_label, generate_binary_structure
    from skimage.morphology import remove_small_objects

    if progress:
        progress("Loading model weights", 0)

    predictor, segmenter = load_model(checkpoint_path, model_type, device)

    Z     = rgb_stack.shape[0]
    stack = []

    for z in range(Z):
        pred = automatic_instance_segmentation(
            predictor=predictor,
            segmenter=segmenter,
            input_path=rgb_stack[z],
            ndim=2,
            tile_shape=TILE_SHAPE,
            halo=HALO,
        )
        stack.append(pred)

        # Tiled inference allocates fresh embedding/decoder tensors on every
        # slice. On small-VRAM GPUs (e.g. 4 GB) these accumulate over a long
        # Z-stack until the driver faults with an unrecoverable native crash
        # rather than a catchable CUDA OOM. Release the cache each iteration
        # to keep memory usage flat across the whole stack.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if progress:
            pct = int(100 * (z + 1) / Z)
            progress(f"Segmenting slice {z + 1}/{Z}", pct)

    pred3d = np.array(stack, dtype=np.uint32)

    if progress:
        progress("3D connected components", 98)

    structure  = generate_binary_structure(3, 3)
    labeled, _ = ndimage_label(pred3d > 0, structure=structure)

    # Remove objects below the minimum voxel count threshold.
    clean_mask = remove_small_objects(
        labeled.astype(bool),
        min_size=BLOB_REMOVAL_THRESHOLD,
        connectivity=1,
    )
    labeled, _ = ndimage_label(clean_mask, structure=structure)

    # Remove objects spanning MIN_Z_SPAN or fewer Z slices.
    # for lbl in np.unique(labeled)[1:]:
    #     z_presence = np.any(labeled == lbl, axis=(1, 2))
    #     if z_presence.sum() <= MIN_Z_SPAN:
    #         labeled[labeled == lbl] = 0

    if progress:
        progress("Final relabelling", 99)

    labeled, n_final = ndimage_label(labeled > 0, structure=structure)

    if progress:
        progress(f"Done — {n_final} boutons detected", 100)

    return labeled.astype(np.uint32)
