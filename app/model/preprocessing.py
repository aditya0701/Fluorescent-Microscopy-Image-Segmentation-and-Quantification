"""
preprocessing.py
Image preprocessing pipeline for BoutonViewer.  Two paths are supported,
selected based on the acquisition type reported by the user at load time.

LSM path
--------
Rolling ball background subtraction (radius 50) → robust percentile
normalisation → Richardson-Lucy deconvolution (15 iterations per channel,
PSF sigma 1.75 for red and 1.50 for green) → combine normalised and
deconvolved results → final per-channel min-max normalisation.
This matches the pipeline documented in image_deconvolve.ipynb.

Airyscan path
-------------
Downsample XY to 1100 px if necessary → robust percentile normalisation.
No deconvolution is applied because Airyscan data already incorporates
computational super-resolution deconvolution at acquisition time.

Both paths emit a progress_callback(step: str, pct: int) so the calling
QThread worker can relay progress to the main-thread progress bar.
"""

from __future__ import annotations
from typing import Callable, Optional
import numpy as np

TARGET_XY = 1100


# ------------------------------------------------------------------
# Shared normalisation utility
# ------------------------------------------------------------------

def normalize_robust(
    image: np.ndarray,
    p_low:  float = 0.0001,
    p_high: float = 99.9999,
) -> np.ndarray:
    """
    Normalises a (Z, C, Y, X) float32 image channel-wise to [0, 1]
    using percentile-based clipping.  Each channel is normalised
    independently across the full Z-stack.
    """
    from skimage.exposure import rescale_intensity

    img        = image.astype(np.float32)
    normalised = np.zeros_like(img)

    for c in range(img.shape[1]):
        channel = img[:, c, :, :]
        lo, hi  = np.percentile(channel, (p_low, p_high))
        normalised[:, c, :, :] = rescale_intensity(
            channel, in_range=(lo, hi), out_range=(0.0, 1.0)
        )
    return normalised


# ------------------------------------------------------------------
# LSM preprocessing steps
# ------------------------------------------------------------------

def _rolling_ball_subtraction(
    image:    np.ndarray,
    radius:   float = 50.0,
    progress: Optional[Callable] = None,
) -> np.ndarray:
    """
    Applies rolling ball background subtraction to each (z, channel) slice
    of a (Z, C, Y, X) float32 array.  Returns a float32 array of the
    same shape with background removed.
    """
    from skimage.restoration import rolling_ball

    result = np.zeros_like(image, dtype=np.float32)
    Z, C   = image.shape[:2]

    for z in range(Z):
        for c in range(C):
            bg              = rolling_ball(image[z, c], radius=radius)
            result[z, c]    = image[z, c] - bg
        if progress is not None:
            pct = int(100 * (z + 1) / Z)
            progress("Background subtraction", pct)

    return result


def _make_psf(sigma: float, kernel_size: int = 15) -> np.ndarray:
    """
    Builds a 2D Gaussian PSF kernel of the given sigma for use with
    Richardson-Lucy deconvolution.
    """
    from skimage.filters import gaussian

    k = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    k[kernel_size // 2, kernel_size // 2] = 1.0
    psf = gaussian(k, sigma=sigma)
    return (psf / psf.sum()).astype(np.float32)


def _richardson_lucy_deconvolution(
    image:    np.ndarray,
    n_iter:   int = 15,
    progress: Optional[Callable] = None,
) -> np.ndarray:
    """
    Applies Richardson-Lucy deconvolution slice-by-slice to a normalised
    (Z, C, Y, X) float32 array.  PSF sigma values match those used in
    image_deconvolve.ipynb: 1.75 for channel 0 (red, BRP-shortcherry)
    and 1.50 for channel 1 (green, KC claws).
    Input must be in [0, 1].  Returns float32 in [0, 1].
    """
    from skimage.restoration import richardson_lucy

    psfs   = [_make_psf(1.75), _make_psf(1.50)]
    deconv = np.zeros_like(image, dtype=np.float32)
    Z, C   = image.shape[:2]

    for z in range(Z):
        for c in range(min(C, 2)):
            deconv[z, c] = richardson_lucy(
                image[z, c].astype(np.float32),
                psfs[c],
                num_iter=n_iter,
                clip=True,
            )
        if progress is not None:
            pct = int(100 * (z + 1) / Z)
            progress("Richardson-Lucy deconvolution", pct)

    # Clip background artefacts introduced by the deconvolution.
    for c in range(deconv.shape[1]):
        ch     = deconv[:, c, :, :]
        thresh = np.percentile(ch, 2)
        deconv[:, c, :, :] = np.where(ch < thresh, 0.0, ch - thresh)

    return deconv


# ------------------------------------------------------------------
# Public preprocessing entry points
# ------------------------------------------------------------------

def preprocess_lsm(
    image:    np.ndarray,
    progress: Optional[Callable] = None,
) -> np.ndarray:
    if image.ndim != 4:
        raise ValueError(f"Expected 4-D (Z, C, Y, X) array, got shape {image.shape}")
    """
    Full LSM preprocessing pipeline matching the procedure in
    image_deconvolve.ipynb:

    1. Rolling ball background subtraction.
    2. Robust percentile normalisation to [0, 1].
    3. Richardson-Lucy deconvolution (15 iterations per channel).
    4. Additive combination of normalised and deconvolved images.
    5. Per-channel min-max rescaling of the combined result to [0, 1].

    Parameters
    ----------
    image : np.ndarray
        Raw (Z, C, Y, X) float32 stack as returned by load_and_standardize.
    progress : callable, optional
        Optional callback with signature (step_name: str, pct: int) used
        to relay progress to the UI.

    Returns
    -------
    np.ndarray
        Preprocessed (Z, C, Y, X) float32 stack in [0, 1].
    """
    if progress:
        progress("Starting background subtraction", 0)

    img_bg   = _rolling_ball_subtraction(image, radius=50.0, progress=progress)
    img_norm = normalize_robust(img_bg)
    del img_bg

    if progress:
        progress("Starting RL deconvolution", 0)

    img_deconv   = _richardson_lucy_deconvolution(img_norm, n_iter=15, progress=progress)
    img_combined = img_norm + img_deconv
    del img_norm, img_deconv

    if progress:
        progress("Finalising preprocessing", 99)

    img_final = np.zeros_like(img_combined, dtype=np.float32)
    for c in range(img_combined.shape[1]):
        ch       = img_combined[:, c, :, :]
        mn, mx   = ch.min(), ch.max()
        img_final[:, c, :, :] = (ch - mn) / (mx - mn) if mx > mn else ch
    del img_combined

    if progress:
        progress("Preprocessing complete", 100)

    return img_final


def preprocess_airyscan(
    image:    np.ndarray,
    progress: Optional[Callable] = None,
) -> np.ndarray:
    """
    Airyscan preprocessing: optional XY downsampling to TARGET_XY (1100 px)
    followed by robust percentile normalisation.  No deconvolution is applied.

    Parameters
    ----------
    image : np.ndarray
        Raw (Z, C, Y, X) float32 stack as returned by load_and_standardize.
    progress : callable, optional
        Progress callback with signature (step_name: str, pct: int).

    Returns
    -------
    np.ndarray
        Preprocessed (Z, C, Y, X) float32 stack in [0, 1].
    """
    from skimage.transform import resize as sk_resize

    if image.ndim != 4:
        raise ValueError(f"Expected 4-D (Z, C, Y, X) array, got shape {image.shape}")

    Z, C, Y, X = image.shape

    # Only downscale; never upscale images that are already at or below TARGET_XY.
    if Y > TARGET_XY or X > TARGET_XY:
        if progress:
            progress("Downsampling to 1100 px", 0)

        resized = np.zeros((Z, C, TARGET_XY, TARGET_XY), dtype=np.float32)
        for z in range(Z):
            for c in range(C):
                resized[z, c] = sk_resize(
                    image[z, c],
                    (TARGET_XY, TARGET_XY),
                    order=1,            # bilinear interpolation
                    anti_aliasing=True,
                    preserve_range=True,
                )
            if progress:
                pct = int(100 * (z + 1) / Z)
                progress("Downsampling to 1100 px", pct)
        image = resized

    if progress:
        progress("Normalising", 99)

    result = normalize_robust(image)

    if progress:
        progress("Preprocessing complete", 100)

    return result


# ------------------------------------------------------------------
# Conversion to MicroSAM RGB format
# ------------------------------------------------------------------

def to_microsam_rgb(image: np.ndarray) -> np.ndarray:
    """
    Converts a preprocessed (Z, C, Y, X) float32 stack to the (Z, Y, X, 3)
    uint8 RGB format expected by micro_sam.automatic_segmentation.
    Channel 0 (BRP-shortcherry red) maps to the R plane, channel 1
    (KC claws green) maps to the G plane.  The B plane is left at zero.
    This function only rescales [0, 1] -> [0, 255]; it does not itself do
    any percentile clipping or normalisation — preprocess_lsm/
    preprocess_airyscan are expected to have already normalised the input
    to [0, 1] via normalize_robust before this is called. Passing in an
    image that isn't already in [0, 1] will produce values outside
    [0, 255] with no clipping.

    Parameters
    ----------
    image : np.ndarray
        Preprocessed (Z, C, Y, X) float32 array, expected to already be in
        [0, 1] (see preprocess_lsm / preprocess_airyscan).

    Returns
    -------
    np.ndarray
        (Z, Y, X, 3) float32 array in [0, 255], ready for slice-wise
        MicroSAM inference (matches the batch inference script's
        preprocess(), which never casts to uint8).
    """
    if image.ndim != 4:
        raise ValueError(f"Expected 4-D (Z, C, Y, X) array, got shape {image.shape}")

    Z, C, Y, X = image.shape
    rgb = np.zeros((Z, Y, X, 3), dtype=np.float32)

    for c in range(min(C, 2)):
        rgb[:, :, :, c] = image[:, c, :, :] * 255.0

    return rgb
