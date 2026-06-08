"""
image_loader.py
Loads TIFF files and standardizes the dimension order to (Z, C, Y, X).
Supports 3D single-channel inputs and 4D two-channel inputs in either
(C, Z, Y, X) or (Z, C, Y, X) layout, which are the two formats produced
by the Zeiss LSM Plus acquisition pipeline used in this project.
"""

from pathlib import Path
import numpy as np
import tifffile


def load_and_standardize(path: str) -> np.ndarray:
    """
    Loads a TIFF file and returns a float32 array with shape (Z, C, Y, X).

    The function handles the two most common dimension orderings encountered
    in this project.  A 4D image whose first axis has length 2 is assumed to
    be in (C, Z, Y, X) order and is transposed accordingly.  A 4D image whose
    second axis has length 2 is assumed to already be in (Z, C, Y, X) order.
    A 3D image is treated as a single-channel stack and a dummy channel axis
    is inserted at position 1.

    Parameters
    ----------
    path : str
        Absolute or relative path to the .tif / .tiff file.

    Returns
    -------
    np.ndarray
        Float32 array with shape (Z, C, Y, X).

    Raises
    ------
    ValueError
        If the image has an unsupported number of dimensions or if the
        channel axis cannot be inferred from the shape.
    """
    path = str(path)
    img = tifffile.imread(path)

    if img.ndim == 3:
        # Single-channel 3D stack: (Z, Y, X) -> (Z, 1, Y, X)
        img = img[:, np.newaxis, :, :]

    elif img.ndim == 4:
        if img.shape[0] == 2 and img.shape[1] != 2:
            # (C, Z, Y, X) -> (Z, C, Y, X)
            img = np.transpose(img, (1, 0, 2, 3))
        elif img.shape[1] == 2:
            # Already in (Z, C, Y, X) — no transpose needed.
            pass
        elif img.shape[0] == 2 and img.shape[1] == 2:
            # Ambiguous: both axes have length 2.  Assume (C, Z, Y, X)
            # since channels are the smaller biological axis.
            img = np.transpose(img, (1, 0, 2, 3))
        else:
            raise ValueError(
                f"Cannot infer channel axis from shape {img.shape}. "
                f"Expected a 4D image with 2 channels along axis 0 or 1."
            )
    else:
        raise ValueError(
            f"Unsupported image dimensionality: {img.ndim}D (shape {img.shape}). "
            f"Expected a 3D or 4D TIFF stack."
        )

    return img.astype(np.float32)


def get_image_info(path: str) -> dict:
    """
    Returns basic metadata about a TIFF file without loading the full array.
    Useful for displaying file information in the UI before committing to a load.
    """
    with tifffile.TiffFile(path) as tif:
        shape = tif.series[0].shape if tif.series else None
        dtype = tif.series[0].dtype if tif.series else None
        n_pages = len(tif.pages)
    return {
        "path": path,
        "filename": Path(path).name,
        "raw_shape": shape,
        "dtype": str(dtype),
        "n_pages": n_pages,
    }
