# BoutonViewer

A desktop tool for segmenting and quantifying synaptic boutons in 3D
fluorescence microscopy stacks. It runs a MicroSAM-based instance
segmentation pipeline on confocal (LSM) or Airyscan TIFF stacks, displays
the raw channels and resulting labels in a [napari](https://napari.org)
viewer, and reports per-bouton volume and surface area in physical units
(µm³ / µm²).

## 1. Requirements

- Python 3.10+ (a dedicated conda environment is recommended)
- A Qt binding for napari (e.g. `PyQt6` or `PyQt5`) — not pulled in
  automatically by `requirements.txt`, see below
- A MicroSAM checkpoint (`best.pt`) for the model variant(s) you want to
  use — these are **not** included in this repository (see §2.3)
- A CUDA-capable GPU is recommended for reasonable inference speed, but
  not required — the code falls back to CPU automatically

## 2. Setup

### 2.1 Create an environment and install dependencies

```bash
conda create -n bouton_viewer python=3.10
conda activate bouton_viewer
pip install -r requirements.txt
```

### 2.2 Install a Qt binding

`requirements.txt` installs `napari` and `qtpy`, but `qtpy` is only an
abstraction layer — napari needs an actual Qt binding underneath. Install
one explicitly, for example:

```bash
pip install PyQt6
```

(`run.bat` sets `QT_API=pyqt6` to make sure napari picks this one if more
than one binding is present in your environment.)

### 2.3 Install micro-sam

micro-sam is not on PyPI in a form pinned by `requirements.txt`. Follow the
official install instructions for your platform:
https://computational-cell-analytics.github.io/micro-sam/micro_sam.html

### 2.4 Get the model checkpoints

The `models/` folder is excluded from this repository via `.gitignore`
(the checkpoint files are too large for GitHub). You need to obtain your
trained `best.pt` checkpoint(s) separately and place them anywhere on
disk — you will point the app at the exact file path from inside the UI,
so the folder location doesn't matter.

## 3. Running the app

```bash
python main.py
```

or, on Windows, double-click `run.bat` (after activating the conda
environment in that same terminal).

This opens a napari viewer with a "Bouton Analysis" dock panel on the
right-hand side.

## 4. Using the app

### 4.1 Load an image

1. In the **Image** section, pick the acquisition type from the dropdown:
   - **LSM** — runs the full preprocessing pipeline (rolling-ball
     background subtraction + Richardson-Lucy deconvolution).
   - **Airyscan** — runs a lighter normalisation-only pipeline, and
     downscales the image to 1100×1100 px first if it's larger.
2. Click **Load TIFF…** and select a `.tif`/`.tiff` stack.
   - Supported layouts: 3D single-channel `(Z, Y, X)`, or 4D two-channel
     stacks in either `(Z, C, Y, X)` or `(C, Z, Y, X)` order — the loader
     infers which from the shape.
3. Each channel appears as its own napari Image layer (green/red for a
   2-channel stack; a single grayscale layer for a single-channel one),
   and the **Voxel size (µm)** boxes auto-fill based on the chosen type
   and the image's actual pixel dimensions (see §4.2).

If you picked the wrong type, you don't need to reload — just change the
dropdown afterward and the app will reclassify the already-loaded image
in place (recomputing voxel size and rescaling the display).

### 4.2 Voxel size (µm)

This determines the physical units used for all volume/surface-area
calculations. It's auto-filled, but you can edit any of the Z/Y/X fields
directly if you know your calibration is different:

- **LSM** always uses the standard confocal pitch: `(0.3, 0.0709, 0.0709)`.
- **Airyscan** is auto-detected from the image's actual size: an image
  larger than 1100 px (e.g. native 1834×1834) gets the finer
  super-resolution pitch `(0.3, 0.0425, 0.0425)`; an image already at or
  below 1100 px gets the LSM-matching pitch `(0.3, 0.0709, 0.0709)`.

Editing these fields after a prediction has already run recomputes the
displayed volume/surface-area stats and rescales the 3D display shortly
after you stop typing/clicking (debounced, so it doesn't recompute on
every keystroke) — no need to re-run Predict.

### 4.3 Set the checkpoint and model variant

1. In the **MicroSAM Checkpoint** section, click **…** and select your
   `best.pt` file.
2. Choose the model variant: **Large (vit_l_lm)** or **Base (vit_b_lm)**.
   - For **LSM** images, the Base variant also skips the deconvolution
     preprocessing step entirely and uses the same lighter
     normalisation-only path as Airyscan — useful if you want a faster,
     lighter run.
   - This is read only at the moment you click Predict, so it's safe to
     change beforehand.

### 4.4 Run prediction

Click **▶ Run Prediction**. A progress bar reports each pipeline stage
(preprocessing, RGB conversion, slice-wise segmentation, 3D linking,
upscaling). When it finishes, a "boutons" Labels layer is added to the
napari viewer and the stats table on the right fills in.

Re-running Predict on the same loaded image with the same image type and
model variant skips preprocessing and reuses the cached result — only
inference reruns. Changing the model variant, image type, or loading a
new image invalidates that cache automatically.

### 4.5 Inspect results

- **Hover** over a bouton in the 3D/2D view to see its ID, volume, and
  surface area in the info box.
- **Click** a bouton to highlight its row in the stats table.
- The table lists every detected bouton with its volume (µm³) and
  surface area (µm²); columns are sortable.

### 4.6 Delete a bouton

Select a row in the table (or click a bouton in the viewer) and click
**Delete Selected Bouton**. After confirming, the label is zeroed out of
the array and removed from the table — this cannot be undone.

## 5. Notes on the pipeline

- Post-processing always applies 3D connected components (26-connectivity)
  and removes objects smaller than 862 voxels.
- Inference releases GPU memory after every Z-slice to avoid out-of-memory
  crashes on small-VRAM GPUs; this adds a small per-slice overhead that's
  the cost of that safety margin.
- `app/model/resource_monitor.py` is a standalone CPU/RAM/GPU profiling
  utility — it is not wired into the app and is only useful if you import
  and use it manually around a block of code.
