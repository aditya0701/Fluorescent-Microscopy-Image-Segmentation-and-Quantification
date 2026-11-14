---
layout: page
title: Model & Data Notes
permalink: /model_notes.html
---

# Model & Data Notes

This page documents the conditions the included MicroSAM checkpoints were
trained and validated under, and the assumptions the preprocessing/
post-processing pipeline makes as a result. Results on data that doesn't
match these conditions are not guaranteed — the app will still run, but
segmentation quality is not validated outside of them.

## 1. Expected input data

The model was trained on **confocal/Airyscan image stacks of *Drosophila
melanogaster* brain tissue**, specifically the mushroom body, with exactly
**two fluorescence channels**:

| Channel | Must contain | Maps to |
|---|---|---|
| 1st | BRP-shortcherry signal | Red (`channel_axis=1`, index 0) |
| 2nd | Kenyon cell (KC) signal | Green (index 1) |

- **A third channel must be dropped before loading.** The app's RGB
  conversion (`to_microsam_rgb`) already adds the third (blue) plane itself
  — it expects exactly 2 input channels and has no use for a real one. If
  your acquisition produced a 3-channel TIFF, remove the extra channel in
  ImageJ/Fiji or with a short script before loading it here. Passing a
  3-channel array in directly isn't supported: `image_loader.py` can only
  infer the channel axis when one axis has length 1, 2, or both axes have
  length 2 — a genuine 3-channel array will fail to load with
  `"Cannot infer channel axis from shape..."`.
- **Channel order matters.** The pipeline hard-codes channel 0 = BRP-
  shortcherry (red) and channel 1 = KC signal (green) — see
  `app/model/preprocessing.py::to_microsam_rgb`. Swapping the channels at
  acquisition/export time will silently feed the model the wrong signal in
  the wrong place; there's no way for the app to detect this from the data
  itself.
- **Single-channel images are technically supported** (the loader inserts
  a dummy channel) but are a distribution shift from training — see the
  caveat in §4.1 of the main README.

## 2. Voxel size and image size assumptions

The default voxel sizes baked into the app (`BoutonStore.VOXEL_SIZE_BY_TYPE`)
are **not universal constants** — they're the specific pixel pitch of the
microscope/objective/zoom settings used to acquire the thesis dataset this
model was trained on:

| Acquisition | Pixel pitch (Z, Y, X) µm | Detected from |
|---|---|---|
| LSM (confocal) | `(0.3, 0.0709, 0.0709)` | Always — fixed for the "LSM" type |
| Airyscan, native resolution | `(0.3, 0.0425, 0.0425)` | Image larger than 1100×1100 px (the thesis data's native size was 1834×1834) |
| Airyscan, downscaled | `(0.3, 0.0709, 0.0709)` | Image at or below 1100×1100 px |

The 1100 px and 1834 px figures are **not arbitrary thresholds** — they are
literally the pixel dimensions the thesis's Airyscan acquisitions came in
at (native 1834×1834, or already downsampled to 1100×1100 by the
acquisition software). The auto-detection in
`BoutonStore.detect_airyscan_voxel_size` picks a pitch by comparing the
loaded image's actual Y/X size against the `TARGET_XY = 1100` constant in
`app/model/preprocessing.py` — this is always the *raw, as-loaded*
dimensions, not the internal 1100×1100 the pipeline resizes to before
feeding the model. That internal resize is undone (`PredictionWorker` Step
4 resizes labels back to the original Y/X) before any volume/surface-area
stat is computed, so it never affects which pitch applies.

**If your microscope, objective, zoom, or binning settings are different**,
these defaults will silently apply the *wrong* physical scale to every
volume/surface-area measurement — the app has no way to know your
acquisition didn't match the thesis setup. Always check the **Voxel size
(µm)** fields after loading and edit them to your own calibration if it
differs (see README §4.2 for how to do this without re-running Predict).

## 3. Preprocessing: LSM vs. Airyscan

The two acquisition types get genuinely different pipelines, not just
different parameters (`app/model/preprocessing.py`):

**LSM** (`preprocess_lsm`) — full pipeline, because confocal data has no
computational deconvolution applied at acquisition:
1. Resize XY to 1100×1100 px, up or down as needed (`resize_to_target`) —
   a no-op if already that size.
2. Rolling-ball background subtraction (radius 50 px, per Z-slice/channel).
3. Robust percentile normalisation to [0, 1] (`normalize_robust`, clipping
   at the 0.0001/99.9999 percentiles).
4. Richardson-Lucy deconvolution, 15 iterations per channel, with a
   Gaussian PSF of sigma 1.75 for channel 0 (red/BRP) and 1.50 for channel
   1 (green/KC) — these sigma values match the point-spread function
   measured for the confocal setup used in the thesis, not a general
   default. **Optional** — toggled via the "Apply RL deconvolution"
   checkbox in the dock; when off, step 3's normalised image is used
   directly instead of being combined with a deconvolved version.
5. The normalised and deconvolved results are added together (skipped if
   deconvolution is off), then per-channel min-max rescaled back to [0, 1].

**Airyscan** (`preprocess_airyscan`) — lighter, because Airyscan acquisition
already performs its own computational super-resolution deconvolution
on-instrument:
1. Resize XY to 1100×1100 px, up or down as needed (same
   `resize_to_target` helper as the LSM path).
2. Robust percentile normalisation to [0, 1] — no deconvolution step at all.

**The deconvolution toggle is independent of the MicroSAM model variant.**
Earlier versions tied deconvolution to the "Base" (`vit_b_lm`) vs. "Large"
(`vit_l_lm`) choice; it is now a separate checkbox so either model variant
can be run with or without deconvolution on LSM data
(`app/model/worker.py`, `BoutonDockWidget.apply_deconvolution`).

## 4. Post-processing

After slice-wise 2D MicroSAM inference, `app/model/predictor.py::run_inference`
applies:

1. **3D connected components** with full 26-connectivity, linking 2D
   per-slice predictions into 3D bouton instances.
2. **Small-object removal**: any connected component under
   `BLOB_REMOVAL_THRESHOLD = 862` voxels is dropped. Note this is a **voxel
   count, not a physical volume** — its effective µm³ cutoff changes with
   whatever voxel size is active (≈1.3 µm³ at the LSM/downscaled-Airyscan
   pitch, ≈0.5 µm³ at the native-Airyscan pitch), since voxel volume itself
   depends on the pitch.
3. **Final relabelling** so the surviving label IDs are contiguous from 1.

A Z-span filter (removing objects present in 3 or fewer Z slices, as
acquisition artefacts) exists in the code but is currently **disabled**
(commented out in `run_inference`) — it isn't applied to the results you
see today.

### No upper volume limit is enforced — by design

Biologically, boutons larger than roughly 40 µm³ are implausible and
usually indicate the model has over-segmented (merged two or more adjacent
boutons into one label). **The software intentionally does not auto-remove
or auto-split these** the way it auto-removes undersized objects. Silently
dropping an oversized prediction would also discard whatever real signal it
does contain, and a fixed cutoff can't tell genuine biological variation
apart from a merge error — that judgment call is left to the user.

Instead, the app gives you the tools to handle this manually once you spot
it in the stats table or by hovering/clicking a suspiciously large bouton
in the viewer (README §4.5–4.6):
- **Delete Selected Bouton** removes a clearly wrong/merged prediction
  entirely.
- For finer correction (e.g. splitting a merged label rather than deleting
  it outright), there's currently no built-in split tool — review and
  re-run with a different model variant or stricter post-processing
  threshold if this happens often on your data.

This is the same philosophy as the missing max-size filter: the app
optimises for not throwing away potentially-correct data automatically,
and trusts the user to review and correct edge cases rather than enforce a
one-size-fits-all cutoff that would inevitably be wrong for some real
boutons too.
