---
layout: home
title: BoutonViewer
---

A desktop tool for segmenting and quantifying synaptic boutons in 3D
fluorescence microscopy stacks. It runs a MicroSAM-based instance
segmentation pipeline on confocal (LSM) or Airyscan TIFF stacks, displays
the raw channels and resulting labels in a [napari](https://napari.org)
viewer, and reports per-bouton volume and surface area in physical units
(µm³ / µm²).

- **[Model & Data Notes](model_notes.html)** — what the model was trained
  on, voxel size assumptions, LSM vs. Airyscan preprocessing, and
  post-processing behavior. Read this before using the app on your own
  data.
- **[Setup and usage instructions](https://github.com/aditya0701/Fluorescent-Microscopy-Image-Segmentation-and-Quantification/blob/main/README.md)**
  are in the main repository README (installing dependencies, getting a
  checkpoint, running the app, and a walkthrough of every UI control).
- **[Source code](https://github.com/aditya0701/Fluorescent-Microscopy-Image-Segmentation-and-Quantification)**
