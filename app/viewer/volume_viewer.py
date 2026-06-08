"""
volume_viewer.py
PyVista-based 3D interactive viewer embedded in the main window via
pyvistaqt.QtInteractor.

Responsibilities
----------------
- Volume rendering the raw image (channel 0, BRP-shortcherry) using
  PyVista's ray-cast volume renderer.
- Generating and displaying one surface mesh per bouton label via marching
  cubes on the binary label mask, with a consistent colour assignment that
  matches the 2D slice view.
- Hover picking: every 100 ms a vtkCellPicker probes the actor under the
  cursor and emits bouton_hovered(label_id).  A throttle prevents excessive
  VTK picks from affecting render performance.
- Click picking: a left-click identifies the bouton under the cursor and
  emits bouton_selected(label_id).
- Highlighting the selected bouton (full opacity) versus unselected ones
  (reduced opacity).
- Removal of individual bouton actors when a bouton is deleted.
"""

from __future__ import annotations
import time
from typing import Dict, Optional
import numpy as np

import vtk
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt6.QtCore import pyqtSignal
from skimage.measure import marching_cubes


# Background colour for the viewer (pure black, matching the 2D viewer).
_BG_COLOR  = "#000000"

# Opacity of unselected bouton meshes.
_OPACITY_NORMAL   = 0.65
# Opacity of the selected bouton mesh.
_OPACITY_SELECTED = 1.00

# Throttle interval in seconds for hover picking to limit VTK CPU usage.
_HOVER_THROTTLE = 0.10


class VolumeViewer(QtInteractor):
    """
    PyVista QtInteractor subclass that manages volume rendering and
    bouton mesh display.  Signals are used to communicate picking results
    to the main window without creating a direct dependency on the UI layer.
    """

    # Emitted with the label_id of the bouton the mouse is currently over,
    # or -1 when the cursor is not over any bouton.
    bouton_hovered  = pyqtSignal(int)

    # Emitted with the label_id of the bouton that was left-clicked,
    # or -1 when the click did not hit a bouton.
    bouton_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

        self._image:          Optional[np.ndarray] = None   # (Z, C, Y, X)
        self._labels:         Optional[np.ndarray] = None   # (Z, Y, X)
        self._voxel_size:     tuple = (0.3, 0.0709, 0.0709) # (z, y, x) µm — placeholder; set_image() supplies the real per-acquisition value
        self._label_colors:   Dict[int, tuple] = {}          # label_id -> (R,G,B)
        self._bouton_actors:  Dict[int, object] = {}         # label_id -> PyVista Actor
        self._volume_actors:  list = []
        self._selected_label: int = -1
        self._last_hovered:   int = -1
        self._last_pick_t:    float = 0.0

        self.set_background(_BG_COLOR)
        self.enable_anti_aliasing("ssaa")
        self.show_axes()

        # Register left-click handler for bouton selection.
        self.track_click_position(callback=self._on_left_click, side="left")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_image(self, image: np.ndarray, voxel_size: tuple):
        """Stores the raw image and renders the 3D volume."""
        self._image      = image
        self._voxel_size = voxel_size
        self._render_volume()

    def set_labels(
        self,
        labels:       np.ndarray,
        label_colors: Dict[int, tuple],
    ):
        """Stores the label array and renders all bouton surface meshes."""
        self._labels       = labels
        self._label_colors = label_colors
        self._render_all_boutons()

    def clear_labels(self):
        """Removes all bouton actors from the scene."""
        for actor in self._bouton_actors.values():
            self.remove_actor(actor)
        self._bouton_actors.clear()
        self._selected_label = -1
        self.render()

    def remove_bouton(self, label_id: int):
        """Removes a single bouton actor from the scene."""
        actor = self._bouton_actors.pop(label_id, None)
        if actor is not None:
            self.remove_actor(actor)
            self.render()

    def highlight_bouton(self, label_id: int):
        """
        Visually selects a bouton by raising its opacity and dimming all others.
        Called either from the viewer's own click picker or from the sidebar
        when the user selects a row in the stats table.
        """
        self._selected_label = label_id
        for lbl, actor in self._bouton_actors.items():
            actor.prop.opacity = (
                _OPACITY_SELECTED if lbl == label_id else _OPACITY_NORMAL
            )
        self.render()

    def pause_rendering(self):
        """
        Suppresses GPU rendering so a background task (CUDA inference) gets
        the full GPU: stops the periodic auto-render timer, sets PyVista's
        suppress_rendering flag (turns every render() call — including the
        timer's — into a no-op that skips render_window.Render()), and
        freezes camera interaction so dragging mid-pause doesn't feel broken
        (nothing would visually update anyway). Pair with resume_rendering().
        """
        self.render_timer.stop()
        self.suppress_rendering = True
        self.disable()

    def resume_rendering(self):
        """
        Reverses pause_rendering(): re-enables camera interaction, clears
        the suppress flag, restarts the render timer at its original
        interval, and forces one render so the final state (e.g. newly
        added bouton meshes) is shown immediately.
        """
        self.enable()
        self.suppress_rendering = False
        self.render_timer.start(self.render_timer.interval())
        self.render()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_volume(self):
        """
        Renders the raw image as a single ray-cast RGBA volume: channel 0
        in green and channel 1 in red, composited per-voxel exactly like
        the 2D view's colour convention (and napari's additive multi-
        channel display) — both channels visible together as one object.

        Why a single RGBA volume rather than add_volume's normal scalar +
        colormap path
        --------------------------------------------------------------
        Two independent vtkVolume actors (one per channel, each with its
        own colormap) do *not* composite in VTK — whichever is added first
        fully occludes the second regardless of blend-mode settings (this
        was verified directly: rendering channel 1 alone showed it fine,
        but adding channel 0 afterwards made channel 1 disappear and vice
        versa). The fix is to build ONE volume whose scalars are already
        direct (N, 4) uint8 RGBA values — add_volume supports this
        natively (it sets independent_components=False and skips the
        colormap/transfer-function machinery), so each voxel's colour and
        opacity are exactly what we computed, with both channels baked
        into a single actor that the mapper can ray-cast in one pass.

        Why mapper="fixed_point"
        ------------------------
        add_volume's default mapper relies on vtkGPUVolumeRayCastMapper,
        which needs 3D-texture / render-to-texture GPU support that's
        inconsistent across hardware. On this machine (4 GB GTX 1650 Ti)
        it produced an actor with nothing visible despite scalars being
        wired correctly (point_data, matching `dimensions` — confirmed
        correct because that exact setup rendered fine in an isolated
        off-screen test). mapper="fixed_point" selects
        vtkFixedPointVolumeRayCastMapper, the classic CPU-side ray caster
        every VTK build supports, sidestepping the GPU-texture path while
        still doing genuine continuous volume rendering. It may feel
        sluggish while orbiting on very large stacks — report back if so.

        Why a display gamma
        -------------------
        Fluorescence stacks are dominated by dim background once stretched
        linearly to [0, 1] — a linear RGBA volume came out almost black
        (peak screen brightness ~57/255 in testing). Raising the
        normalised intensity to a fractional power (gamma 0.45, like the
        "brightness" curve every viewer — including napari — applies
        between data values and screen pixels) and giving alpha a matching
        boost brings the structure into view (~110/255) without
        thresholding a single voxel: every voxel keeps its own continuous
        brightness and colour, it's just remapped onto a more visible
        range — unlike the previous isosurface attempt, which *did* throw
        voxels away (only 3 discrete shells), hence the "blobs" that
        didn't match napari's continuous look.

        Debug output
        ------------
        Prints per-channel intensity stats, percentile clip, the RGBA
        array's value range, the grid summary, and whether the actor was
        created — flushed immediately to the console.
        """
        if self._image is None:
            return

        for actor in self._volume_actors:
            self.remove_actor(actor)
        self._volume_actors = []

        Z, C, Y, X = self._image.shape
        shape = (Z, Y, X)
        vz, vy, vx = self._voxel_size

        # channel index -> (label, normalised [0,1] intensity, flattened in
        # VTK point order: X fastest, then Y, then Z — matches
        # grid.dimensions = (X, Y, Z) below).
        channels = {}
        for c_idx in range(min(C, 2)):
            label = ("green", "red")[c_idx]
            channel = self._image[:, c_idx, :, :].astype(np.float32)
            print(
                f"[VolumeViewer] ch{c_idx} ({label}) shape={channel.shape} "
                f"min={channel.min():.4g} max={channel.max():.4g} "
                f"mean={channel.mean():.4g}",
                flush=True,
            )

            # Same 0.0001/99.9999 percentile clip used throughout the
            # pipeline (to_microsam_rgb / normalize_robust / the 2D
            # viewer's _stretch), so the contrast matches everywhere else.
            lo, hi = np.percentile(channel, (0.0001, 99.9999))
            print(f"[VolumeViewer] ch{c_idx} percentile clip: lo={lo:.4g} hi={hi:.4g}", flush=True)
            if hi <= lo:
                print(f"[VolumeViewer] ch{c_idx} degenerate intensity range — treated as empty", flush=True)
                channels[c_idx] = np.zeros(channel.size, dtype=np.float32)
                continue
            norm = np.clip((channel - lo) / (hi - lo), 0.0, 1.0)
            channels[c_idx] = norm.T.flatten(order="F")

        if not channels:
            print("[VolumeViewer] no channels available — nothing to render", flush=True)
            return

        green = channels.get(0, np.zeros_like(next(iter(channels.values()))))
        red   = channels.get(1, np.zeros_like(next(iter(channels.values()))))

        # Display gamma / alpha mapping. Originally these were pushed well
        # above neutral (gamma 0.45, then 0.6; alpha boosted 3x, then 1.4x)
        # to compensate for "composite" blending, which accumulates opacity
        # along the ray and made a linear mapping look nearly invisible.
        # Now that blending="maximum" (MIP, matching napari's rendering:mip
        # below) just shows the brightest voxel directly, that compensation
        # is no longer needed — neutral values (gamma=1.0, matching napari's
        # own gamma:1.00) let the percentile normalisation above do all the
        # work, exactly as napari's contrast-limits/auto-contrast do. Tune
        # these only if MIP still looks too hot or too faint.
        DISPLAY_GAMMA = 1.0
        ALPHA_BOOST   = 1.0
        g_disp = green ** DISPLAY_GAMMA
        r_disp = red   ** DISPLAY_GAMMA
        alpha  = np.clip(np.maximum(green, red) * ALPHA_BOOST, 0.0, 1.0)

        rgba = np.zeros((green.size, 4), dtype=np.uint8)
        rgba[:, 0] = (r_disp * 255.0).astype(np.uint8)   # channel 1 -> red
        rgba[:, 1] = (g_disp * 255.0).astype(np.uint8)   # channel 0 -> green
        rgba[:, 3] = (alpha  * 255.0).astype(np.uint8)
        print(
            f"[VolumeViewer] rgba ranges: R={rgba[:,0].min()}-{rgba[:,0].max()} "
            f"G={rgba[:,1].min()}-{rgba[:,1].max()} A={rgba[:,3].min()}-{rgba[:,3].max()}",
            flush=True,
        )

        # PyVista ImageData uses (X, Y, Z) axis ordering, so the array was
        # flattened transposed above; dimensions must equal the array shape
        # for the scalars to land in point_data (the mapper defaults to
        # ScalarMode=UsePointData — cell_data with dimensions = shape + 1
        # is silently ignored, which is what made the very first version
        # of this method invisible).
        grid = pv.ImageData()
        grid.dimensions = np.array(shape[::-1])      # (X, Y, Z)
        grid.spacing    = (vx, vy, vz)
        grid.origin     = (0.0, 0.0, 0.0)
        grid.point_data["rgba"] = rgba
        print(
            f"[VolumeViewer] grid dims={tuple(grid.dimensions)} "
            f"bounds={grid.bounds} n_points={grid.n_points}",
            flush=True,
        )

        try:
            actor = self.add_volume(
                grid,
                scalars="rgba",
                mapper="fixed_point",
                # In napari the two channels are independent layers, so one
                # can be "additive" (glows, sums along the ray) and the
                # other "translucent" at once — VTK can't do that (two
                # volume actors don't composite; one fully occludes the
                # other, see the docstring above), so both channels are
                # baked into ONE rgba volume that shares a single blend
                # mode. "additive" sums each channel's contribution along
                # the ray instead of picking a single winning voxel (as
                # "maximum"/MIP does) — this reads as more of a glowing,
                # cohesive single object and is closer to the feel of the
                # additive layer in napari than MIP's crisp x-ray look.
                blending="additive",
                show_scalar_bar=False,
                name="raw_volume",
            )
        except Exception:
            import traceback
            traceback.print_exc()
            actor = None

        if actor is not None:
            print(f"[VolumeViewer] volume actor created: {actor!r}", flush=True)
            self._volume_actors.append(actor)

        print(f"[VolumeViewer] raw-volume actors added: {len(self._volume_actors)}", flush=True)
        self.reset_camera()
        self.render()

    def _render_all_boutons(self):
        """Generates and adds surface meshes for every label in self._labels."""
        self.clear_labels()
        if self._labels is None:
            return

        unique = np.unique(self._labels)
        unique = unique[unique > 0]

        for lbl in unique:
            self._add_bouton_mesh(int(lbl))

        self.render()

    def _add_bouton_mesh(self, label_id: int):
        """
        Generates the marching cubes surface mesh for one bouton label and
        adds it to the scene as a distinct actor.  The mesh carries a
        scalar array 'label_id' used by the cell picker.
        """
        mask = (self._labels == label_id).astype(np.float32)
        if mask.sum() == 0:
            return

        try:
            vz, vy, vx = self._voxel_size

            # _render_volume's pv.ImageData has dimensions=(X,Y,Z) and
            # spacing=(vx,vy,vz), so a data voxel at index (x,y,z) lands at
            # Cartesian position (x*vx, y*vy, z*vz) — data axes map directly
            # onto Cartesian X/Y/Z. marching_cubes, however, returns vertex
            # coordinates as (array_axis_0 * spacing[0], axis_1 * spacing[1],
            # axis_2 * spacing[2]), and PyVista's PolyData treats those three
            # columns as Cartesian (X, Y, Z) directly. Calling it on the mask
            # in its native (Z, Y, X) order with spacing (vz, vy, vx) would
            # therefore place vertices at (z*vz, y*vy, x*vx) — i.e. with the
            # data's Z and X axes swapped relative to the volume (here Z
            # spacing is ~7x the XY pixel size, so this showed up as a large
            # scale/position mismatch, not just a rotation). Transposing the
            # mask to (X, Y, Z) — a pure relabelling of array axes, not a
            # mirror — and passing spacing in the matching (vx, vy, vz) order
            # makes marching_cubes emit vertices already in (x*vx, y*vy,
            # z*vz) Cartesian form, exactly matching the volume grid.
            mask_xyz = np.transpose(mask, (2, 1, 0))
            verts, faces, normals, _ = marching_cubes(
                mask_xyz, level=0.5, spacing=(vx, vy, vz)
            )
            n_faces    = len(faces)
            face_array = np.hstack([
                np.full((n_faces, 1), 3, dtype=int), faces
            ]).flatten()

            mesh               = pv.PolyData(verts, face_array)
            mesh["label_id"]   = np.full(mesh.n_cells, label_id, dtype=np.int32)

            rgb   = self._label_colors.get(label_id, (180, 180, 180))
            color = "#{:02x}{:02x}{:02x}".format(*rgb)

            actor = self.add_mesh(
                mesh,
                color=color,
                opacity=_OPACITY_NORMAL,
                smooth_shading=True,
                name=f"bouton_{label_id}",
            )
            self._bouton_actors[label_id] = actor

        except Exception as exc:
            print(f"[VolumeViewer] Could not build mesh for bouton {label_id}: {exc}")

    # ------------------------------------------------------------------
    # Picking
    # ------------------------------------------------------------------

    def mouseMoveEvent(self, event):
        """
        Intercepts Qt mouse move events to perform throttled VTK cell picking.
        The throttle prevents the vtkCellPicker from being called more than
        once per _HOVER_THROTTLE seconds, which keeps rendering smooth.
        """
        super().mouseMoveEvent(event)

        now = time.perf_counter()
        if now - self._last_pick_t < _HOVER_THROTTLE:
            return
        self._last_pick_t = now

        pos      = event.pos()
        label_id = self._pick_at(pos.x(), pos.y())

        if label_id != self._last_hovered:
            self._last_hovered = label_id
            self.bouton_hovered.emit(label_id)

    def _on_left_click(self, point):
        """
        Called by PyVista's click tracker on every left-click.
        The point argument is a 3D world-space coordinate; we re-run a
        prop picker in screen space to identify the clicked actor.
        """
        # Retrieve the last mouse position in screen coordinates from Qt.
        pos      = self.mapFromGlobal(self.cursor().pos())
        label_id = self._pick_at(pos.x(), pos.y())

        if label_id > 0:
            self._selected_label = label_id
            self.highlight_bouton(label_id)
            self.bouton_selected.emit(label_id)
        else:
            self.bouton_selected.emit(-1)

    def _pick_at(self, screen_x: int, screen_y: int) -> int:
        """
        Uses vtkCellPicker to identify which bouton actor is under the given
        screen coordinates.  Returns the label_id of the hit bouton, or -1
        if the cursor is not over any bouton mesh.

        VTK and Qt use opposite Y conventions: VTK origin is bottom-left,
        Qt origin is top-left.  The Y coordinate is therefore flipped.
        """
        if not self._bouton_actors:
            return -1

        # Build a lookup from VTK C++ object address to label_id so we can
        # identify the picked actor without relying on Python object identity,
        # which is unreliable across VTK wrapper calls.
        addr_to_label = {
            actor.GetAddressAsString(""): lbl
            for lbl, actor in self._bouton_actors.items()
        }

        vtk_y   = self.size().height() - screen_y
        picker  = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)

        # Restrict picking to the bouton meshes only. Without this, the
        # raw-volume actor — which, after the axis-alignment fix, now
        # correctly fills the same bounding box the meshes sit inside —
        # sits in the pick ray's path too. vtkCellPicker considers every
        # prop in the scene by default, and a vtkVolume hit either reports
        # itself (not in addr_to_label) or blocks the ray from reaching
        # the mesh behind it; either way the bouton resolves to "not found".
        # PickFromList makes the picker see only the props we hand it, so
        # the volume can never shadow or be mistaken for a bouton.
        picker.PickFromListOn()
        picker.InitializePickList()
        for actor in self._bouton_actors.values():
            picker.AddPickList(actor)

        picker.Pick(screen_x, vtk_y, 0, self.renderer)

        picked_actor = picker.GetActor()
        if picked_actor is None:
            return -1

        return addr_to_label.get(picked_actor.GetAddressAsString(""), -1)
