"""Interactive 3D preview of the lithophane layers (pure tkinter Canvas).

Renders the 5 meshes (W/C/M/Y/top) with an orthographic projection, colored by
layer, with:
  - left-drag: rotate around X/Y axes
  - right-drag: pan
  - mouse wheel: zoom

This is a lightweight diagnostic viewer for quickly spotting problems
(floating layers, gaps, color placement) without needing a heavy 3D library.
For each mesh we project its vertices once per frame (vectorized via numpy);
~10k vertices per frame is fine for tkinter's ~30 fps.
"""

from __future__ import annotations

import numpy as np
import tkinter as tk

# Layer colors (approximate CMYW), used when no per-vertex color is given.
LAYER_COLORS = {
    "W": "#e0e0e0",
    "C": "#00a8e8",
    "M": "#ff2d95",
    "Y": "#ffe600",
    "top": "#ffffff",
}


class LithoView3D(tk.Canvas):
    """Interactive 3D viewport for a dict of {color: (vertices, faces)}."""

    def __init__(self, master, meshes=None, width=480, height=360, **kw):
        super().__init__(master, width=width, height=height, bg="#1a1a2e", **kw)
        self.meshes = meshes or {}
        self._colors = {}
        self._rx, self._ry = 0.55, -0.7      # rotation (rad)
        self._zoom = 1.0
        self._pan_x, self._pan_y = 0, 0
        self._drag = None                     # (button, last_x, last_y)
        self._cache = None                    # per-frame projected points

        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonPress-3>", self._on_press)
        self.bind("<B3-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<ButtonRelease-3>", self._on_release)
        self.bind("<MouseWheel>", self._on_wheel)

        self.redraw()

    # ------------------------------------------------------------ data
    def set_meshes(self, meshes, colors=None):
        self.meshes = meshes or {}
        self._colors = colors or {}
        self.redraw()

    def _project(self):
        """Rotate all mesh vertices once, return {color: (gx, gy)} screen pts.

        Vertices are downsampled per layer (MAX_PTS_PER_LAYER) so a 69k-vertex
        full build redraws in well under a frame (~10k points total), keeping
        the view interactive.
        """
        MAX_PTS_PER_LAYER = 2500
        rx, ry = self._rx, self._ry
        cosx, sinx = np.cos(rx), np.sin(rx)
        cosy, siny = np.cos(ry), np.sin(ry)
        R = np.array([
            [cosy, 0, siny],
            [sinx * siny, cosx, -sinx * cosy],
            [-cosx * siny, sinx, cosx * cosy],
        ])
        W = float(self.winfo_width() or 480)
        H = float(self.winfo_height() or 360)
        scale = 60.0 * self._zoom
        cx, cy = W / 2 + self._pan_x, H / 2 + self._pan_y
        out = {}
        for color, (v, f) in self.meshes.items():
            if len(v) == 0:
                continue
            vv = np.asarray(v, dtype=np.float64)
            # downsample if too many vertices
            if len(vv) > MAX_PTS_PER_LAYER:
                step = int(np.ceil(len(vv) / MAX_PTS_PER_LAYER))
                vv = vv[::step]
            # centre the mesh about its own bbox centre
            c = 0.5 * (vv.min(axis=0) + vv.max(axis=0))
            rel = vv - c
            rot = rel @ R.T
            # orthographic: keep x,y; use z for depth sort (further = darker)
            gx = rot[:, 0] * scale + cx
            gy = -rot[:, 1] * scale + cy
            out[color] = (gx, gy, rot[:, 2])
        return out

    # ------------------------------------------------------------ draw
    def redraw(self, _evt=None):
        self.delete("all")
        if not self.meshes:
            self.create_text(self.winfo_width() // 2 or 240, self.winfo_height() // 2 or 180,
                             text="No mesh (build first)", fill="#666")
            return
        proj = self._project()
        # Draw per layer: front vertices as small dots, with depth-based shade.
        # Sort layers by mean depth (draw far first).
        order = sorted(proj.items(), key=lambda kv: float(np.mean(kv[1][2])))
        for color, (gx, gy, z) in order:
            col = self._colors.get(color, LAYER_COLORS.get(color, "#888"))
            # shade by mean depth (further = darker)
            zmean = float(np.mean(z))
            try:
                # simple darken
                r = int(col[1:3], 16); g = int(col[3:5], 16); b = int(col[5:7], 16)
                f = max(0.35, min(1.0, 1.0 - (zmean - z.min()) / max(z.max() - z.min(), 1e-6) * 0.5))
                col = f"#{int(r*f):02x}{int(g*f):02x}{int(b*f):02x}"
            except Exception:  # noqa: BLE001
                pass
            # draw points
            for x, y in zip(gx, gy):
                self.create_oval(x - 1, y - 1, x + 1, y + 1, fill=col, outline="")
        # axes hint
        self.create_text(12, 12, anchor="nw", fill="#888",
                         text="drag: rotate | wheel: zoom | right-drag: pan",
                         font=("Consolas", 8))

    # ------------------------------------------------------------ events
    def _on_press(self, evt):
        self._drag = (evt.num, evt.x, evt.y)

    def _on_drag(self, evt):
        if self._drag is None:
            return
        btn, lx, ly = self._drag
        dx, dy = evt.x - lx, evt.y - ly
        if btn == 1:
            # rotate
            self._ry += dx * 0.01
            self._rx += dy * 0.01
        elif btn == 3:
            self._pan_x += dx
            self._pan_y += dy
        self._drag = (btn, evt.x, evt.y)
        self.redraw()

    def _on_release(self, _evt):
        self._drag = None

    def _on_wheel(self, evt):
        # Windows: evt.delta = ±120 per notch
        self._zoom *= 1.1 if evt.delta > 0 else 0.9
        self.redraw()
