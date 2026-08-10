"""Lithophane prototype - M2 color GUI.

Flow:
  1. pick a color image
  2. tweak print parameters (size / layers / layer height / white base / pitch)
     and TD selectivity (the 3 strong channels: C-red, M-green, Y-blue)
  3. "Build + preview" renders the WYSIWYG appearance (what the printed stack
     actually looks like under the backlight) and shows color-fidelity stats.
     The build runs on a background thread so the UI never freezes.
  4. "Export 4 STLs" writes litho_W/C/M/Y.stl for stacking in the slicer

Run:  python litho_gui.py
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from litho_core import LithophaneParams, export_stl, validate_mesh
from litho_color import DEFAULT_TD, build_gamut, color_lithophane_stacked


def _mesh_volume(vertices, faces):
    """Signed volume of a triangle mesh (vectorized, fast). Positive for outward
    winding; used only for the status readout to keep the UI thread responsive."""
    tri = vertices[faces]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    cross = np.cross(b - a, c - a)
    return float(np.sum(np.einsum("ij,ij->i", a, cross))) / 6.0


class LithophaneApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Lithophane Prototype (M2 color - CMYW stacking)")
        self.geometry("980x620")
        self.minsize(860, 560)

        self.rgb: np.ndarray | None = None
        self.image_path: str | None = None
        self._preview_orig = None
        self._preview_reach = None
        self._last = None  # (meshes, dE, params)
        self._worker = None  # background build thread
        self._building = False
        self._result_q = queue.Queue()  # worker -> main thread

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(0, weight=3)
        root.columnconfigure(1, weight=2)
        root.rowconfigure(0, weight=1)

        # --- Left: images ---
        left = ttk.LabelFrame(root, text="Image  |  Printed appearance (WYSIWYG)", padding=8)
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        left.columnconfigure(1, weight=1)

        self.preview_orig = tk.Label(left, text="Original\n(no image)", bg="#222", fg="#aaa",
                                     width=40, height=16, anchor="center")
        self.preview_orig.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.preview_reach = tk.Label(left, text="Printed\n(backlit result)", bg="#222", fg="#aaa",
                                      width=40, height=16, anchor="center")
        self.preview_reach.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        # --- Right: parameters ---
        right = ttk.LabelFrame(root, text="Parameters", padding=8)
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        right.columnconfigure(1, weight=1)

        def param_row(r, label, default, key, lo=0, hi=1000):
            ttk.Label(right, text=label).grid(row=r, column=0, sticky="w", pady=2)
            var = tk.DoubleVar(value=default)
            ent = ttk.Entry(right, textvariable=var, width=10)
            ent.grid(row=r, column=1, sticky="we", padx=4, pady=2)
            setattr(self, key, var)

        param_row(0, "Width (mm)", 144.0, "w_var")
        param_row(1, "Height (mm)", 108.0, "h_var")
        param_row(2, "Max layers / color", 8, "layers_var")
        param_row(3, "Layer height (mm)", 0.08, "layerh_var")
        param_row(4, "White base (mm)", 0.30, "dw_var")
        param_row(5, "Pixel pitch (mm)", 0.3, "pitch_var")

        ttk.Separator(right).grid(row=6, column=0, columnspan=2, sticky="we", pady=6)
        ttk.Label(right, text="TD selectivity (strong channel TD):", font=("", 9, "bold")).grid(
            row=7, column=0, columnspan=2, sticky="w")
        param_row(8, "Cyan absorb R (TD_R)", DEFAULT_TD["C"][0], "tdc_var", 0.05, 2.0)
        param_row(9, "Magenta absorb G (TD_G)", DEFAULT_TD["M"][1], "tdm_var", 0.05, 2.0)
        param_row(10, "Yellow absorb B (TD_B)", DEFAULT_TD["Y"][2], "tdy_var", 0.05, 2.0)
        ttk.Label(right, text="(weak channels fixed at TD=3.0; see README for calibration)",
                  foreground="#777").grid(row=11, column=0, columnspan=2, sticky="w")

        self.exact_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(right, text="Exact mode (slow, ~30s)", variable=self.exact_var).grid(
            row=12, column=0, columnspan=2, sticky="w")

        self.btn_pick = ttk.Button(right, text="1. Choose image...", command=self.pick_image)
        self.btn_pick.grid(row=13, column=0, columnspan=2, sticky="we", pady=(8, 4))
        self.btn_build = ttk.Button(right, text="2. Build + preview (WYSIWYG)", command=self.build,
                                    state="disabled")
        self.btn_build.grid(row=14, column=0, columnspan=2, sticky="we", pady=2)
        self.btn_export = ttk.Button(right, text="3. Export 4 STLs...", command=self.export_stls,
                                     state="disabled")
        self.btn_export.grid(row=15, column=0, columnspan=2, sticky="we", pady=2)

        self.status = tk.Text(right, height=14, width=38, state="disabled",
                              font=("Consolas", 9), background="#111", foreground="#9f9")
        self.status.grid(row=16, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        right.rowconfigure(16, weight=1)

    # ------------------------------------------------------------- actions
    def pick_image(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")])
        if not path:
            return
        try:
            with Image.open(path) as im:
                self.rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
            self.image_path = path
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
            return
        self._show_rgb(self.preview_orig, self.rgb, "Original")
        self._show_rgb(self.preview_reach, np.zeros((8, 8, 3), np.uint8), "Printed")
        self.btn_build.config(state="normal")
        self.btn_export.config(state="disabled")
        self._log(f"Loaded {os.path.basename(path)}: {self.rgb.shape[1]}x{self.rgb.shape[0]} px")

    def build(self):
        if self.rgb is None or self._building:
            return
        try:
            params, td = self._params_from_ui()
        except ValueError as e:
            messagebox.showerror("Invalid parameter", str(e))
            return

        # Snapshot inputs (avoid touching shared state from the worker thread).
        rgb = self.rgb.copy()
        layers = int(self.layers_var.get())
        layer_h = self.layerh_var.get()
        dW = self.dw_var.get()
        exact = self.exact_var.get()

        self._building = True
        self.btn_build.config(state="disabled", text="Building... (UI stays responsive)")
        self.btn_export.config(state="disabled")
        self._log(f"\n=== Building ({'exact' if exact else 'fast'} mode)... ===")

        def worker():
            try:
                t0 = time.time()
                meshes, dE, gamut, reached = color_lithophane_stacked(
                    rgb, params=params, td=td,
                    layers_max=layers, layer_h=layer_h, exact=exact,
                    pitch_cmy=0.8, pitch_top=0.25)
                elapsed = time.time() - t0
                self._result_q.put(("ok", meshes, dE, gamut, reached, elapsed, params))
            except Exception as e:  # noqa: BLE001
                self._result_q.put(("err", str(e)))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()
        # Poll the result queue from the main thread (tkinter is not thread-safe;
        # we never touch widgets from the worker).
        self.after(50, self._poll_result)

    def _poll_result(self):
        try:
            msg = self._result_q.get_nowait()
        except queue.Empty:
            if self._building:
                self.after(50, self._poll_result)
            return
        kind = msg[0]
        if kind == "ok":
            (_, meshes, dE, gamut, reached, elapsed, params) = msg
            self._building = False
            self.btn_build.config(state="normal", text="2. Build + preview (WYSIWYG)")
            self._last = (meshes, dE, params)
            self.btn_export.config(state="normal")
            self._show_rgb(self.preview_reach, reached, "Printed (backlit)")

            # Stats. Use the cheap volume-only path here (full watertight edge
            # check is deferred to export) to keep the UI thread snappy.
            dE_flat = dE.ravel()
            total_vol = sum(_mesh_volume(v, f) for v, f in meshes.values())
            dE_mean = float(dE_flat.mean())
            dE_p90 = float(np.percentile(dE_flat, 90))
            frac6 = float(np.mean(dE_flat <= 6))
            self._log(
                f"Build done in {elapsed:.1f}s\n"
                f"grid        : {gamut['lab'].shape[0]} color combos\n"
                f"color error : mean dE={dE_mean:.2f}  p90={dE_p90:.2f}  <=6:{frac6:.0%}\n"
                f"  (2.3=invisible, 6=acceptable, 10+=soft/dark)\n"
                f"total volume: {total_vol:,.0f} mm^3\n"
                f"PLA weight  : {total_vol*1.24/1000:.1f} g\n"
                f"5 layers    : W base + C/M/Y bands + top_white relief\n"
                f"  (each layer occupies its own Z band -> 1 color per printed layer)")
        else:
            self._building = False
            self.btn_build.config(state="normal", text="2. Build + preview (WYSIWYG)")
            self._log(f"Build failed: {msg[1]}")
            messagebox.showerror("Build failed", msg[1])

    def export_stls(self):
        if self._last is None or self._building:
            return
        meshes, _, _ = self._last
        outdir = filedialog.askdirectory(title="Choose output folder for the 5 STLs")
        if not outdir:
            return
        # Bambu-style 5-layer naming; each STL maps to one AMS slot in Z order.
        names = [("W", "white"), ("C", "cyan"), ("M", "magenta"),
                 ("Y", "yellow"), ("top", "top_white")]
        for key, name in names:
            verts, faces = meshes[key]
            path = os.path.join(outdir, f"litho_{name}.stl")
            export_stl(path, verts, faces, name=f"lithophane_{name}")
            v = validate_mesh(verts, faces)
            self._log(f"  litho_{name}.stl : {v['num_faces']:,} faces, "
                      f"vol={v['volume']:,.0f} mm^3, watertight={v['open_edges']==0}")
        self._log("\nDone. Import the 5 STLs into the slicer and stack them "
                  "aligned on Z: white(bottom) -> cyan -> magenta -> yellow -> "
                  "top_white(top). Each slot = one filament color.")

    # ------------------------------------------------------------- helpers
    def _params_from_ui(self):
        def gv(name, lo, hi, what):
            val = float(getattr(self, name).get())
            if not (lo <= val <= hi):
                raise ValueError(f"{what} must be in [{lo}, {hi}]")
            return val

        params = LithophaneParams(
            width_mm=gv("w_var", 10, 500, "Width"),
            height_mm=gv("h_var", 10, 500, "Height"),
            pixel_pitch_mm=gv("pitch_var", 0.02, 2.0, "Pixel pitch"),
            base_thickness=gv("dw_var", 0.0, 3.0, "White base"),
        )
        layers = int(gv("layers_var", 1, 30, "Max layers"))
        layer_h = gv("layerh_var", 0.02, 0.5, "Layer height")
        td = {
            "C": (gv("tdc_var", 0.05, 2.0, "Cyan TD"), 3.0, 3.0),
            "M": (3.0, gv("tdm_var", 0.05, 2.0, "Magenta TD"), 3.0),
            "Y": (3.0, 3.0, gv("tdy_var", 0.05, 2.0, "Yellow TD")),
            "W": DEFAULT_TD["W"],
        }
        return params, td

    def _show_rgb(self, label, rgb, caption):
        im = Image.fromarray(rgb)
        # Keep aspect within ~420x280.
        scale = min(420 / im.width, 280 / im.height, 1.0)
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BILINEAR)
        photo = ImageTk.PhotoImage(im)
        label.config(image=photo, text=caption, compound="top", fg="#ddd", bg="#333")
        # Keep a reference (prevent GC) depending on which label we're updating.
        if label is self.preview_orig:
            self._preview_orig = photo
        else:
            self._preview_reach = photo

    def _log(self, text):
        self.status.config(state="normal")
        self.status.insert(tk.END, text + "\n")
        self.status.see(tk.END)
        self.status.config(state="disabled")


def main():
    app = LithophaneApp()
    app.mainloop()


if __name__ == "__main__":
    main()
