"""Lithophane prototype - pluggable engine GUI (mode x color-order x live re-run).

Layout:
  - Left panel: TOP image (original) above BOTTOM image (printed WYSIWYG).
  - Right panel: parameters + mode/order dropdowns + build/export buttons.
  - Parameter or mode/order change triggers an automatic background re-build
    (debounced 600 ms) so the user sees the effect immediately.

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
from litho_color import DEFAULT_TD
from litho_engine import LithoMode, ColorOrder, color_lithophane_engine


def _mesh_volume(vertices, faces):
    """Signed volume of a triangle mesh (vectorized, fast). Positive for outward
    winding; used only for the status readout to keep the UI thread responsive."""
    if len(faces) == 0 or len(vertices) == 0:
        return 0.0
    tri = vertices[faces]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    cross = np.cross(b - a, c - a)
    return float(np.sum(np.einsum("ij,ij->i", a, cross))) / 6.0


class LithophaneApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Lithophane Prototype (pluggable engine)")
        self.geometry("1080x720")
        self.minsize(900, 640)

        self.rgb: np.ndarray | None = None
        self.image_path: str | None = None
        self._preview_orig = None
        self._preview_reach = None
        self._last = None
        self._worker = None
        self._building = False
        self._result_q = queue.Queue()
        self._after_id = None  # debounce timer

        self._build_ui()
        # Sync order dropdown to the initial mode (LAYERED -> 6 CMY orders).
        self._on_mode_change()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(0, weight=3)
        root.columnconfigure(1, weight=2)
        root.rowconfigure(0, weight=1)

        # --- Left: previews stacked vertically (top=original, bottom=printed) ---
        left = ttk.LabelFrame(root, text="Preview", padding=8)
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        self.preview_orig = tk.Label(left, text="Original\n(no image)", bg="#222", fg="#aaa",
                                     width=48, height=12, anchor="center")
        self.preview_orig.grid(row=0, column=0, sticky="nsew", pady=(0, 6))

        # Bottom preview: tabs for 2D backlit result and interactive 3D view.
        self.preview_tabs = ttk.Notebook(left)
        self.preview_tabs.grid(row=1, column=0, sticky="nsew")

        self._tab_2d = ttk.Frame(self.preview_tabs)
        self._tab_3d = ttk.Frame(self.preview_tabs)
        self.preview_tabs.add(self._tab_2d, text="Backlit (2D)")
        self.preview_tabs.add(self._tab_3d, text="3D view")

        self.preview_reach = tk.Label(self._tab_2d, text="Printed\n(backlit result)",
                                      bg="#222", fg="#aaa", width=48, height=12, anchor="center")
        self.preview_reach.pack(fill=tk.BOTH, expand=True)

        from litho_view3d import LithoView3D
        self.view3d = LithoView3D(self._tab_3d)
        self.view3d.pack(fill=tk.BOTH, expand=True)

        # --- Right: parameters ---
        right = ttk.LabelFrame(root, text="Parameters", padding=8)
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        right.columnconfigure(1, weight=1)

        def param_row(r, label, default, key, lo=0, hi=1000):
            ttk.Label(right, text=label).grid(row=r, column=0, sticky="w", pady=2)
            var = tk.DoubleVar(value=default)
            ent = ttk.Entry(right, textvariable=var, width=10)
            ent.grid(row=r, column=1, sticky="we", padx=4, pady=2)
            ent.bind("<KeyRelease>", self._schedule_auto)
            setattr(self, key, var)

        row = 0
        param_row(row, "Width (mm)", 144.0, "w_var"); row += 1
        param_row(row, "Height (mm)", 108.0, "h_var"); row += 1
        param_row(row, "Max layers / color", 8, "layers_var"); row += 1
        param_row(row, "Layer height (mm)", 0.2, "layerh_var"); row += 1
        param_row(row, "White base (mm)", 0.30, "dw_var"); row += 1
        param_row(row, "Pixel pitch (mm)", 0.3, "pitch_var"); row += 1

        ttk.Separator(right).grid(row=row, column=0, columnspan=2, sticky="we", pady=6); row += 1

        ttk.Label(right, text="Mode:").grid(row=row, column=0, sticky="w", pady=2)
        self.mode_var = tk.StringVar(value=LithoMode.LAYERED.value)
        self.mode_cb = ttk.Combobox(right, textvariable=self.mode_var, state="readonly",
                                    values=[m.value for m in LithoMode], width=10)
        self.mode_cb.grid(row=row, column=1, sticky="we", padx=4, pady=2)
        self.mode_cb.bind("<<ComboboxSelected>>", self._on_mode_change)
        row += 1

        ttk.Label(right, text="Color order:").grid(row=row, column=0, sticky="w", pady=2)
        self.order_var = tk.StringVar(value=ColorOrder.CMY.value)
        self.order_cb = ttk.Combobox(right, textvariable=self.order_var, state="readonly",
                                     values=[o.value for o in ColorOrder], width=10)
        self.order_cb.grid(row=row, column=1, sticky="we", padx=4, pady=2)
        self.order_cb.bind("<<ComboboxSelected>>", self._schedule_auto)
        row += 1

        ttk.Separator(right).grid(row=row, column=0, columnspan=2, sticky="we", pady=6); row += 1

        ttk.Label(right, text="TD selectivity (strong channel):", font=("", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w"); row += 1
        param_row(row, "Cyan absorb R (TD_R)", DEFAULT_TD["C"][0], "tdc_var", 0.05, 2.0); row += 1
        param_row(row, "Magenta absorb G (TD_G)", DEFAULT_TD["M"][1], "tdm_var", 0.05, 2.0); row += 1
        param_row(row, "Yellow absorb B (TD_B)", DEFAULT_TD["Y"][2], "tdy_var", 0.05, 2.0); row += 1
        ttk.Label(right, text="(weak channels fixed at TD=3.0)", foreground="#777").grid(
            row=row, column=0, columnspan=2, sticky="w"); row += 1

        self.exact_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(right, text="Exact mode (slow)", variable=self.exact_var,
                        command=self._schedule_auto).grid(
            row=row, column=0, columnspan=2, sticky="w"); row += 1

        ttk.Separator(right).grid(row=row, column=0, columnspan=2, sticky="we", pady=6); row += 1

        # Export options: format + vendor/printer. Vendor/Printer are 3MF-only
        # (hidden when format == STL).
        ttk.Label(right, text="Export format:").grid(row=row, column=0, sticky="w", pady=2)
        self.fmt_var = tk.StringVar(value="3MF")
        self.fmt_cb = ttk.Combobox(right, textvariable=self.fmt_var, state="readonly",
                                   values=("3MF", "STL"), width=10)
        self.fmt_cb.grid(row=row, column=1, sticky="we", padx=4, pady=2)
        self.fmt_cb.bind("<<ComboboxSelected>>", lambda e: self._on_fmt_change())
        row += 1

        self.vendor_lbl = ttk.Label(right, text="Vendor:")
        self.vendor_lbl.grid(row=row, column=0, sticky="w", pady=2)
        self.vendor_var = tk.StringVar(value="Snapmaker")
        self.vendor_cb = ttk.Combobox(right, textvariable=self.vendor_var, state="readonly",
                                      values=("Snapmaker", "Bambu Lab"), width=10)
        self.vendor_cb.grid(row=row, column=1, sticky="we", padx=4, pady=2)
        self.vendor_cb.bind("<<ComboboxSelected>>", lambda e: self._sync_printer())
        row += 1

        self.printer_lbl = ttk.Label(right, text="Printer:")
        self.printer_lbl.grid(row=row, column=0, sticky="w", pady=2)
        self.printer_var = tk.StringVar(value="Snapmaker U1")
        self.printer_cb = ttk.Combobox(right, textvariable=self.printer_var, state="readonly",
                                       values=("Snapmaker U1", "Snapmaker A250",
                                               "Snapmaker A350", "Snapmaker Artisan",
                                               "Snapmaker J1", "Bambu Lab X1 Carbon",
                                               "Bambu Lab P1S"), width=16)
        self.printer_cb.grid(row=row, column=1, sticky="we", padx=4, pady=2)
        self.printer_cb.bind("<<ComboboxSelected>>", lambda e: self._sync_printer())
        row += 1

        self.btn_pick = ttk.Button(right, text="1. Choose image...", command=self.pick_image)
        self.btn_pick.grid(row=row, column=0, columnspan=2, sticky="we", pady=(8, 4)); row += 1
        self.btn_build = ttk.Button(right, text="2. Build + preview", command=self.build)
        self.btn_build.grid(row=row, column=0, columnspan=2, sticky="we", pady=2); row += 1
        self.btn_export = ttk.Button(right, text="3. Export...", command=self.export_stls,
                                     state="disabled")
        self.btn_export.grid(row=row, column=0, columnspan=2, sticky="we", pady=2); row += 1
        self.btn_reverse = ttk.Button(right, text="4. Reverse-import (STL/3MF -> preview)",
                                      command=self.reverse_import)
        self.btn_reverse.grid(row=row, column=0, columnspan=2, sticky="we", pady=2); row += 1

        self.status = tk.Text(right, height=16, width=40, state="disabled",
                              font=("Consolas", 9), background="#111", foreground="#9f9")
        self.status.grid(row=row, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        right.rowconfigure(row, weight=1)

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
        self.btn_build.config(state="normal")
        # Default output size: fit the image into a printable size, preserving
        # aspect ratio, with the LONGER side at 144 mm (Bambu standard frame
        # size; matches the 144x108 default params and keeps the top-relief
        # grid under MAX_POINTS so the pitch guard does not kick in).
        # User can still edit afterwards. (A 1px=1mm default made every phone
        # photo a multi-metre object — OrcaSlicer correctly warned "too large".)
        TARGET_LONG_MM = 144.0
        w_px = float(self.rgb.shape[1])
        h_px = float(self.rgb.shape[0])
        scale = TARGET_LONG_MM / max(w_px, h_px)
        w_use = w_px * scale
        h_use = h_px * scale
        self.w_var.set(round(w_use))
        self.h_var.set(round(h_use))
        self._log(f"Loaded {os.path.basename(path)}: {self.rgb.shape[1]}x{self.rgb.shape[0]} px\n"
                  f"Default output size = {w_use:.0f}x{h_use:.0f} mm "
                  f"(long side 144mm, aspect-preserving, editable)\n"
                  f"Mode={self.mode_var.get()} Order={self.order_var.get()}\n"
                  f"Auto-rebuild on parameter change is ON")
        self._schedule_auto()

    def _on_mode_change(self, _evt=None):
        # The order dropdown must match the mode:
        #   LAYERED     -> exactly the 6 CMY permutations
        #   INTERLEAVED -> MIXED (Bambu 方案B, same-band)
        #   GREYSCALE/STACKED -> order locked to CMY (fixed order)
        mode = LithoMode(self.mode_var.get())
        if mode == LithoMode.LAYERED:
            choices = [o.value for o in _ORDER_CMY_ORDER]
            if self.order_var.get() not in choices:
                self.order_var.set(ColorOrder.CMY.value)
        elif mode in (LithoMode.INTERLEAVED, LithoMode.OVERLAP):
            choices = [ColorOrder.MIXED.value]
            self.order_var.set(ColorOrder.MIXED.value)
        else:  # GREYSCALE / STACKED: fixed CMY order
            choices = [ColorOrder.CMY.value]
            self.order_var.set(ColorOrder.CMY.value)
        self.order_cb.config(values=choices)
        self._schedule_auto()

    def _schedule_auto(self, _evt=None):
        """Debounced auto-rebuild after any parameter/mode/order change."""
        if self._after_id is not None:
            self.after_cancel(self._after_id)
        self._after_id = self.after(600, self._auto_build)

    def _auto_build(self):
        self._after_id = None
        if self.rgb is not None and not self._building:
            self.build()

    def _read_ui(self):
        def gv(name, lo, hi, what):
            val = float(getattr(self, name).get())
            if not (lo <= val <= hi):
                raise ValueError(f"{what} must be in [{lo}, {hi}]")
            return val
        params = LithophaneParams(
            width_mm=gv("w_var", 10, 10000, "Width"),
            height_mm=gv("h_var", 10, 10000, "Height"),
            pixel_pitch_mm=gv("pitch_var", 0.02, 20.0, "Pixel pitch"),
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
        mode = LithoMode(self.mode_var.get())
        order = ColorOrder(self.order_var.get())
        exact = self.exact_var.get()
        return params, layers, layer_h, td, mode, order, exact

    def build(self):
        if self.rgb is None or self._building:
            return
        try:
            params, layers, layer_h, td, mode, order, exact = self._read_ui()
        except ValueError as e:
            messagebox.showerror("Invalid parameter", str(e))
            return

        # ---- Grid-size guard (blocks OOM before it happens) ----
        # The solver grid scales as width*height/pitch^2, and the geometry
        # grids (pitch_cmy=0.8, pitch_top=0.25) scale the same way. We clamp the
        # effective pitch so ALL grids stay bounded (MAX_POINTS), which makes
        # large images (e.g. 4000x3000 mm from 1px=1mm default) buildable
        # instead of crashing or hanging.
        from litho_core import thickness_grid_shape
        MAX_POINTS = 600_000
        gx0, gy0 = thickness_grid_shape(self.rgb.shape[0], self.rgb.shape[1], params)
        n0 = gx0 * gy0
        # top geometry grid uses base pitch 0.25 (finer than solver 0.3), so its
        # grid is the binding constraint.
        n_top0 = (int(params.width_mm / 0.25) + 1) * (int(params.height_mm / 0.25) + 1)
        worst = max(n0, n_top0)
        if worst > MAX_POINTS:
            # thickness_grid_shape uses round(width/pitch)+1, so a safety factor
            # (1.1) guarantees the rescaled grids are strictly <= MAX_POINTS.
            scale = (worst / MAX_POINTS) ** 0.5 * 1.1
            # Raise the user pitch by `scale` so grid points drop below MAX.
            base_pitch = max(params.pixel_pitch_mm, 1e-3)
            eff_pitch = base_pitch * scale
            params = LithophaneParams(
                width_mm=params.width_mm, height_mm=params.height_mm,
                pixel_pitch_mm=eff_pitch,
                base_thickness=params.base_thickness,
                depth_range=params.depth_range)
            pitch_cmy = 0.8 * scale
            pitch_top = 0.25 * scale
            self._log(f"[auto] grid {worst:,} pts > {MAX_POINTS:,}; "
                      f"raised pitch to {eff_pitch:.2f}mm to keep it buildable")
        else:
            pitch_cmy, pitch_top = 0.8, 0.25

        rgb = self.rgb.copy()
        self._building = True
        self.btn_build.config(state="disabled", text="Building... (UI responsive)")
        self.btn_export.config(state="disabled")
        self._log(f"\n=== Building mode={mode.value} order={order.value}... ===")

        def worker():
            try:
                t0 = time.time()
                meshes, dE, gamut, reached = color_lithophane_engine(
                    rgb, mode=mode, order=order, params=params, td=td,
                    layers_max=layers, layer_h=layer_h, exact=exact,
                    pitch_cmy=pitch_cmy, pitch_top=pitch_top)
                elapsed = time.time() - t0
                self._result_q.put(("ok", meshes, dE, gamut, reached, elapsed, params, mode, order))
            except Exception as e:  # noqa: BLE001
                self._result_q.put(("err", str(e)))
        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()
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
            (_, meshes, dE, gamut, reached, elapsed, params, mode, order) = msg
            self._building = False
            self.btn_build.config(state="normal", text="2. Build + preview")
            self._last = (meshes, dE, params, mode, order)
            self.btn_export.config(state="normal")
            if reached is not None:
                self._show_rgb(self.preview_reach, reached, f"Printed ({mode.value}/{order.value})")
            # Feed meshes to the interactive 3D view.
            self.view3d.set_meshes(meshes)

            dE_flat = dE.ravel() if reached is not None else np.array([0])
            total_vol = sum(_mesh_volume(v, f) for v, f in meshes.values())
            self._log(
                f"Done in {elapsed:.1f}s  [mode={mode.value} order={order.value}]\n"
                f"color error : mean dE={dE_flat.mean():.2f} p90={np.percentile(dE_flat,90):.2f}\n"
                f"  (2.3=invisible, 6=acceptable, 10+=soft)\n"
                f"total volume: {total_vol:,.0f} mm^3  weight={total_vol*1.24/1000:.1f} g\n"
                f"meshes: " + ", ".join(f"{c}={len(f):,}" for c, (v, f) in meshes.items()))
        else:
            self._building = False
            self.btn_build.config(state="normal", text="2. Build + preview")
            self._log(f"Build failed: {msg[1]}")
            messagebox.showerror("Build failed", msg[1])

    def _sync_printer(self, _evt=None):
        """When vendor changes, offer vendor-appropriate printers."""
        vendor = self.vendor_var.get()
        if vendor == "Snapmaker":
            printers = ["Snapmaker U1", "Snapmaker A250", "Snapmaker A350",
                        "Snapmaker Artisan", "Snapmaker J1"]
            if self.printer_var.get() not in printers:
                self.printer_var.set("Snapmaker U1")
        else:  # Bambu Lab
            printers = ["Bambu Lab X1 Carbon", "Bambu Lab P1S"]
            if self.printer_var.get() not in printers:
                self.printer_var.set("Bambu Lab X1 Carbon")
        self.printer_cb.config(values=printers)

    def _on_fmt_change(self, _evt=None):
        """Hide 3MF-only controls (Vendor/Printer) when format == STL."""
        if self.fmt_var.get() == "STL":
            self.vendor_lbl.grid_remove()
            self.vendor_cb.grid_remove()
            self.printer_lbl.grid_remove()
            self.printer_cb.grid_remove()
        else:
            self.vendor_lbl.grid()
            self.vendor_cb.grid()
            self.printer_lbl.grid()
            self.printer_cb.grid()

    def reverse_import(self):
        """Reverse-import: pick a 3MF (or separate STLs), reconstruct the
        preview, and show it vs the original image."""
        from litho_reverse import load_3mf_colors, reconstruct_from_meshes
        path = filedialog.askopenfilename(
            title="Choose a 3MF (or STL) to reverse-import",
            filetypes=[("3MF / STL", "*.3mf *.stl"), ("All files", "*.*")])
        if not path:
            return
        try:
            if path.lower().endswith(".3mf"):
                cmeshes = load_3mf_colors(path)
            else:
                # Single STL: not enough color info for STACKED; error clearly.
                messagebox.showerror("Single STL",
                                     "A single merged STL has no color identity "
                                     "(C/M/Y overlap per-pixel in stacked mode). "
                                     "Please import the 3MF instead.")
                return
            # Output size: from the W mesh XY span (fallback to 100x100).
            if "W" in cmeshes and len(cmeshes["W"]):
                v = cmeshes["W"].reshape(-1, 3)
                w = float(v[:, 0].max() - v[:, 0].min())
                h = float(v[:, 1].max() - v[:, 1].min())
            else:
                w = h = 100.0
            recon = reconstruct_from_meshes(cmeshes, w, h, pixel_pitch=max(w / 200, 0.5))
            self._log(f"Reverse-imported {os.path.basename(path)}: colors="
                      f"{sorted(cmeshes.keys())}, size={w:.0f}x{h:.0f}mm")
            if self.rgb is not None:
                # Show original (resized) on top, reconstruction below.
                from litho_color import _resample_rgb
                orig_r = _resample_rgb(self.rgb, recon.shape[:2])
                self._show_rgb(self.preview_orig, orig_r, "Original")
            self._show_rgb(self.preview_reach, recon,
                           f"Reconstructed ({os.path.basename(path)})")
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self._log(f"Reverse-import failed: {e}")

    def export_stls(self):
        if self._last is None or self._building:
            return
        meshes, _, _, mode, order = self._last
        fmt = self.fmt_var.get()
        vendor = self.vendor_var.get()
        printer = self.printer_var.get()

        if fmt == "STL":
            # Export 5 separate STLs.
            outdir = filedialog.askdirectory(title="Choose output folder for the 5 STLs")
            if not outdir:
                return
            from litho_core import export_stl
            names = [("W", "white"), ("C", "cyan"), ("M", "magenta"),
                     ("Y", "yellow"), ("top", "top_white")]
            for key, name in names:
                verts, faces = meshes[key]
                if len(faces) == 0:
                    self._log(f"  litho_{name}.stl : skipped (no material)")
                    continue
                path = os.path.join(outdir, f"litho_{name}.stl")
                export_stl(path, verts, faces, name=f"lithophane_{name}")
                self._log(f"  litho_{name}.stl : {len(faces):,} faces")
            self._log("Exported 5 STLs (stack aligned on Z).")
            return

        # 3MF composite export (Snapmaker/OrcaSlicer compatible): one object,
        # per-part filament mapping, 100% infill, machine preset.
        from litho_3mf import assemble_lithophane_parts, write_3mf
        try:
            parts, offsets, names, extruders = assemble_lithophane_parts(meshes)
            outdir = filedialog.askdirectory(title="Choose output folder for the .3mf")
            if not outdir:
                return
            path = os.path.join(outdir, "lithophane.3mf")
            variant = "0.4"
            # Place the model at the centre of the selected printer's bed via
            # the build-item transform. The parts are already XY-centred
            # locally, so translating the build item to the bed centre centres
            # the model on the plate (matches Bambu's exported 3MF, which
            # writes the bed-centre translation in the build item).
            bed_center = _BED_CENTERS_MM.get(printer, (0.0, 0.0))
            write_3mf(path, parts, offsets, extruders, part_names=names,
                      printer_model=printer, printer_variant=variant,
                      printer_settings_id=f"{printer} ({variant} nozzle)",
                      filament_vendor=vendor,
                      build_center_mm=(bed_center[0], bed_center[1], 0.0))
            self._log(f"Exported composite 3MF: {path}")
            self._log(f"  parts={names} extruders={extruders}")
            self._log(f"  printer={printer} vendor={vendor} variant={variant}")
            self._log(f"  infill=100%, per-part filament mapped, model at bed centre "
                      f"({bed_center[0]:.1f},{bed_center[1]:.1f})")
        except Exception as e:  # noqa: BLE001
            self._log(f"3MF export failed: {e}")

    # ------------------------------------------------------------- helpers
    def _show_rgb(self, label, rgb, caption):
        im = Image.fromarray(rgb)
        scale = min(460 / im.width, 240 / im.height, 1.0)
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BILINEAR)
        photo = ImageTk.PhotoImage(im)
        label.config(image=photo, text=caption, compound="top", fg="#ddd", bg="#333")
        if label is self.preview_orig:
            self._preview_orig = photo
        else:
            self._preview_reach = photo

    def _log(self, text):
        self.status.config(state="normal")
        self.status.insert(tk.END, text + "\n")
        self.status.see(tk.END)
        self.status.config(state="disabled")


# Order list used by _on_mode_change to detect fixed CMY orders.
_ORDER_CMY_ORDER = [ColorOrder.CMY, ColorOrder.CYM, ColorOrder.MCY,
                    ColorOrder.MYC, ColorOrder.YMC, ColorOrder.YCM]

# Bed centres (mm) per printer, for placing the exported model at the centre
# of the plate via the 3MF build-item transform. Values from the OrcaSlicer
# machine presets (resources/profiles/Snapmaker/machine/*.json printable_area):
#   A250 230x250 -> (115, 125); A350 320x350 -> (160, 175);
#   Artisan 400x400 -> (200, 200); J1 324x200 -> (162, 100);
#   U1 270.5x271 -> (135.5, 136). Bambu Lab beds: X1C/P1S 256x256 -> (128,128).
_BED_CENTERS_MM = {
    "Snapmaker A250": (115.0, 125.0),
    "Snapmaker A350": (160.0, 175.0),
    "Snapmaker Artisan": (200.0, 200.0),
    "Snapmaker J1": (162.0, 100.0),
    "Snapmaker U1": (135.5, 136.0),
    "Bambu Lab X1 Carbon": (128.0, 128.0),
    "Bambu Lab P1S": (128.0, 128.0),
}


def main():
    app = LithophaneApp()
    app.mainloop()


if __name__ == "__main__":
    main()
