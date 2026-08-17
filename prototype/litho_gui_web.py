"""Lithophane Studio — modern GUI (pywebview + HTML/Three.js).

Python backend exposing a bridge (LithoApi) to the HTML/JS frontend, reusing
the existing engine (litho_engine / litho_3mf / litho_reverse). The frontend
in litho_web/ is a modern dark UI with a Three.js interactive 3D view.

Run:  python litho_gui_web.py
"""

from __future__ import annotations

import base64
import io
import os
import sys
import time
import threading

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from litho_core import LithophaneParams, validate_mesh
from litho_engine import LithoMode, ColorOrder, color_lithophane_engine
from litho_3mf import assemble_lithophane_parts, write_3mf
from litho_color import DEFAULT_TD, _resample_rgb, linear_to_srgb8

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "litho_web")


class LithoApi:
    """Methods callable from the JS frontend via pywebview.api.*"""

    def __init__(self):
        self.rgb = None
        self.image_path = None
        self.last_meshes = None
        self.last_params = None
        self.out_dir = None

    # ------------------------------------------------------------ helpers
    def _img_b64(self, arr, max_side=800):
        """numpy (H,W,3) -> base64 PNG (downscaled for preview)."""
        im = Image.fromarray(arr)
        scale = min(max_side / im.width, max_side / im.height, 1.0)
        if scale < 1.0:
            im = im.resize((int(im.width * scale), int(im.height * scale)), Image.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _mesh_json(self, meshes):
        """meshes {color:(verts,faces)} -> {color:{verts:[...],faces:[...]}}"""
        out = {}
        for c, (v, f) in meshes.items():
            if len(f) == 0:
                out[c] = {"verts": [], "faces": []}
                continue
            out[c] = self._decimate(np.asarray(v, dtype=float), np.asarray(f, dtype=np.int64))
        return out

    @staticmethod
    def _decimate(verts, faces, max_verts=12000):
        """Decimate a structured mesh for 3D preview.

        Height-field / band meshes have vertices on a grid (gx x gy XY points,
        top and bottom share XY -> n = 2*gx*gy). We keep every Nth XY column
        and row, then RE-TRIANGULATE on the kept grid (keeping the top surface
        shape). The full build (top 154k verts / 14MB JSON) drops to ~10k verts
        and <1MB while preserving the visible structure.

        This only approximates the mesh (it re-triangulates the top surface),
        which is fine for a 3D preview. The exported STL/3MF is NOT decimated.
        """
        n = len(verts)
        if n <= max_verts:
            return {"verts": verts.ravel().tolist(),
                    "faces": faces.ravel().tolist()}

        xs = np.round(verts[:, 0], 4)
        ys = np.round(verts[:, 1], 4)
        ux = np.unique(xs)
        uy = np.unique(ys)
        gx, gy = len(ux), len(uy)
        col_of = np.searchsorted(ux, xs)
        row_of = np.searchsorted(uy, ys)

        # Stride in grid coords to reach ~max_verts vertices (target ~0.5 pts).
        n_pts = gx * gy
        step = max(1, int(np.ceil(np.sqrt(n_pts / (max_verts * 0.5)))))

        # Kept grid points: (col, row) with col%step==0 and row%step==0.
        keep_cols = np.arange(0, gx, step)
        keep_rows = np.arange(0, gy, step)

        # Build new vertex list: for each kept (row,col), take the FIRST
        # vertex at that XY (top surface); optionally also bottom (skip for 3D
        # preview — top surface + a flat bottom for solidity).
        # We keep both top and bottom so the preview looks solid.
        new_v = []
        pt_index = {}  # (row,col) -> list of [top_idx, bot_idx] in new_v
        for r in keep_rows:
            for c in keep_cols:
                mask = (col_of == c) & (row_of == r)
                idxs = np.where(mask)[0]
                if len(idxs) == 0:
                    continue
                # sort by z to find top (max) and bottom (min)
                zs = verts[idxs, 2]
                top_i = idxs[np.argmax(zs)]
                bot_i = idxs[np.argmin(zs)]
                new_v.append(verts[top_i])
                new_v.append(verts[bot_i])
                pt_index[(r, c)] = (len(new_v) - 2, len(new_v) - 1)
        new_v = np.array(new_v, dtype=float)

        # Re-triangulate top surface on kept grid: for each kept cell
        # (r, r+step) x (c, c+step), two triangles using top verts.
        # Also emit the bottom surface (using the bottom verts that were
        # otherwise orphaned) so the preview shell is closed and visible
        # from every angle, not just above.
        nf = []
        for ri in range(len(keep_rows) - 1):
            r0 = keep_rows[ri]; r1 = keep_rows[ri + 1]
            for ci in range(len(keep_cols) - 1):
                c0 = keep_cols[ci]; c1 = keep_cols[ci + 1]
                if (r0, c0) not in pt_index or (r0, c1) not in pt_index or \
                   (r1, c0) not in pt_index or (r1, c1) not in pt_index:
                    continue
                a = pt_index[(r0, c0)][0]
                b = pt_index[(r0, c1)][0]
                c_ = pt_index[(r1, c1)][0]
                d = pt_index[(r1, c0)][0]
                # top (CCW, up)
                nf += [(a, b, c_), (a, c_, d)]
                # bottom (CW, down)
                a1 = pt_index[(r0, c0)][1]
                b1 = pt_index[(r0, c1)][1]
                c1_ = pt_index[(r1, c1)][1]
                d1 = pt_index[(r1, c0)][1]
                nf += [(a1, d1, c1_), (a1, c1_, b1)]
        return {"verts": new_v.ravel().tolist(),
                "faces": np.array(nf, dtype=np.int64).ravel().tolist()}

    def _default_outdir(self):
        desk = os.path.join(os.path.expanduser("~"), "Desktop")
        base = os.path.join(desk, "lithophane_exports_web")
        os.makedirs(base, exist_ok=True)
        return base

    # ------------------------------------------------------------ JS API
    def pick_image(self):
        """Open file dialog, load image, return preview thumb + size."""
        import webview
        try:
            # pywebview doesn't ship a file dialog API; use tkinter's silently
            # (headless pick only, no main window).
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            path = filedialog.askopenfilename(
                title="选择图片", filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")])
            root.destroy()
        except Exception:
            path = ""
        if not path:
            return {"ok": False, "error": "未选择"}
        try:
            with Image.open(path) as im:
                self.rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
            self.image_path = path
            w, h = self.rgb.shape[1], self.rgb.shape[0]
            return {"ok": True, "thumb": self._img_b64(self.rgb),
                    "w": w, "h": h, "name": os.path.basename(path)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def build(self, params: dict):
        """Generate meshes for the given params; return previews + 3D data."""
        if self.rgb is None:
            return {"ok": False, "error": "未选择图片"}
        try:
            p = LithophaneParams(
                width_mm=float(params["width_mm"]),
                height_mm=float(params["height_mm"]),
                pixel_pitch_mm=float(params.get("pitch", 0.3)),
                base_thickness=float(params.get("dwhite", 0.3)),
            )
            layer_h = float(params.get("layer_h", 0.2))
            layers = int(params.get("layers", 8))
            mode = LithoMode(params["mode"])
            order = ColorOrder(params.get("order", "CMY"))
            td = {
                "C": (float(params.get("td_c", 0.5)), 3.0, 3.0),
                "M": (3.0, float(params.get("td_m", 0.5)), 3.0),
                "Y": (3.0, 3.0, float(params.get("td_y", 0.5))),
                "W": DEFAULT_TD["W"],
            }
            # Grid guard: cap solver grid like the old GUI (keep interactive).
            from litho_core import thickness_grid_shape
            gx, gy = thickness_grid_shape(self.rgb.shape[0], self.rgb.shape[1], p)
            if gx * gy > 400_000:
                scale = ((gx * gy) / 400_000) ** 0.5
                p.pixel_pitch_mm = max(p.pixel_pitch_mm * scale, 0.3)
                pitch_cmy, pitch_top = 0.8 * scale, 0.25 * scale
            elif mode == LithoMode.BAMBU:
                # Bambu reference resolution: white texture grid 2x finer than
                # CMY (measured 0.22 / 0.44mm in lithophane_谢bro_U1), versus
                # our default 0.25 / 0.8. Keep white ~0.22 but give CMY a
                # moderate 0.44 so the finer CMY grid stays sliceable.
                pitch_cmy, pitch_top = 0.44, 0.22
            else:
                pitch_cmy, pitch_top = 0.8, 0.25

            t0 = time.time()
            meshes, dE, gamut, reached = color_lithophane_engine(
                self.rgb, mode=mode, order=order, params=p, td=td,
                layers_max=layers, layer_h=layer_h, exact=False,
                pitch_cmy=pitch_cmy, pitch_top=pitch_top,
                dW=float(params.get("dwhite", 0.8)),
                top_max=float(params.get("maxthick", 2.0)) - float(params.get("dwhite", 0.8)),
                carve=params.get("carve", "concave"),
                sharpen=float(params.get("sharpen", 2.0)),
                contrast=float(params.get("contrast", 1.5)),
                tone_map=bool(int(float(params.get("tonemap", 1)))))
            elapsed = time.time() - t0

            self.last_meshes = meshes
            self.last_params = (p, layers, layer_h, td, mode, order)

            total_faces = sum(len(f) for _, f in meshes.values())
            dE_med = float(np.median(dE)) if dE is not None else 0.0

            return {
                "ok": True,
                "reached_b64": self._img_b64(reached) if reached is not None else None,
                "meshes": self._mesh_json(meshes),
                "dE_med": round(dE_med, 2),
                "total_faces": total_faces,
                "elapsed": round(elapsed, 1),
                "size": f"{p.width_mm:.0f}×{p.height_mm:.0f}mm",
            }
        except Exception as e:
            import traceback; traceback.print_exc()
            return {"ok": False, "error": str(e)}

    def export_all(self, fmt: str = "BOTH"):
        """Export current build as 3MF and/or STLs to Desktop/lithophane_exports_web."""
        if self.last_meshes is None:
            return {"ok": False, "error": "先构建"}
        try:
            p, layers, layer_h, td, mode, order = self.last_params
            self.out_dir = os.path.join(self._default_outdir(), mode.value)
            os.makedirs(self.out_dir, exist_ok=True)

            if fmt in ("3MF", "BOTH"):
                parts, offsets, names, extruders = assemble_lithophane_parts(self.last_meshes)
                write_3mf(os.path.join(self.out_dir, "lithophane.3mf"), parts, offsets,
                          extruders, part_names=names, printer_model="Snapmaker U1",
                          printer_settings_id="Snapmaker U1 (0.4 nozzle)",
                          build_center_mm=(135.5, 136.0, 0.0))
            if fmt in ("STL", "BOTH"):
                from litho_core import export_stl
                stl_names = [("W","white"),("C","cyan"),("M","magenta"),
                             ("Y","yellow"),("top","top_white")]
                for key, name in stl_names:
                    v, f = self.last_meshes[key]
                    if len(f) == 0:
                        continue
                    export_stl(os.path.join(self.out_dir, f"litho_{name}.stl"), v, f,
                               name=f"lithophane_{name}")
            return {"ok": True, "dir": self.out_dir}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def reverse_import(self):
        """Pick a 3MF, reconstruct preview, return it for comparison."""
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            path = filedialog.askopenfilename(title="选择 3MF",
                                              filetypes=[("3MF", "*.3mf")])
            root.destroy()
        except Exception:
            path = ""
        if not path:
            return {"ok": False, "error": "取消"}
        try:
            from litho_reverse import load_3mf_colors, reconstruct_from_meshes
            cm = load_3mf_colors(path)
            if "W" in cm and len(cm["W"]):
                v = cm["W"].reshape(-1, 3)
                w = float(v[:, 0].max() - v[:, 0].min())
                h = float(v[:, 1].max() - v[:, 1].min())
            else:
                w = h = 100.0
            recon = reconstruct_from_meshes(cm, w, h, pixel_pitch=max(w / 200, 0.5))
            return {"ok": True, "recon_b64": self._img_b64(recon)}
        except Exception as e:
            return {"ok": False, "error": str(e)}


def main():
    import webview
    api = LithoApi()
    window = webview.create_window(
        "Lithophane Studio",
        os.path.join(WEB_DIR, "index.html"),
        js_api=api,
        width=1280, height=800,
        min_size=(1000, 640),
        background_color="#0f1220",
    )
    webview.start()


if __name__ == "__main__":
    main()
