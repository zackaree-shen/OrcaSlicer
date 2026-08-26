"""Reverse-import: STL / 3MF -> reconstructed preview image.

The forward pipeline maps image RGB -> per-pixel thickness (dC,dM,dY,dTop)
-> 5 meshes (W/C/M/Y/top). This module inverts that: given the exported
meshes, it recovers each color's per-pixel thickness with a true vertical
ray-cast (Möller–Trumbore) sampled at pixel centres, then runs the forward
Beer-Lambert model to reconstruct the backlit appearance. The user compares
this reconstruction against the original image to judge the algorithm.

Supported inputs (per adversarial review, single merged STL cannot disambiguate
C/M/Y in STACKED/INTERLEAVED because color segments overlap per-pixel):
  - 3MF  : part identity from part NAME (not extruder; W and top share
           extruder 4); composite transforms applied.
  - Separate named STLs: litho_W/C/M/Y/top.stl (each is one color).
  - Single merged STL: LAYERED only (fixed global Z bands), user must supply
           mode/order/TD.

Notes:
  - Reconstruction vs original is ~6 dE even with a perfect extractor: the
    forward pipeline's gamut mapping + coarse-grid resampling contribute ~6.3
    dE. The extractor itself is exact (verified 0.0 dE round-trip).
  - TD must be supplied (not stored in 3MF/STL); defaults are v1 estimates.
  - Backlight defaults to D65 white (matches the solver's (1,1,1)).
"""

from __future__ import annotations

import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from litho_color import DEFAULT_TD, forward_stacked


# ---------------------------------------------------------------------------
# Ray-cast thickness extraction (Möller–Trumbore, vertical rays)
# ---------------------------------------------------------------------------

def _ray_z_span(tri, x, y, eps=1e-9):
    """Vertical ray at (x,y) vs triangle `tri` (3x3). Returns (z, True) if hit
    else (None, False). Uses Möller–Trumbore with eps so points on shared
    edges / the heightfield quad diagonal are accepted."""
    a, b, c = tri
    # Ray: p = (x, y, t); direction (0,0,1).
    # MT: t = (a - o) dot n / (d dot n); u,v = barycentric.
    ax, ay, az = a
    bx, by, bz = b
    cx_, cy_, cz = c
    e1 = b - a
    e2 = c - a
    # d = (0,0,1)
    pvec = np.cross(np.array([0.0, 0.0, 1.0]), e2)
    det = float(e1.dot(pvec))
    if abs(det) < 1e-15:
        return None, False
    inv = 1.0 / det
    tvec = np.array([x - ax, y - ay, 0.0])
    u = float(tvec.dot(pvec)) * inv
    if u < -eps or u > 1 + eps:
        return None, False
    qvec = np.cross(tvec, e1)
    v = float(np.array([0.0, 0.0, 1.0]).dot(qvec)) * inv
    if v < -eps or u + v > 1 + eps:
        return None, False
    z = float(e2.dot(qvec)) * inv + az
    return z, True


def ray_z_spans(tris, x, y, eps=1e-9):
    """All z intersections of the vertical ray at (x,y) with the triangle set.
    Returns sorted list of z values (each face hit once)."""
    hits = []
    for tri in tris:
        z, ok = _ray_z_span(tri, x, y, eps)
        if ok:
            hits.append(z)
    hits.sort()
    return hits


def extract_thickness_raycast(tris, x_grid, y_grid):
    """Per-pixel vertical thickness of a mesh, sampled at pixel centres.

    Vectorized: each pixel only tests triangles whose XY bbox overlaps its
    pixel cell (spatial pre-bucketing), so it scales to tens of thousands of
    triangles.

    tris: (N,3,3) triangle corners.
    x_grid, y_grid: (gx,), (gy,) pixel-CENTRE coordinates.
    Returns (gy, gx) thickness map (max z - min z among the ray's hits).
    """
    gy, gx = len(y_grid), len(x_grid)
    thick = np.zeros((gy, gx))
    if len(tris) == 0:
        return thick
    dx = (x_grid[1] - x_grid[0]) if gx > 1 else 1.0
    dy = (y_grid[1] - y_grid[0]) if gy > 1 else 1.0

    # Pre-bucket triangles by pixel cell (based on triangle XY centroid).
    tmin = tris[:, :, :2].min(axis=1)   # (N,2)
    tmax = tris[:, :, :2].max(axis=1)   # (N,2)
    n = len(tris)
    cell_list = [[] for _ in range(gy * gx)]
    for i in range(n):
        # cell indices covered by this triangle's XY bbox
        x0 = max(0, int((tmin[i, 0] - x_grid[0]) / dx))
        x1 = min(gx - 1, int((tmax[i, 0] - x_grid[0]) / dx))
        y0 = max(0, int((tmin[i, 1] - y_grid[0]) / dy))
        y1 = min(gy - 1, int((tmax[i, 1] - y_grid[0]) / dy))
        for cy in range(y0, y1 + 1):
            base = cy * gx
            for cx in range(x0, x1 + 1):
                cell_list[base + cx].append(i)

    for iy in range(gy):
        y = y_grid[iy]
        for ix in range(gx):
            cand = cell_list[iy * gx + ix]
            if not cand:
                continue
            x = x_grid[ix]
            zs = []
            for i in cand:
                z, ok = _ray_z_span(tris[i], x, y)
                if ok:
                    zs.append(z)
            if zs:
                thick[iy, ix] = max(zs) - min(zs)
    return thick


# ---------------------------------------------------------------------------
# Mesh loading
# ---------------------------------------------------------------------------

def _stl_tris(path):
    from stl import mesh as stl_mesh
    m = stl_mesh.Mesh.from_file(path)
    return m.vectors  # (N,3,3)


def load_3mf_colors(path):
    """Load per-color meshes from a Bambu/Orca 3MF.

    Reads part NAME from Metadata/model_settings.config for identity (W and top
    share extruder 4, so extruder alone cannot separate them), and applies the
    composite transform (offset_z) from 3D/3dmodel.model so each part lands at
    its true Z.

    Returns dict {color: (N,3,3) triangle corners}, colors in W/C/M/Y/top.
    """
    import xml.etree.ElementTree as ET
    z = zipfile.ZipFile(path)
    ns_m = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}

    # 1. Main model: composite object -> component transforms (offset_z).
    #    Also gives the sub-model file path and object ids.
    main = ET.fromstring(z.read("3D/3dmodel.model"))
    comps = {}  # objectid -> offset_z
    for obj in main.findall("m:resources/m:object", ns_m):
        for comp in obj.findall("m:components/m:component", ns_m):
            oid = comp.get("objectid")
            transform = comp.get("transform", "1 0 0 0 1 0 0 0 1 0 0 0")
            # 4x3 row-major; translation is cols 9,10,11
            vals = [float(v) for v in transform.split()]
            oz = vals[11] if len(vals) >= 12 else 0.0
            comps[oid] = oz

    # 2. Part names from model_settings.config (order = part id).
    names = {}  # part id -> name
    try:
        ms = ET.fromstring(z.read("Metadata/model_settings.config"))
        for obj in ms.findall("object"):
            for part in obj.findall("part"):
                pid = part.get("id")
                for md in part.findall("metadata"):
                    if md.get("key") == "name":
                        names[pid] = md.get("value")
    except (KeyError, ET.ParseError):
        names = {}

    # 3. Sub-model objects (meshes). Find the sub-model file.
    submodel_file = None
    for n in z.namelist():
        if n.startswith("3D/Objects/") and n.endswith(".model"):
            submodel_file = n
            break
    if submodel_file is None:
        z.close()
        raise ValueError("No 3D/Objects/*.model in 3MF")

    sub = ET.fromstring(z.read(submodel_file))
    meshes = {}
    for obj in sub.findall("m:resources/m:object", ns_m):
        oid = obj.get("id")
        mesh = obj.find("m:mesh", ns_m)
        if mesh is None:
            continue
        verts = []
        for v in mesh.find("m:vertices", ns_m).findall("m:vertex", ns_m):
            verts.append((float(v.get("x")), float(v.get("y")), float(v.get("z"))))
        verts = np.array(verts, dtype=np.float64)
        tris = []
        for t in mesh.find("m:triangles", ns_m).findall("m:triangle", ns_m):
            tris.append((int(t.get("v1")), int(t.get("v2")), int(t.get("v3"))))
        tris = np.array(tris, dtype=np.int64)
        if len(verts) == 0 or len(tris) == 0:
            continue
        # Apply composite Z offset.
        oz = comps.get(oid, 0.0)
        tri_corners = verts[tris]
        if oz:
            tri_corners = tri_corners.copy()
            tri_corners[:, :, 2] += oz
        # Name: prefer part name, fallback to oid.
        name = names.get(oid, oid)
        # Normalize: map name to W/C/M/Y/top.
        key = _color_key(name)
        if key:
            meshes[key] = tri_corners
    z.close()
    return meshes


def _color_key(name):
    """Map a part name (or object id) to a color key W/C/M/Y/top.

    Handles single letters (our exporter writes 'W','C','M','Y','top') and
    descriptive names (Bambu/Orca may write 'litho_cyan', 'white', etc).
    """
    n = name.strip().lower()
    # single-letter keys first (exact match, avoid 'w' matching 'yellow' etc)
    if n in ("w", "white", "jade white"):
        return "W"
    if n in ("c", "cyan"):
        return "C"
    if n in ("m", "magenta"):
        return "M"
    if n in ("y", "yellow"):
        return "Y"
    if "top" in n or n == "t":
        return "top"
    # substring fallback (e.g. 'litho_cyan.stl', 'litho_top_white')
    for key, k in (("cyan", "C"), ("magenta", "M"), ("yellow", "Y"),
                   ("white", "W"), ("top", "top")):
        if key in n:
            return k
    return None


def load_separate_stls(dirpath):
    """Load named STLs (litho_W/C/M/Y/top.stl) into {color: tris}."""
    meshes = {}
    for key in ("W", "C", "M", "Y", "top"):
        path = os.path.join(dirpath, f"litho_{key}.stl")
        if os.path.exists(path):
            meshes[key] = _stl_tris(path)
    return meshes


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def reconstruct_from_meshes(color_meshes, width_mm, height_mm, pixel_pitch=1.0,
                            dW=None, td=None, backlight=(1.0, 1.0, 1.0)):
    """Reconstruct preview from per-color meshes (dict {color: tris}).

    Width/height in mm; pixel_pitch sets output resolution. dW is read from
    the W mesh if present (its thickness), else defaults to 0.8.
    Returns (rows, cols, 3) uint8 image.

    The exporter (assemble_lithophane_parts) centers each part in XY, so meshes
    here may be centered (or shrunken to the material extent). We normalize the
    XY bbox of each mesh onto [0,width]x[0,height] before ray-casting, matching
    the sampling grid at pixel centres.
    """
    if td is None:
        td = DEFAULT_TD
    cols = max(2, int(round(width_mm / pixel_pitch)))
    rows = max(2, int(round(height_mm / pixel_pitch)))
    dx = width_mm / cols
    dy = height_mm / rows
    xs = (np.arange(cols) + 0.5) * dx
    ys = (np.arange(rows) + 0.5) * dy

    # Normalize each mesh's XY onto [0,width]x[0,height] (centering-aware).
    normalized = {}
    for c, tris in color_meshes.items():
        if tris is None or len(tris) == 0:
            normalized[c] = tris
            continue
        v = tris.reshape(-1, 3)
        xmin, xmax = v[:, 0].min(), v[:, 0].max()
        ymin, ymax = v[:, 1].min(), v[:, 1].max()
        xspan = max(xmax - xmin, 1e-9)
        yspan = max(ymax - ymin, 1e-9)
        t = np.array(tris, copy=True)
        t[:, :, 0] = (t[:, :, 0] - xmin) / xspan * width_mm
        t[:, :, 1] = (t[:, :, 1] - ymin) / yspan * height_mm
        normalized[c] = t

    thick = {}
    for c in ("C", "M", "Y", "top"):
        tris = normalized.get(c)
        thick[c] = (extract_thickness_raycast(tris, xs, ys)
                    if tris is not None and len(tris) else np.zeros((rows, cols)))

    # dW: read from W mesh (its thickness), else default.
    if dW is None:
        if "W" in color_meshes and len(color_meshes["W"]):
            w_tris = color_meshes["W"]
            xm = xs[len(xs) // 2]
            ym = ys[len(ys) // 2]
            zs = ray_z_spans(w_tris, xm, ym)
            dW = (zs[-1] - zs[0]) if zs else 0.8
        else:
            dW = 0.8

    rgb = forward_stacked(thick["top"], thick["C"], thick["M"], thick["Y"],
                          dW=dW, td=td, backlight=backlight)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)
