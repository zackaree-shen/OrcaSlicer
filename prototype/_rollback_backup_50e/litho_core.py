"""Lithophane prototype - core algorithm.

Faithful Python port of src/libslic3r/Lithophane.cpp (M1, monochrome greyscale).

Conventions identical to the C++ implementation:
  - thickness(x,y) = base_thickness + (1 - brightness/255) * depth_range
    dark pixel -> thick (lets less backlight through)
    white pixel -> thin  (lets most backlight through)
  - bilinear sampling from the greyscale image into the output grid
  - solid closed by front height-field (z = thickness), back plane (z = 0)
    and 4 side walls; all faces wound CCW outward so the signed volume is
    positive (verified by validate_mesh).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class LithophaneParams:
    width_mm: float = 144.0       # Bambu standard frame size
    height_mm: float = 108.0      # Bambu standard frame size
    base_thickness: float = 0.8   # thinnest point (white), v1 coarse default
    depth_range: float = 2.0      # extra thickness range (black = base+depth)
    pixel_pitch_mm: float = 0.2   # mm per output grid step
    mirror: bool = False


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_greyscale(path: str) -> np.ndarray:
    """Read an image as 8-bit greyscale (H, W) uint8."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Thickness map
# ---------------------------------------------------------------------------

def thickness_grid_shape(img_rows, img_cols, params):
    """Output grid resolution (gy, gx) for the given image and params."""
    pitch = max(params.pixel_pitch_mm, 1e-3)
    gx = max(2, int(round(params.width_mm / pitch)) + 1)
    gy = max(2, int(round(params.height_mm / pitch)) + 1)
    return gx, gy


def thickness_map(grey: np.ndarray, params: LithophaneParams) -> np.ndarray:
    """Map greyscale to thickness in mm with bilinear sampling.

    Output grid has (gy, gx) points where gx = round(width/pitch)+1 etc.
    Returns a (gy, gx) float array of thickness in mm.
    """
    cols = grey.shape[1]
    rows = grey.shape[0]
    if cols == 0 or rows == 0:
        raise ValueError("empty image")

    gx, gy = thickness_grid_shape(rows, cols, params)

    # Grid coordinates in image space.
    ix = np.linspace(0.0, cols - 1.0, gx)
    iy = np.linspace(0.0, rows - 1.0, gy)
    if params.mirror:
        ix = cols - 1.0 - ix
    X, Y = np.meshgrid(ix, iy)  # (gy, gx)

    # Bilinear interpolation of brightness.
    x0 = np.floor(X).astype(np.int64)
    y0 = np.floor(Y).astype(np.int64)
    x1 = np.minimum(x0 + 1, cols - 1)
    y1 = np.minimum(y0 + 1, rows - 1)
    fx = (X - x0).astype(np.float64)
    fy = (Y - y0).astype(np.float64)

    b00 = grey[y0, x0].astype(np.float64)
    b10 = grey[y0, x1].astype(np.float64)
    b01 = grey[y1, x0].astype(np.float64)
    b11 = grey[y1, x1].astype(np.float64)

    top = b00 * (1 - fx) + b10 * fx
    bottom = b01 * (1 - fx) + b11 * fx
    bright = top * (1 - fy) + bottom * fy  # 0..1, 1 = white

    return params.base_thickness + (1.0 - bright / 255.0) * params.depth_range


# ---------------------------------------------------------------------------
# Mesh construction (identical winding to Lithophane.cpp)
# ---------------------------------------------------------------------------

def build_mesh(grey: np.ndarray, params: LithophaneParams):
    """Build the closed solid mesh.

    Returns (vertices, faces):
      vertices: (N, 3) float array
      faces:    (M, 3) int array of vertex indices, CCW outward.
    """
    h = thickness_map(grey, params)
    return heightfield_to_mesh(h, params)


def heightfield_to_mesh(heights, params, z_offset=0.0):
    """Build a closed solid mesh from a thickness map.

    heights: (gy, gx) float array of thickness in mm (>= 0). The solid is
    closed by front height-field (z = z_offset + heights), back plane
    (z = z_offset) and four side walls; all faces CCW outward so signed volume
    is positive. z_offset shifts the whole slab up so stacked layers each live
    in their own Z band (strict Z separation).

    Returns (vertices, faces):
      vertices: (N, 3) float array
      faces:    (M, 3) int array of vertex indices, CCW outward.
    """
    h = np.asarray(heights, dtype=np.float64)
    bottom = np.full_like(h, z_offset)
    top = h + z_offset
    return heightfield_band_mesh(bottom, top, params)


def heightfield_band_mesh(bottom, top, params):
    """Build a closed solid mesh between two height-fields.

    bottom, top: (gy, gx) float arrays of z height in mm (bottom <= top).
    The solid is closed by the top surface (z = top), the bottom surface
    (z = bottom) and four side walls along the grid boundary. All faces CCW
    outward so signed volume is positive. This generalizes heightfield_to_mesh
    (constant bottom) to a variable bottom — used e.g. for a top relief layer
    whose underside follows the actual C/M/Y fill height underneath, so the
    layer never floats in air.

    Returns (vertices, faces).
    """
    bottom = np.asarray(bottom, dtype=np.float64)
    top = np.asarray(top, dtype=np.float64)
    gy, gx = top.shape
    assert bottom.shape == (gy, gx), "bottom and top must have same shape"

    dx = params.width_mm / float(gx - 1)
    dy = params.height_mm / float(gy - 1)

    xs = np.arange(gx, dtype=np.float64) * dx
    ys = np.arange(gy, dtype=np.float64) * dy
    X, Y = np.meshgrid(xs, ys)          # (gy, gx)

    # Top surface vertices (row-major, iy*gx+ix), z = top.
    front = np.stack([X, Y, top], axis=-1).reshape(-1, 3)
    # Bottom surface vertices, z = bottom.
    back = np.stack([X, Y, bottom], axis=-1).reshape(-1, 3)
    vertices = np.vstack([front, back])  # (2*gx*gy, 3)

    def _quads(a, b, c, d):
        t1 = np.stack([a, b, c], axis=-1)
        t2 = np.stack([a, c, d], axis=-1)
        return np.concatenate([t1, t2], axis=0).reshape(-1, 3)

    iy = np.arange(gy - 1)[:, None]
    ix = np.arange(gx - 1)[None, :]
    a = iy * gx + ix
    b = a + 1
    c = a + gx + 1
    d = a + gx
    back_base = gy * gx
    ba, bb, bc, bd = a + back_base, b + back_base, c + back_base, d + back_base

    parts = []
    # Top surface, normal +Z.
    parts.append(_quads(a, b, c, d))
    # Bottom surface, normal -Z (reverse winding).
    parts.append(_quads(ba, bd, bc, bb))
    # Side walls: same boundary strips as the constant-bottom case; the vertex
    # indices are identical, only the bottom vertices' z varies with `bottom`.
    xe = np.arange(gx - 1)
    parts.append(_quads(xe, back_base + xe, back_base + xe + 1, xe + 1))  # y=0, -Y
    atop = (gy - 1) * gx + xe
    parts.append(_quads(atop, atop + 1, back_base + atop + 1, back_base + atop))  # y=gy-1, +Y
    ye = np.arange(gy - 1) * gx
    parts.append(_quads(ye, ye + gx, back_base + ye + gx, back_base + ye))  # x=0, -X
    aright = ye + (gx - 1)
    parts.append(_quads(aright, back_base + aright, back_base + aright + gx, aright + gx))  # x=gx-1, +X

    faces = np.concatenate(parts, axis=0).astype(np.int64)
    return vertices, faces


# Validation
# ---------------------------------------------------------------------------

def validate_mesh(vertices, faces, eps: float = 1e-5):
    """Verify the mesh is a closed, outward-oriented solid.

    Returns dict with:
      volume: signed volume (must be > 0)
      open_edges: number of edges shared by != 2 triangles (0 = watertight)
      degenerate: number of zero-area triangles
    """
    # Signed volume via the divergence theorem.
    tri = vertices[faces]  # (M,3,3)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    cross = np.cross(b - a, c - a)
    # Volume = (1/6) sum(dot(a, cross(b-a, c-a))) for CCW-oriented triangles.
    volume = float(np.sum(np.einsum("ij,ij->i", a, cross))) / 6.0

    # Edge uniqueness: each directed edge must appear exactly once (one triangle
    # contributes (a,b),(b,c),(c,a) all in consistent outward orientation).
    edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ])  # (3M, 2)
    directed = (edges[:, 0] * (vertices.shape[0] + 1) + edges[:, 1]).astype(np.int64)
    uniq, counts = np.unique(directed, return_counts=True)
    open_edges = int(np.sum(counts != 1))

    # Degenerate triangles.
    area = 0.5 * np.linalg.norm(cross, axis=1)
    degenerate = int(np.sum(area < eps))

    return {
        "volume": volume,
        "open_edges": open_edges,
        "degenerate": degenerate,
        "num_vertices": int(vertices.shape[0]),
        "num_faces": int(faces.shape[0]),
    }


# ---------------------------------------------------------------------------
# STL export
# ---------------------------------------------------------------------------

def export_stl(path: str, vertices, faces, name: str = "lithophane"):
    """Write a binary STL. Uses numpy-stl if available, else a minimal writer."""
    try:
        from stl import mesh as stl_mesh
        tri = vertices[faces]  # (M,3,3)
        m = stl_mesh.Mesh(np.zeros(tri.shape[0], dtype=stl_mesh.Mesh.dtype))
        for i in range(3):
            m.vectors[:, i, :] = tri[:, i, :]
        m.save(path)
        return "binary"
    except ImportError:
        # Minimal binary STL writer (IEEE little-endian floats).
        tri = vertices[faces].astype("<f4")
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nlen = np.linalg.norm(normals, axis=1, keepdims=True)
        nlen[nlen == 0] = 1.0
        normals = normals / nlen
        with open(path, "wb") as f:
            header = name.encode()[:80].ljust(80, b"\0")
            f.write(header)
            f.write(np.uint32(tri.shape[0]))
            for n, t in zip(normals.astype("<f4"), tri):
                f.write(n.tobytes())
                f.write(t.tobytes())
                f.write(np.uint16(0).tobytes())
        return "binary-minimal"
