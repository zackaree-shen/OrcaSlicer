"""Snapmaker 3MF exporter for the lithophane generator.

Produces a BambuStudio/OrcaSlicer-compatible 3MF with:
  - a single composite object (one object, multiple parts/volumes), each part
    being one color layer (white / cyan / magenta / yellow / top_white)
  - per-part filament mapping via Metadata/model_settings.config
    (part-level `extruder`), object-level `sparse_infill_density=100%`
  - the full print-settings JSON in Metadata/project_settings.config

File layout mirrors what BambuStudio writes (see
C:\\Users\\snapmaker\\Downloads\\interleaved_mixed_Test.3mf as reference):

  [Content_Types].xml
  _rels/.rels
  3D/3dmodel.model                 <- composite object + build
  3D/_rels/3dmodel.model.rels
  3D/Objects/litho.stl_1.model     <- the 5 part meshes (objects 1..5)
  Metadata/model_settings.config   <- part extruders + 100% infill
  Metadata/project_settings.config <- print settings JSON
"""

from __future__ import annotations

import json
import os
import uuid
import zipfile
from typing import Iterable

import numpy as np

# ---------------------------------------------------------------------------
# XML helpers (escape-only, we control all values)
# ---------------------------------------------------------------------------

def _xml_escape(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _fmt(x):
    """Format a float the way BambuStudio does (shortest round-trip)."""
    return f"{x:.6g}"


# ---------------------------------------------------------------------------
# 3MF writer
# ---------------------------------------------------------------------------

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
 <Default Extension="png" ContentType="image/png"/>
 <Default Extension="gcode" ContentType="text/x.gcode"/>
</Types>"""

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>"""

MODEL_RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/Objects/litho.stl_1.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>"""


def _mesh_xml(vertices, faces, indent="   "):
    """Serialize a (V,3)/(F,3) mesh to 3MF <mesh> XML. Triangles CCW."""
    lines = []
    lines.append(indent + "<mesh>")
    lines.append(indent + "  <vertices>")
    for x, y, z in vertices:
        lines.append(indent + f'   <vertex x="{_fmt(x)}" y="{_fmt(y)}" z="{_fmt(z)}"/>')
    lines.append(indent + "  </vertices>")
    lines.append(indent + "  <triangles>")
    for a, b, c in faces:
        lines.append(indent + f'   <triangle v1="{int(a)}" v2="{int(b)}" v3="{int(c)}"/>')
    lines.append(indent + "  </triangles>")
    lines.append(indent + "</mesh>")
    return "\n".join(lines)


def build_submodel(parts):
    """parts: list of (name, vertices, faces). Writes 3D/Objects/litho.stl_1.model.

    Each part becomes one <object> with a unique id (1..N). Meshes are in the
    coordinate system of the part itself (absolute coords as generated); the
    composite object positions them via component transforms.
    """
    out = []
    out.append('<?xml version="1.0" encoding="UTF-8"?>')
    out.append('<model unit="millimeter" xml:lang="en-US" '
               'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
               'xmlns:BambuStudio="http://schemas.bambulab.com/package/2021" '
               'xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" '
               'requiredextensions="p">')
    out.append(' <metadata name="BambuStudio:3mfVersion">1</metadata>')
    out.append(' <resources>')
    for i, (name, vertices, faces) in enumerate(parts, start=1):
        out.append(f'  <object id="{i}" p:UUID="{str(uuid.uuid4()).upper()}" type="model">')
        out.append(_mesh_xml(vertices, faces))
        out.append('  </object>')
    out.append(' </resources>')
    out.append('</model>')
    return "\n".join(out)


def build_mainmodel(composite_id, component_ids, offsets_mm, part_names,
                    build_center_mm=None):
    """composite_id: object id of the composite (e.g. 6).
    component_ids: list of object ids referenced.
    offsets_mm: list of (x,y,z) translation per component (part placed at).
    part_names: list of names (for metadata only).
    build_center_mm: (cx, cy, z0) placed into the build item transform so the
        model is centered on the plate. None -> identity (OrcaSlicer centers)."""
    out = []
    out.append('<?xml version="1.0" encoding="UTF-8"?>')
    out.append('<model unit="millimeter" xml:lang="en-US" '
               'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
               'xmlns:BambuStudio="http://schemas.bambulab.com/package/2021" '
               'xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" '
               'requiredextensions="p">')
    out.append(' <metadata name="Application">OrcaSlicer</metadata>')
    out.append(' <metadata name="BambuStudio:3mfVersion">1</metadata>')
    out.append(' <resources>')
    out.append(f'  <object id="{composite_id}" p:UUID="{str(uuid.uuid4()).upper()}" type="model">')
    out.append('   <components>')
    for cid, (ox, oy, oz), name in zip(component_ids, offsets_mm, part_names):
        # 3MF component transform: 4x3 matrix in row-major (rotation+translation).
        out.append(f'    <component p:path="/3D/Objects/litho.stl_1.model" '
                   f'objectid="{cid}" p:UUID="{str(uuid.uuid4()).upper()}" '
                   f'transform="1 0 0 0 1 0 0 0 1 {_fmt(ox)} {_fmt(oy)} {_fmt(oz)}"/>')
    out.append('   </components>')
    out.append('  </object>')
    out.append(' </resources>')
    if build_center_mm is None:
        bx, by, bz = 0.0, 0.0, 0.0
    else:
        bx, by, bz = build_center_mm
    out.append(f' <build p:UUID="{str(uuid.uuid4()).upper()}">')
    out.append(f'  <item objectid="{composite_id}" p:UUID="{str(uuid.uuid4()).upper()}" '
               f'transform="1 0 0 0 1 0 0 0 1 {_fmt(bx)} {_fmt(by)} {_fmt(bz)}" printable="1"/>')
    out.append(' </build>')
    out.append('</model>')
    return "\n".join(out)


def build_model_settings(composite_id, part_names, extruders, infill="100%"):
    """Metadata/model_settings.config — per-part filament + 100% infill.

    part_names: list of part names (match submodel object order 1..N).
    extruders: list of extruder ids (1-based) per part.
    """
    out = []
    out.append('<?xml version="1.0" encoding="UTF-8"?>')
    out.append('<config>')
    out.append(f'  <object id="{composite_id}">')
    out.append(f'    <metadata key="name" value="{_xml_escape(part_names[0])}"/>')
    out.append('    <metadata key="extruder" value="0"/>')
    out.append(f'    <metadata key="sparse_infill_density" value="{infill}"/>')
    for i, (name, extruder) in enumerate(zip(part_names, extruders), start=1):
        out.append(f'    <part id="{i}" subtype="normal_part">')
        out.append(f'      <metadata key="name" value="{_xml_escape(name)}"/>')
        # identity matrix (parts already placed by composite transform)
        out.append('      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>')
        out.append(f'      <metadata key="extruder" value="{extruder}"/>')
        out.append('      <mesh_stat edges_fixed="0" degenerate_facets="0" facets_removed="0" facets_reversed="0" backwards_edges="0"/>')
        out.append('    </part>')
    out.append('  </object>')
    out.append('</config>')
    return "\n".join(out)


def build_project_settings(filament_colours=None, filament_types=None,
                           sparse_infill_density="100%", layer_height="0.08",
                           printer_model="Snapmaker U1", printer_variant="0.4",
                           printer_settings_id=None, filament_vendor="Snapmaker"):
    """Metadata/project_settings.config — print-settings JSON (minimal but valid).
    OrcaSlicer reads filament_colour for the AMS color display and
    printer_model / printer_variant for machine matching."""
    if filament_colours is None:
        filament_colours = ["#0080FF", "#FF0080", "#FFFF00", "#FFFFFF"]
    if filament_types is None:
        filament_types = ["PLA", "PLA", "PLA", "PLA"]
    if printer_settings_id is None:
        printer_settings_id = f"{printer_model} ({printer_variant} nozzle)"
    settings = {
        "filament_colour": list(filament_colours),
        "filament_type": list(filament_types),
        "filament_vendor": [filament_vendor] * len(filament_types),
        "sparse_infill_density": sparse_infill_density,
        "layer_height": layer_height,
        "initial_layer_print_height": "0.2",
        "printer_model": printer_model,
        "printer_variant": printer_variant,
        "printer_settings_id": printer_settings_id,
        "printer_technology": "FFF",
    }
    return json.dumps(settings, indent=4)


def write_3mf(path, parts, offsets_mm, extruders, filament_colours=None,
              filament_types=None, sparse_infill_density="100%",
              layer_height="0.08", composite_id=6, part_names=None,
              printer_model="Snapmaker U1", printer_variant="0.4",
              printer_settings_id=None, filament_vendor="Snapmaker",
              build_center_mm=None):
    """Write a complete composite 3MF.

    parts:        list of (vertices, faces) per part (submodel objects 1..N)
    offsets_mm:   list of (x, y, z) component translations (absolute placement)
    extruders:    list of 1-based extruder ids per part
    filament_colours: list of hex colours for the AMS (index = extruder-1)
    part_names:   display names per part (default: part_1..part_N)
    printer_model / printer_variant / printer_settings_id / filament_vendor:
                  written into Metadata/project_settings.config so OrcaSlicer
                  can match the machine preset.
    build_center_mm: (cx, cy, z0) placed into the build item transform so the
                  model lands centered on the plate. If None, build transform
                  is identity (OrcaSlicer centers on import).
    """
    if part_names is None:
        part_names = [f"part_{i}" for i in range(1, len(parts) + 1)]
    if filament_colours is None:
        n = max(max(extruders), 4) if extruders else 4
        filament_colours = ["#0080FF", "#FF0080", "#FFFF00", "#FFFFFF"][:n]

    component_ids = list(range(1, len(parts) + 1))

    submodel = build_submodel(list(zip(part_names, *zip(*parts))))
    mainmodel = build_mainmodel(composite_id, component_ids, offsets_mm, part_names,
                                build_center_mm=build_center_mm)
    model_settings = build_model_settings(composite_id, part_names, extruders,
                                          sparse_infill_density)
    project_settings = build_project_settings(
        filament_colours, filament_types, sparse_infill_density, layer_height,
        printer_model=printer_model, printer_variant=printer_variant,
        printer_settings_id=printer_settings_id, filament_vendor=filament_vendor)

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", ROOT_RELS)
        z.writestr("3D/3dmodel.model", mainmodel)
        z.writestr("3D/_rels/3dmodel.model.rels", MODEL_RELS)
        z.writestr("3D/Objects/litho.stl_1.model", submodel)
        z.writestr("Metadata/model_settings.config", model_settings)
        z.writestr("Metadata/project_settings.config", project_settings)
    return path


# ---------------------------------------------------------------------------
# Convenience: assemble the 5 lithophane parts with proper stacking
# ---------------------------------------------------------------------------

def assemble_lithophane_parts(meshes, order=None, filament_map=None):
    """Convert the engine's meshes dict into 3MF parts + offsets + extruders.

    meshes: dict {color: (vertices, faces)} from color_lithophane_engine.
    order:  list of color keys in stacking order (bottom to top). Default
            ["W","C","M","Y","top"].
    filament_map: dict color -> extruder id. Default:
            W=4, C=1, M=2, Y=3, top=4  (white shared by W and top).
    Returns (parts, offsets_mm, part_names, extruders).

    All parts are XY-centered around the origin (each part's XY centroid is
    shifted so the printed plate is centered on the bed), and stacked along Z
    from z=0 upward with no overlap.
    """
    if order is None:
        order = ["W", "C", "M", "Y", "top"]
    if filament_map is None:
        filament_map = {"W": 4, "C": 1, "M": 2, "Y": 3, "top": 4}

    parts, offsets, names, extruders = [], [], [], []
    z_cursor = 0.0
    for color in order:
        verts, faces = meshes[color]
        if len(verts) == 0 or len(faces) == 0:
            continue
        # Normalize per part: shift XY to center on origin, and Z to start at 0.
        # (The engine emits absolute Z; we strip the Z offset and let the
        # composite transform stack the parts.)
        verts = np.asarray(verts, dtype=np.float64)
        local = np.array(verts, copy=True)
        cx = 0.5 * (local[:, 0].min() + local[:, 0].max())
        cy = 0.5 * (local[:, 1].min() + local[:, 1].max())
        zmin = float(local[:, 2].min())
        local[:, 0] -= cx
        local[:, 1] -= cy
        local[:, 2] -= zmin
        parts.append((local, faces))
        offsets.append((0.0, 0.0, z_cursor))
        names.append(color)
        extruders.append(filament_map[color])
        z_cursor += float(local[:, 2].max())
    return parts, offsets, names, extruders
