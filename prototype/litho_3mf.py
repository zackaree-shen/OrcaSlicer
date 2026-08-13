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


def build_model_settings(composite_id, part_names, extruders, infill="100%",
                         layer_height="0.2"):
    """Metadata/model_settings.config — per-part filament + 100% infill.

    Mirrors the Bambu reference (lithophane_谢bro_U1.3mf): object-level
    layer_height (object config OVERRIDES the process preset at slice time),
    sparse_infill_density=100%, sparse_infill_pattern=zig-zag, and part-level
    extruder mapping. part_names: list of part names (match submodel object
    order 1..N); extruders: list of extruder ids (1-based) per part.
    """
    out = []
    out.append('<?xml version="1.0" encoding="UTF-8"?>')
    out.append('<config>')
    out.append(f'  <object id="{composite_id}">')
    out.append(f'    <metadata key="name" value="{_xml_escape(part_names[0])}"/>')
    out.append('    <metadata key="extruder" value="0"/>')
    out.append(f'    <metadata key="layer_height" value="{layer_height}"/>')
    out.append(f'    <metadata key="sparse_infill_density" value="{infill}"/>')
    out.append('    <metadata key="sparse_infill_pattern" value="zig-zag"/>')
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


# ---------------------------------------------------------------------------
# Preset templates (filament / process / machine) — extracted from the
# measured Bambu reference 3MF (lithophane_谢bro_U1) and parameterized.
# OrcaSlicer matches presets by *exact* name; the three _settings_1.config
# files provide the presets, and project_settings.config links them via
# filament_settings_id / print_settings_id / printer_settings_id.
# ---------------------------------------------------------------------------

_PRESET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "3mf_templates")


def _load_preset_template(name):
    """Load a JSON preset template (filament/process/machine) from 3mf_templates/."""
    path = os.path.join(_PRESET_DIR, f"{name}_settings_1.config")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _set_keys(d, pairs):
    for k, v in pairs:
        if k in d:
            d[k] = v


def build_filament_settings(preset_suffix="", filament_type="PLA",
                            filament_vendor="Generic",
                            filament_colours=None):
    """Metadata/filament_settings_1.config — full PLA preset (from reference)."""
    d = _load_preset_template("filament")
    suffix = f"({preset_suffix})" if preset_suffix else ""
    n = 5 if filament_colours is None else len(filament_colours)
    sid = [f"{filament_vendor} {filament_type}{suffix}"] * n
    _set_keys(d, [
        ("name", f"{filament_vendor} {filament_type}{suffix}"),
        ("filament_settings_id", sid),
        ("filament_type", [filament_type] * n),
        ("filament_vendor", [filament_vendor] * n),
    ])
    if filament_colours is not None:
        d["default_filament_colour"] = list(filament_colours)
    return json.dumps(d, indent=4, ensure_ascii=False)


def build_process_settings(preset_suffix="", layer_height="0.2",
                           sparse_infill_density="100%",
                           printer_model="Snapmaker U1",
                           printer_variant="0.4"):
    """Metadata/process_settings_1.config — full process preset (from reference).

    The reference preset targets BBL A1; we REWRITE compatible_printers to the
    target machine so OrcaSlicer applies it (exact-name matching). Without
    this the process preset is judged incompatible and silently dropped.
    """
    d = _load_preset_template("process")
    suffix = f"({preset_suffix})" if preset_suffix else ""
    machine = f"{printer_model} ({printer_variant} nozzle)"
    _set_keys(d, [
        ("name", f"0.20mm Standard @{machine}{suffix}"),
        ("print_settings_id", f"0.20mm Standard @{machine}{suffix}"),
        ("compatible_printers", [machine]),
        ("compatible_printers_condition", ""),
        ("layer_height", str(layer_height)),
        ("initial_layer_print_height", str(layer_height)),
        ("sparse_infill_density", sparse_infill_density),
    ])
    return json.dumps(d, indent=4, ensure_ascii=False)


def build_machine_settings(preset_suffix="", printer_model="Snapmaker U1",
                           printer_variant="0.4",
                           printer_settings_id=None):
    """Metadata/machine_settings_1.config — full machine preset (from reference).

    printer_settings_id must MATCH project_settings.config exactly: we use the
    system-preset name 'Snapmaker U1 (0.4 nozzle)' (parenthesized, matches the
    installed preset) so the machine is selected on import. (The reference is
    internally inconsistent here — its embedded machine preset is dead code.)
    """
    d = _load_preset_template("machine")
    suffix = f"({preset_suffix})" if preset_suffix else ""
    if printer_settings_id is None:
        printer_settings_id = f"{printer_model} ({printer_variant} nozzle)"
    machine = f"{printer_model} ({printer_variant} nozzle)"
    _set_keys(d, [
        ("name", f"{printer_model} {printer_variant} nozzle{suffix}"),
        ("printer_model", printer_model),
        ("printer_variant", printer_variant),
        ("printer_settings_id", printer_settings_id),
        ("default_print_profile", f"0.20mm Standard @{machine}{suffix}"),
    ])
    return json.dumps(d, indent=4, ensure_ascii=False)


def build_project_settings(filament_colours=None, filament_types=None,
                           sparse_infill_density="100%", layer_height="0.2",
                           printer_model="Snapmaker U1", printer_variant="0.4",
                           printer_settings_id=None, filament_vendor="Generic",
                           preset_suffix="", filament_settings_id=None,
                           print_settings_id=None):
    """Metadata/project_settings.config — print-settings JSON.

    Contains the preset-name links that make OrcaSlicer select the machine /
    process / filament presets: printer_settings_id (machine), print_settings_id
    (process), filament_settings_id (list, one per AMS slot), plus
    print_compatible_printers so the process preset's own compatible_printers
    is overridden (B1: without this, the process preset is dropped for U1).
    """
    if filament_colours is None:
        # Reference slot order: slot1=white(W), slot2=cyan(C), slot3=magenta(M),
        # slot4=yellow(Y), slot5=black(support). Matches extruder map W=1,C=2,
        # M=3,Y=4,top=1.
        filament_colours = ["#FFFFFF", "#0086D6", "#EC008C", "#F4EE2A", "#222222"]
    if filament_types is None:
        filament_types = ["PLA", "PLA", "PLA", "PLA", "PLA"]
    n = len(filament_colours)
    if printer_settings_id is None:
        printer_settings_id = f"{printer_model} ({printer_variant} nozzle)"
    suffix = f"({preset_suffix})" if preset_suffix else ""
    if filament_settings_id is None:
        filament_settings_id = [f"{filament_vendor} {filament_types[0]}{suffix}"] * n
    if print_settings_id is None:
        print_settings_id = f"0.20mm Standard @{printer_model} ({printer_variant} nozzle){suffix}"
    settings = {
        "filament_colour": list(filament_colours),
        "filament_type": list(filament_types),
        "filament_vendor": [filament_vendor] * n,
        "filament_settings_id": list(filament_settings_id),
        "print_settings_id": print_settings_id,
        "print_compatible_printers": [printer_settings_id],
        "sparse_infill_density": sparse_infill_density,
        "layer_height": str(layer_height),
        "initial_layer_print_height": str(layer_height),
        "printer_model": printer_model,
        "printer_variant": printer_variant,
        "printer_settings_id": printer_settings_id,
        "printer_technology": "FFF",
    }
    return json.dumps(settings, indent=4)


def write_3mf(path, parts, offsets_mm, extruders, filament_colours=None,
              filament_types=None, sparse_infill_density="100%",
              layer_height="0.2", composite_id=6, part_names=None,
              printer_model="Snapmaker U1", printer_variant="0.4",
              printer_settings_id=None, filament_vendor="Generic",
              build_center_mm=None, preset_suffix=""):
    """Write a complete composite 3MF.

    parts:        list of (vertices, faces) per part (submodel objects 1..N)
    offsets_mm:   list of (x, y, z) component translations (absolute placement)
    extruders:    list of 1-based extruder ids per part
    filament_colours: list of hex colours for the AMS (index = extruder-1)
    part_names:   display names per part (default: part_1..part_N)
    printer_model / printer_variant / printer_settings_id / filament_vendor:
                  written into the preset configs so OrcaSlicer matches presets.
    preset_suffix: e.g. project name -> '(suffix)' appended to the preset ids.
    build_center_mm: (cx, cy, z0) placed into the build item transform so the
                  model lands centered on the plate. If None, build transform
                  is identity (OrcaSlicer centers on import).

    Writes the full 5-file preset set (filament/process/machine/project/model
    settings) mirroring the measured Bambu reference 3MF.
    """
    if part_names is None:
        part_names = [f"part_{i}" for i in range(1, len(parts) + 1)]
    if filament_colours is None:
        # Reference slots: 1=W(white), 2=C(cyan), 3=M(magenta), 4=Y(yellow),
        # 5=support(black). Default map W=1,C=2,M=3,Y=4,top=1 -> 5 slots.
        filament_colours = ["#FFFFFF", "#0086D6", "#EC008C", "#F4EE2A", "#222222"]

    component_ids = list(range(1, len(parts) + 1))

    submodel = build_submodel(list(zip(part_names, *zip(*parts))))
    mainmodel = build_mainmodel(composite_id, component_ids, offsets_mm, part_names,
                                build_center_mm=build_center_mm)
    model_settings = build_model_settings(composite_id, part_names, extruders,
                                          sparse_infill_density, layer_height)
    project_settings = build_project_settings(
        filament_colours, filament_types, sparse_infill_density, layer_height,
        printer_model=printer_model, printer_variant=printer_variant,
        printer_settings_id=printer_settings_id, filament_vendor=filament_vendor,
        preset_suffix=preset_suffix)
    filament_settings = build_filament_settings(
        preset_suffix, filament_vendor=filament_vendor,
        filament_colours=filament_colours)
    process_settings = build_process_settings(
        preset_suffix, layer_height, sparse_infill_density,
        printer_model, printer_variant)
    machine_settings = build_machine_settings(
        preset_suffix, printer_model, printer_variant, printer_settings_id)

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", ROOT_RELS)
        z.writestr("3D/3dmodel.model", mainmodel)
        z.writestr("3D/_rels/3dmodel.model.rels", MODEL_RELS)
        z.writestr("3D/Objects/litho.stl_1.model", submodel)
        z.writestr("Metadata/model_settings.config", model_settings)
        z.writestr("Metadata/project_settings.config", project_settings)
        z.writestr("Metadata/filament_settings_1.config", filament_settings)
        z.writestr("Metadata/process_settings_1.config", process_settings)
        z.writestr("Metadata/machine_settings_1.config", machine_settings)
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
        # Bambu reference mapping (measured lithophane_谢bro_U1 model_settings):
        # W texture=1, margin=1, C=2, M=3, Y=4. Our 'top' is white like W, so it
        # shares extruder 1 (reference has no separate top part).
        filament_map = {"W": 1, "C": 2, "M": 3, "Y": 4, "top": 1}

    parts, offsets, names, extruders = [], [], [], []
    for color in order:
        verts, faces = meshes[color]
        if len(verts) == 0 or len(faces) == 0:
            continue
        # Center XY on the origin. For Z: keep each part's ORIGINAL base height
        # as the composite offset and normalize the local Z to start at 0. This
        # preserves per-part absolute Z placement: LAYERED bands stay disjoint
        # (offsets 0.85/1.54/...), while INTERLEAVED/OVERLAP C/M/Y all share the
        # same base offset so their boxes genuinely overlap in Z — which the
        # slicer's clip_multipart_objects resolves by part order. (Adversarial
        # finding: stripping Z entirely collapsed the overlap.)
        verts = np.asarray(verts, dtype=np.float64)
        local = np.array(verts, copy=True)
        cx = 0.5 * (local[:, 0].min() + local[:, 0].max())
        cy = 0.5 * (local[:, 1].min() + local[:, 1].max())
        zbase = float(local[:, 2].min())
        local[:, 0] -= cx
        local[:, 1] -= cy
        local[:, 2] -= zbase
        parts.append((local, faces))
        offsets.append((0.0, 0.0, zbase))
        names.append(color)
        extruders.append(filament_map[color])
    return parts, offsets, names, extruders
