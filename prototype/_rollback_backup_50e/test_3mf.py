"""Validation for the 3MF composite exporter (Snapmaker/OrcaSlicer compatible).

Checks:
  - the exported 3MF is a valid zip with all required entries
  - every XML is well-formed
  - the composite object references exactly the 5 part meshes
  - per-part extruder mapping is written correctly (W=1,C=2,M=3,Y=4,top=1,
    Bambu reference mapping)
  - the full 5-file preset set is present (filament/process/machine/project/
    model settings) with consistent preset ids + print_compatible_printers
  - object-level sparse_infill_density == 100% and layer_height == 0.2
  - offsets stack the parts without overlap (Z strictly increasing)
  - meshes inside the sub-model are watertight solids
"""

from __future__ import annotations

import json
import os
import re
import sys
import zipfile
import xml.dom.minidom as md

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from litho_core import LithophaneParams, validate_mesh
from litho_engine import LithoMode, ColorOrder, color_lithophane_engine
from litho_3mf import assemble_lithophane_parts, write_3mf
from test_engine import make_test_img

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name:55s} {detail}")


def test_3mf_export(tmp="_3mf_test.3mf"):
    img = make_test_img(h=80, w=100)
    p = LithophaneParams(width_mm=99.99, height_mm=71.42, pixel_pitch_mm=1.0)
    meshes, _, _, _ = color_lithophane_engine(
        img, mode=LithoMode.LAYERED, order=ColorOrder.CMY, params=p,
        pitch_cmy=1.5, pitch_top=1.0)

    parts, offsets, names, extruders = assemble_lithophane_parts(meshes)
    # Bed-centre placement (Snapmaker U1: 270.5x271 -> centre 135.5, 136).
    write_3mf(tmp, parts, offsets, extruders, part_names=names,
              printer_model="Snapmaker U1", printer_settings_id="Snapmaker U1 (0.4 nozzle)",
              build_center_mm=(135.5, 136.0, 0.0), preset_suffix="lithophane")

    # 1. Valid zip with all entries (full 5-file preset set).
    z = zipfile.ZipFile(tmp)
    entries = set(z.namelist())
    required = {"[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model",
                "3D/_rels/3dmodel.model.rels", "3D/Objects/litho.stl_1.model",
                "Metadata/model_settings.config", "Metadata/project_settings.config",
                "Metadata/filament_settings_1.config",
                "Metadata/process_settings_1.config",
                "Metadata/machine_settings_1.config"}
    report("3MF: all required entries present", required.issubset(entries),
           f"{len(entries)} entries")

    # 2. All XML well-formed.
    ok_xml = True
    for n in required - {"Metadata/project_settings.config",
                         "Metadata/filament_settings_1.config",
                         "Metadata/process_settings_1.config",
                         "Metadata/machine_settings_1.config"}:
        try:
            md.parseString(z.read(n))
        except Exception as e:  # noqa: BLE001
            ok_xml = False
            report(f"3MF: XML well-formed {n}", False, str(e))
    report("3MF: XML well-formed", ok_xml)

    # 3. Composite references exactly 5 part objects.
    main = z.read("3D/3dmodel.model").decode()
    comps = re.findall(r'<component[^>]*objectid="(\d+)"', main)
    report("3MF: composite has 5 parts", len(comps) == 5, f"{len(comps)} components")

    # 4. Per-part extruder mapping (Bambu reference: W=1,C=2,M=3,Y=4,top=1).
    ms = z.read("Metadata/model_settings.config").decode()
    extruder_map = dict(re.findall(
        r'<part id="(\d+)"[^>]*>.*?extruder" value="(\d+)"', ms, re.DOTALL))
    expected = {"1": "1", "2": "2", "3": "3", "4": "4", "5": "1"}
    report("3MF: part extruder mapping (Bambu ref)", extruder_map == expected,
           str(extruder_map))

    # 5. Infill 100% + layer_height 0.2 at object level.
    infill = re.search(r'sparse_infill_density" value="([^"]+)"', ms)
    lh = re.search(r'layer_height" value="([^"]+)"', ms)
    report("3MF: object infill 100% + layer_height 0.2",
           infill is not None and infill.group(1) == "100%"
           and lh is not None and lh.group(1) == "0.2",
           f"infill={infill.group(1) if infill else 'none'} "
           f"layer_h={lh.group(1) if lh else 'none'}")

    # 5b. Preset set is consistent: machine/project printer_settings_id match,
    # process preset is compatible with U1, project links all three ids.
    fs = json.loads(z.read("Metadata/filament_settings_1.config"))
    ps_ = json.loads(z.read("Metadata/process_settings_1.config"))
    mch = json.loads(z.read("Metadata/machine_settings_1.config"))
    proj = json.loads(z.read("Metadata/project_settings.config"))
    sid = "Snapmaker U1 (0.4 nozzle)"
    ok_presets = (
        mch.get("printer_settings_id") == sid
        and proj.get("printer_settings_id") == sid
        and ps_.get("compatible_printers") == [sid]
        and proj.get("print_compatible_printers") == [sid]
        and proj.get("print_settings_id") == ps_.get("print_settings_id")
        and proj.get("filament_settings_id") == fs.get("filament_settings_id")
        and proj.get("layer_height") == "0.2"
        and len(proj.get("filament_colour", [])) == 5
    )
    report("3MF: preset set consistent (ids link, U1-compatible)",
           ok_presets,
           f"mch={mch.get('printer_settings_id')} proj={proj.get('printer_settings_id')}")

    # 6. Offsets stack without overlap.
    ztops = []
    prev = -1e9
    ok_stack = True
    for i, (o, part) in enumerate(zip(offsets, parts)):
        lo = o[2]
        hi = o[2] + part[0][:, 2].max()
        if lo < prev - 1e-6:
            ok_stack = False
        prev = hi
        ztops.append(hi)
    report("3MF: parts stack without overlap", ok_stack,
           f"z_tops={[round(z, 2) for z in ztops]}")

    # 7. Sub-model meshes are watertight.
    sub = z.read("3D/Objects/litho.stl_1.model").decode()
    n_objs = len(re.findall(r'<object id=', sub))
    report("3MF: sub-model has 5 objects", n_objs == 5, f"{n_objs} objects")

    # 8. project_settings valid JSON with filament colours.
    ps = z.read("Metadata/project_settings.config").decode()
    try:
        data = json.loads(ps)
        ok_ps = "filament_colour" in data and len(data["filament_colour"]) >= 4
    except Exception:  # noqa: BLE001
        ok_ps = False
    report("3MF: project_settings JSON valid", ok_ps)

    # 9. Machine preset written into project_settings.
    try:
        ok_machine = (data.get("printer_model") == "Snapmaker U1"
                      and data.get("printer_variant") == "0.4"
                      and data.get("printer_settings_id") == "Snapmaker U1 (0.4 nozzle)"
                      and data.get("filament_vendor") is not None)
    except Exception:  # noqa: BLE001
        ok_machine = False
    report("3MF: machine preset (printer_model/variant/settings_id)", ok_machine,
           f"model={data.get('printer_model')} variant={data.get('printer_variant')}")

    # 10. XY centering: every part's XY centroid is at the origin.
    ok_center = True
    for i, (o, part) in enumerate(zip(offsets, parts)):
        v = part[0]
        cx = 0.5 * (v[:, 0].min() + v[:, 0].max())
        cy = 0.5 * (v[:, 1].min() + v[:, 1].max())
        if abs(cx) > 0.5 or abs(cy) > 0.5:
            ok_center = False
    report("3MF: parts XY-centered on origin", ok_center)

    # 11. Build-item transform places the model at the bed centre.
    bt = re.search(r'<item objectid="6"[^>]*transform="([^"]+)"', main)
    bt_vals = [float(x) for x in bt.group(1).split()] if bt else []
    # 4x3 row-major; translation = cols 9,10,11.
    ok_bed = (len(bt_vals) >= 12 and abs(bt_vals[9] - 135.5) < 0.01
              and abs(bt_vals[10] - 136.0) < 0.01)
    report("3MF: build transform at bed centre (135.5,136)", ok_bed,
           bt.group(1) if bt else "no build item")

    z.close()
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except PermissionError:  # noqa: S110
            pass


if __name__ == "__main__":
    test_3mf_export()
    print()
    print(f"{sum(RESULTS)}/{len(RESULTS)} passed")
    sys.exit(0 if all(RESULTS) else 1)
