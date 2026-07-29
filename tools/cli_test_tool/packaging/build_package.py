"""Build a self-contained zip package for distribution.

Layout in zip:
  snapmaker-orca.exe          # CLI binary + all DLLs at root (needed for DLL resolution)
  *.dll
  resources/                  # CLI data dir (profiles, calib, printers, fonts)
    profiles/
    calib/
    printers/
    fonts/
  tools/
    cli_test_tool/            # Test tool web app
      app.py
      translator.py
      templates/index.html
      start.bat
      guide.html, guide_zh.html, guide_en.html
    gcode-diff.exe            # Quality scoring engine
    Snapmaker-CLI-Test-Tool-*.exe  # Electron wrapper (optional)
"""
import pathlib, zipfile, time

ROOT = pathlib.Path("C:/workDir/programing/SnapMakerOracaRequirements/OrcaSlicer")
BUILD_DIR = ROOT / "build" / "src" / "Release"
RESOURCES = ROOT / "resources"
TOOL_DIR = ROOT / "tools" / "cli_test_tool"
GCODE_DIFF = pathlib.Path("C:/workDir/programing/Gcode-diff/gcode-diff-rs/target/release/gcode-diff.exe")
ELECTRON_EXE = TOOL_DIR / "electron" / "dist" / "Snapmaker-CLI-Test-Tool-1.0.0-x64.exe"
OUTPUT = TOOL_DIR / "packaging" / "Snapmaker-CLI-Package-1.0.0.zip"

ESSENTIAL_RESOURCES = ["profiles", "calib", "printers", "fonts"]
EXCLUDE_EXTS = {".pdb", ".exp", ".lib"}


def should_include(f):
    return f.suffix.lower() not in EXCLUDE_EXTS


print("Building package...")
t0 = time.time()

with zipfile.ZipFile(str(OUTPUT), "w", zipfile.ZIP_DEFLATED) as zf:
    # 1. CLI binary + runtime DLLs at root (required for DLL resolution)
    n_cli = 0
    for f in sorted(BUILD_DIR.iterdir()):
        if f.is_file() and should_include(f):
            zf.write(str(f), f.name)
            n_cli += 1
    print(f"  CLI: {n_cli} files")

    # 2. Resources under resources/ wrapper (so --datadir resources works)
    for res in ESSENTIAL_RESOURCES:
        src = RESOURCES / res
        if not src.exists():
            print(f"  WARNING: resources/{res} not found")
            continue
        n = 0
        for f in sorted(src.rglob("*")):
            if f.is_file():
                rel = f.relative_to(RESOURCES)
                zf.write(str(f), "resources/" + str(rel))
                n += 1
        print(f"  resources/{res}/: {n} files")

    # 3. Test tool (all files under tools/cli_test_tool/)
    tool_files = [
        "app.py", "translator.py", "start.bat", "README.md",
        "guide.html", "guide_zh.html", "guide_en.html",
    ]
    for fname in tool_files:
        fpath = TOOL_DIR / fname
        if fpath.exists():
            zf.write(str(fpath), "tools/cli_test_tool/" + fname)
    # Templates go inside tools/cli_test_tool/templates/
    for f in sorted((TOOL_DIR / "templates").rglob("*")):
        if f.is_file():
            rel = f.relative_to(TOOL_DIR)
            zf.write(str(f), "tools/cli_test_tool/" + str(rel))
    print("  tools/cli_test_tool/: app.py + translator.py + templates + guides")

    # 4. gcode-diff quality engine
    if GCODE_DIFF.exists():
        zf.write(str(GCODE_DIFF), "tools/gcode-diff.exe")
        print("  tools/gcode-diff.exe")
    else:
        print("  WARNING: gcode-diff.exe not found")

    # 5. Electron wrapper (optional)
    if ELECTRON_EXE.exists():
        zf.write(str(ELECTRON_EXE), "tools/" + ELECTRON_EXE.name)
        print("  Electron exe")

elapsed = time.time() - t0
size_mb = OUTPUT.stat().st_size / (1024 * 1024)
print(f"\nDone: {OUTPUT.name} ({size_mb:.1f} MB, {elapsed:.1f}s)")
