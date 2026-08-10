#ifndef slic3r_Lithophane_hpp_
#define slic3r_Lithophane_hpp_

#include "libslic3r.h"
#include <string>
#include <vector>
#include "TriangleMesh.hpp"

namespace Slic3r {

// Lithophane (透光浮雕) generator.
//
// M1 scope: monochrome greyscale only. An image is read as 8-bit greyscale,
// each pixel is mapped to a thickness on a height-field, and the field is
// closed into a watertight solid ready for slicing.
//
// Physical convention: a lithophane is viewed with a backlight behind it.
// Darker pixels must let less light through, so they are printed thicker.
// thickness(x,y) = base_thickness + (1 - gray/255) * depth_range.
//   - white (gray=255) -> thinnest = base_thickness (most light)
//   - black  (gray=0)  -> thickest = base_thickness + depth_range (least light)
struct LithophaneParams
{
    // Output size in millimetres. The image is resampled to fit width/height
    // preserving aspect ratio; the smaller dimension determines pixel pitch.
    double width_mm  = 144.0; // Bambu standard frame size (Bambu wiki)
    double height_mm = 108.0; // Bambu standard frame size (Bambu wiki)

    // Minimum (white) and maximum (black) thickness in millimetres.
    // base_thickness gives structural integrity to the thinnest point;
    // (base_thickness + depth_range) is the thickest, darkest point.
    // v1 coarse values, informed by itslithophane.com defaults and Bambu wiki
    // (top/bottom shell = 3 layers). Calibrate per printer/material later.
    double base_thickness = 0.8;
    double depth_range    = 2.0;

    // Output pixel pitch in millimetres per image pixel. Smaller = sharper but
    // heavier mesh. ~0.1 mm/pixel is near photographic at 0.08 mm layer height.
    double pixel_pitch_mm = 0.2;

    // Mirror the image horizontally (for printing the viewed side down).
    bool mirror = false;
};

// Read an image file as greyscale (PNG/JPG via OpenCV imgcodecs) and build a
// single-material lithophane mesh. Returns an empty mesh on failure.
// The mesh is watertight and orientation-corrected via repair().
TriangleMesh make_lithophane_mesh(const std::string &image_path, const LithophaneParams &params);

// Build a lithophane mesh directly from an 8-bit single-channel greyscale
// buffer (row-major, rows * cols bytes). Exposed for unit testing and for UI
// paths that already hold the image in memory.
TriangleMesh make_lithophane_mesh_from_grey(const std::vector<uint8_t> &grey, size_t cols, size_t rows, const LithophaneParams &params);

} // namespace Slic3r

#endif // slic3r_Lithophane_hpp_
