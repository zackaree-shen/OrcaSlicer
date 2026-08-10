#include "Lithophane.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

// OpenCV is linked PUBLIC via opencv_world (deps/OpenCV/OpenCV.cmake builds with
// BUILD_LIST=core,imgcodecs,imgproc,world). imread lives in imgcodecs, so include
// the specific header rather than the heavy opencv.hpp umbrella.
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <boost/log/trivial.hpp>

namespace Slic3r {

namespace {

// Interpolate a thickness at continuous (x, y) in image space using bilinear
// sampling over the greyscale buffer. x in [0, cols-1], y in [0, rows-1].
// Returns thickness in mm: dark pixels -> thick, bright pixels -> thin.
float sample_thickness_bilinear(const std::vector<uint8_t> &grey, size_t cols, size_t rows,
                                double x, double y, double base_mm, double depth_mm)
{
    // Clamp to valid interpolation domain.
    if (cols < 2 || rows < 2) {
        uint8_t g = grey.empty() ? 0 : grey[0];
        return float(base_mm + (1.0 - g / 255.0) * depth_mm);
    }
    double xf = std::clamp(x, 0.0, double(cols - 1));
    double yf = std::clamp(y, 0.0, double(rows - 1));
    size_t x0 = size_t(xf);
    size_t y0 = size_t(yf);
    size_t x1 = (x0 + 1 < cols) ? x0 + 1 : x0;
    size_t y1 = (y0 + 1 < rows) ? y0 + 1 : y0;
    double fx = xf - double(x0);
    double fy = yf - double(y0);

    auto at = [&](size_t ix, size_t iy) -> double {
        return grey[iy * cols + ix] / 255.0;
    };
    double top    = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
    double bottom = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
    double bright = top * (1.0 - fy) + bottom * fy; // 0..1, 1 = white
    // dark (bright->0) => thick
    return float(base_mm + (1.0 - bright) * depth_mm);
}

// Append a triangle to the face list. Vertices are expected in CCW order when
// viewed from outside the solid.
inline void push_triangle(std::vector<Vec3i32> &faces, int a, int b, int c)
{
    faces.emplace_back(a, b, c);
}

} // namespace

TriangleMesh make_lithophane_mesh_from_grey(const std::vector<uint8_t> &grey, size_t cols, size_t rows, const LithophaneParams &params)
{
    if (cols == 0 || rows == 0 || grey.size() < cols * rows) {
        BOOST_LOG_TRIVIAL(error) << "Lithophane: invalid greyscale buffer (" << cols << "x" << rows << ", " << grey.size() << " bytes)";
        return TriangleMesh();
    }

    // Resolve output grid resolution from pixel pitch, clamped by requested size.
    const double pitch = std::max(params.pixel_pitch_mm, 1e-3);
    // Sample grid covers the requested width x height at the given pitch.
    const size_t gx = std::max<size_t>(2, size_t(std::round(params.width_mm / pitch)) + 1);
    const size_t gy = std::max<size_t>(2, size_t(std::round(params.height_mm / pitch)) + 1);

    const double phys_w = params.width_mm;
    const double phys_h = params.height_mm;
    const double dx = phys_w / double(gx - 1);
    const double dy = phys_h / double(gy - 1);

    std::vector<Vec3f> vertices;
    std::vector<Vec3i32> faces;

    // Reserve a rough upper bound: front grid (gx*gy) + back plane (gx*gy) +
    // 4 side strips. Faces: front (gx-1)*(gy-1)*2, back same, sides ~ 2*(gx+gy)*2.
    vertices.reserve(gx * gy * 2);
    faces.reserve((gx - 1) * (gy - 1) * 4 + (gx + gy) * 4);

    // ---- Front height-field vertices (top surface, z = thickness) ----
    // Index base 0..(gx*gy-1). Vertex (ix, iy) at column-major row-major order:
    //   v_front(ix,iy) = iy*gx + ix
    auto v_front = [gx](size_t ix, size_t iy) -> int { return int(iy * gx + ix); };
    for (size_t iy = 0; iy < gy; ++iy) {
        for (size_t ix = 0; ix < gx; ++ix) {
            // Map grid coord to image coord (image origin top-left, y down).
            double img_x = (double(ix) / double(gx - 1)) * double(cols - 1);
            double img_y = (double(iy) / double(gy - 1)) * double(rows - 1);
            if (params.mirror) img_x = double(cols - 1) - img_x;
            float z = sample_thickness_bilinear(grey, cols, rows, img_x, img_y, params.base_thickness, params.depth_range);
            float px = float(double(ix) * dx);
            float py = float(double(iy) * dy);
            vertices.emplace_back(px, py, z);
        }
    }

    // ---- Back plane vertices (z = 0), indices gx*gy .. 2*gx*gy-1 ----
    const int back_base = int(gx * gy);
    auto v_back = [back_base](size_t ix, size_t iy) -> int { return back_base + int(iy * gx + ix); };
    for (size_t iy = 0; iy < gy; ++iy)
        for (size_t ix = 0; ix < gx; ++ix)
            vertices.emplace_back(float(double(ix) * dx), float(double(iy) * dy), 0.0f);

    // Helper to push two CCW triangles for a quad given 4 corner vertex indices
    // in CCW order as seen from outside.
    auto push_quad_ccw = [&](int a, int b, int c, int d) {
        // a-b-c / a-c-d (both CCW)
        push_triangle(faces, a, b, c);
        push_triangle(faces, a, c, d);
    };

    // ---- Front surface quads ----
    // Front normal points +Z. Walking ix,iy with right = +X, up = +Y (toward
    // increasing iy), CCW order of a quad (ix,iy),(ix+1,iy),(ix+1,iy+1),(ix,iy+1)
    // is a,b,c,d as below (viewed from +Z looking down -Z).
    for (size_t iy = 0; iy + 1 < gy; ++iy) {
        for (size_t ix = 0; ix + 1 < gx; ++ix) {
            int a = v_front(ix,     iy);
            int b = v_front(ix + 1, iy);
            int c = v_front(ix + 1, iy + 1);
            int d = v_front(ix,     iy + 1);
            push_quad_ccw(a, b, c, d);
        }
    }

    // ---- Back surface quads (normal points -Z, so reverse winding) ----
    for (size_t iy = 0; iy + 1 < gy; ++iy) {
        for (size_t ix = 0; ix + 1 < gx; ++ix) {
            int a = v_back(ix,     iy);
            int b = v_back(ix + 1, iy);
            int c = v_back(ix + 1, iy + 1);
            int d = v_back(ix,     iy + 1);
            // Reverse winding so normal points -Z.
            push_quad_ccw(a, d, c, b);
        }
    }

    // ---- Side walls: stitch front edge to back edge, CCW outward ----
    // Edge along iy = 0     (front: y=0),     outward normal -Y
    // Edge along iy = gy-1  (front: y=phys_h),outward normal +Y
    // Edge along ix = 0     (front: x=0),     outward normal -X
    // Edge along ix = gx-1  (front: x=phys_w),outward normal +X
    // For each edge we emit a strip of quads connecting front to back.

    // Bottom edge iy=0: walking ix 0..gx-1, outward = -Y. Quad corners viewed
    // from outside (-Y): front(ix,0) -> back(ix,0) -> back(ix+1,0) -> front(ix+1,0)
    for (size_t ix = 0; ix + 1 < gx; ++ix) {
        int a = v_front(ix,     0);
        int b = v_back (ix,     0);
        int c = v_back (ix + 1, 0);
        int d = v_front(ix + 1, 0);
        push_quad_ccw(a, b, c, d);
    }
    // Top edge iy=gy-1: outward = +Y. Reverse sense.
    for (size_t ix = 0; ix + 1 < gx; ++ix) {
        int a = v_front(ix,     gy - 1);
        int b = v_front(ix + 1, gy - 1);
        int c = v_back (ix + 1, gy - 1);
        int d = v_back (ix,     gy - 1);
        push_quad_ccw(a, b, c, d);
    }
    // Left edge ix=0: outward = -X.
    for (size_t iy = 0; iy + 1 < gy; ++iy) {
        int a = v_front(0, iy);
        int b = v_front(0, iy + 1);
        int c = v_back (0, iy + 1);
        int d = v_back (0, iy);
        push_quad_ccw(a, b, c, d);
    }
    // Right edge ix=gx-1: outward = +X.
    for (size_t iy = 0; iy + 1 < gy; ++iy) {
        int a = v_front(gx - 1, iy);
        int b = v_back (gx - 1, iy);
        int c = v_back (gx - 1, iy + 1);
        int d = v_front(gx - 1, iy + 1);
        push_quad_ccw(a, b, c, d);
    }

    // Build the indexed_triangle_set, then weld coincident vertices and fix
    // orientation BEFORE constructing TriangleMesh. Doing it before construction
    // means the TriangleMesh constructor's fill_initial_stats (TriangleMesh.cpp:58)
    // computes volume/size over the corrected mesh, so mesh.stats()/volume()/size()
    // are all consistent afterwards.
    //
    // TriangleMesh has no repair() member in this codebase; the free functions
    // its_merge_vertices / its_volume / its_flip_triangles (TriangleMesh.hpp:209,
    // :321, :204) are the available tools.
    //
    //   - its_merge_vertices welds the shared grid corners (front/back/side walls
    //     reuse identical xy positions) so the mesh is watertight.
    //   - A negative volume means winding is inside-out; the slicer would treat an
    //     inside-out solid as air, so flip the winding.
    indexed_triangle_set its;
    its.vertices = std::move(vertices);
    its.indices  = std::move(faces);
    its_merge_vertices(its);
    if (its_volume(its) < 0.f) {
        BOOST_LOG_TRIVIAL(warning) << "Lithophane: negative volume after weld; flipping facets";
        its_flip_triangles(its);
    }

    TriangleMesh mesh(std::move(its));
    BOOST_LOG_TRIVIAL(info) << "Lithophane: generated mesh " << mesh.stats().number_of_facets << " facets, "
                            << "volume " << mesh.volume() << " mm^3, size " << mesh.size().x() << "x" << mesh.size().y()
                            << "x" << mesh.size().z() << " mm";
    return mesh;
}

TriangleMesh make_lithophane_mesh(const std::string &image_path, const LithophaneParams &params)
{
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        BOOST_LOG_TRIVIAL(error) << "Lithophane: failed to read image: " << image_path;
        return TriangleMesh();
    }
    // Ensure continuous single-channel 8-bit.
    if (img.type() != CV_8U) {
        img.convertTo(img, CV_8U);
    }
    const size_t cols = size_t(img.cols);
    const size_t rows = size_t(img.rows);
    std::vector<uint8_t> grey(size_t(img.total()));
    if (img.isContinuous()) {
        std::memcpy(grey.data(), img.data, grey.size());
    } else {
        for (size_t r = 0; r < rows; ++r)
            std::memcpy(&grey[r * cols], img.ptr<uchar>(int(r)), cols);
    }
    return make_lithophane_mesh_from_grey(grey, cols, rows, params);
}

} // namespace Slic3r
