#ifndef _LITHOPHANE_DIALOG_H_
#define _LITHOPHANE_DIALOG_H_

#include "GUI_App.hpp"
#include "GUI_Utils.hpp"
#include "Widgets/Button.hpp"
#include "libslic3r/Lithophane.hpp"

#include <wx/slider.h>
#include <wx/stattext.h>
#include <wx/statbmp.h>
#include <wx/textctrl.h>

class Button;

// Monochrome lithophane generator dialog (M1 scope).
//
// Flow: pick an image -> tweak params with live preview -> OK generates a
// TriangleMesh via Slic3r::make_lithophane_mesh and loads it onto the plate.
// Generation happens after ShowModal() returns wxID_OK (matching the
// SLAImportDialog pattern), so the menu callback owns the load step.
class LithophaneDialog : public Slic3r::GUI::DPIDialog
{
public:
    LithophaneDialog(wxWindow* parent);
    ~LithophaneDialog() override = default;

    void on_dpi_changed(const wxRect& suggested_rect) override;

    // True when the user confirmed with a valid image selected; the caller
    // should then call take_mesh() to consume the generated mesh.
    bool has_result() const { return m_has_result; }
    // Consumes the generated mesh. Empty if generation failed.
    Slic3r::TriangleMesh take_mesh();

private:
    void build_dialog();
    void on_pick_image(wxCommandEvent& evt);
    void on_ok(wxCommandEvent& evt);
    void on_cancel(wxCommandEvent& evt);
    void update_preview();
    void rebuild_params_text();

    // Image (8-bit greyscale) held in memory for preview + mesh generation.
    std::vector<uint8_t> m_grey;
    size_t               m_img_cols = 0;
    size_t               m_img_rows = 0;
    wxString             m_image_path;

    // Controls
    Button*         m_pick_btn        = nullptr;
    wxStaticBitmap* m_preview_bmp     = nullptr;
    wxStaticText*   m_image_info      = nullptr;
    wxSlider*       m_width_slider    = nullptr;
    wxSlider*       m_height_slider   = nullptr;
    wxSlider*       m_depth_slider    = nullptr;
    wxSlider*       m_pixel_slider    = nullptr;
    wxTextCtrl*     m_width_value     = nullptr;
    wxTextCtrl*     m_height_value    = nullptr;
    wxTextCtrl*     m_depth_value     = nullptr;
    wxTextCtrl*     m_pixel_value     = nullptr;
    wxCheckBox*     m_mirror_chk      = nullptr;

    // Generated on OK, consumed by take_mesh().
    Slic3r::TriangleMesh m_mesh;
    bool                 m_has_result = false;
};

#endif // _LITHOPHANE_DIALOG_H_
