#include "LithophaneDialog.hpp"

#include "I18N.hpp"
#include "GUI.hpp"
#include "GUI_ObjectList.hpp"
#include "Plater.hpp"
#include "Widgets/DialogButtons.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <boost/log/trivial.hpp>

namespace {

// Slider ranges (DPI-independent units). Sliders map to real values linearly.
constexpr int WIDTH_MIN_MM  = 40;
constexpr int WIDTH_MAX_MM  = 220;
constexpr int HEIGHT_MIN_MM = 40;
constexpr int HEIGHT_MAX_MM = 220;
constexpr int DEPTH_MIN_MM  = 1;   // depth_range, 0.1 mm granularity via /10
constexpr int DEPTH_MAX_MM  = 40;  // -> 0.1 .. 4.0 mm
constexpr int PIXEL_MIN_UM  = 8;   // pixel pitch, 0.01 mm granularity via /100
constexpr int PIXEL_MAX_UM  = 40;  // -> 0.08 .. 0.40 mm

wxString double_fmt(double v, int prec = 1)
{
    return wxString::Format("%.*f", prec, v);
}

} // namespace

LithophaneDialog::LithophaneDialog(wxWindow* parent)
    : DPIDialog(parent ? parent : static_cast<wxWindow*>(wxGetApp().mainframe),
                wxID_ANY,
                _L("Lithophane Generator"),
                wxDefaultPosition,
                wxDefaultSize,
                wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER)
{
    SetFont(wxGetApp().normal_font());
    SetBackgroundColour(*wxWHITE);
    build_dialog();
    wxGetApp().UpdateDlgDarkUI(this);
}

void LithophaneDialog::on_dpi_changed(const wxRect& /*suggested_rect*/)
{
    // Layout-only; child controls are DPI-scaled via FromDIP at construction.
    Layout();
    Refresh();
}

void LithophaneDialog::build_dialog()
{
    auto *main_sizer = new wxBoxSizer(wxVERTICAL);

    // ---- Image pick row ----
    auto *img_box = new wxStaticBoxSizer(wxVERTICAL, this, _L("Image"));
    auto *pick_row = new wxBoxSizer(wxHORIZONTAL);
    m_pick_btn = new Button(this, _L("Choose image..."));
    m_pick_btn->Bind(wxEVT_BUTTON, &LithophaneDialog::on_pick_image, this);
    m_image_info = new wxStaticText(this, wxID_ANY, _L("No image selected"));
    pick_row->Add(m_pick_btn, 0, wxALL | wxALIGN_CENTER_VERTICAL, FromDIP(8));
    pick_row->Add(m_image_info, 1, wxALL | wxALIGN_CENTER_VERTICAL, FromDIP(8));
    img_box->Add(pick_row, 0, wxEXPAND);

    // Preview bitmap (square-ish).
    m_preview_bmp = new wxStaticBitmap(this, wxID_ANY, wxBitmap(FromDIP(200), FromDIP(150)));
    img_box->Add(m_preview_bmp, 0, wxALL | wxALIGN_CENTER, FromDIP(8));
    main_sizer->Add(img_box, 0, wxEXPAND | wxALL, FromDIP(10));

    // ---- Parameters ----
    auto *param_box = new wxStaticBoxSizer(wxVERTICAL, this, _L("Parameters"));
    auto add_param_row = [&](const wxString &label, int slider_min, int slider_init, int slider_max,
                             wxSlider *&slider, wxTextCtrl *&value, int prec, double scale_to_mm) {
        auto *row = new wxBoxSizer(wxHORIZONTAL);
        auto *lbl = new wxStaticText(this, wxID_ANY, label);
        lbl->SetMinSize(wxSize(FromDIP(90), -1));
        slider = new wxSlider(this, wxID_ANY, slider_init, slider_min, slider_max,
                              wxDefaultPosition, wxSize(FromDIP(180), -1), wxSL_HORIZONTAL);
        value = new wxTextCtrl(this, wxID_ANY, double_fmt(slider_init * scale_to_mm, prec),
                               wxDefaultPosition, wxSize(FromDIP(70), -1), wxTE_READONLY);
        row->Add(lbl, 0, wxALL | wxALIGN_CENTER_VERTICAL, FromDIP(4));
        row->Add(slider, 1, wxALL | wxALIGN_CENTER_VERTICAL, FromDIP(4));
        row->Add(value, 0, wxALL | wxALIGN_CENTER_VERTICAL, FromDIP(4));
        param_box->Add(row, 0, wxEXPAND | wxALL, FromDIP(2));
    };

    // Defaults match LithophaneParams: width=144, height=108, depth=2.0, pixel=0.2.
    add_param_row(_L("Width (mm)"),  WIDTH_MIN_MM,  144, WIDTH_MAX_MM,  m_width_slider,  m_width_value,  0, 1.0);
    add_param_row(_L("Height (mm)"), HEIGHT_MIN_MM, 108, HEIGHT_MAX_MM, m_height_slider, m_height_value, 0, 1.0);
    add_param_row(_L("Depth (mm)"),  DEPTH_MIN_MM,  20,  DEPTH_MAX_MM,  m_depth_slider,  m_depth_value,  1, 0.1);
    add_param_row(_L("Pixel pitch (mm)"), PIXEL_MIN_UM, 20, PIXEL_MAX_UM, m_pixel_slider, m_pixel_value, 2, 0.01);

    auto *slider_bind = [&](wxSlider *s, wxTextCtrl *t, double scale, int prec) {
        s->Bind(wxEVT_SLIDER, [this, s, t, scale, prec](wxCommandEvent&) {
            t->SetValue(double_fmt(s->GetValue() * scale, prec));
        });
    };
    slider_bind(m_width_slider,  m_width_value,  1.0,  0);
    slider_bind(m_height_slider, m_height_value, 1.0,  0);
    slider_bind(m_depth_slider,  m_depth_value,  0.1,  1);
    slider_bind(m_pixel_slider,  m_pixel_value,  0.01, 2);

    m_mirror_chk = new wxCheckBox(this, wxID_ANY, _L("Mirror horizontally (print viewed side down)"));
    param_box->Add(m_mirror_chk, 0, wxALL, FromDIP(4));
    main_sizer->Add(param_box, 0, wxEXPAND | wxALL, FromDIP(10));

    // ---- Buttons ----
    auto *btns = new DialogButtons(this, {"OK", "Cancel"});
    btns->GetOK()->Bind(wxEVT_BUTTON, &LithophaneDialog::on_ok, this);
    btns->GetCANCEL()->Bind(wxEVT_BUTTON, &LithophaneDialog::on_cancel, this);
    main_sizer->Add(btns, 0, wxEXPAND | wxALL, FromDIP(10));

    SetSizer(main_sizer);
    Layout();
    main_sizer->SetSizeHints(this);
    Fit();

    Bind(wxEVT_CLOSE_WINDOW, [this](wxCloseEvent&) { EndModal(wxID_CANCEL); });
}

void LithophaneDialog::on_pick_image(wxCommandEvent& /*evt*/)
{
    // PNG/JPG wildcard. No dedicated image FileType in GUI_App.hpp, hand-written.
    wxString wildcard = _L("Image files") + " (*.png;*.jpg;*.jpeg;*.bmp)|*.png;*.jpg;*.jpeg;*.bmp";
    wxFileDialog dlg(this, _L("Choose an image"), "", "", wildcard, wxFD_OPEN | wxFD_FILE_MUST_EXIST);
    if (dlg.ShowModal() != wxID_OK) return;

    m_image_path = dlg.GetPath();
    std::string path_u8 = m_image_path.ToUTF8().data();
    // Read once as colour, convert to greyscale once, keep in m_grey for both
    // preview and mesh generation (single source of truth, no re-reads).
    cv::Mat img = cv::imread(path_u8, cv::IMREAD_COLOR);
    if (img.empty()) {
        m_image_info->SetLabel(_L("Failed to load image"));
        m_grey.clear();
        m_img_cols = m_img_rows = 0;
        return;
    }
    cv::Mat grey;
    cv::cvtColor(img, grey, cv::COLOR_BGR2GRAY);
    m_img_cols = size_t(grey.cols);
    m_img_rows = size_t(grey.rows);
    m_grey.assign(grey.datastart, grey.dataend);
    m_image_info->SetLabel(wxString::Format(_L("Image: %dx%d"), int(m_img_cols), int(m_img_rows)));
    update_preview();
}

void LithophaneDialog::update_preview()
{
    if (m_grey.empty() || m_img_cols == 0 || m_img_rows == 0) return;

    // Render the greyscale buffer into a wxImage for the static bitmap.
    const int pw = FromDIP(200);
    const int ph = FromDIP(150);
    cv::Mat src(int(m_img_rows), int(m_img_cols), CV_8U, const_cast<uint8_t*>(m_grey.data()));
    cv::Mat dst;
    cv::resize(src, dst, cv::Size(pw, ph), 0, 0, cv::INTER_AREA);
    wxImage img(pw, ph, false);
    // wxImage wants RGB; replicate the single grey channel into R, G, B.
    unsigned char *rgb = img.GetData();
    for (int i = 0; i < pw * ph; ++i) {
        unsigned char g = dst.data[i];
        rgb[3 * i] = rgb[3 * i + 1] = rgb[3 * i + 2] = g;
    }
    m_preview_bmp->SetBitmap(wxBitmap(img));
}

void LithophaneDialog::on_ok(wxCommandEvent& /*evt*/)
{
    if (m_img_cols == 0 || m_img_rows == 0 || m_grey.empty()) {
        // No image picked: just close without result.
        m_has_result = false;
        EndModal(wxID_CANCEL);
        return;
    }

    Slic3r::LithophaneParams p;
    p.width_mm      = double(m_width_slider->GetValue());
    p.height_mm     = double(m_height_slider->GetValue());
    p.depth_range   = m_depth_slider->GetValue() * 0.1;
    p.pixel_pitch_mm = m_pixel_slider->GetValue() * 0.01;
    p.mirror        = m_mirror_chk->GetValue();

    BOOST_LOG_TRIVIAL(info) << "Lithophane: generating mesh from " << m_img_cols << "x" << m_img_rows
                            << " image, params w=" << p.width_mm << " h=" << p.height_mm
                            << " depth=" << p.depth_range << " pitch=" << p.pixel_pitch_mm;
    m_mesh = Slic3r::make_lithophane_mesh_from_grey(m_grey, m_img_cols, m_img_rows, p);
    m_has_result = m_mesh.facets_count() > 0;
    if (!m_has_result) {
        BOOST_LOG_TRIVIAL(error) << "Lithophane: mesh generation produced an empty mesh";
    }
    EndModal(wxID_OK);
}

void LithophaneDialog::on_cancel(wxCommandEvent& /*evt*/)
{
    m_has_result = false;
    EndModal(wxID_CANCEL);
}

Slic3r::TriangleMesh LithophaneDialog::take_mesh()
{
    m_has_result = false;
    return std::move(m_mesh);
}
