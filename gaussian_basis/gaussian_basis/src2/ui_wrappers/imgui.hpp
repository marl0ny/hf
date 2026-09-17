
// #include "parameters.hpp"

#ifndef _IMGUI_CONTROLS_
#define _IMGUI_CONTROLS_
// using namespace sim_3d;

#include "gl_wrappers.hpp"

#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"
#include "common_imgui.hpp"

/* #include <functional>
#include <set>

#include "parameters.hpp"

static std::function<void(int, Uniform)> s_sim_params_set;
static std::function<void(int, int, std::string)> s_sim_params_set_string;
static std::function<Uniform(int)> s_sim_params_get;
static std::function<void(int, std::string, float)> s_user_edit_set_value;
static std::function<float(int, std::string)> s_user_edit_get_value;
static std::function<std::string(int)>
    s_user_edit_get_comma_separated_variables;
static std::function<void(int)> s_button_pressed;
static std::function<void(int, int)> s_selection_set;
static std::function<void(
    int, const std::string &image_data, int, int)> s_image_set;
static std::function<unsigned char *()> s_bmp_image;
static std::function<unsigned int ()> s_bmp_image_size;
static std::function<void (int, bool)> s_configure_bmp_recording;
static std::function<void(int, std::string, float)>
    s_sim_params_set_user_float_param;

static ImGuiIO global_io;
static std::map<int, std::string> global_labels;

void edit_label_display(int c, std::string text_content) {
    global_labels[c] = text_content;
}

void display_parameters_as_sliders(
    int c, std::set<std::string> variables, 
    std::set<std::string> do_not_show={ 
    }
    ) {
    std::string string_val = "[";
    for (auto &e: variables)
        string_val += """ + e + "", ";
    string_val += "]";
    string_val 
        = "modifyUserSliders(" + std::to_string(c) + ", " + string_val + ");";
    // TODO
}

void download_bmp_image(std::string postfix_name) {
    unsigned char *image_data = s_bmp_image();
    int image_size = s_bmp_image_size();
    std::string time = std::to_string(
        std::chrono::system_clock().now().time_since_epoch().count());
    std::string fname = time + postfix_name + ".bmp";
    FILE *f = fopen(&fname[0], "wb");
    // TODO: check file!
    fwrite(image_data, 1, image_size, f);
    // TOO: check file writing!

}

void start_gui(void *window) {
    bool show_controls_window = true;
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    ImGui::StyleColorsClassic();
    ImGui_ImplGlfw_InitForOpenGL((GLFWwindow *)window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}*/

void imgui_controls(void *void_params) {
    SimParams *params = (SimParams *)void_params;
    for (auto &e: global_labels)
        params->set(e.first, 0, e.second);
    if (ImGui::BeginMenu("Mouse usage")) {
        if (ImGui::MenuItem("Rotate only"))
            s_selection_set(params->MOUSE_SELECTOR, 0);
        if (ImGui::MenuItem("Place atom"))
            s_selection_set(params->MOUSE_SELECTOR, 1);
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Grid discretization size")) {
        if (ImGui::MenuItem("64x64x64"))
            s_selection_set(params->TEXEL_SIDE_LENGTH_SELECTOR, 0);
        if (ImGui::MenuItem("128x128x128"))
            s_selection_set(params->TEXEL_SIDE_LENGTH_SELECTOR, 1);
        if (ImGui::MenuItem("256x256x256"))
            s_selection_set(params->TEXEL_SIDE_LENGTH_SELECTOR, 2);
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Preset Compounds")) {
        if (ImGui::MenuItem("Hydrogen molecule"))
            s_selection_set(params->PRESET_COMPOUNDS_DROPDOWN, 0);
        if (ImGui::MenuItem("Water"))
            s_selection_set(params->PRESET_COMPOUNDS_DROPDOWN, 1);
        if (ImGui::MenuItem("Carbon Dioxide"))
            s_selection_set(params->PRESET_COMPOUNDS_DROPDOWN, 2);
        if (ImGui::MenuItem("Oxygen Molecule"))
            s_selection_set(params->PRESET_COMPOUNDS_DROPDOWN, 3);
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Atom dropdown")) {
        if (ImGui::MenuItem("H"))
            s_selection_set(params->PRESET_ATOMS, 0);
        if (ImGui::MenuItem("He"))
            s_selection_set(params->PRESET_ATOMS, 1);
        if (ImGui::MenuItem("Li"))
            s_selection_set(params->PRESET_ATOMS, 2);
        if (ImGui::MenuItem("Be"))
            s_selection_set(params->PRESET_ATOMS, 3);
        if (ImGui::MenuItem("B"))
            s_selection_set(params->PRESET_ATOMS, 4);
        if (ImGui::MenuItem("C"))
            s_selection_set(params->PRESET_ATOMS, 5);
        if (ImGui::MenuItem("N"))
            s_selection_set(params->PRESET_ATOMS, 6);
        if (ImGui::MenuItem("O"))
            s_selection_set(params->PRESET_ATOMS, 7);
        if (ImGui::MenuItem("F"))
            s_selection_set(params->PRESET_ATOMS, 8);
        if (ImGui::MenuItem("Ne"))
            s_selection_set(params->PRESET_ATOMS, 9);
        if (ImGui::MenuItem("Na"))
            s_selection_set(params->PRESET_ATOMS, 10);
        if (ImGui::MenuItem("Mg"))
            s_selection_set(params->PRESET_ATOMS, 11);
        if (ImGui::MenuItem("Al"))
            s_selection_set(params->PRESET_ATOMS, 12);
        if (ImGui::MenuItem("Si"))
            s_selection_set(params->PRESET_ATOMS, 13);
        if (ImGui::MenuItem("P"))
            s_selection_set(params->PRESET_ATOMS, 14);
        if (ImGui::MenuItem("S"))
            s_selection_set(params->PRESET_ATOMS, 15);
        if (ImGui::MenuItem("Cl"))
            s_selection_set(params->PRESET_ATOMS, 16);
        if (ImGui::MenuItem("Ar"))
            s_selection_set(params->PRESET_ATOMS, 17);
        if (ImGui::MenuItem("K"))
            s_selection_set(params->PRESET_ATOMS, 18);
        if (ImGui::MenuItem("Ca"))
            s_selection_set(params->PRESET_ATOMS, 19);
        if (ImGui::MenuItem("Sc"))
            s_selection_set(params->PRESET_ATOMS, 20);
        if (ImGui::MenuItem("Ti"))
            s_selection_set(params->PRESET_ATOMS, 21);
        if (ImGui::MenuItem("V"))
            s_selection_set(params->PRESET_ATOMS, 22);
        if (ImGui::MenuItem("Cr"))
            s_selection_set(params->PRESET_ATOMS, 23);
        if (ImGui::MenuItem("Mn"))
            s_selection_set(params->PRESET_ATOMS, 24);
        if (ImGui::MenuItem("Fe"))
            s_selection_set(params->PRESET_ATOMS, 25);
        if (ImGui::MenuItem("Co"))
            s_selection_set(params->PRESET_ATOMS, 26);
        if (ImGui::MenuItem("Ni"))
            s_selection_set(params->PRESET_ATOMS, 27);
        if (ImGui::MenuItem("Cu"))
            s_selection_set(params->PRESET_ATOMS, 28);
        if (ImGui::MenuItem("Zn"))
            s_selection_set(params->PRESET_ATOMS, 29);
        if (ImGui::MenuItem("Ga"))
            s_selection_set(params->PRESET_ATOMS, 30);
        if (ImGui::MenuItem("Ge"))
            s_selection_set(params->PRESET_ATOMS, 31);
        if (ImGui::MenuItem("As"))
            s_selection_set(params->PRESET_ATOMS, 32);
        if (ImGui::MenuItem("Se"))
            s_selection_set(params->PRESET_ATOMS, 33);
        if (ImGui::MenuItem("Br"))
            s_selection_set(params->PRESET_ATOMS, 34);
        if (ImGui::MenuItem("Kr"))
            s_selection_set(params->PRESET_ATOMS, 35);
        ImGui::EndMenu();
    }
    if (ImGui::SliderInt("Max # of SCF steps", &params->maxNumberOfIterations, 0, 30))
            s_sim_params_set(params->MAX_NUMBER_OF_ITERATIONS, params->maxNumberOfIterations);
    if (ImGui::Button("Solve"))
           s_button_pressed(params->SOLVE);
    if (ImGui::Button("Clear"))
           s_button_pressed(params->CLEAR);
    if (ImGui::BeginMenu("Method type")) {
        if (ImGui::MenuItem("All shells closed"))
            s_selection_set(params->SHELL_METHOD_TYPE, 0);
        if (ImGui::MenuItem("Unrestricted"))
            s_selection_set(params->SHELL_METHOD_TYPE, 1);
        ImGui::EndMenu();
    }
    if (ImGui::SliderFloat("Zoom out level", &params->sizeScale, 0.5, 10.0))
           s_sim_params_set(params->SIZE_SCALE, params->sizeScale);
    if (ImGui::Checkbox("Show total electron density", &params->showDensity))
            s_sim_params_set(params->SHOW_DENSITY, params->showDensity);
    if (ImGui::SliderInt("Which orbital", &params->whichOrbitalSliderVal, 0, 20))
            s_sim_params_set(params->WHICH_ORBITAL_SLIDER_VAL, params->whichOrbitalSliderVal);
    if (ImGui::SliderInt("Particle count upon reset", &params->numberOfParticles, 8192, 1048576))
            s_sim_params_set(params->NUMBER_OF_PARTICLES, params->numberOfParticles);
    if (ImGui::TreeNode("Visualization Controls")) {
    if (ImGui::BeginMenu("Visualization select")) {
        if (ImGui::MenuItem("Volume render"))
            s_selection_set(params->VISUALIZATION_SELECT, 0);
        if (ImGui::MenuItem("Three orthogonal planar slices"))
            s_selection_set(params->VISUALIZATION_SELECT, 1);
        ImGui::EndMenu();
    }
    if (ImGui::Checkbox("Use perspective projection", &params->usePerspectiveProjection))
            s_sim_params_set(params->USE_PERSPECTIVE_PROJECTION, params->usePerspectiveProjection);
    if (ImGui::SliderFloat("Overall scaling", &params->brightness, 0.0, 0.5))
           s_sim_params_set(params->BRIGHTNESS, params->brightness);
    if (ImGui::TreeNode("Volume Render Controls")) {
    if (ImGui::Checkbox("Linear interpolation", &params->useLinear))
            s_sim_params_set(params->USE_LINEAR, params->useLinear);
    if (ImGui::SliderFloat("Alpha brightness", &params->alphaBrightness, 0.0, 10.0))
           s_sim_params_set(params->ALPHA_BRIGHTNESS, params->alphaBrightness);
    if (ImGui::SliderFloat("Color brightness", &params->colorBrightness, 0.0, 10.0))
           s_sim_params_set(params->COLOR_BRIGHTNESS, params->colorBrightness);
    ImGui::Text("Volume dimensions (volumeTexelDimensions3D)");
    if (ImGui::SliderInt("volumeTexelDimensions3D[0]", &params->volumeTexelDimensions3D.ind[0], 16, 512))
            s_sim_params_set(params->VOLUME_TEXEL_DIMENSIONS3_D, params->volumeTexelDimensions3D);
    if (ImGui::SliderInt("volumeTexelDimensions3D[1]", &params->volumeTexelDimensions3D.ind[1], 16, 512))
            s_sim_params_set(params->VOLUME_TEXEL_DIMENSIONS3_D, params->volumeTexelDimensions3D);
    if (ImGui::SliderInt("volumeTexelDimensions3D[2]", &params->volumeTexelDimensions3D.ind[2], 16, 512))
            s_sim_params_set(params->VOLUME_TEXEL_DIMENSIONS3_D, params->volumeTexelDimensions3D);
    if (ImGui::Checkbox("Enable bloom", &params->applyBlur))
            s_sim_params_set(params->APPLY_BLUR, params->applyBlur);
    if (ImGui::SliderInt("Bloominess", &params->blurSize, 0, 10))
            s_sim_params_set(params->BLUR_SIZE, params->blurSize);
    ImGui::TreePop();
    }
 
    if (ImGui::TreeNode("Three Orthogonal Planar Slices Controls")) {
    ImGui::Text("Planar slices offsets (in normalized coordinates) for xy, yz, xz");
    if (ImGui::SliderFloat("planarNormCoordOffsets[0]", &params->planarNormCoordOffsets.ind[0], 0.0, 1.0))
           s_sim_params_set(params->PLANAR_NORM_COORD_OFFSETS, params->planarNormCoordOffsets);
    if (ImGui::SliderFloat("planarNormCoordOffsets[1]", &params->planarNormCoordOffsets.ind[1], 0.0, 1.0))
           s_sim_params_set(params->PLANAR_NORM_COORD_OFFSETS, params->planarNormCoordOffsets);
    if (ImGui::SliderFloat("planarNormCoordOffsets[2]", &params->planarNormCoordOffsets.ind[2], 0.0, 1.0))
           s_sim_params_set(params->PLANAR_NORM_COORD_OFFSETS, params->planarNormCoordOffsets);
    ImGui::TreePop();
    }
 
    if (ImGui::TreeNode("Arrows Plot")) {
    ImGui::Text("Arrows dimensions");
    if (ImGui::SliderInt("arrowDimensions[0]", &params->arrowDimensions.ind[0], 8, 128))
            s_sim_params_set(params->ARROW_DIMENSIONS, params->arrowDimensions);
    if (ImGui::SliderInt("arrowDimensions[1]", &params->arrowDimensions.ind[1], 8, 128))
            s_sim_params_set(params->ARROW_DIMENSIONS, params->arrowDimensions);
    if (ImGui::SliderInt("arrowDimensions[2]", &params->arrowDimensions.ind[2], 8, 128))
            s_sim_params_set(params->ARROW_DIMENSIONS, params->arrowDimensions);
    if (ImGui::Checkbox("Use conical arrows", &params->useCones))
            s_sim_params_set(params->USE_CONES, params->useCones);
    ImGui::TreePop();
    }
 
    ImGui::TreePop();
    }
 
    if (ImGui::Checkbox("Take screenshots at every frame (uncompressed bitmap)", &params->takeScreenshots.is_recording))
            s_configure_bmp_recording(params->TAKE_SCREENSHOTS, params->takeScreenshots.is_recording);

}

// bool outside_gui() {
//     return !global_io.WantCaptureMouse;
// }

void display_gui(void *data) {
    global_io = ImGui::GetIO();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    bool val = true;
    ImGui::Begin("Controls", &val);
    ImGui::Text("WIP AND INCOMPLETE");
    imgui_controls(data);
    ImGui::End();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

#endif
