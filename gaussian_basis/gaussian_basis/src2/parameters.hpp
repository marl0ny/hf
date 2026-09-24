#include "gl_wrappers.hpp"

namespace sim_3d {

#ifndef _PARAMETERS_
#define _PARAMETERS_

struct Button {};

struct UploadImage {};

typedef std::string Label;

typedef bool BoolRecord;

struct BMPRecord {
    bool is_recording;
    int width, height;
};

typedef std::vector<std::string> EntryBoxes;

struct SelectionList {
    int selected;
    std::vector<std::string> options;
};

struct LineDivider {};

struct SubSectionStart {};

struct SubSectionEnd {};

struct HoveringCanvasLabel { std::string contents; };

struct LinkedLabel { std::string contents; };

struct KaTeXLabel {};

struct NotUsed {};

struct SimParams {
    LinkedLabel link = {"https://github.com/marl0ny/hf"};
    SelectionList mouseSelector = SelectionList{0, {"Rotate only", "Place atom"}};
    int texelSideLength = (int)(64);
    SelectionList texelSideLengthSelector = SelectionList{0, {"64x64x64", "128x128x128", "256x256x256"}};
    IVec3 dataTexelDimensions3D = (IVec3)(IVec3 {.ind={64, 64, 64}});
    SelectionList presetCompoundsDropdown = SelectionList{1, {"Hydrogen molecule", "Water", "Carbon Dioxide", "Oxygen Molecule", "Methane", "Acetylene", "Benzene (Will take a loong time!)"}};
    SelectionList presetAtoms = SelectionList{0, {"H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr"}};
    int maxNumberOfIterations = (int)(20);
    Button solve = Button{};
    Button clear = Button{};
    SelectionList shellMethodType = SelectionList{0, {"All shells closed", "Unrestricted"}};
    float sizeScale = (float)(1.0F);
    bool showDensity = (bool)(true);
    int whichOrbitalSliderVal = (int)(3);
    int numberOfParticles = (int)(65536);
    SubSectionStart visualizationControlsStart = SubSectionStart{};
    SelectionList visualizationSelect = SelectionList{0, {"Volume render", "Three orthogonal planar slices"}};
    bool usePerspectiveProjection = (bool)(true);
    float brightness = (float)(1.0F);
    SubSectionStart volumeRenderSectionStart = SubSectionStart{};
    bool useLinear = (bool)(false);
    float alphaBrightness = (float)(2.0F);
    float colorBrightness = (float)(1.0F);
    IVec3 volumeTexelDimensions3D = (IVec3)(IVec3 {.ind={128, 128, 192}});
    bool applyBlur = (bool)(true);
    int blurSize = (int)(5);
    SubSectionEnd volumeRenderSectionEnd = SubSectionEnd{};
    SubSectionStart planarSlicesSectionStart = SubSectionStart{};
    Vec3 planarNormCoordOffsets = (Vec3)(Vec3 {.ind={0.5, 0.5, 0.5}});
    SubSectionEnd planarSlicesSectionEnd = SubSectionEnd{};
    SubSectionStart arrows3DLineSectionStart = SubSectionStart{};
    IVec3 arrowDimensions = (IVec3)(IVec3 {.ind={8, 8, 8}});
    bool useCones = (bool)(false);
    SubSectionEnd arrows3DLineSectionEnd = SubSectionEnd{};
    SubSectionEnd visualizationControlsEnd = SubSectionEnd{};
    BMPRecord takeScreenshots = BMPRecord{false, 1440, 1440};
    HoveringCanvasLabel canvasHoverDisplay = HoveringCanvasLabel{};
    int dummyValue = (int)(0);
    enum {
        LINK=0,
        MOUSE_SELECTOR=1,
        TEXEL_SIDE_LENGTH=2,
        TEXEL_SIDE_LENGTH_SELECTOR=3,
        DATA_TEXEL_DIMENSIONS3_D=4,
        PRESET_COMPOUNDS_DROPDOWN=5,
        PRESET_ATOMS=6,
        MAX_NUMBER_OF_ITERATIONS=7,
        SOLVE=8,
        CLEAR=9,
        SHELL_METHOD_TYPE=10,
        SIZE_SCALE=11,
        SHOW_DENSITY=12,
        WHICH_ORBITAL_SLIDER_VAL=13,
        NUMBER_OF_PARTICLES=14,
        VISUALIZATION_CONTROLS_START=15,
        VISUALIZATION_SELECT=16,
        USE_PERSPECTIVE_PROJECTION=17,
        BRIGHTNESS=18,
        VOLUME_RENDER_SECTION_START=19,
        USE_LINEAR=20,
        ALPHA_BRIGHTNESS=21,
        COLOR_BRIGHTNESS=22,
        VOLUME_TEXEL_DIMENSIONS3_D=23,
        APPLY_BLUR=24,
        BLUR_SIZE=25,
        VOLUME_RENDER_SECTION_END=26,
        PLANAR_SLICES_SECTION_START=27,
        PLANAR_NORM_COORD_OFFSETS=28,
        PLANAR_SLICES_SECTION_END=29,
        ARROWS3_D_LINE_SECTION_START=30,
        ARROW_DIMENSIONS=31,
        USE_CONES=32,
        ARROWS3_D_LINE_SECTION_END=33,
        VISUALIZATION_CONTROLS_END=34,
        TAKE_SCREENSHOTS=35,
        CANVAS_HOVER_DISPLAY=36,
        DUMMY_VALUE=37,
    };
    void set(int enum_val, Uniform val) {
        switch(enum_val) {
            case TEXEL_SIDE_LENGTH:
            texelSideLength = val.i32;
            break;
            case DATA_TEXEL_DIMENSIONS3_D:
            dataTexelDimensions3D = val.ivec3;
            break;
            case MAX_NUMBER_OF_ITERATIONS:
            maxNumberOfIterations = val.i32;
            break;
            case SIZE_SCALE:
            sizeScale = val.f32;
            break;
            case SHOW_DENSITY:
            showDensity = val.b32;
            break;
            case WHICH_ORBITAL_SLIDER_VAL:
            whichOrbitalSliderVal = val.i32;
            break;
            case NUMBER_OF_PARTICLES:
            numberOfParticles = val.i32;
            break;
            case USE_PERSPECTIVE_PROJECTION:
            usePerspectiveProjection = val.b32;
            break;
            case BRIGHTNESS:
            brightness = val.f32;
            break;
            case USE_LINEAR:
            useLinear = val.b32;
            break;
            case ALPHA_BRIGHTNESS:
            alphaBrightness = val.f32;
            break;
            case COLOR_BRIGHTNESS:
            colorBrightness = val.f32;
            break;
            case VOLUME_TEXEL_DIMENSIONS3_D:
            volumeTexelDimensions3D = val.ivec3;
            break;
            case APPLY_BLUR:
            applyBlur = val.b32;
            break;
            case BLUR_SIZE:
            blurSize = val.i32;
            break;
            case PLANAR_NORM_COORD_OFFSETS:
            planarNormCoordOffsets = val.vec3;
            break;
            case ARROW_DIMENSIONS:
            arrowDimensions = val.ivec3;
            break;
            case USE_CONES:
            useCones = val.b32;
            break;
            case DUMMY_VALUE:
            dummyValue = val.i32;
            break;
        }
    }
    Uniform get(int enum_val) const {
        switch(enum_val) {
            case TEXEL_SIDE_LENGTH:
            return {(int)texelSideLength};
            case DATA_TEXEL_DIMENSIONS3_D:
            return {(IVec3)dataTexelDimensions3D};
            case MAX_NUMBER_OF_ITERATIONS:
            return {(int)maxNumberOfIterations};
            case SIZE_SCALE:
            return {(float)sizeScale};
            case SHOW_DENSITY:
            return {(bool)showDensity};
            case WHICH_ORBITAL_SLIDER_VAL:
            return {(int)whichOrbitalSliderVal};
            case NUMBER_OF_PARTICLES:
            return {(int)numberOfParticles};
            case USE_PERSPECTIVE_PROJECTION:
            return {(bool)usePerspectiveProjection};
            case BRIGHTNESS:
            return {(float)brightness};
            case USE_LINEAR:
            return {(bool)useLinear};
            case ALPHA_BRIGHTNESS:
            return {(float)alphaBrightness};
            case COLOR_BRIGHTNESS:
            return {(float)colorBrightness};
            case VOLUME_TEXEL_DIMENSIONS3_D:
            return {(IVec3)volumeTexelDimensions3D};
            case APPLY_BLUR:
            return {(bool)applyBlur};
            case BLUR_SIZE:
            return {(int)blurSize};
            case PLANAR_NORM_COORD_OFFSETS:
            return {(Vec3)planarNormCoordOffsets};
            case ARROW_DIMENSIONS:
            return {(IVec3)arrowDimensions};
            case USE_CONES:
            return {(bool)useCones};
            case DUMMY_VALUE:
            return {(int)dummyValue};
        }
        return Uniform(0);
    }
    void set(int enum_val, int index, std::string val) {
        switch(enum_val) {
        }
    }
};
#endif
}
