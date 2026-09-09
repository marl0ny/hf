
#include "gl_wrappers.hpp"

#ifndef _QUADS_SCATTER3D_
#define _QUADS_SCATTER3D_


namespace quads_scatter3d {

WireFrame get_scatter_wire_frame(
    IVec2 d_2d);

struct Programs {
    unsigned int quads_scatter;
    Programs();
};

struct Frames {
    WireFrame quads;
    IVec2 dimensions2d;
    void reset_dimensions(IVec2 d_2d);
    Frames(const TextureParams &default_texture_params,
           IVec2 d_2d);
};

class Scatter {
    Programs m_programs;
    Frames m_frames;
    int points_per_cone_circle;
    public:
    Scatter(
        IVec2 d_2d, 
        TextureParams default_tex_params);
    void view(
        RenderTarget &dst, const Quad &src,
        float scale, Quaternion rotation,
        // IVec3 src_texel_dimensions3d,
        Vec3 simulation_dimensions3d,
        Uniforms additional_uniforms = {});
    void view(
        RenderTarget &dst, const Quad &src,
        float scale, Quaternion rotation,
        IVec2 scatter_dimensions2d,
        // IVec3 src_texel_dimensions3d,
        Vec3 simulation_dimensions3d,
        Uniforms additional_uniforms = {});
};

}

#endif