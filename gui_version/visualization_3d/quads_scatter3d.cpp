#include "quads_scatter3d.hpp"

static std::vector<float> get_vertices_set_elements(std::vector<int> &elements, IVec2 d_2d) {
    std::vector<float> vertices {};
    int element_count = 0;
    for (int i = 0; i < d_2d[1]; i++) {
        for (int j = 0; j < d_2d[0]; j++) {
            float u = (float(j) + 0.5F)/float(d_2d[0]);
            float v = (float(i) + 0.5F)/float(d_2d[1]);
            int lb, lu, ru, rb;
            vertices.push_back(u);
            vertices.push_back(v);
            vertices.push_back(-1.0F);
            vertices.push_back(-1.0F);
            lb = element_count;
            element_count++;
            vertices.push_back(u);
            vertices.push_back(v);
            vertices.push_back(-1.0F);
            vertices.push_back(1.0F);
            lu = element_count;
            element_count++;
            vertices.push_back(u);
            vertices.push_back(v);
            vertices.push_back(1.0F);
            vertices.push_back(1.0F);
            ru = element_count;
            element_count++;
            vertices.push_back(u);
            vertices.push_back(v);
            vertices.push_back(1.0F);
            vertices.push_back(-1.0F);
            rb = element_count;
            element_count++;
            for (const int &e: {rb, lb, lu})
                elements.push_back(e);
            for (const int &e: {rb, lu, ru})
                elements.push_back(e);

        }
    }
    return vertices;
}

WireFrame quads_scatter3d::get_scatter_wire_frame(IVec2 d_2d) {
    Attributes attributes = {
        {"position", {
            .size=4, .type=GL_FLOAT, .normalized=false, .stride=0, .offset=0
    }}};
    std::vector<int> elements {};
    std::vector<float> vertices = get_vertices_set_elements(elements, d_2d);
    return WireFrame(attributes, vertices, elements, WireFrame::TRIANGLES);
}

quads_scatter3d::Programs::Programs() {
    this->quads_scatter = make_program_from_paths(
        "./shaders/quad-scatter/quads.vert",
        "./shaders/util/uniform-color.frag"
    );
}

quads_scatter3d::Frames::Frames(
    const TextureParams &default_texture_params, IVec2 d_2d
): quads(quads_scatter3d::get_scatter_wire_frame(d_2d)), 
    dimensions2d(d_2d) {}


void quads_scatter3d::Frames
::reset_dimensions(IVec2 d_2d) {
    this->dimensions2d = d_2d;
    this->quads = get_scatter_wire_frame(d_2d);
}

quads_scatter3d::Scatter
::Scatter(IVec2 d_2d, TextureParams default_tex_params):
    m_programs(quads_scatter3d::Programs()),
    m_frames(quads_scatter3d::Frames(default_tex_params, d_2d)) {}

void quads_scatter3d::Scatter::view(
    RenderTarget &dst, const Quad &src,
    float scale, Quaternion rotation,
    IVec2 scatter_dimensions2d,
    // IVec3 src_texel_dimensions3d,
    Vec3 simulation_dimensions3d,
    Uniforms additional_uniforms
) {
    if (scatter_dimensions2d.x != m_frames.dimensions2d.x ||
        scatter_dimensions2d.y != m_frames.dimensions2d.y)
        m_frames.reset_dimensions(scatter_dimensions2d);
    this->view(
        dst, src, scale, rotation, 
        simulation_dimensions3d,
        additional_uniforms);
}

void quads_scatter3d::Scatter::view(
    RenderTarget &dst, const Quad &src,
    float scale, Quaternion rotation,
    // IVec3 src_texel_dimensions3d,
    Vec3 simulation_dimensions3d,
    Uniforms additional_uniforms
) {
    // IVec3 id3d = src_texel_dimensions3d;
    // IVec2 id2d = get_2d_from_3d_dimensions(id3d);
    Vec3 d3d = simulation_dimensions3d;
    // IVec2 scatter_2d = get_2d_from_3d_dimensions(m_frames.dimensions2d);
    Uniforms uniforms {
        {"scatterTex", &src},
        {"quadScale", float(0.001)},
        {"rotation", rotation},
        {"scale", float(scale)},
        {"screenDimensions", dst.texture_dimensions()},
        // {"texelDimensions2D", id2d},
        // {"texelDimensions3D", id3d},
        {"simulationDimensions3D", d3d},
        // {"scatterDimensions2D", m_frames.dimensions2d},
        {"color", Vec4{.r=1.0, .g=1.0, .b=1.0, .a=1.0}}
    };
    for (auto &e: additional_uniforms)
        uniforms.insert(e);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    dst.draw(
        m_programs.quads_scatter,
        uniforms,
        m_frames.quads
    );
    glDisable(GL_DEPTH_TEST);
}