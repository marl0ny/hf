#include "gl_wrappers.hpp"
#include "parameters.hpp"
#include "volume_render.hpp"
#include "planar_slice.hpp"
#include "line_arrows3d.hpp"
#include "conical_arrow3d.hpp"
#include "quads_scatter3d.hpp"

#include "spatial.hpp"
#include "simple_system.hpp"

#ifndef _SIMULATION_
#define _SIMULATION_

using namespace sim_3d;

struct Frames {
    TextureParams sim_tex_params;
    TextureParams particles_tex_params;
    TextureParams data_reduce_tex_params;
    Quad data_reduce;
    /* struct {
        Quad psi[3];
        Quad potential;
        Quad tmp;
    } wave_sim;
    struct {
        Quad particles;
        Quad tmp;
        Quad rk4[5];
    } particles_sim; */
    RenderTarget render_tmp;
    RenderTarget render;
    WireFrame quad_wire_frame;
    // WireFrame arrows3d_frame;
    // WireFrame conical_arrows3d_frame;
    Frames(const TextureParams &default_tex_params, const SimParams &params);
    // void reset_data_reduce_dimensions(IVec3 texel_dimensions3d);
    void reset_simulation_discretization_dimensions(IVec3 texel_dimensions3d);
    void reset_particles_dimensions(int number_of_particles);
};

struct Programs {
    unsigned int copy;
    unsigned int add2, add3;
    struct {
        unsigned int domain_color;
        unsigned int blur;
        unsigned int gradient;
        unsigned int cube_outline;
        unsigned int cursor_outline;
        unsigned int axes_3d;
        unsigned int axes_labels_3d;
    } visualization;
    unsigned int orbital;
    /* struct {
        unsigned int new_wavepacket;
        unsigned int time_step;
    } wave_sim;
    struct {
        unsigned int forward_euler;
        unsigned int rk4;
        unsigned int guide;
        unsigned int guide_rk4;
    } particles_sim;*/
    // unsigned int user_defined;
    Programs();
    void set_orbital_viewer(unsigned int program);
};

class Simulation {
    volume_render::VolumeRender m_volume_render;
    planar_slice::PlanarSlices m_planar_slices;
    line_arrows3d::Arrows m_arrows3d;
    conical_arrows3d::Arrows m_conical_arrows3d;
    quads_scatter3d::Scatter m_quads_scatter3d;
    Vec3 m_cursor_location;
    Programs m_programs;
    Frames m_frames;
    System m_system;
    array_helpers::Array2D m_orbitals;
    array_helpers::Array1D m_energies;
    int m_time_step_count;
    int last, curr, next;
    float m_rad;
    std::vector<unsigned char> m_image_rgba_arr;
    std::vector<unsigned char> m_image_data;
    enum {VOL_RENDER_VIEW=0, PLANAR_SLICES_VIEW=1};

    const RenderTarget
    &view_volume_render(
        SimParams &params, ::Quaternion rotation, float scale);
    const RenderTarget
    &view_planar_slices(
        SimParams &params, ::Quaternion rotation, float scale);

    // void compute_guide(
    //     Quad &q2, const Quad &wave, const Quad &q,
    //     const SimParams &params);
    // void compute_guide(
    //     Quad &q2,
    //     const Quad &wave,
    //     const Quad &q, double dt, const Quad &q_dot,
    //     const SimParams &params);
    // void trajectories_time_step_rk4(const SimParams &params);

    void arrows_view(const SimParams &params,
        const std::optional<Vec2> &hover,
        ::Quaternion rotation, float scale);

    void new_particles(
        const SimParams &params, Vec3 tex_position);
    void new_particles(
        const SimParams &params, Quaternion rotation, float scale,
        const Vec2 &cursor_pos);
    
    // void new_wave_function_from_cursor_positions(
    //     const SimParams &sim_params,
    //     Quaternion rotate, float scale,
    //     const Vec2 &cursor_pos1, const Vec2 &cursor_pos2, float sigma);
    // void new_wave_function_from_cursor_positions(
    //     const SimParams &sim_params,
    //     Quaternion rotate, float scale,
    //     const Vec3 &cursor_pos1, const Vec3 &cursor_pos2, float sigma);

    public:

    Simulation(const TextureParams &default_tex_params,
               const SimParams &params);

    bool modify_from_mouse_touch_input_volumetric(
        const SimParams &params, Quaternion rotation, float scale,
        const std::vector<Vec2> &cursor_positions
    );
    bool modify_from_mouse_touch_input_planar_slices(
        const SimParams &params, Quaternion rotation, float scale,
        const std::vector<Vec2> &cursor_positions
    );
    bool modify_from_mouse_touch_input(
        const SimParams &params, Quaternion rotation, float scale,
        const std::vector<Vec2> &cursor_positions
    );

    spatial::Vector get_position_of_cursor(
        const SimParams &params,
        const Vec2 &cursor_positions,
        Quaternion rotation, float scale
    ) const;

    void add_atom(
        const SimParams &params,
        unsigned int z, spatial::Vector position);
    void solve(const SimParams &params);
    void clear_atoms(const SimParams &params);
    void set_preset_system(
        const SimParams &params, int preset_label);
    // double get_total_energy() const;

    const RenderTarget
    &view(const SimParams &params,
        const std::optional<Vec2> &hover,
        ::Quaternion rotation, float scale);

    // Reset dimensions
    void reset_simulation_discretization_dimensions(
        IVec3 texel_dimensions_3d);
    void reset_data_reduce_dimensions(IVec3 texel_dimensions_3d);
    void reset_volume_dimensions(IVec3 volume_dimensions_3d);
    void reset_volume_filtering(unsigned int filtering);
    void reset_particles_dimensions(int number_of_particles);
    Vec3 get_cursor_location() const;
    Vec3 get_scaled_cursor_location(const SimParams &params) const;
    bool is_inside(
        const SimParams &params, Quaternion rotate, float scale,
        const Vec3 &r) const;
    bool is_inside(
        const SimParams &params, Quaternion rotate, float scale,
        const Vec2 &cursor_pos) const;
    
    std::vector<unsigned char> &get_image_data();

    // For debugging.
    const RenderTarget &view_data_texture(
        SimParams &params, ::Quaternion rotation, float scale
    );
    const RenderTarget &view_volume_texture(
        SimParams &params, ::Quaternion rotation, float scale
    );
};

#endif