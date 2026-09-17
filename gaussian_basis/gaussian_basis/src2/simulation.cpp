#include "simulation.hpp"
#include "cube_outline.hpp"
#include "cursor_outline3d.hpp"
#include "axes3d.hpp"
#include "metropolis.hpp"
#include "quads_scatter3d.hpp"
#include "bmp.hpp"
#include "nuclear_charges.hpp"
#include "orbitals_description.hpp"
#include "converge.hpp"
#include "build_arrays.hpp"
#include "orbital_shader_creator.hpp"
#include "compute_energies.hpp"

#include <vector>

using namespace sim_3d;


static const std::vector<float> QUAD_VERTICES = {
    -1.0, -1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, -1.0, 0.0};
static const std::vector<int> QUAD_ELEMENTS = {0, 1, 2, 0, 2, 3};
static WireFrame get_quad_wire_frame() {
    return WireFrame(
        {{"position", Attribute{
            3, GL_FLOAT, false,
            0, 0}}},
        QUAD_VERTICES, QUAD_ELEMENTS,
        WireFrame::TRIANGLES
    );
}

static struct IVec2 decompose(unsigned int n) {
    struct IVec2 d = {.ind={(int)n, 1}};
    int i = 1;
    for (; i*i < n; i++) {}
    for (; n % i; i--) {}
    d.ind[0] = ((n / i) > i)? (n / i): i;
    d.ind[1] = ((n / i) < i)? (n / i): i;
    return d;
}

Programs::Programs() {
    this->copy = Quad::make_program_from_path(
        "./shaders/util/copy.frag"
    );
    this->visualization.gradient = Quad::make_program_from_path(
        "./shaders/gradient/gradient3d.frag"
    );
    this->visualization.domain_color = Quad::make_program_from_path(
        "./shaders/vol-render/domain-coloring.frag"
    );
    this->visualization.blur = Quad::make_program_from_path(
        "./shaders/util/blur.frag"
    );
    this->visualization.cube_outline = make_program_from_paths(
        "./shaders/cube-outline/cube-outline.vert",
        "./shaders/util/uniform-color.frag"
    );
    this->visualization.cursor_outline = make_program_from_paths(
        "./shaders/cursor/outline3d.vert",
        "./shaders/util/uniform-color.frag"
    );
    this->visualization.axes_3d = make_program_from_paths(
        "./shaders/axes/axes3d.vert",
        "./shaders/axes/axes3d.frag"
    );
    this->visualization.axes_labels_3d = make_program_from_paths(
        "./shaders/axes/axes-labels3d.vert",
        "./shaders/axes/axes3d.frag"
    );
    this->orbital = 0;
}

void Programs
::set_orbital_viewer(unsigned int program) {
    this->orbital = program;
}

Frames::
Frames(const TextureParams &default_tex_params, const SimParams &params):
    sim_tex_params({
        .format=GL_RG32F,
        .width=(unsigned int)get_2d_from_3d_dimensions(
            IVec3{
                .ind{params.texelSideLength,
                     params.texelSideLength,
                     params.texelSideLength}})[0],
        .height=(unsigned int)get_2d_from_3d_dimensions(
            IVec3{
                .ind{params.texelSideLength,
                     params.texelSideLength,
                     params.texelSideLength}})[1],
        .generate_mipmap=1, // default_tex_params.generate_mipmap,
        .min_filter=default_tex_params.min_filter,
        .mag_filter=default_tex_params.mag_filter,
        .wrap_s=GL_REPEAT,
        .wrap_t=GL_REPEAT
    }),
    particles_tex_params({
        .format=GL_RGBA32F,
        .width=(unsigned int)decompose(params.numberOfParticles)[0],
        .height=(unsigned int)decompose(params.numberOfParticles)[1],
        .generate_mipmap=1, // default_tex_params.generate_mipmap,
        .min_filter=default_tex_params.min_filter,
        .mag_filter=default_tex_params.mag_filter,
        .wrap_s=GL_REPEAT,
        .wrap_t=GL_REPEAT
    }),
    data_reduce_tex_params({
        .format=GL_RGBA32F,
        .width=(unsigned int)get_2d_from_3d_dimensions(
            params.dataTexelDimensions3D)[0],
        .height=(unsigned int)get_2d_from_3d_dimensions(
            params.dataTexelDimensions3D)[1],
        .generate_mipmap=default_tex_params.generate_mipmap,
        .min_filter=default_tex_params.min_filter,
        .mag_filter=default_tex_params.mag_filter,
        .wrap_s=GL_REPEAT,
        .wrap_t=GL_REPEAT
    }),
    data_reduce(data_reduce_tex_params),
    /* wave_sim {
        .psi = {
            wave_sim_tex_params, wave_sim_tex_params, wave_sim_tex_params},
        .potential{wave_sim_tex_params},
        .tmp{wave_sim_tex_params}
    },
    particles_sim {
        .particles={trajectories_tex_params},
        .tmp={trajectories_tex_params},
        .rk4={
            trajectories_tex_params, trajectories_tex_params,
            trajectories_tex_params, trajectories_tex_params,
            trajectories_tex_params
        }
    },*/
    render_tmp(default_tex_params),
    // render_tmp2(default_tex_params),
    render(default_tex_params),
    quad_wire_frame(get_quad_wire_frame())
    /* arrows3d_frame(
        line_arrows3d::get_3d_vector_field_wire_frame(
            params.arrowDimensions)),
    conical_arrows3d_frame(
        conical_arrows3d::get_3d_vector_field_wire_frame(
            params.arrowDimensions, 13)) */
    {
}

// void Frames::reset_data_reduce_dimensions(IVec3 texel_dimensions3d) {
//     IVec2 texel_dimensions2d = get_2d_from_3d_dimensions(texel_dimensions3d);
//     this->data_reduce_tex_params.width = texel_dimensions2d[0];
//     this->data_reduce_tex_params.height = texel_dimensions2d[1];
//     data_reduce.reset(this->data_reduce_tex_params);
// }

void 
Frames::reset_simulation_discretization_dimensions(
    IVec3 texel_dimensions3d) {
    IVec2 texel_dimensions2d = get_2d_from_3d_dimensions(texel_dimensions3d);
    // this->wave_sim_tex_params.width = texel_dimensions2d[0];
    // this->wave_sim_tex_params.height = texel_dimensions2d[1];
    // for (int i = 0; i < 3; i++)
    //     this->wave_sim.psi[i].reset(this->wave_sim_tex_params);
    // this->wave_sim.potential.reset(this->wave_sim_tex_params);
    // this->wave_sim.tmp.reset(this->wave_sim_tex_params);
}

void
Frames::reset_particles_dimensions(int number_of_particles) {
    IVec2 d_2d = decompose(number_of_particles);
    // this->trajectories_tex_params.width = d_2d[0];
    // this->trajectories_tex_params.height = d_2d[1];
    // particles_sim.particles.reset(this->trajectories_tex_params);
    // particles_sim.tmp.reset(this->trajectories_tex_params);
    // for (int i = 0; i < 5; i++)
    //     particles_sim.rk4[i].reset(this->trajectories_tex_params);
}

static Vec3 scale_rotate(Vec3 r, float scale, Quaternion rotation) {
    Vec3 r2 = r - Vec3{.x=0.5, 0.5, 0.0};
    Quaternion q = rotate(
        Quaternion{.real=1.0, r2.x, r2.y, r2.z},
        rotation.conj());
    return Vec3{.x=q.i/scale, q.j/scale, q.k/scale};
}

static void take_screenshot(
    const SimParams &params, RenderTarget &render,
    std::vector<unsigned char> &image_data,
    std::vector<unsigned char> &image_rgba_arr) {
    if (params.takeScreenshots.is_recording) {
        BMPHeader header (
            params.takeScreenshots.width, 
            params.takeScreenshots.height);
        memcpy(
            (unsigned char *)&image_data[0], 
            &header, sizeof(BMPHeader));
        render.fill_array_with_contents(
            (unsigned char *)&image_rgba_arr[0]);
        for (int i = 0; i < params.takeScreenshots.height; i++) {
            for (int j = 0; j < params.takeScreenshots.width; j++) {
                // unsigned char a = image_rgba_arr[
                //     4*(i*params.takeScreenshots.width + j)];
                unsigned char r = image_rgba_arr[
                    4*(i*params.takeScreenshots.width + j) + 2];
                unsigned char g = image_rgba_arr[
                    4*(i*params.takeScreenshots.width + j) + 1];
                unsigned char b = image_rgba_arr[
                    4*(i*params.takeScreenshots.width + j)];
                image_data[
                    54 + 3*(i*params.takeScreenshots.width + j)
                ] = r;
                image_data[
                    54 + 3*(i*params.takeScreenshots.width + j) + 1
                ] = g;
                image_data[
                    54 + 3*(i*params.takeScreenshots.width + j) + 2
                ] = b;
            }
        }
        BMPHeader *header_ptr = (BMPHeader *)(&image_data[0]);
        int max_val = 0;
        for (int i = 54; i < image_data.size(); i++)
            max_val = (image_data[i] > max_val)? image_data[i]: max_val;
        // printf("Max val: %d\n", max_val);
        print_bmp_header(*header_ptr);
    }
}

Simulation::
Simulation(const TextureParams &default_tex_params, const SimParams &params
) : m_volume_render(default_tex_params,
    params.volumeTexelDimensions3D,
    params.dataTexelDimensions3D),
    m_planar_slices(default_tex_params),
    m_arrows3d(params.arrowDimensions, default_tex_params),
    m_conical_arrows3d(params.arrowDimensions, 13, default_tex_params),
    m_quads_scatter3d(decompose(params.numberOfParticles), default_tex_params),
    m_frames(default_tex_params, params),
    m_orbitals(1, 1),
    m_energies(1) {
    // m_image_data = std::vector<unsigned char>(
    //     54 + get_bmp_row_byte_size(params.takeScreenshots.width)
    //     *params.takeScreenshots.height, 0
    // );
    // m_image_rgba_arr = std::vector<unsigned char>(
    //     4*params.takeScreenshots.width*params.takeScreenshots.height, 0);
}

void Simulation::add_atom(
    const SimParams &params,
    unsigned int z, spatial::Vector position) {
    this->m_system.add_atom(z, position);
}

void Simulation::solve(const SimParams &params) {
    BasisFunctionArray arr = orbital_description_data::get_basis_function_array(
        this->m_system.get_atomic_orbital_description_list());
    int n = arr.get_number_of_basis_functions();
    NuclearChargesArray nuclear_charges
         = this->m_system.get_nuclear_charges();
    array_helpers::SquareArray overlap(n);
    array_helpers::SquareArray kinetic(n);
    array_helpers::SquareArray nuclear(n);
    array_helpers::Symmetric4 repulsion_exchange(n);

    // struct timespec frame_time[2];
    // clock_gettime(CLOCK_MONOTONIC, &frame_time[0]);
    build_arrays::fill(
        overlap, kinetic, nuclear, repulsion_exchange, 
        arr, nuclear_charges);
    // clock_gettime(CLOCK_MONOTONIC, &frame_time[1]);
    // double delta_t = frame_time[1].tv_sec - frame_time[0].tv_sec;
    // std::cout << "Construction time: " << delta_t << "s \n";

    array_helpers::SquareArray h = kinetic + nuclear;
    int n_electrons = this->m_system.electron_count();
    printf("Electron count: %d\n", n_electrons);
    printf("Selected: %d\n", params.shellMethodType.selected);
    arr.print();
    if (params.shellMethodType.selected == 1) {
        unsigned int u_count = m_system.get_up_count();
        unsigned int d_count = m_system.get_down_count();
        printf("Up count: %d\n", u_count);
        printf("Down count: %d\n", d_count);
        array_helpers::Array1D energies_up(u_count);
        array_helpers::Array1D energies_down(d_count);
        array_helpers::Array2D orbitals_up
            = orbital_description_data
            ::get_orbital_basis_function_coefficients(
            u_count, 
            this->m_system.get_atomic_orbital_description_list());
        array_helpers::Array2D orbitals_down
            = orbital_description_data
            ::get_orbital_basis_function_coefficients(
            d_count, 
            this->m_system.get_atomic_orbital_description_list());
        converge::open(
            energies_up, orbitals_up, u_count,
            energies_down, orbitals_down, d_count,
            overlap, h, repulsion_exchange,
            params.maxNumberOfIterations, true
        );
        array_helpers::Array2D orbitals = array_helpers::row_stack(
            orbitals_up, orbitals_down);
        printf("Electron count: %d\n", orbitals.row_count());
        double ke = compute_energies::kinetic(kinetic, orbitals)/2.0;
        double pe = compute_energies::nuclear_potential(nuclear, orbitals)/2.0;
        double re = compute_energies::repulsion(repulsion_exchange, orbitals);
        double ex_up = compute_energies::exchange(
            repulsion_exchange, orbitals_up);
        double ex_down = compute_energies::exchange(
            repulsion_exchange, orbitals_down);
        printf("Total energy: %g\n", ke + pe + (re - ex_up - ex_down)/2.0);
        std::string orbital_shader = get_shader_text(orbitals, arr);
        int status;
        uint32_t program;
        Quad::make_program_from_source(program, status, orbital_shader);
        if (status == GL_TRUE)
            this->m_programs.set_orbital_viewer(program);
    } else {
        array_helpers::Array1D energies(n_electrons/2);
        array_helpers::Array2D orbitals
            = orbital_description_data
            ::get_orbital_basis_function_coefficients(
            n_electrons/2, 
            this->m_system.get_atomic_orbital_description_list());
        converge::closed(
            energies, orbitals, n_electrons/2, overlap,
            h, repulsion_exchange, params.maxNumberOfIterations);
        std::string orbital_shader = get_shader_text(orbitals, arr);
        int status;
        uint32_t program;
        Quad::make_program_from_source(program, status, orbital_shader);
        if (status == GL_TRUE)
            this->m_programs.set_orbital_viewer(program);
        double ke = compute_energies::kinetic(kinetic, orbitals);
        double pe = compute_energies::nuclear_potential(
            nuclear, orbitals);
        double re = compute_energies::repulsion_exchange(
            repulsion_exchange, orbitals);
        double ne = nuclear_charges.get_energy();
        printf("Total energy: %g\n", ke + pe + re + ne // + mp2e
        );
    }

}

void Simulation::clear_atoms(const SimParams &params) {
    this->m_programs.orbital = 0;
    this->m_system.clear();
}


void Simulation::set_preset_system(
    const SimParams &params, int preset_label) {
    this->clear_atoms(params);
    int H_MOLECULE = 0, WATER = 1, CO2 = 2, O2 = 3;
    #define vec(x, y, z) spatial::Vector{.t=0.0, x, y, z}
    if (preset_label == H_MOLECULE) {
        this->add_atom(params, 1, vec(1.37, 0.0, 0.0));
        this->add_atom(params, 1, vec(0.0, 0.0, 0.0));
    }
    else if (preset_label == WATER) {
        this->add_atom(params, 1, vec(-1.93044664,  0.82666546,  0.0));
        this->add_atom(params, 1, vec(0.82666546, -1.93044664,  0.0));
        this->add_atom(params, 8, vec(0.0, 0.0, 0.0));
    } else if (preset_label == CO2) {
        this->add_atom(params, 8, vec(-2.2, 0.0,  0.0));
        this->add_atom(params, 8, vec(2.2, 0.0, 0.0));
        this->add_atom(params, 6, vec(0.0, 0.0, 0.0));
    } else if (preset_label == O2) {
        this->add_atom(params, 8, vec(2.31, 0.0,  0.0));
        this->add_atom(params, 8, vec(0.0, 0.0, 0.0));
    }
    this->solve(params);
    #undef vec3
}

void Simulation::arrows_view(const SimParams &params,
    const std::optional<Vec2> &hover,
    ::Quaternion rotation, float scale) {
    IVec3 arrows_d3d = params.arrowDimensions;
    IVec2 arrows_d2d = get_2d_from_3d_dimensions(arrows_d3d);
    IVec3 tex_d3d = params.dataTexelDimensions3D;
    IVec2 tex_d2d = get_2d_from_3d_dimensions(tex_d3d);
    Vec3 dr = Vec3{.x=1.0F, .y=1.0F, .z=1.0F};
    // this->m_frames.render_tmp.clear();
    // this->m_frames.render.clear();
    if (params.useCones) {
        m_conical_arrows3d.view(
            this->m_frames.render, this->m_frames.data_reduce,
            2.0*scale, rotation,
            params.arrowDimensions,
            params.dataTexelDimensions3D,
            {
                {"useOrthogonalProjection", 
                        (params.usePerspectiveProjection)? int(0): int(1)},
                {"rescaleZ", int(0)}
            });
    } else {
        m_arrows3d.view(
            this->m_frames.render, this->m_frames.data_reduce,
            2.0*scale, rotation, 
            params.arrowDimensions,
            params.dataTexelDimensions3D,
            {
                {"useOrthogonalProjection", 
                        (params.usePerspectiveProjection)? int(0): int(1)},
                {"rescaleZ", int(0)}
            });
    }
}

/* void Simulation::new_wave_function_from_cursor_positions(
    const SimParams &sim_params,
    Quaternion rotate, float scale,
    const Vec2 &cursor_pos1, const Vec2 &cursor_pos2, float sigma) {
    IVec2 tex_dims = m_frames.render.texture_dimensions();
    Vec3 r0 = Vec3{
        .x=cursor_pos1.x,
        .y=cursor_pos1.y*(float(tex_dims[1])/float(tex_dims[0]))
            + 0.5F*(1.0F - float(tex_dims[1])/float(tex_dims[0])),
        .z=0.0};
    r0 = scale_rotate(r0, scale, rotate) + Vec3{.x=0.5, 0.5, 0.5};
    Vec3 r1 = Vec3{
        .x=cursor_pos2.x,
        .y=cursor_pos2.y*(float(tex_dims[1])/float(tex_dims[0]))
            + 0.5F*(1.0F - float(tex_dims[1])/float(tex_dims[0])),
        .z=0.0};
    r1 = scale_rotate(r1, scale, rotate) + Vec3{.x=0.5, 0.5, 0.5};
    Vec3 r = r1 - r0;
    Vec3 wavenum = {.ind{
        float(int(r.x*sim_params.texelSideLength)),
        float(int(r.y*sim_params.texelSideLength)),
        float(int(r.z*sim_params.texelSideLength))
    }};
    for (int i = 0; i < 3; i++) {
        wavenum[i] = (wavenum[i] > sim_params.texelSideLength/4.0F)?
            sim_params.texelSideLength/4.0F: wavenum[i];
        wavenum[i] = (wavenum[i] < -sim_params.texelSideLength/4.0F)?
            -sim_params.texelSideLength/4.0F: wavenum[i];
    }
    // printf("%g, %g, %g\n", r.x, r.y, r.z);
    this->new_wave_function(sim_params, r0, wavenum);
}*/

/* void Simulation::new_wave_function_from_cursor_positions(
    const SimParams &sim_params,
    Quaternion rotate, float scale,
    const Vec3 &r0, const Vec3 &r1, float sigma) {
    // IVec2 tex_dims = m_frames.render.texture_dimensions();
    Vec3 r = r1 - r0;
    Vec3 wavenum = {.ind{
        float(int(r.x*sim_params.texelSideLength)),
        float(int(r.y*sim_params.texelSideLength)),
        float(int(r.z*sim_params.texelSideLength))
    }};
    for (int i = 0; i < 3; i++) {
        wavenum[i] = (wavenum[i] > sim_params.texelSideLength/4.0F)?
            sim_params.texelSideLength/4.0F: wavenum[i];
        wavenum[i] = (wavenum[i] < -sim_params.texelSideLength/4.0F)?
            -sim_params.texelSideLength/4.0F: wavenum[i];
    }
    // printf("%g, %g, %g\n", r.x, r.y, r.z);
    this->new_wave_function(sim_params, r0, wavenum);
} */

// void Simulation::compute_guide(
//     Quad &q2, const Quad &wave, const Quad &q, const SimParams &params) {
//     Vec3 d_3d = {
//         .x=params.sideLength, .y=params.sideLength, .z=params.sideLength};
//     IVec3 id_3d = {
//         .x=params.texelSideLength,
//         .y=params.texelSideLength, .z=params.texelSideLength};
//     IVec2 id_2d = get_2d_from_3d_dimensions(id_3d);
//     q2.draw(
//         m_programs.particles_sim.guide,
//         {
//             {"hbar", params.hbar},
//             {"m", params.m},
//             {"psiTex", &wave},
//             {"qTex", &q},
//             {"dt", 0.0F},
//             {"dimensions3D", d_3d},
//             {"texelDimensions2D", id_2d},
//             {"texelDimensions3D", id_3d},
//             // {"potentialTex",&m_frames.potential},
//             {"imposeAbsorbingBoundaries", int(0)}
//                 // int(params.addAbsorbingBoundaries)}
//         }
//     );
// }

// void Simulation::compute_guide(
//     Quad &q2,
//     const Quad &wave,
//     const Quad &q, double dt, const Quad &q_dot,
//     const SimParams &params) {
//     // printf("Use nearest sampling: %d\n", use_nearest_sampling);
//     Vec3 d_3d = {
//         .x=params.sideLength, .y=params.sideLength, .z=params.sideLength};
//     IVec3 id_3d = {
//         .x=params.texelSideLength,
//         .y=params.texelSideLength, .z=params.texelSideLength};
//     IVec2 id_2d = get_2d_from_3d_dimensions(id_3d);
//     q2.draw(
//         m_programs.particles_sim.guide,
//         {
//             {"hbar", params.hbar},
//             {"m", params.m},
//             {"psiTex", &wave},
//             {"qTex", &q},
//             {"dt", float(dt)},
//             {"qDotTex", &q_dot},
//             {"dimensions3D", d_3d},
//             {"texelDimensions2D", id_2d},
//             {"texelDimensions3D", id_3d},
//             {"imposeAbsorbingBoundaries", int(0)}
//         }
//     );
// }

// void Simulation::trajectories_time_step_rk4(const SimParams &params) {
//     float dt = params.dt;
//     float hbar = params.hbar, m = params.m;
//     Vec3 d_3d = {
//         .x=params.sideLength, .y=params.sideLength, .z=params.sideLength};
//     IVec3 id_3d = {
//         .x=params.texelSideLength,
//         .y=params.texelSideLength, .z=params.texelSideLength};
//     IVec2 id_2d = get_2d_from_3d_dimensions(id_3d);
//     m_frames.particles_sim.tmp.draw(
//         m_programs.particles_sim.guide_rk4,
//         {
//             {"hbar", hbar},
//             {"m", m},
//             {"psiLastTex", &m_frames.wave_sim.psi[this->last]},
//             {"psiCurrTex", &m_frames.wave_sim.psi[this->curr]},
//             {"psiNextTex", &m_frames.wave_sim.psi[this->next]},
//             {"qTex", &m_frames.particles_sim.particles},
//             {"dt", dt},
//             {"dimensions3D", d_3d},
//             {"texelDimensions2D", id_2d},
//             {"texelDimensions3D", id_3d},
//             {"periodicizeResult", int(1)},
//             {"imposeAbsorbingBoundaries", int(1)}
//         }
//     );
//     m_frames.particles_sim.particles.draw(
//         m_programs.copy,
//         {
//             {"tex", &m_frames.particles_sim.tmp}
//         }
//     );
//     /*
//     m_frames.particles_sim.rk4[0].draw(
//         m_programs.copy,
//         {
//             {"tex", &m_frames.particles_sim.particles}
//         });
//     // q1
//     this->compute_guide(
//         m_frames.particles_sim.rk4[1],
//         m_frames.wave_sim.psi[this->last],
//         m_frames.particles_sim.particles, 
//         params);
//     // q2
//     this->compute_guide(
//         m_frames.particles_sim.rk4[2], 
//         m_frames.wave_sim.psi[this->curr],
//         m_frames.particles_sim.particles, dt/2.0, m_frames.particles_sim.rk4[1],
//         params);
//     // q3
//     this->compute_guide(
//         m_frames.particles_sim.rk4[3], 
//         m_frames.wave_sim.psi[this->curr],
//         m_frames.particles_sim.particles, dt/2.0, m_frames.particles_sim.rk4[2],
//         params);
//     // q4
//     this->compute_guide(
//         m_frames.particles_sim.rk4[4], 
//         m_frames.wave_sim.psi[this->next],
//         m_frames.particles_sim.particles, dt, m_frames.particles_sim.rk4[3],
//         params);
//     m_frames.particles_sim.particles.draw(
//         m_programs.particles_sim.rk4,
//         {
//             {"qTex", &m_frames.particles_sim.rk4[0]},
//             {"qDotTex1", &m_frames.particles_sim.rk4[1]},
//             {"qDotTex2", &m_frames.particles_sim.rk4[2]},
//             {"qDotTex3", &m_frames.particles_sim.rk4[3]},
//             {"qDotTex4", &m_frames.particles_sim.rk4[4]},
//             {"dt", params.dt},
//             {"periodicizeResult", int(true)},
//             {"minBoundaryVal", Vec4{.ind={0.0}}},
//             {"domainDimensions", Vec4{.ind{
//                    params.sideLength,
//                    params.sideLength,
//                    params.sideLength,
//                    params.sideLength}}},
//         }
//     );*/
// }

struct Gaussian3DParams {
    Vec3 tex_offset;
    double sigma;
};

static double gaussian(const std::vector<double> &r, void *void_params) {
    Gaussian3DParams *params = (Gaussian3DParams *)void_params;
    double sigma = params->sigma;
    double x0 = (double)params->tex_offset[0];
    double y0 = (double)params->tex_offset[1];
    double z0 = (double)params->tex_offset[2];
    double gx = std::exp(-0.5*pow((r[0] - x0)/sigma, 2.0));
    double gy = std::exp(-0.5*pow((r[1] - y0)/sigma, 2.0));
    double gz = std::exp(-0.5*pow((r[2] - z0)/sigma, 2.0));
    return gx*gy*gz;
}

void Simulation::new_particles(
    const SimParams &params, Quaternion rotation, float scale,
    const Vec2 &cursor_pos
) {
    IVec2 tex_dims = m_frames.render.texture_dimensions();
    Vec3 r = Vec3{
        .x=cursor_pos.x,
        .y=cursor_pos.y*(float(tex_dims[1])/float(tex_dims[0]))
            + 0.5F*(1.0F - float(tex_dims[1])/float(tex_dims[0])),
        .z=0.0};
    r = scale_rotate(r, scale, rotation) + Vec3{.x=0.5, 0.5, 0.5};
    new_particles(params, r);
    
}

void Simulation::new_particles(
    const SimParams &params, Vec3 tex_position
) {
    /* int old_number_of_particles 
        = m_frames.trajectories_tex_params.width
            *m_frames.trajectories_tex_params.height;
    if (old_number_of_particles != params.numberOfParticles)
        m_frames.reset_trajectories_dimensions(params.numberOfParticles);
    std::vector<double> x0 = {
        (double)tex_position.x, 
        (double)tex_position.y,
        (double)tex_position.z};
    std::vector<double> delta = {
        1.5*params.sigma, 
        1.5*params.sigma,
        1.5*params.sigma};
    std::vector <double>configs 
        = std::vector<double>(params.numberOfParticles*3, 0.0);
    Gaussian3DParams gaussian_params = Gaussian3DParams {
        .tex_offset=tex_position,
        .sigma=params.sigma
    };
    metropolis(
        configs, x0, delta,
        gaussian, params.numberOfParticles, 
        (void *)&gaussian_params);
    std::vector<float> configs_f 
        = std::vector<float>(params.numberOfParticles*4); 
    for (int i = 0; i < params.numberOfParticles; i++) {
        float x = configs[3*i];
        float y = configs[3*i + 1];
        float z = configs[3*i + 2];
        configs_f[4*i] = x*params.sideLength;
        configs_f[4*i + 1] = y*params.sideLength;
        configs_f[4*i + 2] = z*params.sideLength;
        configs_f[4*i + 3] = 0.0;
        // printf("%g, %g, %g\n", x, y, z);
    }
    m_frames.particles_sim.particles.set_pixels(&configs_f[0]);
    // if (params.showTrails)
    //     m_frames.particles_sim.*/
}

/* void Simulation::new_wave_function(
    const SimParams &params, Vec3 tex_position, Vec3 wave_num) {
    Vec3 d_3d = {
        .x=params.sideLength, .y=params.sideLength, .z=params.sideLength};
    IVec3 id_3d = {
        .x=params.texelSideLength,
        .y=params.texelSideLength, .z=params.texelSideLength};
    IVec2 id_2d = get_2d_from_3d_dimensions(id_3d);
    this->last = 0;
    this->curr = 1;
    this->next = 2;
    this->m_time_step_count = 1;
    // params.t = 0.0;
    this->m_frames.wave_sim.psi[0].draw(
        m_programs.wave_sim.new_wavepacket,
        {
            {"waveNumber", wave_num},
            {"offsetTexCoord", tex_position},
            {"amplitude", 1.0F},
            {"sigmaTexCoord",
                Vec3{
                    .x=params.sigma, .y=params.sigma,
                    .z=params.sigma}},
            {"spinor", Vec4{.x=1.0, 0.0, 0.0, 0.0}},
            {"texelDimensions3D", id_3d},
            {"texelDimensions2D", id_2d},
            {"dimensions3D", d_3d}
        }
    );
    m_frames.wave_sim.psi[1].draw(
        m_programs.wave_sim.time_step,
        {
            {"m", params.m},
            {"hbar", params.hbar},
            {"dt", float(params.dt/2.0)},
            {"psi0Tex", &m_frames.wave_sim.psi[this->last]},
            {"psi1Tex", &m_frames.wave_sim.psi[this->last]},
            {"potentialTex", &m_frames.wave_sim.potential},
            {"enableVectorPotential", int(0)},
            {"dimensions3D", d_3d},
            {"texelDimensions2D", id_2d},
            {"texelDimensions3D", id_3d}
        }
    );
}*/

bool Simulation::modify_from_mouse_touch_input_planar_slices(
    const SimParams &params, Quaternion rotation, float scale,
    const std::vector<Vec2> &cursor_positions
    ) {
    IVec2 tex_dims = m_frames.render.texture_dimensions();
    Vec2 cursor_pos0_2d =  Vec2{
        .x=cursor_positions[0].x,
        .y=cursor_positions[0].y*(float(tex_dims[1])/float(tex_dims[0]))
            + 0.5F*(1.0F - float(tex_dims[1])/float(tex_dims[0])),
    };
    Vec2 cursor_pos1_2d =  Vec2{
        .x=cursor_positions[cursor_positions.size() - 1].x,
        .y=cursor_positions[cursor_positions.size() - 1].y
            *(float(tex_dims[1])/float(tex_dims[0]))
            + 0.5F*(1.0F - float(tex_dims[1])/float(tex_dims[0])),
    };
    Vec3 cursor_pos0 = m_planar_slices.most_perpendicular_intersection(
        params.dataTexelDimensions3D,
        rotation, scale,
        int(params.dataTexelDimensions3D.z
            * params.planarNormCoordOffsets[0]),
        int(params.dataTexelDimensions3D.x
            * params.planarNormCoordOffsets[1]),
        int(params.dataTexelDimensions3D.y
            * params.planarNormCoordOffsets[2]),
        cursor_pos0_2d);
    cursor_pos0 = cursor_pos0/2.0 + Vec3{.x=0.5, 0.5, 0.5};
    Vec3 cursor_pos1 = m_planar_slices.most_perpendicular_intersection(
        params.dataTexelDimensions3D,
        rotation, scale,
        int(params.dataTexelDimensions3D.z
            * params.planarNormCoordOffsets[0]),
        int(params.dataTexelDimensions3D.x
            * params.planarNormCoordOffsets[1]),
        int(params.dataTexelDimensions3D.y
            * params.planarNormCoordOffsets[2]),
        cursor_pos1_2d);
    cursor_pos1 = cursor_pos1/2.0 + Vec3{.x=0.5, 0.5, 0.5};
    Vec3 cursor_pos_last;
    if (cursor_positions.size() > 1) {
        Vec2 cursor_pos_last_2d =  Vec2{
            .x=cursor_positions[cursor_positions.size() - 2].x,
            .y=cursor_positions[cursor_positions.size() - 2].y
                *(float(tex_dims[1])/float(tex_dims[0]))
                + 0.5F*(1.0F - float(tex_dims[1])/float(tex_dims[0])),
        };
        cursor_pos_last = m_planar_slices.most_perpendicular_intersection(
            params.dataTexelDimensions3D,
            rotation, scale,
            int(params.dataTexelDimensions3D.z
                * params.planarNormCoordOffsets[0]),
            int(params.dataTexelDimensions3D.x
                * params.planarNormCoordOffsets[1]),
            int(params.dataTexelDimensions3D.y
                * params.planarNormCoordOffsets[2]),
            cursor_pos_last_2d);
        cursor_pos_last = cursor_pos_last/2.0 + Vec3{.x=0.5, 0.5, 0.5};
    }
    if (params.mouseSelector.selected == 1 && 
        this->is_inside(
            params, rotation, scale, 
            cursor_pos0 - Vec3{.x=0.5, 0.5, 0.5})) {
        // printf("%g, %g, %g\n", cursor_pos0.x, cursor_pos0.y, cursor_pos0.z);
        // Vec3 cursor_location = sim.get_cursor_location();
        // this->new_wave_function_from_cursor_positions(
        //     params, rotation,
        //     scale, 
        //     cursor_pos0, 
        //     cursor_pos1,
        //     params.sigma);
        if (cursor_positions.size() == 1)
            this->new_particles(params, cursor_pos0);
        return true;
    } else if ((params.mouseSelector.selected == 2 
        || params.mouseSelector.selected == 3
        || params.mouseSelector.selected == 4
        || params.mouseSelector.selected == 5)
        && this->is_inside(params, rotation, scale, 
            cursor_pos0 - Vec3{.x=0.5, 0.5, 0.5})
    ) {
        return true;
    }
    return false;
}

bool Simulation::modify_from_mouse_touch_input_volumetric(
    const SimParams &params, Quaternion rotation, float scale,
    const std::vector<Vec2> &cursor_positions
    ) {
    Vec2 cursor_pos0 = cursor_positions[0];
    Vec2 cursor_pos1 = cursor_positions[cursor_positions.size() - 1];
    if (params.mouseSelector.selected == 1 && 
        this->is_inside(params, rotation, 
        scale, cursor_pos0)) {
        // Vec3 cursor_location = sim.get_cursor_location();
        // this->new_wave_function_from_cursor_positions(
        //     params, rotation,
        //     scale, 
        //     cursor_pos0, 
        //     cursor_pos1,
        //     params.sigma);
        if (cursor_positions.size() == 1)
            this->new_particles(
                params, rotation, scale, cursor_pos0);
        return true;
    } else if ((params.mouseSelector.selected == 2 
        || params.mouseSelector.selected == 3
        || params.mouseSelector.selected == 4
        || params.mouseSelector.selected == 5)
        && this->is_inside(params, rotation, scale, cursor_pos0)
    ) {
        return true;
    }
    return false;
}

bool Simulation::modify_from_mouse_touch_input(
    const SimParams &params, Quaternion rotation, float scale,
    const std::vector<Vec2> &cursor_positions
) {
    if (params.visualizationSelect.selected == VOL_RENDER_VIEW) {
        return this->modify_from_mouse_touch_input_volumetric(
            params, rotation, scale, cursor_positions);
    }
    if (params.visualizationSelect.selected == PLANAR_SLICES_VIEW) {
        return this->modify_from_mouse_touch_input_planar_slices(
            params, rotation, scale, cursor_positions);
    }
    return false;
}

/* void Simulation::time_step(const SimParams &params) {
    Vec3 d_3d = {
        .x=params.sideLength, .y=params.sideLength, .z=params.sideLength};
    IVec3 id_3d = {
        .x=params.texelSideLength,
        .y=params.texelSideLength, .z=params.texelSideLength};
    IVec2 id_2d = get_2d_from_3d_dimensions(id_3d);
    m_frames.wave_sim.psi[this->next].draw(
        m_programs.wave_sim.time_step,
        {
            {"m", params.m},
            {"hbar", params.hbar},
            {"dt", params.dt},
            {"psi0Tex", &m_frames.wave_sim.psi[this->last]},
            {"psi1Tex", &m_frames.wave_sim.psi[this->curr]},
            {"potentialTex", &m_frames.wave_sim.potential},
            {"enableVectorPotential", int(0)},
            {"dimensions3D", d_3d},
            {"texelDimensions2D", id_2d},
            {"texelDimensions3D", id_3d}
        }
    );
    int new_time_slice_positions[3] = {
        this->curr, this->next, this->last
    };
    // printf("Frames: %d, %d, %d\n", this->last, this->curr, this->next);
    this->last = new_time_slice_positions[0];
    this->curr = new_time_slice_positions[1];
    this->next = new_time_slice_positions[2];
    if (m_time_step_count % 2 && m_time_step_count != 0) {
        trajectories_time_step_rk4(params);
    }
    m_time_step_count++;
} */


const RenderTarget &Simulation
::view(
    const SimParams &params,
    const std::optional<Vec2> &hover,
    ::Quaternion rotation, float scale) {
    // printf("This is some text.\n");
    if (m_programs.orbital > 0) {
        spatial::Vector center
             = this->m_system.get_nuclear_charges().get_center();
        float rad
            = this->m_system.get_nuclear_charges().furthest_from_center();
        if (this->m_system.get_nuclear_charges().size() <= 1) { 
            rad = 1.0;
        } else {

        }
        rad *= params.sizeScale;
        int max_e_count = this->m_system.electron_count()/2;
        if (params.shellMethodType.selected == 1)
            max_e_count = 
                this->m_system.get_up_count() 
                    + this->m_system.get_down_count();
        this->m_frames.data_reduce.draw(
            m_programs.orbital,
            {
                {"showTotalDensity", int(params.showDensity)},
                {"orbitalCount", max_e_count},
                {"orbitalIndex", int(std::min(
                    params.whichOrbitalSliderVal,
                    max_e_count))},
                {"texelDimensions3D", params.dataTexelDimensions3D},
                {"texelDimensions2D",
                    IVec2{.ind{
                        (int)m_frames.data_reduce_tex_params.width,
                        (int)m_frames.data_reduce_tex_params.height
                    }}},
            {"center", Vec3{.ind{2.0F*rad, 2.0F*rad, 2.0F*rad}}},
            {"dimensions3D", Vec3{
                .ind{4.0F*rad, 4.0F*rad, 4.0F*rad}}}
                
            }
        );
    }
    switch(params.visualizationSelect.selected) {
        case PLANAR_SLICES_VIEW: {
            this->m_frames.render.clear();
            this->m_frames.render_tmp.clear();
            Vec2 scaled_hover;
            if (hover.has_value()) {
                IVec2 tex_dims = m_frames.render.texture_dimensions();
                scaled_hover = Vec2{
                    .x=hover->x,
                    .y=hover->y*(float(tex_dims[1])/float(tex_dims[0]))
                        + 0.5F*(1.0F - float(tex_dims[1])/float(tex_dims[0])) 
                };
            }
            m_planar_slices.view(
                this->m_frames.render,
                this->m_frames.data_reduce, params.dataTexelDimensions3D,
                rotation, scale,
                int(params.dataTexelDimensions3D.z
                    *params.planarNormCoordOffsets[0]),
                int(params.dataTexelDimensions3D.x
                    *params.planarNormCoordOffsets[1]),
                int(params.dataTexelDimensions3D.y
                    *params.planarNormCoordOffsets[2]),
                (hover.has_value())? scaled_hover: Vec2{.ind {0.0, 0.0}},
                params.usePerspectiveProjection
            );
            this->m_cursor_location = m_planar_slices.most_perpendicular_intersection(
                params.dataTexelDimensions3D,
                rotation, scale,
                int(params.dataTexelDimensions3D.z
                    *params.planarNormCoordOffsets[0]),
                int(params.dataTexelDimensions3D.x
                    *params.planarNormCoordOffsets[1]),
                int(params.dataTexelDimensions3D.y
                    *params.planarNormCoordOffsets[2]),
                (hover.has_value())? scaled_hover: Vec2{.ind {0.0, 0.0}}
            );
            WireFrame axes = axes3d::get_axes_wireframe();
            WireFrame axes_labels = axes3d::get_xyz_axes_labels_wireframe();
            /* m_quads_scatter3d.view(
                this->m_frames.render,
                this->m_frames.particles_sim.particles,
                scale, rotation,
                Vec3{.ind{
                    params.sideLength, params.sideLength, 
                    params.sideLength
                }});
            axes3d::draw_axes(
                this->m_frames.render,
                {
                    .axes=m_programs.visualization.axes_3d,
                    .labels=m_programs.visualization.axes_labels_3d},
                axes, axes_labels,
                rotation, 110, 0.0F,
                params.usePerspectiveProjection, 
                m_frames.render.texture_dimensions());*/
            take_screenshot(
                params, m_frames.render, 
                m_image_data, m_image_rgba_arr);
            return m_frames.render;
        }
        case VOL_RENDER_VIEW: {
            this->m_frames.render.clear();
            this->m_frames.render_tmp.clear();
            // this->m_frames.render_tmp2.clear();
            WireFrame cube_outline = get_cube_outline_wire_frame();
            m_volume_render.view(
                this->m_frames.render, this->m_frames.data_reduce,
                scale, rotation,
                params.alphaBrightness, 
                params.colorBrightness,
                params.usePerspectiveProjection
                // {{"noiseScale", params.noiseScale}}
            );
            if (params.blurSize >= 1 && params.applyBlur) { 
                this->m_frames.render_tmp.draw(
                    m_programs.visualization.blur,
                    {{"tex", {this->m_frames.render}}, 
                    {"textureDimensions2D",
                            m_frames.render.texture_dimensions()},
                    {"orientation", int(0)},
                    {"size", int(params.blurSize)}},
                    m_frames.quad_wire_frame
                );
                this->m_frames.render.draw(
                    m_programs.visualization.blur,
                    {{"tex", {this->m_frames.render_tmp}}, 
                    {"textureDimensions2D",
                        m_frames.render.texture_dimensions()},
                    {"orientation", int(1)},
                    {"size", int(params.blurSize)}},
                    m_frames.quad_wire_frame
                );
            }
            this->m_frames.render.draw(
                m_programs.visualization.cube_outline,
                {
                    {"rotation", rotation},
                    {"viewScale", scale},
                    {"color", Vec4{.ind{1.0, 1.0, 1.0, 0.5}}},
                    {"usePerspectiveProjection", 
                            int(params.usePerspectiveProjection)},
                    {"screenDimensions", m_frames.render.texture_dimensions()}
                },
                cube_outline
            );
            /* m_quads_scatter3d.view(
                this->m_frames.render,
                this->m_frames.particles_sim.particles,
                scale, rotation,
                Vec3{.ind{
                    params.sideLength, params.sideLength, 
                    params.sideLength
            }});*/
            if (hover.has_value()) {
                IVec2 tex_dims = m_frames.render.texture_dimensions();
                Vec3 r = Vec3{
                    .x=hover->x,
                    .y=hover->y*(float(tex_dims[1])/float(tex_dims[0]))
                        + 0.5F*(1.0F - float(tex_dims[1])/float(tex_dims[0])),
                    .z=0.0};
                r = 2.0*scale_rotate(r, scale, rotation);
                if (r.x >= -1.0 && r.x < 1.0 && 
                    r.y >= -1.0 && r.y < 1.0 &&
                    r.z >= -1.0 && r.z < 1.0) {
                    this->m_cursor_location = r;
                    // std::cout << r.x << ", " << r.y << ", " << r.z << std::endl;
                    WireFrame cursor_frame 
                        = cursor_outline3d::get_cursor_wire_frame();
                    this->m_frames.render.draw(
                        m_programs.visualization.cursor_outline,
                        {
                            {"rotation", rotation},
                            {"viewScale", scale},
                            {"cursorPosition", r},
                            {"color", Vec4{.ind{0.3, 0.3, 0.3, 0.1}}},
                            {"usePerspectiveProjection", 
                                    int(params.usePerspectiveProjection)},
                            {"screenDimensions", m_frames.render.texture_dimensions()}
                        },
                        cursor_frame
                    );
                }
            }
            WireFrame axes = axes3d::get_axes_wireframe();
            WireFrame axes_labels = axes3d::get_xyz_axes_labels_wireframe();
            axes3d::draw_axes(
                this->m_frames.render,
                {.axes=m_programs.visualization.axes_3d, .labels=m_programs.visualization.axes_labels_3d},
                axes, axes_labels,
                rotation, 110, 0.0F,
                params.usePerspectiveProjection, 
                m_frames.render.texture_dimensions());
            take_screenshot(
                params, m_frames.render, 
                m_image_data, m_image_rgba_arr);
            return this->m_frames.render;
        }
    }
}

const RenderTarget &Simulation
::view_data_texture(SimParams &params, ::Quaternion rotation, float scale) {
    m_frames.render.draw(
        m_programs.copy,
        {{"tex", &m_frames.data_reduce}},
        m_frames.quad_wire_frame
    );
    return m_frames.render;
}

const RenderTarget &Simulation
::view_volume_texture(
    SimParams &params, ::Quaternion rotation, float scale
    ) {
    m_frames.render.draw(
        m_programs.copy,
        {{"tex", &m_frames.data_reduce}},
        m_frames.quad_wire_frame
    );
    return m_frames.render;
}

void Simulation::reset_data_reduce_dimensions(IVec3 texel_dimensions_3d) {
    m_volume_render.reset_data_dimensions(texel_dimensions_3d);
    // m_frames.reset_data_reduce_dimensions(texel_dimensions_3d);
}

void 
Simulation::reset_simulation_discretization_dimensions(
    IVec3 texel_dimensions3d) {
    m_frames.reset_simulation_discretization_dimensions(texel_dimensions3d);
}

void Simulation::reset_volume_dimensions(IVec3 texel_dimensions_3d) {
    m_volume_render.reset_volume_dimensions(texel_dimensions_3d);
}

void Simulation::reset_volume_filtering(unsigned int filtering) {
    m_volume_render.reset_filtering(filtering);
}

void
Simulation::reset_particles_dimensions(int number_of_particles) {
    // this->m_frames.reset_trajectories_dimensions(number_of_particles);
}

Vec3 Simulation::get_cursor_location() const {
    return m_cursor_location;
}

Vec3 Simulation::get_scaled_cursor_location(const SimParams &params) const {
    // float side_length = params.sideLength;
    float side_length = (float)params.texelSideLength;
    return Vec3{
        .x=m_cursor_location.x*side_length/2.0F,
        .y=m_cursor_location.y*side_length/2.0F,
        .z=m_cursor_location.z*side_length/2.0F,
    };
}


bool Simulation::is_inside(
    const SimParams &params, Quaternion rotate, float scale,
    const Vec3 &r) const {
    // return true;
    return (r.x > -0.5 && r.y > -0.5 && r.z > -0.5 && 
            r.x < 0.5 && r.y < 0.5 && r.z < 0.5);
}

bool Simulation::is_inside(
    const SimParams &params, Quaternion rotate, float scale,
    const Vec2 &cursor_pos) const {
    IVec2 tex_dims = m_frames.render.texture_dimensions();
    Vec3 r = Vec3{
        .x=cursor_pos.x,
        .y=cursor_pos.y*(float(tex_dims[1])/float(tex_dims[0]))
        + 0.5F*(1.0F - float(tex_dims[1])/float(tex_dims[0])),
        .z=0.0};
    r = scale_rotate(r, scale, rotate);
    return (r.x > -0.5 && r.y > -0.5 && r.z > -0.5 && 
            r.x < 0.5 && r.y < 0.5 && r.z < 0.5);
}

std::vector<unsigned char> &Simulation::get_image_data() {
    return m_image_data;
}


