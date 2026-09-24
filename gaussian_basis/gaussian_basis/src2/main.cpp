#include "gl_wrappers.hpp"
#include "glfw_window.hpp"
#include "parameters.hpp"
#include "interactor.hpp"
#include "simulation.hpp"

#include "simple_examples.hpp"

#include <GLFW/glfw3.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/bind.h>
#include "ui_wrappers/wasm.hpp"
#else
#include "ui_wrappers/imgui.hpp"
#endif

#include <functional>
#include <utility>
#include <iostream>
#include <vector>

#include "spatial.hpp"


static std::function <void()> s_loop;
#ifdef __EMSCRIPTEN__
static void s_main_loop() {
    s_loop();
}
#endif

using namespace sim_3d;



void simulation_ui_interface_handler(
    MainGLFWQuad main_render,
    TextureParams default_tex_params,  // Default texture parameters
    SimParams &params  // Parameters of the simulations
) {
    Interactor interactor(main_render.get_window());
    Simulation sim(default_tex_params, params);
    SimParams modified_params {};

    // For handling mouse or touch interation.
    std::optional<Vec2> hover_position;
    std::optional<Vec2> start_position;
    std::vector<Vec2> cursor_positions {};
    std::optional<std::pair<Vec2, Vec2>> start_double_touches;
    std::vector<std::pair<Vec2, Vec2>> double_touches_positions {};
    Quaternion rotation // = Quaternion::rotator(0.25*PI, Vec3{.x=0.0, 1.0, 0.0});
        = Quaternion::rotator(-1.0, Vec3{.x=1.0, 1.0, 0.0});

    {
        /* Set those parameters of the Parameters struct that are treated
        as uniforms by GLSL shaders.*/
        s_sim_params_set = [&params, &sim]
            (int c, Uniform u) {
            if (c == params.DATA_TEXEL_DIMENSIONS3_D) {
                // sim.reset_data_dimensions(u.ivec3);
                IVec2 d = get_2d_from_3d_dimensions(u.ivec3);
                printf("Dimensions (%d, %d)\n",
                       d[0], d[1]);
            }
            if (c == params.VOLUME_TEXEL_DIMENSIONS3_D) {
                sim.reset_volume_dimensions(u.ivec3);
                IVec2 d = get_2d_from_3d_dimensions(u.ivec3);
                printf("Dimensions (%d, %d)\n",
                       d[0], d[1]);
            }
            // if (c == params.BRIGHTNESS)
            //     user_text_edit_potential.queue_current();
            if (c == params.USE_LINEAR) {
                if (u.b32)
                    sim.reset_volume_filtering(GL_LINEAR);
                else
                    sim.reset_volume_filtering(GL_NEAREST);
            }
            // if (c == params.NUMBER_OF_PARTICLES) {
            //     sim.reset_trajectories_dimensions(u.i32);
            // }
            params.set(c, u);
        };
        /* Get those parameters of the Parameters struct that can be
        inputed as uniforms to GLSL shaders.*/
        s_sim_params_get = [&params]
            (int c) -> Uniform {
            return params.get(c);
        };
        /* String parametres can't be configured as uniforms, so
        are set using a different function.*/
        // s_sim_params_set_string = [&params]
        //     (int c, int index, std::string val) {
        //     params.set(c, index, val);
        // };
        /* Perform an action upon the press of a button. */
        s_button_pressed = [&params, &sim]
            (int param_code) {
            if (param_code == params.SOLVE) {
                sim.solve(params);
            }
            if (param_code == params.CLEAR) {
                printf("Clear\n");
                sim.clear_atoms(params);
            }
        };
        /* Floating-point value parameters and their associated sliders
        can be created by the user. This notifies and keeps track of any
        newly created user-defined parameter. The user defined paramters are
        not part of the Parameters struct, so are stored separately.*/
        // s_sim_params_set_user_float_param = []
        //     (int c, std::string var_name, float value) {
        // };
        /* Upon a change of a dropdown or selection menu, change its
        corresponding selection parameter in the Parameters struct so that
        it matches the dropdown.*/
        s_selection_set = [&params, &sim]
            (int c, int val) {
            if (c == params.MOUSE_SELECTOR) {
                params.mouseSelector.selected = val;
            }
            if (c == params.PRESET_ATOMS) {
                params.presetAtoms.selected = val;
                // sim.clear_atoms(params);
                // sim.add_atom(params, val + 1, spatial::Vector{
                //     .t=0.0, 0.0, 0.0, 0.0
                // });
                // sim.solve(params);
            }
            if (c == params.PRESET_COMPOUNDS_DROPDOWN) {
                sim.set_preset_system(params, val);
            }
            if (c == params.SHELL_METHOD_TYPE) {
                params.shellMethodType.selected = val;
                sim.solve(params);
            }
            if (c == params.VISUALIZATION_SELECT) {
                params.visualizationSelect.selected = val;
            }
            if (c == params.TEXEL_SIDE_LENGTH_SELECTOR) {
                    params.texelSideLengthSelector.selected = val;
                int texel_side_length = 64;
                if (val == 1)
                    texel_side_length = 128;
                else if (val == 2)
                    texel_side_length = 256;
                params.texelSideLength = texel_side_length;
                sim.reset_simulation_discretization_dimensions(
                    IVec3{.ind{
                        texel_side_length, texel_side_length,
                        texel_side_length}});
                sim.reset_data_reduce_dimensions(IVec3{.ind{
                    texel_side_length, texel_side_length, texel_side_length
                }});
                params.dataTexelDimensions3D.x = texel_side_length;
                params.dataTexelDimensions3D.y = texel_side_length;
                params.dataTexelDimensions3D.z = texel_side_length;
                // params.sideLength = (float)texel_side_length;
            }
        };
        // /* Upon change of a user-defined parameter, change its value. */
        // s_user_edit_set_value = [&user_text_edit_potential]
        //     (int c, std::string var_name, float value) {
        // };
        // /* Upon change of a user-defined parameter, get its value. */
        // s_user_edit_get_value = [&user_text_edit_potential]
        //     (int c, std::string var_name) -> float {
        // };
        /* Retrieve the new image that was set by the user. */
        // s_image_set = [&params]
        //     (int c, const std::string &image_data, int w, int h) {
        // };
        s_configure_bmp_recording = [&params](int c, bool is_recording) {
            if (c == params.TAKE_SCREENSHOTS) {
                params.takeScreenshots.is_recording = is_recording;
            }
        };
        s_bmp_image = [&sim] () {
            std::vector<unsigned char> &image_data = sim.get_image_data();
            return (unsigned char *)&image_data[0];
        };
        s_bmp_image_size = [&sim]() {
            std::vector<unsigned char> &image_data = sim.get_image_data();
            return image_data.size();
        };
    }

    { // Initial configuration from the default preset option
        // TODO
        sim.set_preset_system(params, 1);
    }

    start_gui(main_render.get_window());
    s_loop = [&] {

        if (start_position.has_value()) {
            if (cursor_positions.size() > 0) {
                Vec2 delta_2d = interactor.get_mouse_delta();
                Vec3 delta {.ind={delta_2d[0], delta_2d[1], 0.0}};
                if (s_is_on_touch_screen() && delta.length() > 0.01)
                    delta = 0.01*delta/delta.length();
                Vec3 view_vec {.ind={0.0, 0.0, -1.0}};
                Vec3 axis = cross_product(delta, view_vec);
                Quaternion rot = Quaternion::rotator(
                    3.0*axis.length(), axis);
                if (!sim.modify_from_mouse_touch_input(
                    params, rotation, 0.01*Interactor::get_scroll(), cursor_positions))
                    rotation = rotation*rot;
            }
        }
        /* if (!user_text_edit_potential.program_queued() && user_text_edit_potential.is_time_dependent()) {
            params.t += 0.01;
            user_text_edit_potential.queue_current();
        }
        if (user_text_edit_potential.program_queued()) {
            UserDefinedProgram user_defined = user_text_edit_potential.expend_program();
            sim.add_user_defined_potential(
                params, user_defined.program, user_defined.uniforms);

        }
        for (int i = 0; i < params.stepsPerFrame; i++) {
            if (cursor_positions.size() > 0 && params.mouseSelector.selected == 1
                && sim.is_inside(params, rotation, 
                        0.01*Interactor::get_scroll(), cursor_positions[0]))
                break;
            sim.time_step(params);
            params.t += 0.5*params.dt;
        }*/
        if (cursor_positions.size() == 1 && params.mouseSelector.selected == 1
            && sim.is_inside(params, rotation, 
                        0.01*Interactor::get_scroll(), cursor_positions[0])) {
            spatial::Vector position = sim.get_position_of_cursor(
                params,
                cursor_positions[0], 
                rotation, 0.01*Interactor::get_scroll()
            );
            sim.add_atom(params, params.presetAtoms.selected + 1, position);
            main_render.draw(
                sim.view(params, cursor_positions[0], 
                    rotation, 0.01*Interactor::get_scroll()));
        } else if (cursor_positions.size() > 1 && params.mouseSelector.selected == 1
            && sim.is_inside(params, rotation, 
                        0.01*Interactor::get_scroll(), cursor_positions[0])) {
            main_render.draw(
                sim.view(params, cursor_positions[0], 
                    rotation, 0.01*Interactor::get_scroll()));
        } else {
                main_render.draw(
                sim.view(params, hover_position, 
                    rotation, 0.01*Interactor::get_scroll()));
        }

        if (hover_position.has_value()) {
            Vec3 loc = sim.get_cursor_location();
            Vec3 scaled_loc = sim.get_scaled_cursor_location(params);
            if (loc.x >= -1.0 && loc.x < 1.0 && loc.y >= -1.0 && loc.y < 1.0
                && loc.z >= -1.0 && loc.z < 1.0) {
                #ifdef __EMSCRIPTEN__
                edit_hovering_canvas_label_display(
                    SimParams::CANVAS_HOVER_DISPLAY,
                    "x: " + std::to_string(scaled_loc.x) + ", "
                    + "y: " + std::to_string(scaled_loc.y) + ", "
                    + "z: " + std::to_string(scaled_loc.z)
                    // + "\n" + std::to_string(sim.get_total_energy())
                );
                #endif
                // edit_hovering_canvas_visibility_top_left_offset(
                //     SimParams::CANVAS_HOVER_DISPLAY, true, 50, 50
                // );
            }
        }

        if (params.takeScreenshots.is_recording)
            download_bmp_image("hf");

        auto poll_events = [&] {
            // Tell GLFW to poll events
            glfwPollEvents();

            // Get user interaction events
            interactor.click_update(main_render.get_window());

            // Handle mouse or single touch events
            Vec2 pos = interactor.get_mouse_position();
            if (outside_gui() && pos.x > 0.0 && pos.x < 1.0 && 
                pos.y > 0.0 && pos.y < 1.0) { 
                if (interactor.left_pressed()) {
                    if (!start_position.has_value())
                        start_position = pos;
                    cursor_positions.push_back(pos);
                }
                hover_position = pos;
            } else {
                hover_position.reset();
            }
            if (interactor.left_released()) {
                if (start_position.has_value()) {
                    start_position.reset();
                    cursor_positions.clear();
                }
            }

            // Handle double touch events
            Vec2 double_touches[2];
            double_touches[0] = interactor.get_double_touch_position(0);
            double_touches[1] = interactor.get_double_touch_position(1);
            if (interactor.double_touch_active()
                && double_touches[0].x > 0.0 && double_touches[0].x < 1.0
                && double_touches[0].y > 0.0 && double_touches[0].y < 1.0
                && double_touches[1].x > 0.0 && double_touches[1].x < 1.0
                && double_touches[1].y > 0.0 && double_touches[1].y < 1.0) {
                if (!start_double_touches.has_value())
                    start_double_touches = 
                        {double_touches[0], double_touches[1]};
                double_touches_positions.push_back(
                        {double_touches[0], double_touches[1]});
            }
            if (interactor.double_touch_released()) {
                if (start_double_touches.has_value()) {
                    start_double_touches.reset();
                    double_touches_positions.clear();
                }
            }

            #ifndef __EMSCRIPTEN__
            #endif
        };
        display_gui(&params);
        poll_events();

        glfwSwapBuffers(main_render.get_window());
    };

    #ifdef __EMSCRIPTEN__
    emscripten_set_main_loop(s_main_loop, 0, true);
    #else
    while (!glfwWindowShouldClose(main_render.get_window()))
        s_loop();
    #endif
}


int main(int argc, char **argv) {
    // #ifndef __EMSCRIPTEN__
    struct timespec frame_time[2];
    clock_gettime(CLOCK_MONOTONIC, &frame_time[0]);
    // #endif
    // array_helpers::test1();
    // array_helpers::test2();
    // array_helpers::test3();
    // array_helpers::test4();
    // array_helpers::test5();
    // array_helpers::test6();
    // array_helpers::test7();
    // array_helpers::test8();
    // array_helpers::test9();
    // array_helpers::test10();
    // array_helpers::test11();
    // array_helpers::test12();
    // h2_example();
    // h2o_example();
    // benzene_example();
    // co2_example();
    // o2_example();
    // for (int i = 1; i <= 36; i++) {
    //     printf("Atomic number: %d:\n", i);
    //     // printf("Closed:\n");
    //     // closed_shell_element_example(i);
    //     printf("Unrestricted Open:\n");
    //     unrestricted_element_example(i);
    //     puts("############################################################");
    // }
    // std::vector<std::pair<unsigned int, spatial::Vector>> atom_list = {
    //     {11, {.t=0.0, 0.0, 0.0, 0.0}},
    //     {8, {.t=0.0, 0.5, 0.0, 0.0}},
    //     {7, {.t=0.0, 1.0, 0.0, 0.0}},
    //     {1, {.t=0.0, 1.25, 0.25, 0.0}},
    //     {9, {.t=0.0, 1.25, -0.25, 0.0}}
    // };
    // simple_system(atom_list,
    //     true, 10);
    // #ifndef __EMSCRIPTEN__
    clock_gettime(CLOCK_MONOTONIC, &frame_time[1]);
    double delta_t = frame_time[1].tv_sec - frame_time[0].tv_sec;
    std::cout << "Time taken: " << delta_t << "s \n";
    // #endif

    int window_width = 1440, window_height = 1440;
    if (argc >= 3) {
        window_width = std::atoi(argv[1]);
        window_height = std::atoi(argv[2]);
    }
    int filter_type = GL_LINEAR;
    if (argc >= 4) {
        std::string s(argv[3]);
        if (s == "nearest")
            filter_type = GL_NEAREST;
    }
    if (argc >= 5) {
        std::string s(argv[4]);
        s_is_on_touch_screen = []() {
            return true;
        };
    } else {
        std::string s(argv[4]);
        s_is_on_touch_screen = []() {
            return false;
        };
    }
    SimParams params {};
    TextureParams default_tex_params = {
        .format=GL_RGBA16F,
        .width=(unsigned int)window_width,
        .height=(unsigned int)window_height,
        .generate_mipmap=false,
        // .generate_mipmap=!(filter_type == GL_NEAREST),
        .wrap_s=GL_CLAMP_TO_EDGE,
        .wrap_t=GL_CLAMP_TO_EDGE,
        .mag_filter=(unsigned int)filter_type,
        .min_filter=(unsigned int)filter_type
    };
    MainGLFWQuad 
    main_render (default_tex_params.width, default_tex_params.height);
    simulation_ui_interface_handler(
        main_render, default_tex_params, params);
    
    return 0;
}