#include "build_arrays.hpp"

#include <pthread.h>
#include "gl_wrappers.hpp"

#include <set>

// #define THREAD_COUNT 0

#ifndef __EMSCRIPTEN__
#define THREAD_COUNT 1
#else
#define THREAD_COUNT 1
#endif

struct FillArraysThreadData {
    array_helpers::SquareArray *overlap;
    array_helpers::SquareArray *kinetic;
    array_helpers::SquareArray *nuclear;
    array_helpers::HypercubeArray *repulsion_exchange;
    const NuclearChargesArray *nuclear_charges;
    const BasisFunctionArray *basis_functions;
    int start, end, n;
};

static void *fill_arrays_inner(void *data) {
    struct FillArraysThreadData *thread_data 
        = (struct FillArraysThreadData *)data;
    array_helpers::SquareArray *overlap = thread_data->overlap;
    array_helpers::SquareArray *kinetic = thread_data->kinetic;
    array_helpers::SquareArray *nuclear = thread_data->nuclear;
    array_helpers::HypercubeArray *repulsion_exchange 
        = thread_data->repulsion_exchange;
    const NuclearChargesArray *nuclear_charges = thread_data->nuclear_charges;
    const BasisFunctionArray *basis_functions
        = thread_data->basis_functions;
    int start = thread_data->start;
    int end = thread_data->end;
    int n = thread_data->n;
    for (int i = start; i < end; i++)
        for (int j = i; j < n; j++)
            overlap->operator()(i, j) = basis_functions->overlap(i, j);
    for (int i = start; i < end; i++)
        for (int j = i; j < n; j++)
            kinetic->operator()(i, j) = basis_functions->kinetic(i, j);
    for (int i = start; i < end; i++)
        for (int j = i; j < n; j++)
            nuclear->operator()(i, j) = 
                basis_functions->nuclear(i, j, *nuclear_charges);

    for (int i = start; i < end; i++)
        for (int j = i + 1; j < n; j++)
            overlap->operator()(j, i) = overlap->operator()(i, j);
    for (int i = start; i < end; i++)
        for (int j = i + 1; j < n; j++)
            kinetic->operator()(j, i) = kinetic->operator()(i, j);
    for (int i = start; i < end; i++)
        for (int j = i + 1; j < n; j++)
            nuclear->operator()(j, i) = nuclear->operator()(i, j);
    
    for (int i = start; i < end; i++) {
        for (int j = i; j < n; j++) {
            // overlap->operator()(i, j) = basis_functions->overlap(i, j);
            // kinetic->operator()(i, j) = basis_functions->kinetic(i, j);
            // nuclear->operator()(i, j) = basis_functions->nuclear(
            //     i, j, *nuclear_charges);
            int outer = i*n + j;
            for (int k = outer / n; k < n; k++) {
                int start = (k == (outer / n))?
                    std::max(outer % n, k): k;
                for (int l = start; l < n; l++) {
                    int inner = k*n + l;
                    double val
                        = basis_functions->repulsion_exchange(i, j, k, l);
                    repulsion_exchange->operator()(i, j, k, l) = val;
                    if (l > k)
                        repulsion_exchange->operator()(i, j, l, k) = val;
                    if (inner > outer) {
                        // if (k <= l) {
                        repulsion_exchange->operator()(l, k, i, j) = val;
                        repulsion_exchange->operator()(l, k, j, i) = val;
                        // }
                        // if (l <= k) {
                        repulsion_exchange->operator()(k, l, i, j) = val;
                        repulsion_exchange->operator()(k, l, j, i) = val;
                        // }
                    }
                }
            }
        }
    }
    for (int i = start; i < end; i++)
        for (int j = i + 1; j < n; j++)
            repulsion_exchange->operator()(
                    j, i, repulsion_exchange->operator()(i, j));
    return NULL;
}

void build_arrays::fill(
    array_helpers::SquareArray &overlap,
    array_helpers::SquareArray &kinetic,
    array_helpers::SquareArray &nuclear,
    array_helpers::HypercubeArray &repulsion_exchange,
    const BasisFunctionArray &basis_functions,
    const NuclearChargesArray &nuclear_charges
) {
    int n = basis_functions.get_number_of_basis_functions();
    // printf("Number of basis functions: %d\n", n);
    if (THREAD_COUNT >= 4 && n > (2*THREAD_COUNT)) {
        int op_count = n*n;
        int ops_per_thread = op_count / THREAD_COUNT;
        std::vector<pthread_t> threads {THREAD_COUNT};
        std::vector <FillArraysThreadData> thread_data {THREAD_COUNT};
        for (int i = THREAD_COUNT - 1, k = 0, end = n; i >= 0; i--, k++) {
            int start = n - round(sqrt(ops_per_thread + (end - n)*(end - n)));
            start = std::max(0, start);
            if (k == THREAD_COUNT - 1 && start != 0)
                start = 0;
            thread_data[k].overlap = &overlap;
            thread_data[k].kinetic = &kinetic;
            thread_data[k].nuclear = &nuclear;
            thread_data[k].repulsion_exchange = &repulsion_exchange;
            thread_data[k].basis_functions = &basis_functions;
            thread_data[k].nuclear_charges = &nuclear_charges;
            thread_data[k].start = start;
            printf("Thread number: %d\n", k);
            printf("Start index: %d\n", start);
            printf("Stop index: %d\n", end);
            thread_data[k].end = end;
            thread_data[k].n = n;
            pthread_create(&threads[k], NULL,
                fill_arrays_inner, (void *)&thread_data[k]);
            end = start;
        }
        for (int k = THREAD_COUNT - 1; k >= 0; k--)
            pthread_join(threads[k], NULL);
        return;
    }
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            overlap(i, j) = basis_functions.overlap(i, j);
            kinetic(i, j) = basis_functions.kinetic(i, j);
            nuclear(i, j) = basis_functions.nuclear(
                i, j, nuclear_charges);
            int outer = i*n + j;
            for (int k = outer / n; k < n; k++) {
                int start = (k == (outer / n))?
                    std::max(outer % n, k): k;
                for (int l = start; l < n; l++) {
                    int inner = k*n + l;
                    double val
                        = basis_functions.repulsion_exchange(i, j, k, l);
                    repulsion_exchange(i, j, k, l) = val;
                    if (l > k)
                        repulsion_exchange(i, j, l, k) = val;
                    if (inner > outer) {
                        repulsion_exchange(l, k, i, j) = val;
                        repulsion_exchange(k, l, i, j) = val;
                        repulsion_exchange(l, k, j, i) = val;
                        repulsion_exchange(k, l, j, i) = val;
                    }
                }
            }
            if (j > i) {
                overlap(j, i) = overlap(i, j);
                kinetic(j, i) = kinetic(i, j);
                nuclear(j, i) = nuclear(i, j);
                repulsion_exchange(j, i, repulsion_exchange(i, j));
            }
        }
    }
}

struct FillArraysThreadData2 {
    array_helpers::SquareArray *overlap;
    array_helpers::SquareArray *kinetic;
    array_helpers::SquareArray *nuclear;
    array_helpers::Symmetric4 *repulsion_exchange;
    const array_helpers::SquareArray *re_abab;
    const NuclearChargesArray *nuclear_charges;
    const BasisFunctionArray *basis_functions;
    int start, end, n;
};

static void *fill_arrays_inner2(void *data) {
    struct FillArraysThreadData2 *thread_data 
        = (struct FillArraysThreadData2 *)data;
    array_helpers::SquareArray *overlap = thread_data->overlap;
    array_helpers::SquareArray *kinetic = thread_data->kinetic;
    array_helpers::SquareArray *nuclear = thread_data->nuclear;
    array_helpers::Symmetric4 *repulsion_exchange 
        = thread_data->repulsion_exchange;
    const NuclearChargesArray *nuclear_charges = thread_data->nuclear_charges;
    const BasisFunctionArray *basis_functions
        = thread_data->basis_functions;
    int start = thread_data->start;
    int end = thread_data->end;
    int n = thread_data->n;
    for (int i = start; i < end; i++)
        for (int j = i; j < n; j++)
            overlap->operator()(i, j) = basis_functions->overlap(i, j);
    for (int i = start; i < end; i++)
        for (int j = i; j < n; j++)
            kinetic->operator()(i, j) = basis_functions->kinetic(i, j);
    for (int i = start; i < end; i++)
        for (int j = i; j < n; j++)
            nuclear->operator()(i, j) = 
                basis_functions->nuclear(i, j, *nuclear_charges);

    for (int i = start; i < end; i++)
        for (int j = i + 1; j < n; j++)
            overlap->operator()(j, i) = overlap->operator()(i, j);
    for (int i = start; i < end; i++)
        for (int j = i + 1; j < n; j++)
            kinetic->operator()(j, i) = kinetic->operator()(i, j);
    for (int i = start; i < end; i++)
        for (int j = i + 1; j < n; j++)
            nuclear->operator()(j, i) = nuclear->operator()(i, j);
    
    for (int i = start; i < end; i++) {
        for (int j = i; j < n; j++) {
            int outer = i*n + j;
            for (int k = outer / n; k < n; k++) {
                int start = (k == (outer / n))?
                    std::max(outer % n, k): k;
                for (int l = start; l < n; l++) {
                    double val
                        = basis_functions->repulsion_exchange(
                            i, j, k, l, 
                            *thread_data->re_abab);
                    repulsion_exchange->operator()(i, j, k, l) = val;
                }
            }
        }
    }
    return NULL;
}

static void fill_overlap_kinetic_nuclear(
    array_helpers::SquareArray &overlap,
    array_helpers::SquareArray &kinetic,
    array_helpers::SquareArray &nuclear,
    const BasisFunctionArray &basis_functions,
    const NuclearChargesArray &nuclear_charges
) {
    int n = basis_functions.get_number_of_basis_functions();
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            overlap(i, j) = basis_functions.overlap(i, j);
            kinetic(i, j) = basis_functions.kinetic(i, j);
            nuclear(i, j) = basis_functions.nuclear(
                i, j, nuclear_charges);
            if (j > i) {
                overlap(j, i) = overlap(i, j);
                kinetic(j, i) = kinetic(i, j);
                nuclear(j, i) = nuclear(i, j);
                // repulsion_exchange(j, i, repulsion_exchange(i, j));
            }
        }
    }
}

static void compute_repulsion_exchange_in_shell(
    array_helpers::Symmetric4 &repulsion_exchange,
    array_helpers::SquareArray &shell_ij_ij,
    const BasisFunctionArray &basis_functions, 
    const array_helpers::SquareArray &re_abab,
    int i, const ShellData &shell_i,
    int j, const ShellData &shell_j) {
    int i_offset = shell_i.basis_functions.offset;
    int i_count = shell_i.basis_functions.count;
    int j_offset = shell_j.basis_functions.offset;
    int j_count = shell_j.basis_functions.count;
    double max_abs_val = 0.0;
    for (int a = i_offset; a < (i_offset + i_count); a++) {
        for (int b = j_offset; b < (j_offset + j_count); b++) {
            for (int c = i_offset; c < (i_offset + i_count); c++) {
                for (int d = j_offset; d < (j_offset + j_count); d++) {
                    int row_size = repulsion_exchange.row_size();
                    // if (b >= a && d >= c && 
                    //     (c*row_size + d) >= (a*row_size + b)) {
                        double val
                            = basis_functions.repulsion_exchange(
                                a, b, c, d, re_abab);
                        repulsion_exchange(a, b, c, d) = val;
                        if (abs(val) > max_abs_val) {
                            max_abs_val = val;
                        }
                    // }
                }
            }
        }
    }
    shell_ij_ij(i, j) = max_abs_val;
}

static void compute_repulsion_exchange_in_shell(
    array_helpers::Symmetric4 &repulsion_exchange,
    const BasisFunctionArray &basis_functions,
    const array_helpers::SquareArray &re_abab,
    const array_helpers::SquareArray &shell_ij_ij, 
    int i, const ShellData &shell_i,
    int j, const ShellData &shell_j,
    int n, const ShellData &shell_n,
    int m, const ShellData &shell_m) {
    if (
        (i == n && j == m) 
        || (i == m && j == n)
    )
        return;
    double four_e1 = shell_ij_ij(i, j);
    double four_e2 = shell_ij_ij(n, m);
    if (sqrt(four_e1 * four_e2) < 1e-3)
        return;
    int i_offset = shell_i.basis_functions.offset;
    int i_count = shell_i.basis_functions.count;
    int j_offset = shell_j.basis_functions.offset;
    int j_count = shell_j.basis_functions.count;
    int n_offset = shell_n.basis_functions.offset;
    int n_count = shell_n.basis_functions.count;
    int m_offset = shell_m.basis_functions.offset;
    int m_count = shell_m.basis_functions.count;
    for (int a = i_offset; a < (i_offset + i_count); a++) {
        for (int b = j_offset; b < (j_offset + j_count); b++) {
            for (int c = n_offset; c < (n_offset + n_count); c++) {
                for (int d = m_offset; d < (m_offset + m_count); d++) {
                    int row_size = repulsion_exchange.row_size();
                    // if (b >= a && d >= c && 
                    //     (c*row_size + d) >= (a*row_size + b)) {
                        double val
                            = basis_functions.repulsion_exchange(
                                a, b, c, d, re_abab);
                        repulsion_exchange(a, b, c, d) = val;
                    // }
                }
            }
        }
    }
}

static void fill_repulsion_exchange_cull_vanishing_shells(
    array_helpers::Symmetric4 &repulsion_exchange,
    const BasisFunctionArray &basis_functions,
    const array_helpers::SquareArray &re_abab
) {
    int shell_count = basis_functions.number_of_shells();
    array_helpers::SquareArray shell_ij_ij(shell_count);
    for (int i = 0; i < shell_count; i++) {
        for (int j = i; j < shell_count; j++) {
            ShellData shell_i = basis_functions.get_shell(i);
            ShellData shell_j = basis_functions.get_shell(j);
            compute_repulsion_exchange_in_shell(
                repulsion_exchange, shell_ij_ij,
                basis_functions, re_abab,
                i, shell_i, j, shell_j);
            shell_ij_ij(j, i) = shell_ij_ij(i, j); 
        }
    }
    for (int i = 0; i < shell_count; i++) {
        for (int j = i; j < shell_count; j++) {
            int outer = i*shell_count + j;
            for (int k = outer / shell_count; k < shell_count; k++) {
                int start = (k == (outer / shell_count))? 
                    std::max(outer % shell_count, k): k;
                for (int l = start; l < shell_count; l++) {
                    ShellData shell_i = basis_functions.get_shell(i);
                    ShellData shell_j = basis_functions.get_shell(j);
                    ShellData shell_k = basis_functions.get_shell(k);
                    ShellData shell_l = basis_functions.get_shell(l);
                    compute_repulsion_exchange_in_shell(
                        repulsion_exchange, 
                        basis_functions, re_abab, shell_ij_ij,
                        i, shell_i, j, shell_j, k, shell_k, l, shell_l);
                }
            }
        }
    }
}

static void modify_re_arrays(
    std::vector<float> &bf_spec1_arr,
    std::vector<float> &bf_spec2_arr,
    std::vector<float> &primitives_arr,
    int &max_primitives_count,
    std::vector<float> &indices,
    int &total_count,
    const array_helpers::Symmetric4 &repulsion_exchange,
    const array_helpers::SquareArray &re_abab,
    const BasisFunctionArray &basis_functions
) {
    int n = basis_functions.get_number_of_basis_functions();
    bf_spec1_arr = std::vector<float>(0);
    bf_spec2_arr = std::vector<float>(0);
    max_primitives_count = 0;
    for (int i = 0; i < n; i++) {
        spatial::UByte4 angular = basis_functions.get_angular(i);
        spatial::Vector position = basis_functions.get_position(i);
        int count = basis_functions.primitive_count_at(i);
        bf_spec1_arr.push_back(position.x);
        bf_spec1_arr.push_back(position.y);
        bf_spec1_arr.push_back(position.z);
        bf_spec1_arr.push_back(float(count));
        max_primitives_count 
            = (count > max_primitives_count)? count: max_primitives_count;
        bf_spec2_arr.push_back(angular.x);
        bf_spec2_arr.push_back(angular.y);
        bf_spec2_arr.push_back(angular.z);
        bf_spec2_arr.push_back(0.0F);
    }
    primitives_arr = std::vector<float>(2*n*max_primitives_count, 0.0);
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < max_primitives_count; k++) {
            if (k < int(bf_spec1_arr[4*i + 3])) {
                Gaussian3D p = basis_functions.get_primitive(i, k);
                primitives_arr[2*(i*max_primitives_count + k)] 
                    = p.amplitude();
                primitives_arr[2*(i*max_primitives_count + k) + 1]
                    = p.orbital_exponent();
            }
        }
    }
    int curr = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            int outer = i*n + j;
            for (int k = outer / n; k < n; k++) {
                int start = (k == (outer / n))?
                    std::max(outer % n, k): k;
                for (int l = start; l < n; l++) {
                    double four_e1 = re_abab(i, j);
                    double four_e2 = re_abab(k, l);
                    if (sqrt(four_e1 * four_e2) > 1e-4) {
                        // printf("%d %d %d %d\n", i, j, k, l);
                        indices[curr] = (float)i;
                        curr++;
                        indices[curr] = (float)j;
                        curr++;
                        indices[curr] = (float)k;
                        curr++;
                        indices[curr] = (float)l;
                        curr++;
                    }
                }   
            }
        }
    }
    total_count = curr/4;
}

static void print_spec_arrs(
    std::vector<float> &bf_spec1_arr,
    std::vector<float> &bf_spec2_arr,
    std::vector<float> &primitives_arr,
    int &max_primitives_count,
    const BasisFunctionArray &basis_functions
) {
    for (int i = 0; 
         i < basis_functions.get_number_of_basis_functions(); i++) {
        printf("Basis function %d\n", i);
        printf("position:\t%g, %g, %g\n",
            bf_spec1_arr[4*i], bf_spec1_arr[4*i + 1], bf_spec1_arr[4*i + 2]);
        printf("angular:\t%g, %g, %g\n",
            bf_spec2_arr[4*i], bf_spec2_arr[4*i + 1], bf_spec2_arr[4*i + 2]);
        int count = int(bf_spec1_arr[4*i + 3]);
        printf("primitives count:\t%d\n", count);
        printf("coefficients\texponents\n");
        for (int k = 0; k < count; k++) {
            float c = primitives_arr[
                2*(i*max_primitives_count + k)];
            float e = primitives_arr[
                2*(i*max_primitives_count + k) + 1];
            printf("%g \t %g\n", c, e);
        }
        printf("\n");
    }
}

void get_symmetric_re(
    array_helpers::SquareArray &re_abab,
    const BasisFunctionArray &basis_functions
) {
    int n = basis_functions.get_number_of_basis_functions();
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            re_abab(i, j) 
                = basis_functions.repulsion_exchange(i, j, i, j);
            re_abab(j, i) = re_abab(i, j);
        }
    }
}

void build_arrays::fill(
    Quad &bf_spec1, Quad &bf_spec2,
    Quad &primitives, Quad &indices_q, Quad &rep_exc,
    array_helpers::SquareArray &overlap,
    array_helpers::SquareArray &kinetic,
    array_helpers::SquareArray &nuclear,
    array_helpers::Symmetric4 &repulsion_exchange,
    const BasisFunctionArray &basis_functions,
    const NuclearChargesArray &nuclear_charges,
    const unsigned int program
) {
    array_helpers::SquareArray re_abab(
        basis_functions.get_number_of_basis_functions());
    get_symmetric_re(re_abab, basis_functions);
    fill_overlap_kinetic_nuclear(
        overlap, kinetic, nuclear,
        basis_functions, nuclear_charges);
    std::vector<float> bf_spec1_arr;
    std::vector<float> bf_spec2_arr;
    std::vector<float> primitives_arr;
    int max_primitives_count;
    int total_count;
    std::vector<float> indices (10000);
    modify_re_arrays(
        bf_spec1_arr,
        bf_spec2_arr,
        primitives_arr,
        max_primitives_count,
        indices,
        total_count,
        repulsion_exchange, re_abab, basis_functions);
    int re_tex_width = std::ceil(std::sqrt(total_count));
    print_spec_arrs(
        bf_spec1_arr, bf_spec2_arr,
        primitives_arr, 
        max_primitives_count, basis_functions);
    // for (int n = 0; n < total_count; n++) {
    //     printf("%d, %d, %d, %d\n", 
    //         (int)indices[4*n], (int)indices[4*n + 1], 
    //         (int)indices[4*n + 2], (int)indices[4*n + 3]);
    // }
    TextureParams primitives_tex_params = {
        .format=GL_RG32F,
        .width=(unsigned int)
            max_primitives_count,
        .height=(unsigned int)
            basis_functions.get_number_of_basis_functions(),
        .generate_mipmap=false,
        .wrap_s=GL_REPEAT, .wrap_t=GL_REPEAT,
        .min_filter=GL_NEAREST, .mag_filter=GL_NEAREST
    };
    TextureParams bf_tex_params = {
        .format=GL_RGBA32F,
        .width=(unsigned int)
            basis_functions.get_number_of_basis_functions(),
        .height=1,
        .generate_mipmap=false,
        .wrap_s=GL_REPEAT, .wrap_t=GL_REPEAT,
        .min_filter=GL_NEAREST, .mag_filter=GL_NEAREST
    };
    TextureParams re_tex_params = {
        .format=GL_R32F,
        .width=(unsigned int)re_tex_width,
        .height=(unsigned int)re_tex_width,
        .generate_mipmap=false,
        .wrap_s=GL_REPEAT, .wrap_t=GL_REPEAT,
        .min_filter=GL_NEAREST, .mag_filter=GL_NEAREST
    };
    TextureParams indices_tex_params = {
        .format=GL_RGBA32F,
        .width=(unsigned int)re_tex_width,
        .height=(unsigned int)re_tex_width,
        .generate_mipmap=false,
        .wrap_s=GL_REPEAT, .wrap_t=GL_REPEAT,
        .min_filter=GL_NEAREST, .mag_filter=GL_NEAREST
    };
    primitives.reset(primitives_tex_params);
    bf_spec1.reset(bf_tex_params);
    bf_spec2.reset(bf_tex_params);
    indices_q.reset(indices_tex_params);
    rep_exc.reset(re_tex_params);
    primitives.set_pixels((float *)&primitives_arr[0]);
    bf_spec1.set_pixels((float *)&bf_spec1_arr[0]);
    bf_spec2.set_pixels((float *)&bf_spec2_arr[0]);
    printf("The re_tex_width is: %d\n", re_tex_width);
    std::vector<float> rep_exc_arr(
        re_tex_width*re_tex_width, 0.0);
    std::vector<float> indices_final(
        re_tex_width*re_tex_width, 0.0);
    indices_q.set_pixels((float *)&indices[0]);
    rep_exc.draw(
        program,
        {
            {"basisFunctionSpec1Tex", &bf_spec1},
            {"basisFunctionSpec2Tex", &bf_spec2},
            {"primitivesTex", &primitives},
            {"indicesTex", &indices_q},
            {"numberOfBasisFunctions", 
                    int(basis_functions.get_number_of_basis_functions())}
        }
    );
    printf("This is reached.\n");
    rep_exc.fill_array_with_contents((float *)&rep_exc_arr[0]);
    indices_q.fill_array_with_contents((float *)&indices_final[0]);
    for (int n = 0; n < total_count; n++) {
        int i = indices[4*n], j = indices[4*n + 1];
        int k = indices[4*n + 2], l = indices[4*n + 3];
        double val = rep_exc_arr[n];
        repulsion_exchange(i, j, k, l) = val;
        if (i == j && k == l) {
            int i = int(indices_final[4*n]);
            int j = int(indices_final[4*n + 1]);
            int k = int(indices_final[4*n + 2]);
            int l = int(indices_final[4*n + 3]);
            printf("%d, %d, %d, %d\t%g\n", i, j, k, l, val);
        }
    }
}

void build_arrays::fill(
        array_helpers::SquareArray &overlap,
        array_helpers::SquareArray &kinetic,
        array_helpers::SquareArray &nuclear,
        array_helpers::Symmetric4 &repulsion_exchange,
        const BasisFunctionArray &basis_functions,
        const NuclearChargesArray &nuclear_charges
    ) {
    int n = basis_functions.get_number_of_basis_functions();
    fill_overlap_kinetic_nuclear(
        overlap, kinetic, nuclear,
        basis_functions, nuclear_charges);
    // printf("Number of basis functions: %d\n", n);
    if (THREAD_COUNT >= 4 && n > (2*THREAD_COUNT)) {
        int op_count = n*n;
        int ops_per_thread = op_count / THREAD_COUNT;
        std::vector<pthread_t> threads {THREAD_COUNT};
        std::vector <FillArraysThreadData2> thread_data {THREAD_COUNT};
        array_helpers::SquareArray re_abab(n);
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                re_abab(i, j) 
                    = basis_functions.repulsion_exchange(i, j, i, j);
                re_abab(j, i) = re_abab(i, j);
            }
        }
        for (int i = THREAD_COUNT - 1, k = 0, end = n; i >= 0; i--, k++) {
            int start = n - round(sqrt(ops_per_thread + (end - n)*(end - n)));
            start = std::max(0, start);
            if (k == THREAD_COUNT - 1 && start != 0)
                start = 0;
            thread_data[k].overlap = &overlap;
            thread_data[k].kinetic = &kinetic;
            thread_data[k].nuclear = &nuclear;
            thread_data[k].repulsion_exchange = &repulsion_exchange;
            thread_data[k].basis_functions = &basis_functions;
            thread_data[k].nuclear_charges = &nuclear_charges;
            thread_data[k].start = start;
            thread_data[k].re_abab = &re_abab;
            printf("Thread number: %d\n", k);
            printf("Start index: %d\n", start);
            printf("Stop index: %d\n", end);
            thread_data[k].end = end;
            thread_data[k].n = n;
            pthread_create(&threads[k], NULL,
                fill_arrays_inner2, (void *)&thread_data[k]);
            end = start;
        }
        for (int k = THREAD_COUNT - 1; k >= 0; k--)
            pthread_join(threads[k], NULL);
        return;
    }

    array_helpers::SquareArray re_abab(n);
    get_symmetric_re(
        re_abab, basis_functions); 

    fill_repulsion_exchange_cull_vanishing_shells(
        repulsion_exchange, basis_functions, re_abab);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                for (int l = 0; l < n; l++) {
                    if (i == j && k == l) {
                        double val = repulsion_exchange(i, j, k, l);
                        printf("%d, %d, %d, %d\t%g\n", i, j, k, l, val);
                    }
                }
            }
        }
    }


    // for (int i = 0; i < n; i++) {
    //     for (int j = i; j < n; j++) {
    //         int outer = i*n + j;
    //         for (int k = outer / n; k < n; k++) {
    //             int start = (k == (outer / n))?
    //                 std::max(outer % n, k): k;
    //             for (int l = start; l < n; l++) {
    //                 double val
    //                     = basis_functions.repulsion_exchange(
    //                         i, j, k, l, re_abab);
    //                 repulsion_exchange(i, j, k, l) = val;
    //             }
    //         }
    //     }
    // }
}