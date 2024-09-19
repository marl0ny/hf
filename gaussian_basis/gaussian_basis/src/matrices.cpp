#include "integrals3d.hpp"
#include "gaussian3d.hpp"
#include "matrices.hpp"
#include "basis_function.hpp"

#include <stdio.h>
#ifdef __APPLE__
#include <pthread.h>
#endif

#define AT_INDEX(n, i, j, k, l) \
    (i)*(n)*(n)*(n) + (j)*(n)*(n) + (k)*(n) + (l)

static void copy_block_to_other_block(
    double *arr, int n, 
    int dst_0, int dst_1, int src_0, int src_1  
) {
    for (int k = 0; k < n; k++) {
        for (int l = 0; l < n; l++) {
            arr[AT_INDEX(n, dst_0, dst_1, k, l)]
                = arr[AT_INDEX(n, src_0, src_1, k, l)];
        }
    }
}

static double get_overlap_element(const struct BasisFunction &a,
                                  const struct BasisFunction &b) {
    double sum = 0.0;
    for (int i = 0; i < a.count; i++) {
        if (a == b)
            for (int j = i; j < b.count; j++)
                sum += ((j == i)? 1.0: 2.0)*overlap(a[i], b[j]);
        else
            for (int j = 0; j < b.count; j++)
                sum += overlap(a[i], b[j]);
    }
    return sum;
}

void set_overlap_elements(double *mat, const struct BasisFunction *w, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            mat[i*n + j] = get_overlap_element(w[i], w[j]);
            if (j > i)
                mat[j*n + i] = mat[i*n + j];
        }
    }
}


static double get_kinetic_element(const struct BasisFunction &a,
                                  const struct BasisFunction &b) {
    double sum = 0.0;
    for (int i = 0; i < a.count; i++) {
        if (a == b)
            for (int j = i; j < b.count; j++)
                sum += ((j == i)? 1.0: 2.0)*kinetic(a[i], b[j]);
        else
            for (int j = 0; j < b.count; j++)
                sum += kinetic(a[i], b[j]);
    }
    return sum;
}

void set_kinetic_elements(double *mat, const BasisFunction *w, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            mat[i*n + j] = get_kinetic_element(w[i], w[j]);
            if (j > i)
                mat[j*n + i] = mat[i*n + j];
        }
    }
}

static double get_nuclear_potential_element(
    const struct BasisFunction &a,
    const struct BasisFunction &b,
    Vec3 *nuc_loc, int *charges, int charge_count
) {
    double sum = 0.0;
    for (int k = 0; k < charge_count; k++) {
        for (int i = 0; i < a.count; i++) {
            if (a == b) {
                for (int j = i; j < b.count; j++) {
                    double val = ((double)charges[k])
                        *nuclear_single_charge(a[i], a[j], nuc_loc[k]);
                    if (j > i)
                        val *= 2.0;
                    sum += val;
                }
            } else {
                for (int j = 0; j < b.count; j++)
                    sum += ((double)charges[k])
                        *nuclear_single_charge(a[i], b[j], nuc_loc[k]);
            }
        }
    }
    return sum;
}

void set_nuclear_potential_elements(
    double *mat, const BasisFunction *w, int n,
    Vec3 *nuc_loc, int *charges, int charge_count
) {
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            mat[i*n + j] = get_nuclear_potential_element(
                w[i], w[j], nuc_loc, charges, charge_count);
            if (j > i)
                mat[j*n + i] = mat[i*n + j];
        }
    }
}

static double get_two_electron_integrals_element_inner(
    int i, int j,
    const struct BasisFunction &a, const struct BasisFunction &b,
    const struct BasisFunction &c, const struct BasisFunction &d
) {
    double sum = 0.0;
    for (int n = 0; n < c.count; n++) {
        int start_index = (c == d)? n: 0;
        for (int m = start_index; m < d.count; m++) {
            double val = repulsion(a[i], b[j], c[n], d[m]);
            if (c == d && m > n) val *= 2.0;
            sum += val;
        }
    }
    return sum;
}

static double get_two_electron_integrals_element(
    const struct BasisFunction &a, const struct BasisFunction &b,
    const struct BasisFunction &c, const struct BasisFunction &d) {
    double sum = 0.0;
    for (int i = 0; i < a.count; i++) {
        int start_index = (a == b)? i: 0;
        for (int j = start_index; j < b.count; j++) {
            double val = get_two_electron_integrals_element_inner(
                i, j, a, b, c, d);
            if (a == b && j > i) val *= 2.0;
            sum += val;
        }
    }
    return sum;
}

static void set_two_electron_integrals_inner(
    int i, int j, double *arr, const BasisFunction *w, int n) {
    for (int k = 0; k < n; k++) {
        for (int l = k; l < n; l++) {
            arr[AT_INDEX(n, i, j, k, l)] 
                = get_two_electron_integrals_element(
                    w[i], w[j], w[k], w[l]);
            if (l > k)
                arr[AT_INDEX(n, i, j, l, k)] = arr[AT_INDEX(n, i, j, k, l)];
        }
    }
}

// #define SINGLE_THREADED_ONLY 1

struct BasisFunctionThreadData {
    double *arr;
    const BasisFunction *basis_functions;
    int i;
    int n;
};

void *threaded_basis_func_set_two_electron_integrals_elements_inner(
    void *data) {
    struct BasisFunctionThreadData *thread_data
        = (struct BasisFunctionThreadData *)data;
    double *arr = thread_data->arr;
    const BasisFunction *w = thread_data->basis_functions;
    int i = thread_data->i;
    int n = thread_data->n;
    for (int j = i; j < n; j++) {
        set_two_electron_integrals_inner(i, j, arr, w, n);
        if (j > i)
            copy_block_to_other_block(arr, n, j, i, i, j);
    }
    return NULL;
}

#define MAX_SUPPORTED_THREADS 500

pthread_t s_basis_func_threads[MAX_SUPPORTED_THREADS] = {};
struct BasisFunctionThreadData
    s_basis_func_thread_data[MAX_SUPPORTED_THREADS] = {};

// #define SINGLE_THREADED_ONLY

void set_two_electron_integrals_elements(
    double *arr, const BasisFunction *w, int n
) {
    #if defined(SINGLE_THREADED_ONLY)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            set_two_electron_integrals_inner(i, j, arr, w, n);
        }
    }
    #elif defined(__APPLE__)
    for (int i = 0; i < n; i++) {
        s_basis_func_thread_data[i].arr = arr;
        s_basis_func_thread_data[i].basis_functions = w;
        s_basis_func_thread_data[i].i = i;
        s_basis_func_thread_data[i].n = n;
        pthread_create(&s_basis_func_threads[i], NULL,
            threaded_basis_func_set_two_electron_integrals_elements_inner,
            (void *)&s_basis_func_thread_data[i]);
    }
    for (int i = 0; i < n; i++)
        pthread_join(s_basis_func_threads[i], NULL);
    #else
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            set_two_electron_integrals_inner(i, j, arr, w, n);
            if (j > i) {
                copy_block_to_other_block(arr, n, j, i, i, j);
            }
        }
    }
    #endif
}

void set_overlap_elements(double *mat, const Gaussian3D *g, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            mat[i*n + j] = overlap(g[i], g[j]);
            if (j > i)
                mat[j*n + i] = mat[i*n + j];
        }
    }
}

void set_kinetic_elements(double *mat, const Gaussian3D *g, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            mat[i*n + j] = kinetic(g[i], g[j]);
            if (j > i)
                mat[j*n + i] = mat[i*n + j];
        }
    }
}

void set_nuclear_potential_elements(
    double *mat, const Gaussian3D *g, int n,
    Vec3 *nuc_loc, int *charges, int charge_count
) {
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            for (int k = 0; k < charge_count; k++) {
                mat[i*n + j] 
                    += ((double)charges[k])
                    *nuclear_single_charge(g[i], g[j], nuc_loc[k]);
            }
            if (j > i)
                mat[j*n + i] = mat[i*n + j];
        }
    }
}

static void set_two_electron_integrals_inner(
    int i, int j, double *arr, const Gaussian3D *g, int n) {
    for (int k = 0; k < n; k++) {
        for (int l = k; l < n; l++) {
            arr[AT_INDEX(n, i, j, k, l)] 
                = repulsion(g[i], g[j], g[k], g[l]);
            if (l > k)
                arr[AT_INDEX(n, i, j, l, k)] = arr[AT_INDEX(n, i, j, k, l)];
        }
    }
}


// #define SINGLE_THREADED_ONLY

#ifdef __APPLE__
struct PrimitiveTwoElectronIntegralsThreadData {
    double *arr;
    const Gaussian3D *g;
    int i;
    int n;
};


static void *threaded_set_two_electron_integrals_elements_inner(void *data) {
    struct PrimitiveTwoElectronIntegralsThreadData 
        *thread_data = (struct PrimitiveTwoElectronIntegralsThreadData *)data;
    double *arr = thread_data->arr;
    const Gaussian3D *g = thread_data->g;
    int i = thread_data->i;
    int n = thread_data->n;
    for (int j = i; j < n; j++) {
        set_two_electron_integrals_inner(i, j, arr, g, n);
        if (j > i) {
            copy_block_to_other_block(arr, n, j, i, i, j);
        }
    }
    return NULL;
}


pthread_t s_primitive_threads[MAX_SUPPORTED_THREADS] = {};
struct PrimitiveTwoElectronIntegralsThreadData 
    s_primitive_thread_data[MAX_SUPPORTED_THREADS] = {};
#endif


void set_two_electron_integrals_elements(
    double *arr, const Gaussian3D *g, int n
) {
    #ifdef SINGLE_THREADED_ONLY
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            set_two_electron_integrals_inner(i, j, arr, g, n);
            if (j > i) {
                copy_block_to_other_block(arr, n, j, i, i, j);
            }
        }
    }
    #else
    #ifndef __APPLE__
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            set_two_electron_integrals_inner(i, j, arr, g, n);
            if (j > i) {
                copy_block_to_other_block(arr, n, j, i, i, j);
            }
        }
    }
    #else
    for (int i = 0; i < n; i++) {
        s_primitive_thread_data[i].arr = arr;
        s_primitive_thread_data[i].g = g;
        s_primitive_thread_data[i].i = i;
        s_primitive_thread_data[i].n = n;
        pthread_create(&s_primitive_threads[i], NULL, 
                       threaded_set_two_electron_integrals_elements_inner,
                       (void *)&s_primitive_thread_data[i]);
    }
    for (int i = 0; i < n; i++)
        pthread_join(s_primitive_threads[i], NULL);
    #endif
    #endif
}

#undef AT_INDEX
