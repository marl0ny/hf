#include "integrals3d.hpp"
#include "gaussian3d.hpp"
#include "matrices.hpp"

#include <stdio.h>
#ifdef __APPLE__
#include <pthread.h>
#endif

void set_overlap_elements(double *mat, const Gaussian3D *g, int n) {
    for (int i = 0; i < n; i++) {
        // printf("Gaussian %d\n", i);
        // printf("Amplitude: %g\n", g[i].amplitude());
        // printf("Orbital exponent: %g\n", g[i].orbital_exponent());
        // printf("Angular number: %g, %g, %g\n", 
        //        g[i].angular()[0], g[i].angular()[1], g[i].angular()[2]);
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

#define AT_INDEX(n, i, j, k, l) \
    (i)*(n)*(n)*(n) + (j)*(n)*(n) + (k)*(n) + (l)

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

// #define SINGLE_THREADED_ONLY

#ifdef __APPLE__
struct ThreadData {
    double *arr;
    const Gaussian3D *g;
    int i;
    int n;
};


static void *threaded_set_two_electron_integrals_elements_inner(void *data) {
    struct ThreadData *thread_data = (struct ThreadData *)data;
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


#define MAX_SUPPORTED_THREADS 500
pthread_t s_threads[MAX_SUPPORTED_THREADS] = {};
struct ThreadData s_thread_data[MAX_SUPPORTED_THREADS] = {};
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
        s_thread_data[i].arr = arr;
        s_thread_data[i].g = g;
        s_thread_data[i].i = i;
        s_thread_data[i].n = n;
        pthread_create(&s_threads[i], NULL, 
                       threaded_set_two_electron_integrals_elements_inner,
                       (void *)&s_thread_data[i]);
    }
    for (int i = 0; i < n; i++)
        pthread_join(s_threads[i], NULL);
    #endif
    #endif
}

#undef AT_INDEX
