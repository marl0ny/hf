#include "build_arrays.hpp"

#include <pthread.h>


#ifndef __EMSCRIPTEN__
#define THREAD_COUNT 10
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
    for (int i = start; i < end; i++) {
        for (int j = i; j < n; j++) {
            overlap->operator()(i, j) = basis_functions->overlap(i, j);
            kinetic->operator()(i, j) = basis_functions->kinetic(i, j);
            nuclear->operator()(i, j) = basis_functions->nuclear(
                i, j, *nuclear_charges);
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
                        repulsion_exchange->operator()(l, k, i, j) = val;
                        repulsion_exchange->operator()(l, k, j, i) = val;
                        repulsion_exchange->operator()(k, l, i, j) = val;
                        repulsion_exchange->operator()(k, l, j, i) = val;
                    }
                }
            }
            if (j > i) {
                overlap->operator()(j, i) = overlap->operator()(i, j);
                kinetic->operator()(j, i) = kinetic->operator()(i, j);
                nuclear->operator()(j, i) = nuclear->operator()(i, j);
                repulsion_exchange->operator()(
                    j, i, repulsion_exchange->operator()(i, j));
            }
        }
    }
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
