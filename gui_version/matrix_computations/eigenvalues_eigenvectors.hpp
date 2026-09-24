#include "array_helpers.hpp"

#ifndef _EIGENVALUES_EIGENVECTORS_
#define _EIGENVALUES_EIGENVECTORS_

void compute_eigenvalues_eigenvectors(
    array_helpers::Array1D &eigenvalues,
    array_helpers::Array2D &eigenvectors,
    const array_helpers::SquareArray &overlap,
    const array_helpers::SquareArray &matrix);

#endif

