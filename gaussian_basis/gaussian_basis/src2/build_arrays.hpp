#include "basis_function_array.hpp"
#include "orbitals_description.hpp"
#include "array_helpers.hpp"

#ifndef _BUILD_ARRAYS_
#define _BUILD_ARRAYS_

namespace build_arrays {

    void fill(
        array_helpers::SquareArray &overlap,
        array_helpers::SquareArray &kinetic,
        array_helpers::SquareArray &nuclear,
        array_helpers::HypercubeArray &repulsion_exchange,
        const BasisFunctionArray &basis_functions,
        const NuclearChargesArray &nuclear_charges
    );

}

#endif