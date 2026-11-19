#include "basis_function_array.hpp"
#include "orbitals_description.hpp"
#include "array_helpers.hpp"
#include "gl_wrappers.hpp"

#include <set>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <iostream>

#ifndef _BUILD_ARRAYS_
#define _BUILD_ARRAYS_

namespace build_arrays {


    void fill(
        array_helpers::SquareArray &overlap,
        array_helpers::SquareArray &kinetic,
        array_helpers::SquareArray &nuclear,
        array_helpers::Symmetric4 &repulsion_exchange,
        const BasisFunctionArray &basis_functions,
        const NuclearChargesArray &nuclear_charges
    );

    void fill(
        Quad &bf_spec1, Quad &bf_spec2,
        Quad &primitives,
        Quad &indices_q, Quad &shell_indices_q,
        Quad &shell_rep_exc, Quad &rep_exc1, Quad &rep_exc2,
        array_helpers::SquareArray &overlap,
        array_helpers::SquareArray &kinetic,
        array_helpers::SquareArray &nuclear,
        array_helpers::Symmetric4 &repulsion_exchange,
        const BasisFunctionArray &basis_functions,
        const NuclearChargesArray &nuclear_charges,
        const unsigned int boys_program,
        const unsigned int re_program
    );

}

#endif