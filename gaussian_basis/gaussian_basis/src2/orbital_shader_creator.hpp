#include "array_helpers.hpp"
#include "basis_function_array.hpp"


#ifndef _ORBITAL_SHADER_CREATOR_
#define _ORBITAL_SHADER_CREATOR_

std::string express_orbitals_as_function(
    const array_helpers::Array2D &orbitals,
    const BasisFunctionArray &arr);

void write_orbital_to_file(
    const array_helpers::Array2D &orbitals,
    const BasisFunctionArray &arr);

#endif