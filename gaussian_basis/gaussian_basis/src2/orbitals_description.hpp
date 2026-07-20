#include <vector>

#include "basis_function_array.hpp"
#include "array_helpers.hpp"

#ifndef _ORBITALS_DESCRIPTION_
#define _ORBITALS_DESCRIPTION_


namespace orbital_description_data {

struct Primitives {
    std::vector<double> coefficients;
    std::vector<double> exponents;
};

struct BasisFunction {
    double coefficient;
    Primitives primitives;
};

struct Orbital {
    std::string name;
    std::vector<BasisFunction> basis_functions;
};

struct OrbitalsData {
    std::vector<Orbital> orbitals;
};

struct PositionedOrbitalsData: OrbitalsData {
    spatial::Vector position;
    spatial::Vector get_position() const;
};

unsigned int get_number_of_primitives(
    const orbital_description_data::OrbitalsData &d);

unsigned int get_number_of_basis_functions(
    const orbital_description_data::OrbitalsData &d);

BasisFunctionArray get_basis_function_array(
    const orbital_description_data::OrbitalsData &d,
    spatial::Vector position);

BasisFunctionArray get_basis_function_array(
    const std::vector<PositionedOrbitalsData> &o_array);

array_helpers::Array2D get_orbital_basis_function_coefficients(
    unsigned int orbital_count,
    const std::vector<PositionedOrbitalsData> &o_array);

void add_to_basis_function_array(
    BasisFunctionArray &basis_functions,
    const orbital_description_data::OrbitalsData &d,
    spatial::Vector position);

}

#endif
