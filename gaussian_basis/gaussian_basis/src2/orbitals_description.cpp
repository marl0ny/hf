#include "orbitals_description.hpp"
#include "basis_function_array.hpp"


static unsigned int get_angular_number(const std::string &orbital_letter) {
    if (orbital_letter.substr(1) == "s") {
        return 0;
    }
    else if (orbital_letter.substr(1) == "p") {
        return 1;
    }
    else if (orbital_letter.substr(1) == "d") {
        return 2;
    }
    else if (orbital_letter.substr(1) == "d") {
        return 3;
    }
    return 0;
}

static unsigned int get_angular_multiplicity(const std::string &orbital_letter) {
    if (orbital_letter.substr(1) == "s") {
        return 1;
    }
    else if (orbital_letter.substr(1) == "p") {
        return 3;
    }
    else if (orbital_letter.substr(1) == "d") {
        return 5;
    }
    else if (orbital_letter.substr(1) == "d") {
        return 7;
    }
    return 0;
}

unsigned int 
orbital_description_data::get_number_of_primitives(
    const orbital_description_data::OrbitalsData &d) {
    using namespace orbital_description_data;
    unsigned int size = 0;
    for (const Orbital &o: d.orbitals) {
        unsigned int multiplicity = get_angular_multiplicity(o.name);
        for (const BasisFunction &b: o.basis_functions)
            size += multiplicity*b.primitives.coefficients.size();
    }
    return size;
}

unsigned int 
orbital_description_data::get_number_of_basis_functions(
    const orbital_description_data::OrbitalsData &d) {
    using namespace orbital_description_data;
    unsigned int size = 0;
    for (const Orbital &o: d.orbitals) {
        unsigned int multiplicity = get_angular_multiplicity(o.name);
        size += multiplicity*o.basis_functions.size();
    }
    return size;
}

BasisFunctionArray 
orbital_description_data::get_basis_function_array(
    const orbital_description_data::OrbitalsData &o,
    spatial::Vector position) {
    int number_of_primitives 
        = orbital_description_data::get_number_of_primitives(o);
    int number_of_basis_functions
        = orbital_description_data::get_number_of_basis_functions(o);
    BasisFunctionArray arr(number_of_primitives, number_of_basis_functions);
    for (int orbital_index = 0; 
         orbital_index < o.orbitals.size(); orbital_index++) {
        orbital_description_data::Orbital orbital = o.orbitals[orbital_index];
        int angular_number = get_angular_number(orbital.name);
        int multiplicity = get_angular_multiplicity(orbital.name);
        for (int i = 0; i < multiplicity; i++) {
            spatial::UByte4 angular
                 = spatial::UByte4{.ind{0, 0, 0}};
            if (multiplicity <= 3) {
                angular.ind[i] = angular_number;
            } else if (multiplicity == 5) {
                if (i == 0)
                    angular = spatial::UByte4{.ind{0, 2, 0}};
                if (i == 1)
                    angular = spatial::UByte4{.ind{0, 0, 2}};
                if (i == 2)
                    angular = spatial::UByte4{.ind{1, 1, 0}};
                if (i == 3)
                    angular = spatial::UByte4{.ind{1, 0, 1}};
                if (i == 4)
                    angular = spatial::UByte4{.ind{0, 1, 1}};
            } else if (multiplicity == 7) {
                if (i == 0)
                    angular = spatial::UByte4{.ind{3, 0, 0}};
                if (i == 1)
                    angular = spatial::UByte4{.ind{0, 3, 0}};
                if (i == 2)
                    angular = spatial::UByte4{.ind{0, 0, 3}};
                if (i == 3)
                    angular = spatial::UByte4{.ind{2, 0, 1}};
                if (i == 4)
                    angular = spatial::UByte4{.ind{0, 2, 1}};
                if (i == 5)
                    angular = spatial::UByte4{.ind{0, 1, 2}};
            }
            for (auto &basis_function: orbital.basis_functions) {
                arr.add_basis_function(
                    position, angular,
                    basis_function.primitives.coefficients,
                    basis_function.primitives.exponents);
                // for (auto &e: basis_function.primitives.coefficients)
                //     printf("%g\n", e);
                // printf("\n");
            }
        }
    }
    return arr;
}

BasisFunctionArray 
orbital_description_data::get_basis_function_array(
    const std::vector<orbital_description_data::PositionedOrbitalsData>
     &o_arr) {
    int number_of_primitives = 0;
    int number_of_basis_functions = 0;
    for (const auto &o: o_arr) {
        number_of_primitives += 
            orbital_description_data::get_number_of_primitives(o);
        number_of_basis_functions +=
            orbital_description_data::get_number_of_basis_functions(o);
    }
    BasisFunctionArray basis_func_arr(
        number_of_primitives, number_of_basis_functions);
    for (const auto &o: o_arr) {
        for (int orbital_index = 0; 
            orbital_index < o.orbitals.size(); orbital_index++) {
            orbital_description_data::Orbital orbital = o.orbitals[orbital_index];
            int angular_number = get_angular_number(orbital.name);
            int multiplicity = get_angular_multiplicity(orbital.name);
            for (int i = 0; i < multiplicity; i++) {
                spatial::UByte4 angular
                    = spatial::UByte4{.ind{0, 0, 0}};
                if (multiplicity <= 3) {
                    angular.ind[i] = angular_number;
                } else if (multiplicity == 5) {
                    if (i == 0)
                        angular = spatial::UByte4{.ind{0, 2, 0}};
                    if (i == 1)
                        angular = spatial::UByte4{.ind{0, 0, 2}};
                    if (i == 2)
                        angular = spatial::UByte4{.ind{1, 1, 0}};
                    if (i == 3)
                        angular = spatial::UByte4{.ind{1, 0, 1}};
                    if (i == 4)
                        angular = spatial::UByte4{.ind{0, 1, 1}};
                } else if (multiplicity == 7) {
                    if (i == 0)
                        angular = spatial::UByte4{.ind{3, 0, 0}};
                    if (i == 1)
                        angular = spatial::UByte4{.ind{0, 3, 0}};
                    if (i == 2)
                        angular = spatial::UByte4{.ind{0, 0, 3}};
                    if (i == 3)
                        angular = spatial::UByte4{.ind{2, 0, 1}};
                    if (i == 4)
                        angular = spatial::UByte4{.ind{0, 2, 1}};
                    if (i == 5)
                        angular = spatial::UByte4{.ind{0, 1, 2}};
                }
                for (auto &basis_function: orbital.basis_functions) {
                    spatial::Vector position = o.position;
                    basis_func_arr.add_basis_function(
                        position, angular,
                        basis_function.primitives.coefficients,
                        basis_function.primitives.exponents);
                    // for (auto &e: basis_function.primitives.coefficients)
                    //     printf("%g\n", e);
                    // printf("\n");
                }
            }
        }
    }
    return basis_func_arr;
}

array_helpers::Array2D
orbital_description_data::get_orbital_basis_function_coefficients(
    unsigned int orbital_count,
    const std::vector
    <orbital_description_data::PositionedOrbitalsData> &o_arr) {
    int number_of_basis_functions = 0;
    int basis_func_index = 0;
    for (const auto &o: o_arr) {
        number_of_basis_functions +=
            orbital_description_data::get_number_of_basis_functions(o);
    }
    array_helpers::Array2D coeffs(orbital_count, number_of_basis_functions);
    for (const auto &o: o_arr) {
        for (int orbital_index = 0; 
            orbital_index < o.orbitals.size(); orbital_index++) {
            orbital_description_data::Orbital orbital = o.orbitals[orbital_index];
            int multiplicity = get_angular_multiplicity(orbital.name);
            for (int i = 0; i < multiplicity; i++) {
                for (auto &basis_function: orbital.basis_functions) {
                    double coeff = basis_function.coefficient;
                    coeffs(std::min(basis_func_index, int(orbital_count-1)),
                           basis_func_index) = coeff;
                    basis_func_index++;
                }
            }
        }
    }
    return coeffs;
}