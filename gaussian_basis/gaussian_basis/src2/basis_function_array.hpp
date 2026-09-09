#include "spatial.hpp"
#include "nuclear_charges.hpp"
#include "integrals/gaussian3d.hpp"
#include "array_helpers.hpp"
#include <vector>

#ifndef _BASIS_FUNCTION_
#define _BASIS_FUNCTION_

struct PrimitiveData {
    double amplitude;
    double exponent;
};

struct BasisFunctionData {
    struct {
        unsigned int offset;
        unsigned int count;
    } primitives;
    spatial::Vector position;
    spatial::UByte4 angular;
};

struct ShellData {
    unsigned int angular;
    struct {
        unsigned int offset;
        unsigned int count;
    } basis_functions;
    unsigned long hash;
};

class BasisFunctionArray {
    std::vector <PrimitiveData> m_gaussian_primitive_data;
    std::vector <BasisFunctionData> m_basis_function_data;
    std::vector <ShellData> m_shell_data;
    int m_primitive_count = 0;
    int m_basis_function_count = 0;
    int m_shell_count = 0;
    Gaussian3D primitive(
        const BasisFunctionData &basis_function_data, int index) const;
    Gaussian3D primitive(
        int basis_function_index, int primitive_index) const;
    public:
    BasisFunctionArray(
        int number_of_primitives, int number_of_basis_functions);
    // void extend_size(
    //     int number_of_primitives, int number_of_basis_functions);
    void add_basis_function(
        spatial::Vector &position, spatial::UByte4 &angular,
        const std::vector<double> &primitive_amplitudes,
        const std::vector<double> &primitive_exponents
    );
    double overlap(int i, int j) const;
    double kinetic(int i, int j) const;
    double nuclear(int i, int j, 
        const NuclearChargesArray &nuclear_charges) const;
    double repulsion_exchange(int i, int j, int k, int l) const;
    double repulsion_exchange(
        int i, int j, int k, int l,
        const array_helpers::SquareArray &re_abab) const;
    int get_number_of_basis_functions() const;
    double evaluate_at(
        int index, const spatial::Vector &r) const;
    double evaluate_at(
        const array_helpers::Array2D &orbitals,
        int index, const spatial::Vector &r
    ) const;
    void print() const;
};

class ShellArray {
    BasisFunctionArray m_basis_functions;

};

#endif
