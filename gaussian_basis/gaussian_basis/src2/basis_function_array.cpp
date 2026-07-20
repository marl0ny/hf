#include "basis_function_array.hpp"
#include "integrals/gaussian3d.hpp"
#include "integrals/integrals3d.hpp"

#include <iostream>


Gaussian3D BasisFunctionArray::primitive(const BasisFunctionData &b, int i) const {
    PrimitiveData p = this->m_gaussian_primitive_data[i];
    double amplitude = p.amplitude;
    double exponent = p.exponent;
    spatial::Vector position = b.position;
    spatial::UByte4 angular = b.angular;
    return Gaussian3D(
        exponent, amplitude, angular.x, angular.y, angular.z, position);
}

BasisFunctionArray::BasisFunctionArray(
    int number_of_primitives, int number_of_basis_functions) {
    m_gaussian_primitive_data
        = std::vector<PrimitiveData>(number_of_primitives);
    m_basis_function_data 
        = std::vector<BasisFunctionData>(number_of_basis_functions);
    m_primitive_count = 0;
    m_basis_function_count = 0;
}

void BasisFunctionArray::add_basis_function(
    spatial::Vector &position, spatial::UByte4 &angular,
    const std::vector<double> &primitive_amplitudes,
    const std::vector<double> &primitive_exponents
    ) {
    m_basis_function_data[m_basis_function_count].angular = angular;
    m_basis_function_data[m_basis_function_count].position = position;
    m_basis_function_data[m_basis_function_count].primitives.count
        = (unsigned int)primitive_amplitudes.size();
    m_basis_function_data[m_basis_function_count].primitives.offset
        = (unsigned int)m_primitive_count;
    for (int i = 0; i < primitive_amplitudes.size(); i++) {
        m_gaussian_primitive_data[m_primitive_count].amplitude 
            = primitive_amplitudes[i];
        m_gaussian_primitive_data[m_primitive_count].exponent
            = primitive_exponents[i];
        m_primitive_count++;
    }
    m_basis_function_count++;

}

double BasisFunctionArray::overlap(int a, int b) const {
    BasisFunctionData basis_func_a = m_basis_function_data[a];
    BasisFunctionData basis_func_b = m_basis_function_data[b];
    double sum = 0.0;
    for (int i = 0; i < basis_func_a.primitives.count; i++) {
        if (a == b)
            for (int j = i; j < basis_func_b.primitives.count; j++)
                sum += ((j == i)? 1.0: 2.0)*::overlap(
                    primitive(basis_func_a, i),
                    primitive(basis_func_b, j));
        else
            for (int j = 0; j < basis_func_b.primitives.count; j++)
                sum += ::overlap(
                    primitive(basis_func_a, i),
                    primitive(basis_func_b, j));
    }
    return sum;
}


double BasisFunctionArray::kinetic(int a, int b) const {
    double sum = 0.0;
    BasisFunctionData basis_func_a = m_basis_function_data[a];
    BasisFunctionData basis_func_b = m_basis_function_data[b];
    for (int i = 0; i < basis_func_a.primitives.count; i++) {
        if (a == b)
            for (int j = i; j < basis_func_b.primitives.count; j++)
                sum += ((j == i)? 1.0: 2.0)*::kinetic(
                    primitive(basis_func_a, i),
                    primitive(basis_func_b, j)
                );
        else
            for (int j = 0; j < basis_func_b.primitives.count; j++)
                sum += ::kinetic(
                    primitive(basis_func_a, i), primitive(basis_func_b, j));
    }
    return sum;
}

double BasisFunctionArray
::nuclear(int a, int b, const NuclearChargesArray &nuclear_charges) const {
    BasisFunctionData basis_func_a = m_basis_function_data[a];
    BasisFunctionData basis_func_b = m_basis_function_data[b];
    double sum = 0.0;
    for (int k = 0; k < nuclear_charges.size(); k++) {
        for (int i = 0; i < basis_func_a.primitives.count; i++) {
            if (a == b) {
                for (int j = i; j < basis_func_b.primitives.count; j++) {
                    double val = ((double)nuclear_charges.strength(k))
                        *nuclear_single_charge(
                            primitive(basis_func_a, i),
                            primitive(basis_func_b, j),
                            nuclear_charges.location(k));
                    if (j > i)
                        val *= 2.0;
                    sum += val;
                }
            } else {
                for (int j = 0; j < basis_func_b.primitives.count; j++)
                    sum += ((double)nuclear_charges.strength(k))
                        *nuclear_single_charge(
                            primitive(basis_func_a, i),
                            primitive(basis_func_b, j),
                            nuclear_charges.location(k));
            }
        }
    }
    return sum;
}

double 
BasisFunctionArray::repulsion_exchange(int a, int b, int c, int d) const {
    BasisFunctionData basis_func_a = m_basis_function_data[a];
    BasisFunctionData basis_func_b = m_basis_function_data[b];
    BasisFunctionData basis_func_c = m_basis_function_data[c];
    BasisFunctionData basis_func_d = m_basis_function_data[d];
    double sum = 0.0;
    int start_index_j = 0, start_index_k = 0, start_index_l = 0;
    for (int i = 0; i < basis_func_a.primitives.count; i++) {
        // int start_index_j = (a == b)? i: 0;
        for (int j = start_index_j;
             j < basis_func_b.primitives.count; j++) {
            for (int k = start_index_k;
                 k < basis_func_c.primitives.count; k++) {
                for (int l = start_index_l; 
                     l < basis_func_d.primitives.count; l++) {
                    double val = repulsion(
                        primitive(basis_func_a, i),
                        primitive(basis_func_b, j),
                        primitive(basis_func_c, k),
                        primitive(basis_func_d, l)
                    );
                    sum += val;
                }
            }
        }
    }
    return sum;
}

int BasisFunctionArray::get_number_of_basis_functions() const {
    return m_basis_function_count;
}

void BasisFunctionArray::print() const {
    for (int i = 0; i < m_basis_function_count; i++) {
        BasisFunctionData basis_function = m_basis_function_data[i];
        std::cout << "Basis function " << i << std::endl;
        // std::cout >> "coefficient: " << basis_function.
        std::cout << "position: ";
        std::cout << basis_function.position.x << ", ";
        std::cout << basis_function.position.y << ", ";
        std::cout << basis_function.position.z << std::endl;
        std::cout << "angular: ";
        std::cout << (int)basis_function.angular.x << ", ";
        std::cout << (int)basis_function.angular.y << ", ";
        std::cout << (int)basis_function.angular.z << std::endl;
        std::cout << "primitives count: ";
        std::cout << basis_function.primitives.count << std::endl;
        std::cout << "primitives offset: ";
        std::cout << basis_function.primitives.offset << std::endl;
        int primitive_count = basis_function.primitives.count;
        std::cout << "primitives:\n";
        std::cout << "coefficients\texponents\n";
        for (int j = 0; j < primitive_count; j++) {
            PrimitiveData primitive = 
                m_gaussian_primitive_data[
                    basis_function.primitives.offset + j];
            double amplitude = primitive.amplitude;
            double exponent = primitive.exponent;
            std::cout << amplitude << "\t" << exponent << std::endl;
        }
        if (i != m_basis_function_count - 1)
            std::cout << "\n";
    }
}