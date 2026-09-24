#include "basis_function_array.hpp"
#include "integrals/gaussian3d.hpp"
#include "integrals/integrals3d.hpp"

#include <iostream>


Gaussian3D BasisFunctionArray::primitive(const BasisFunctionData &b, int i) const {
    PrimitiveData p = this->m_gaussian_primitive_data[i + b.primitives.offset];
    double amplitude = p.amplitude;
    double exponent = p.exponent;
    spatial::Vector position = b.position;
    spatial::UByte4 angular = b.angular;
    return Gaussian3D(
        exponent, amplitude, 
        (int)angular.x, (int)angular.y, (int)angular.z, position);
}

BasisFunctionArray::BasisFunctionArray(
    int number_of_primitives, int number_of_basis_functions) {
    m_gaussian_primitive_data
        = std::vector<PrimitiveData>(number_of_primitives);
    m_basis_function_data 
        = std::vector<BasisFunctionData>(number_of_basis_functions);
    m_shell_data = std::vector<ShellData> (0);
    m_shell_data.reserve(number_of_basis_functions/4);
    m_primitive_count = 0;
    m_basis_function_count = 0;
    m_shell_count = 0;
}

void BasisFunctionArray::add_basis_function(
    spatial::Vector &position, spatial::UByte4 &angular,
    const std::vector<double> &primitive_amplitudes,
    const std::vector<double> &primitive_exponents
    ) {
    spatial::Vector prev_pos 
        = m_basis_function_data[m_basis_function_count - 1].position;
    spatial::UByte4 prev_angular
        = m_basis_function_data[m_basis_function_count - 1].angular;
    unsigned int angular_curr
        = (unsigned int)(angular.ind[0]) 
            + (unsigned int)(angular.ind[1])
            + (unsigned int)(angular.ind[2]) 
            + (unsigned int)(angular.ind[3]);
    unsigned int angular_prev_s
        = (unsigned int)(prev_angular.ind[0])
            + (unsigned int)(prev_angular.ind[1])
            + (unsigned int)(prev_angular.ind[2])
            + (unsigned int)(prev_angular.ind[3]);
    double diff2 = spatial::dot(prev_pos - position, prev_pos - position);
    if (m_shell_count == 0 ||
        diff2 > 1e-20 || angular_prev_s != angular_curr) {
        m_shell_data.push_back({
            .angular=angular_curr,
            .basis_functions.count=1,
            .basis_functions.offset=(unsigned int)m_basis_function_count,
            .hash=0
        });
        this->m_shell_count++;
    } else {
        m_shell_data[m_shell_count - 1].basis_functions.count++;
    }
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
    const BasisFunctionData &basis_func_a = m_basis_function_data[a];
    const BasisFunctionData &basis_func_b = m_basis_function_data[b];
    const BasisFunctionData &basis_func_c = m_basis_function_data[c];
    const BasisFunctionData &basis_func_d = m_basis_function_data[d];
    double sum = 0.0;
    int start_j = 0, start_l = 0;
    int count_a = basis_func_a.primitives.count;
    int count_b = basis_func_b.primitives.count;
    int count_c = basis_func_c.primitives.count;
    int count_d = basis_func_d.primitives.count;
    for (int i = 0; i < count_a; i++) {
        // start_j = (a == b)? i: 0;
        for (int j = start_j; j < count_b; j++) {
            for (int k = 0; k < count_c; k++) {
                // start_l = (c == d)? k: 0;
                for (int l = start_l; l < count_d; l++) {
                    double val = repulsion(
                        primitive(basis_func_a, i),
                        primitive(basis_func_b, j),
                        primitive(basis_func_c, k),
                        primitive(basis_func_d, l)
                    );
                    // if (a == b && start_j > i)
                    //     val *= 2.0;
                    // if (c == d && start_l > k)
                    //     val *= 2.0;
                    sum += val;
                }
            }
        }
    }
    return sum;
}


/* For finding info that explained and motivated the Schwarz
Inequality's usage in Computational Chemistry, I found this
article's review of these topics helpful:

"A New Scalable Parallel Algorithm for Fock Matrix Construction"
Lui X., Patel A., Chow E.
2014 IEEE 28th International Parallel and Distributed Processing Symposium.
902-914 (2014).

See Section II "Background" D on screening. I've only consulted
this article for this explanation of this background material,
though perhaps I might try to implement this article's algorithm in the
future.
*/
double 
BasisFunctionArray::repulsion_exchange(
    int a, int b, int c, int d,
    const array_helpers::SquareArray &re_nm_nm) const {
    if ((a == c && b == d) || (a == d && b == c)) {
        return re_nm_nm(a, b);
    }
    double four_e1 = re_nm_nm(a, b);
    double four_e2 = re_nm_nm(c, d);
    if (sqrt(four_e1 * four_e2) < 1e-10) {
        // printf("Integral screened (%d, %d, %d, %d)\n", a, b, c, d);
        return 0.0;
    }
    const BasisFunctionData &basis_func_a = m_basis_function_data[a];
    const BasisFunctionData &basis_func_b = m_basis_function_data[b];
    const BasisFunctionData &basis_func_c = m_basis_function_data[c];
    const BasisFunctionData &basis_func_d = m_basis_function_data[d];
    double sum = 0.0;
    int start_j = 0, start_l = 0;
    int count_a = basis_func_a.primitives.count;
    int count_b = basis_func_b.primitives.count;
    int count_c = basis_func_c.primitives.count;
    int count_d = basis_func_d.primitives.count;
    for (int i = 0; i < count_a; i++) {
        // start_j = (a == b)? i: 0;
        for (int j = start_j; j < count_b; j++) {
            for (int k = 0; k < count_c; k++) {
                // start_l = (c == d)? k: 0;
                for (int l = start_l; l < count_d; l++) {
                    double val = repulsion(
                        primitive(basis_func_a, i),
                        primitive(basis_func_b, j),
                        primitive(basis_func_c, k),
                        primitive(basis_func_d, l)
                    );
                    // if (a == b && start_j > i)
                    //     val *= 2.0;
                    // if (c == d && start_l > k)
                    //     val *= 2.0;
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

double BasisFunctionArray
::evaluate_at(int index, const spatial::Vector &r1) const {
    BasisFunctionData b = m_basis_function_data[index];
    double val = 0.0;
    for (int i = 0; i < b.primitives.count; i++) {
        Gaussian3D g = this->primitive(b, i);
        spatial::Vector r0 = g.position();
        spatial::Vector angular = g.angular();
        double exponent = g.orbital_exponent();
        double x1 = r1.x, y1 = r1.y, z1 = r1.z;
        double x0 = r0.x, y0 = r0.y, z0 = r0.z;
        spatial::Vector r = r1 - r0;
        val += 
            pow(x1 - x0, angular.x)
            *pow(y1 - y0, angular.y)
            *pow(z1 - z0, angular.z)*
            exp(-exponent*dot(r, r));
    }
    return val;
}

double BasisFunctionArray::evaluate_at(
    const array_helpers::Array2D &orbitals,
    int index, const spatial::Vector &r
) const {
    double value = 0.0;
    for (int i = 0; i < orbitals.row_size(); i++) {
        double c = orbitals(index, i);
        if (std::abs(c) > 1e-40)
            value += c*this->evaluate_at(i, r);
    }
    return value;
}

int BasisFunctionArray::primitive_count_at(int index) const {
    BasisFunctionData b = m_basis_function_data[index];
    return b.primitives.count;
}

spatial::Vector
BasisFunctionArray::get_position(int index) const {
    return m_basis_function_data[index].position;
}

spatial::UByte4
BasisFunctionArray::get_angular(int index) const {
    return m_basis_function_data[index].angular;
}

Gaussian3D BasisFunctionArray::
get_primitive(int ind_bf, int ind_p) const {
    return primitive(m_basis_function_data[ind_bf], ind_p);
}

void BasisFunctionArray::print() const {
    for (int i = 0, shell_ind = 0; i < m_basis_function_count; i++) {
        BasisFunctionData basis_function = m_basis_function_data[i];
        if (m_shell_data[shell_ind].basis_functions.offset == i) {
            std::cout << "##############################";
            std::cout << "##############################" << std::endl;
            std::cout << "Shell " << shell_ind << std::endl;
            std::cout << "angular: ";
            std::cout << m_shell_data[shell_ind].angular << std::endl;
            std::cout << "position: ";
            std::cout << basis_function.position.x << ", ";
            std::cout << basis_function.position.y << ", ";
            std::cout << basis_function.position.z << std::endl;
            std::cout << std::endl;
            shell_ind++;
        }
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
    std::cout << "##############################";
    std::cout << "##############################" << std::endl;
}