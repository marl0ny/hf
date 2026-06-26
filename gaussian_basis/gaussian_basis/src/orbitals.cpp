#include "orbitals.hpp"
#include "parse_subset_of_json.hpp"
#include "vec3.hpp"
#include "matrices.hpp"

// #include <iostream>

static void encode(
    BasisFunction *basis_functions, Gaussian3D *primitives,
    const std::vector<AtomicOrbitalsParameters> &atomic_orbitals_params_list
    ) {
    int primitives_ind = 0;
    int basis_func_ind = 0;
    for (const AtomicOrbitalsParameters& 
         atomic_orbitals_params: atomic_orbitals_params_list) {
        spatial::Vec3 position = atomic_orbitals_params.get_position();
        for (const auto &orbital_param: atomic_orbitals_params) {
            int multiplicity = orbital_param.get_multiplicity();
            for (int k = 0; k < multiplicity; k++) {
                short angular[3] = {0, 0, 0};
                angular[k] = orbital_param.get_angular_number();
                for (const auto &basis_func_param: orbital_param) {
                    basis_functions[basis_func_ind] = {
                        .count=basis_func_param.number_of_primitives(),
                        .primitives=&primitives[primitives_ind]};
                    for (const auto &primitive_param: basis_func_param) {
                        primitives[primitives_ind] = Gaussian3D(
                            primitive_param.exponent,
                            primitive_param.coefficient,
                            angular[0], angular[1], angular[2],
                            position);
                        primitives_ind++;
                    }
                    basis_func_ind++;
                }
            }
        }
    }
}

const BasisFunction *Orbitals::get_basis_functions() const {
    return (const BasisFunction *)&m_data[0];
}

Orbitals::Orbitals(const MolecularOrbitalsParameters &params) {
    m_primitives_count = params.multiplicity_adjusted_number_of_primitives();
    m_basis_func_count 
        = params.multiplicity_adjusted_number_of_basis_functions();
    m_data = std::vector<uint8_t> (
        m_basis_func_count*sizeof(BasisFunction)
         + m_primitives_count*sizeof(Gaussian3D));
    int j = 0;
    encode(
        (BasisFunction *)&m_data[0], 
        (Gaussian3D *)((BasisFunction *)&m_data[0] + m_basis_func_count),
        params.get_all_atomic_orbitals());
    m_basis_func_coefficients = std::vector<double>(0);
    for (const AtomicOrbitalsParameters &atomics_params: params) {
        for (const OrbitalParameters &orbital_params: atomics_params) {
            int multiplicity = orbital_params.get_multiplicity();
            for (const BasisFunctionParameters 
                 &basis_func_parameters: orbital_params) {
                double coefficient = basis_func_parameters.get_coefficient();
                m_basis_func_coefficients.push_back(coefficient);
            }
        }
    }
}

void Orbitals::print() const {
    BasisFunction *basis_functions = (BasisFunction *)&m_data[0];
    Gaussian3D *primitives
        = (Gaussian3D *)((BasisFunction *)&m_data[0] + m_basis_func_count);
    for (int i = 0; i < m_basis_func_count; i++) {
        printf("Basis function %d\n", i);
        Gaussian3D *primitives = basis_functions[i].primitives;
        for (int j = 0; j < basis_functions[i].count; j++) {
            printf("Primitive %d\n", j);
            printf("Amplitude: %g\n",
                primitives[j].amplitude());
            printf("Orbital exponent: %g\n",
                primitives[j].orbital_exponent());
            printf("Position: (%g, %g, %g)\n",
                primitives[j].position()[0],
                primitives[j].position()[1],
                primitives[j].position()[2]);
            printf("Angular numbers: (%g, %g, %g)\n",
                primitives[j].angular()[0],
                primitives[j].angular()[1],
                primitives[j].angular()[2]);
        }
        printf("\n");
    }

}

using namespace molecular_geometry;
using namespace parse_subset_of_json;

Orbitals Orbitals::
from_geometry_and_atomic_json_data(
    const molecular_geometry::MolecularGeometry &geom, 
    const std::map<molecular_geometry::AtomicSymbol, std::string>
    &atomic_json_data) {
    MolecularOrbitalsParameters molecular_params;
    for (int e = int(AtomicSymbol::H); e <= int(AtomicSymbol::CA); e++) {
        AtomicSymbol element_symbol = AtomicSymbol(e);
        if (geom[element_symbol].size() > 0) {
            string json_data = atomic_json_data.at(element_symbol);
            AtomicOrbitalsParameters atomic_params 
                = get_atomic_orbitals_parameters(
                        Vec3{}, json_data);
            for (const spatial::Vec3 &position: geom[element_symbol]) {
                atomic_params.set_position(position);
                molecular_params.add_atomic_orbitals(atomic_params);
            }

        }
    }
    return (molecular_params);
}

int Orbitals::number_of_basis_functions() const {
    return m_basis_func_count;
}

std::vector<double> Orbitals::get_overlap() const {
    std::vector<double> 
    overlap_matrix(m_basis_func_count*m_basis_func_count);
    set_overlap_elements(
        &overlap_matrix[0], get_basis_functions(), m_basis_func_count);
    return overlap_matrix;

}

std::vector<double> Orbitals::get_kinetic() const {
    std::vector<double>
    kinetic_matrix(m_basis_func_count*m_basis_func_count);
    set_kinetic_elements(
        &kinetic_matrix[0], get_basis_functions(), m_basis_func_count);
    return kinetic_matrix;
}


std::vector<double> Orbitals::get_two_electron_integrals() const {
    int n = m_basis_func_count;
    std::vector<double> two_electron_integrals(n*n*n*n);
    set_two_electron_integrals_elements(
        &two_electron_integrals[0], get_basis_functions(), n);
    return two_electron_integrals;
}


std::vector<double> Orbitals::get_nuclear_potential(
    const NuclearConfiguration &nuc_config) const {
    std::vector<double> nuclear_potential(
        m_basis_func_count, m_basis_func_count);
    set_nuclear_potential_elements(
        &nuclear_potential[0],
        get_basis_functions(), m_basis_func_count,
        (const Nuclear *)&nuc_config[0], nuc_config.size());
    return nuclear_potential;
}
