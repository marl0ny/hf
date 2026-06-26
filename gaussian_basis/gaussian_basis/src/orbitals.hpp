#include "orbitals_parameters.hpp"
#include "molecular_geometry.hpp"
#include "basis_function.hpp"
#include "gaussian3d.hpp"
#include "nuclear.hpp"

#ifndef _ORBITALS_
#define _ORBITALS_

class Orbitals {
    std::vector<uint8_t> m_data;
    std::vector<double> m_basis_func_coefficients;
    int m_primitives_count;
    int m_basis_func_count;
    const BasisFunction *get_basis_functions() const;
    public:
    Orbitals(const MolecularOrbitalsParameters &params);
    std::vector<double> get_overlap() const;
    std::vector<double> get_kinetic() const;
    std::vector<double> get_two_electron_integrals() const;
    std::vector<double> get_nuclear_potential(
        const NuclearConfiguration &n) const;
    void print() const;
    // MolecularOrbitalsParameters get_parameters() const;
    static Orbitals from_geometry_and_atomic_json_data(
        const molecular_geometry::MolecularGeometry &geom,
        const std::map<molecular_geometry::AtomicSymbol,std::string>
            &atomic_json_data);
    int number_of_basis_functions() const;
};


// get_orbitals_from_
// const molecular_geometry::MolecularGeometry &geom,
//     const std::map<
//         molecular_geometry::AtomicSymbol, 
//         AtomicOrbitalsParameters> &atoms);


#endif
