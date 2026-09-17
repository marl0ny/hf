#include "basis_function_array.hpp"
#include "orbitals_description.hpp"
#include "molecular_geometry.hpp"
#include "array_helpers.hpp"
#include "eigenvalues_eigenvectors.hpp"
#include "build_arrays.hpp"
#include "converge.hpp"
#include "compute_energies.hpp"
#include "orbital_shader_creator.hpp"

#include <iostream>
#include <string>

/*
 For implementing MP2, 
 I've followed the Wikipedia page "Moller-Plesset perturbation theory",
 as well as Chapter 23, pg 833 of Boudreau and Swanson (see README).
*/
double mp2(const array_helpers::Array1D &energies, 
           const array_helpers::Array2D &orbitals, 
           const array_helpers::HypercubeArray &repulsion_exchange,
           int occupation_count) {
    int total_count = orbitals.row_size();
    double sum = 0.0;
    for (int occ_ind1 = 0; occ_ind1 < occupation_count; occ_ind1++) {
        for (int occ_ind2 = 0; occ_ind2 < occupation_count; occ_ind2++) {
            for (int virt_ind1 = occupation_count; 
                 virt_ind1 < total_count; virt_ind1++) {
                for (int virt_ind2 = virt_ind1; 
                     virt_ind2 < total_count; virt_ind2++) {
                    double occ_e1 = energies(occ_ind1);
                    double occ_e2 = energies(occ_ind2);
                    double virt_e1 = energies(virt_ind1);
                    double virt_e2 = energies(virt_ind2);
                    double occ12_virt12 = repulsion_exchange.reduce(
                        0, 2,
                        orbitals.c_ptr(occ_ind1), orbitals.c_ptr(occ_ind2),
                        1, 3,
                        orbitals.c_ptr(virt_ind1), orbitals.c_ptr(virt_ind2));
                    // double virt12_occ12 = occ12_virt12;
                    double virt12_occ12 = repulsion_exchange.reduce(
                        0, 2,
                        orbitals.c_ptr(virt_ind1), orbitals.c_ptr(virt_ind2),
                        1, 3,
                        orbitals.c_ptr(occ_ind1), orbitals.c_ptr(occ_ind2));
                    double virt12_occ21 = repulsion_exchange.reduce(
                        0, 2,
                        orbitals.c_ptr(virt_ind1), orbitals.c_ptr(virt_ind2),
                        1, 3,
                        orbitals.c_ptr(occ_ind2), orbitals.c_ptr(occ_ind1));
                    double 
                    term = 2.0*occ12_virt12*virt12_occ12/(
                        occ_e1 + occ_e2 - virt_e1 - virt_e2);
                    term -= occ12_virt12*virt12_occ21/(
                        occ_e1 + occ_e2 - virt_e1 - virt_e2); 
                    // sum += term;
                    sum += ((virt_ind2 == virt_ind1)? term: 2.0*term);
                }
            }
        }
    }
    return sum;
}

/* array_helpers::Array2D construct_ci_matrix(
    const array_helpers::Array1D &energies,
    const array_helpers::Array2D &orbitals,
    const array_helpers::SquareArray &kinetic,
    const array_helpers::SquareArray &nuclear,
    const array_helpers::HypercubeArray &repulsion_exchange,
    int occupation_count
) {
    int total_count = orbitals.row_size();
    for (int i = occupation_count; i < total_count; i++) {

    }
    double kinetic_energy = get_kinetic_energy(kinetic, orbitals, orbitals);
    double nuclear_potential = get_nuclear_potential_energy(nuclear, orbitals, orbitals);
    get_repulsion_exchange_energy(repulsion_exchange, orbitals, orbitals);


}*/