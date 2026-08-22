#include "basis_function_array.hpp"
#include "orbitals_description.hpp"
#include "atomic_data.hpp"
#include "molecular_geometry.hpp"
#include "array_helpers.hpp"
#include "eigenvalues_eigenvectors.hpp"


void iteration(
    array_helpers::Array1D &energies,
    array_helpers::Array2D &next_orbitals,
    const array_helpers::SquareArray &overlap,
    const array_helpers::SquareArray &kinetic_nuclear,
    const array_helpers::HypercubeArray &repulsion_exchange_tensor,
    const array_helpers::Array2D &orbitals) {
    array_helpers::SquareArray
    repulsion = repulsion_exchange_tensor.reduce(
        2, 3, orbitals, orbitals
    );
    array_helpers::SquareArray
    exchange = repulsion_exchange_tensor.reduce(
        1, 3, orbitals, orbitals
    );
    array_helpers::SquareArray
    fock = kinetic_nuclear + (2.0*repulsion - exchange);

    compute_eigenvalues_eigenvectors(
        energies, next_orbitals, overlap, fock);

}

void iteration(
    array_helpers::Array1D &energies_u,
    array_helpers::Array2D &next_orbitals_u,
    array_helpers::Array1D &energies_d,
    array_helpers::Array2D &next_orbitals_d,
    const array_helpers::SquareArray &overlap,
    const array_helpers::SquareArray &kinetic_nuclear,
    const array_helpers::HypercubeArray &repulsion_exchange_tensor,
    const array_helpers::Array2D &orbitals_u,
    const array_helpers::Array2D &orbitals_d
) {
    array_helpers::Array2D orbitals = array_helpers::row_stack(
        orbitals_u, orbitals_d);
    array_helpers::SquareArray
    repulsion = repulsion_exchange_tensor.reduce(
        2, 3, orbitals, orbitals
    );
    array_helpers::SquareArray
    exchange_u = repulsion_exchange_tensor.reduce(
        1, 3, orbitals_u, orbitals_u
    );
    array_helpers::SquareArray
    exchange_d = repulsion_exchange_tensor.reduce(
        1, 3, orbitals_d, orbitals_d
    );
    // printf("Row sizes: %d, %d, %d\n", 
    //        kinetic_nuclear.row_size(), repulsion.row_size(), exchange_u.row_size());
    array_helpers::SquareArray
    fock_u = kinetic_nuclear + (repulsion - exchange_u);
    array_helpers::SquareArray
    fock_d = kinetic_nuclear + (repulsion - exchange_d);

    compute_eigenvalues_eigenvectors(
        energies_u, next_orbitals_u, overlap, fock_u);
    compute_eigenvalues_eigenvectors(
        energies_d, next_orbitals_d, overlap, fock_d);
}

double get_kinetic_energy(
    const array_helpers::SquareArray &kinetic,
    const array_helpers::Array2D &orbitals) {
    return 2.0*kinetic.reduce(orbitals);
}

double get_kinetic_energy(
    const array_helpers::SquareArray &kinetic,
    const array_helpers::Array2D &orbitals1,
    const array_helpers::Array2D &orbitals2) {
    return 2.0*kinetic.reduce(orbitals1, orbitals2);
}

double get_nuclear_potential_energy(
    const array_helpers::SquareArray &nuclear,
    const array_helpers::Array2D &orbitals) {
    return 2.0*nuclear.reduce(orbitals);
}

double get_nuclear_potential_energy(
    const array_helpers::SquareArray &nuclear,
    const array_helpers::Array2D &orbitals1,
    const array_helpers::Array2D &orbitals2) {
    return 2.0*nuclear.reduce(orbitals1, orbitals2);
}

double get_repulsion_energy(
    const array_helpers::HypercubeArray &repulsion_exchange,
    const array_helpers::Array2D &orbitals
) {
    return repulsion_exchange.reduce(
        0, 1, orbitals, orbitals, 2, 3, orbitals, orbitals
    );
}

double get_exchange_energy(
    const array_helpers::HypercubeArray &repulsion_exchange,
    const array_helpers::Array2D &orbitals
) {
    return repulsion_exchange.reduce(
        0, 2, orbitals, orbitals, 1, 3, orbitals, orbitals
    );
}

double get_repulsion_exchange_energy(
    const array_helpers::HypercubeArray &repulsion_exchange,
    const array_helpers::Array2D &orbitals) {
    double repulsion = repulsion_exchange.reduce(
        0, 1, orbitals, orbitals, 2, 3, orbitals, orbitals
    );
    double exchange = repulsion_exchange.reduce(
        0, 2, orbitals, orbitals, 1, 3, orbitals, orbitals
    );
    return 2.0*repulsion - exchange;
}

double get_repulsion_exchange_energy(
    const array_helpers::HypercubeArray &repulsion_exchange,
    const array_helpers::Array2D &orbitals1,
    const array_helpers::Array2D &orbitals2) {
    double repulsion = repulsion_exchange.reduce(
        0, 1, orbitals1, orbitals1, 2, 3, orbitals2, orbitals2
    );
    double exchange = repulsion_exchange.reduce(
        0, 2, orbitals1, orbitals1, 1, 3, orbitals2, orbitals2
    );
    return 2.0*repulsion - exchange;
}

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

void si_example() {
    orbital_description_data::PositionedOrbitalsData si
    { atomic_data_descriptions::ORB_14P14E_1S4_2S4_2P4_3S4_3P4_3D4,
     .position={.t=0.0, .x=0.0, .y=0.0, .z=0.0}};
    NuclearChargesArray nuclear_charges = NuclearChargesArray({
        {{.t=0.0, 0.0, 0.0,  0.0}, 14}
        // {{.t=0.0, .x=0.0, .y=0.0, .z=0.0}, 8},
    });
    BasisFunctionArray arr = orbital_description_data::get_basis_function_array(
        {si});
    int n = arr.get_number_of_basis_functions();
    array_helpers::SquareArray overlap(n);
    array_helpers::SquareArray kinetic(n);
    array_helpers::SquareArray nuclear(n);
    array_helpers::HypercubeArray repulsion_exchange(n);
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            overlap(i, j) = arr.overlap(i, j);
            kinetic(i, j) = arr.kinetic(i, j);
            nuclear(i, j) = arr.nuclear(i, j, nuclear_charges);
            for (int k = 0; k < n; k++) {
                for (int l = k; l < n; l++) {
                    repulsion_exchange(i, j, k, l) 
                        = arr.repulsion_exchange(i, j, k, l);
                    if (l > k) {
                        repulsion_exchange(i, j, l, k)
                            = repulsion_exchange(i, j, k, l);
                    }
                }
            }
            if (j > i) {
                overlap(j, i) = overlap(i, j);
                kinetic(j, i) = kinetic(i, j);
                nuclear(j, i) = nuclear(i, j);
                repulsion_exchange(j, i, repulsion_exchange(i, j));
            }
        }
    }
    array_helpers::SquareArray h = kinetic + nuclear;
    array_helpers::Array1D energies(7);
    array_helpers::Array2D orbitals
        = orbital_description_data::get_orbital_basis_function_coefficients(
            7, {si});
    array_helpers::Array2D orbitals_final
        = array_helpers::Array2D(orbitals.row_size(), orbitals.row_size());
    array_helpers::Array1D energies_final(orbitals.row_size());
    for (int i = 0; i < orbitals.col_size(); i++) {
        for (int j = 0; j < orbitals.row_size(); j++) {
            printf("%g ", orbitals(i, j));
        }
        printf("\n");
    }
    // for (int i = 5; i < overlap.row_size(); i++)
    //     orbitals(4, i) = 1.0;
    for (int i = 0; i < 20; i++) {
        iteration(energies, orbitals, overlap, h,
                repulsion_exchange, orbitals);
        if (i == 19)
            iteration(energies_final, orbitals_final, overlap, h,
                repulsion_exchange, orbitals);
        for (int i = 0; i < energies_final.size(); i++)
            printf("%g ", energies_final(i));
        printf("\n");
    }
    double ke = get_kinetic_energy(kinetic, orbitals);
    double pe = get_nuclear_potential_energy(nuclear, orbitals);
    double re = get_repulsion_exchange_energy(repulsion_exchange, orbitals);
    double ne = nuclear_charges.get_energy();
    double mp2e = mp2(energies_final, orbitals_final, repulsion_exchange, 1);
    printf("kinetic and potential energies: %g, %g\n", ke, pe);
    printf("Total energy: %g\n", ke + pe + re + ne + mp2e
    );
    for (int i = 0; i < orbitals_final.col_size(); i++) {
        for (int j = 0; j < orbitals_final.row_size(); j++) {
            printf("%g ", orbitals(i, j));
        }
        printf("\n");
    }
}

void si_unrestricted_example() {
    orbital_description_data::PositionedOrbitalsData si
    { atomic_data_descriptions::ORB_14P14E_1S4_2S4_2P4_3S4_3P4_3D4,
     .position={.t=0.0, .x=0.0, .y=0.0, .z=0.0}};
    NuclearChargesArray nuclear_charges = NuclearChargesArray({
        {{.t=0.0, 0.0, 0.0,  0.0}, 14}
        // {{.t=0.0, .x=0.0, .y=0.0, .z=0.0}, 8},
    });
    BasisFunctionArray arr = orbital_description_data::get_basis_function_array(
        {si});
    int n = arr.get_number_of_basis_functions();
    array_helpers::SquareArray overlap(n);
    array_helpers::SquareArray kinetic(n);
    array_helpers::SquareArray nuclear(n);
    array_helpers::HypercubeArray repulsion_exchange(n);
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            overlap(i, j) = arr.overlap(i, j);
            kinetic(i, j) = arr.kinetic(i, j);
            nuclear(i, j) = arr.nuclear(i, j, nuclear_charges);
            for (int k = 0; k < n; k++) {
                for (int l = k; l < n; l++) {
                    repulsion_exchange(i, j, k, l) 
                        = arr.repulsion_exchange(i, j, k, l);
                    if (l > k) {
                        repulsion_exchange(i, j, l, k)
                            = repulsion_exchange(i, j, k, l);
                    }
                }
            }
            if (j > i) {
                overlap(j, i) = overlap(i, j);
                kinetic(j, i) = kinetic(i, j);
                nuclear(j, i) = nuclear(i, j);
                repulsion_exchange(j, i, repulsion_exchange(i, j));
            }
        }
    }
    array_helpers::SquareArray h = kinetic + nuclear;
    array_helpers::Array1D energies_up(8);
    array_helpers::Array2D orbitals_up
        = orbital_description_data::get_orbital_basis_function_coefficients(
            8, {si});
    array_helpers::Array1D energies_down(6);
    array_helpers::Array2D orbitals_down
        = orbital_description_data::get_orbital_basis_function_coefficients(
            6, {si});
    // for (int i = 5; i < overlap.row_size(); i++)
    //     orbitals(4, i) = 1.0;
    for (int i = 0; i < 20; i++) {
        iteration(
            energies_up, orbitals_up, 
            energies_down, orbitals_down,
            overlap, 
            h, repulsion_exchange,
            orbitals_up, orbitals_down);
        for (int i = 0; i < energies_up.size(); i++) {
            printf("%g\n", energies_up(i));
            if (i < energies_down.size())
                printf("%g\n", energies_down(i));
        }
        printf("\n");
    }
    array_helpers::Array2D orbitals = array_helpers::row_stack(
        orbitals_up, orbitals_down);
    printf("Orbitals up dimensions: %d, %d,\n", 
        orbitals_up.row_count(), orbitals_up.column_count());
    printf("Orbitals down dimensions: %d, %d,\n", 
        orbitals_down.row_count(), orbitals_down.column_count());
    printf("Orbitals dimensions: %d, %d,\n", 
        orbitals.row_count(), orbitals.column_count());
    // double ke_u = get_kinetic_energy(kinetic, orbitals_up);
    // double ke_d = get_kinetic_energy(kinetic, orbitals_down);
    // double pe_u =  get_nuclear_potential_energy(nuclear, orbitals_up);
    // double pe_d =  get_nuclear_potential_energy(nuclear, orbitals_down);
    double ke = get_kinetic_energy(kinetic, orbitals)/2.0;
    double pe = get_nuclear_potential_energy(nuclear, orbitals)/2.0;
    double re = get_repulsion_energy(repulsion_exchange, orbitals);
    double ex_up = get_exchange_energy(repulsion_exchange, orbitals_up);
    double ex_down = get_exchange_energy(repulsion_exchange, orbitals_down);
    printf("Kinetic Energy: %g\n", ke);
    printf("Nuclear Energy: %g\n", pe);
    printf("Total energy: %g\n", ke + pe + re - ex_up - ex_down);
    for (int i = 0; i < orbitals.col_size(); i++) {
        for (int j = 0; j < orbitals.row_size(); j++) {
            printf("%g ", orbitals(i, j));
        }
        printf("\n");
    }
}

void h2_example() {
    orbital_description_data::PositionedOrbitalsData h1
        {// atomic_data_descriptions::ORB_1P1E_1S4, 
         atomic_data_descriptions::ORB_1P1E_1S22_2S22_2P22,
            {.t=0.0, 1.37,  0.0,  0.0}}; 
    orbital_description_data::PositionedOrbitalsData h2
        {// atomic_data_descriptions::ORB_1P1E_1S4, 
         atomic_data_descriptions::ORB_1P1E_1S22_2S22_2P22,
            {.t=0.0, 0.0, 0.0,  0.0}}; 
    NuclearChargesArray nuclear_charges = NuclearChargesArray({
        {{.t=0.0, 1.37,  0.0,  0.0}, 1},
        {{.t=0.0, 0.0, 0.0,  0.0}, 1},
        // {{.t=0.0, .x=0.0, .y=0.0, .z=0.0}, 8},
    });
    BasisFunctionArray arr = orbital_description_data::get_basis_function_array(
        {h1, h2});
    arr.print();
    int n = arr.get_number_of_basis_functions();
    array_helpers::SquareArray overlap(n);
    array_helpers::SquareArray kinetic(n);
    array_helpers::SquareArray nuclear(n);
    array_helpers::HypercubeArray repulsion_exchange(n);
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            overlap(i, j) = arr.overlap(i, j);
            kinetic(i, j) = arr.kinetic(i, j);
            nuclear(i, j) = arr.nuclear(i, j, nuclear_charges);
            for (int k = 0; k < n; k++) {
                for (int l = k; l < n; l++) {
                    repulsion_exchange(i, j, k, l) 
                        = arr.repulsion_exchange(i, j, k, l);
                    if (l > k) {
                        repulsion_exchange(i, j, l, k)
                            = repulsion_exchange(i, j, k, l);
                    }
                }
            }
            if (j > i) {
                overlap(j, i) = overlap(i, j);
                kinetic(j, i) = kinetic(i, j);
                nuclear(j, i) = nuclear(i, j);
                repulsion_exchange(j, i, repulsion_exchange(i, j));
            }
        }
    }
    array_helpers::SquareArray h = kinetic + nuclear;
    array_helpers::Array1D energies(1);
    array_helpers::Array2D orbitals
        = orbital_description_data::get_orbital_basis_function_coefficients(
            1, {h1, h2});
    array_helpers::Array2D orbitals_final
        = array_helpers::Array2D(orbitals.row_size(), orbitals.row_size());
    array_helpers::Array1D energies_final(orbitals.row_size());
    for (int i = 0; i < orbitals.col_size(); i++) {
        for (int j = 0; j < orbitals.row_size(); j++) {
            printf("%g ", orbitals(i, j));
        }
        printf("\n");
    }
    // for (int i = 5; i < overlap.row_size(); i++)
    //     orbitals(4, i) = 1.0;
    for (int i = 0; i < 20; i++) {
        iteration(energies, orbitals, overlap, h,
                repulsion_exchange, orbitals);
        if (i == 19)
            iteration(energies_final, orbitals_final, overlap, h,
                repulsion_exchange, orbitals);
        for (int i = 0; i < energies_final.size(); i++)
            printf("%g ", energies_final(i));
        printf("\n");
    }
    double ke = get_kinetic_energy(kinetic, orbitals);
    double pe = get_nuclear_potential_energy(nuclear, orbitals);
    double re = get_repulsion_exchange_energy(repulsion_exchange, orbitals);
    double ne = nuclear_charges.get_energy();
    double mp2e = mp2(energies_final, orbitals_final, repulsion_exchange, 1);
    printf("Total energy: %g\n", ke + pe + re + ne + mp2e);
    for (int i = 0; i < orbitals_final.col_size(); i++) {
        for (int j = 0; j < orbitals_final.row_size(); j++) {
            printf("%g ", orbitals(i, j));
        }
        printf("\n");
    }

}

void h2o_example() {
    orbital_description_data::PositionedOrbitalsData h1
        { atomic_data_descriptions::ORB_1P1E_1S21_2S21_2P21,
         // atomic_data_descriptions::ORB_1P1E_1S22_2S22_2P22,
        // atomic_data_descriptions::ORB_1P1E_1S4_2S4_2P4,
            {.t=0.0, -1.93044664,  0.82666546,  0.0}}; 
    orbital_description_data::PositionedOrbitalsData h2
        { atomic_data_descriptions::ORB_1P1E_1S21_2S21_2P21,
         // atomic_data_descriptions::ORB_1P1E_1S22_2S22_2P22,
        // atomic_data_descriptions::ORB_1P1E_1S4_2S4_2P4,
            {.t=0.0, 0.82666546, -1.93044664,  0.0}}; 
    orbital_description_data::PositionedOrbitalsData o 
        {// atomic_data_descriptions::ORB_10P10E_1S5_2S311_2P311,
         // atomic_data_descriptions::ORB_8P8E_1S5_2S32_2P32,
         atomic_data_descriptions::ORB_8P8E_1S6_2S6_2P3111,
            {.ind{0.0, 0.0, 0.0, 0.0}}};
    NuclearChargesArray nuclear_charges = NuclearChargesArray({
        {{.t=0.0, -1.93044664,  0.82666546,  0.0}, 1},
        {{.t=0.0, 0.82666546, -1.93044664,  0.0}, 1},
        {{.t=0.0, .x=0.0, .y=0.0, .z=0.0}, 8},
    });
    BasisFunctionArray arr = orbital_description_data::get_basis_function_array(
        {h1, h2, o});
    arr.print();
    int n = arr.get_number_of_basis_functions();
    array_helpers::SquareArray overlap(n);
    array_helpers::SquareArray kinetic(n);
    array_helpers::SquareArray nuclear(n);
    array_helpers::HypercubeArray repulsion_exchange(n);
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            overlap(i, j) = arr.overlap(i, j);
            kinetic(i, j) = arr.kinetic(i, j);
            nuclear(i, j) = arr.nuclear(i, j, nuclear_charges);
            for (int k = 0; k < n; k++) {
                for (int l = k; l < n; l++) {
                    repulsion_exchange(i, j, k, l) 
                        = arr.repulsion_exchange(i, j, k, l);
                    if (l > k) {
                        repulsion_exchange(i, j, l, k)
                            = repulsion_exchange(i, j, k, l);
                    }
                }
            }
            if (j > i) {
                overlap(j, i) = overlap(i, j);
                kinetic(j, i) = kinetic(i, j);
                nuclear(j, i) = nuclear(i, j);
                repulsion_exchange(j, i, repulsion_exchange(i, j));
            }
        }
    }
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("Overlap %d, %d: %g\n", i, j, overlap(i, j));
    //         printf("Kinetic %d, %d: %g\n", i, j, kinetic(i, j));
    //         printf("Nuclear %d, %d: %g\n", i, j, nuclear(i, j));
    //     }
    // }
    array_helpers::SquareArray h = kinetic + nuclear;
    array_helpers::Array1D energies(5);
    array_helpers::Array2D orbitals
        = orbital_description_data::get_orbital_basis_function_coefficients(
            5, {h1, h2, o});
    array_helpers::Array2D orbitals_final = 
        array_helpers::Array2D(orbitals.row_size(), orbitals.row_size());
    array_helpers::Array1D energies_final(orbitals.row_size());
    for (int i = 0; i < orbitals.col_size(); i++) {
        for (int j = 0; j < orbitals.row_size(); j++) {
            printf("%g ", orbitals(i, j));
        }
        printf("\n");
    }
    // for (int i = 5; i < overlap.row_size(); i++)
    //     orbitals(4, i) = 1.0;
    for (int i = 0; i < 20; i++) {
        iteration(energies, orbitals, overlap, h, 
            repulsion_exchange, orbitals);
        if (i == 19)
            iteration(energies_final, orbitals_final, overlap, h,
                repulsion_exchange, orbitals);
        for (int i = 0; i < energies.size(); i++)
            printf("%g ", energies(i));
        printf("\n");
    }
    double ke = get_kinetic_energy(kinetic, orbitals);
    double pe = get_nuclear_potential_energy(nuclear, orbitals);
    double re = get_repulsion_exchange_energy(repulsion_exchange, orbitals);
    double ne = nuclear_charges.get_energy();
    double mp2e = mp2(energies_final, orbitals_final, repulsion_exchange, 5);
    printf("Total energy: %g\n", ke + pe + re + ne + mp2e
    );
    for (int i = 0; i < orbitals.col_size(); i++) {
        for (int j = 0; j < orbitals.row_size(); j++) {
            printf("%g ", orbitals(i, j));
        }
        printf("\n");
    }
}

void co2_example() {
    orbital_description_data::PositionedOrbitalsData o1
        {atomic_data_descriptions::ORB_8P8E_1S5_2S32_2P32,
         // atomic_data_descriptions::ORB_8P8E_1S6_2S6_2P3111,
            {.ind{0.0, -2.2, 0.0, 0.0}}};
    orbital_description_data::PositionedOrbitalsData o2
        {atomic_data_descriptions::ORB_8P8E_1S5_2S32_2P32,
         // atomic_data_descriptions::ORB_8P8E_1S6_2S6_2P3111,
            {.ind{0.0, 2.2, 0.0, 0.0}}};
    orbital_description_data::PositionedOrbitalsData c
        {atomic_data_descriptions::ORB_7P7E_1S6_2S42_2P42,
            {.ind{0.0, 0.0, 0.0, 0.0}}};
    NuclearChargesArray nuclear_charges = NuclearChargesArray({
        {{.t=0.0, -2.2, 0.0,  0.0}, 8},
        {{.t=0.0, 2.2, 0.0,  0.0}, 8},
        {{.t=0.0, .x=0.0, .y=0.0, .z=0.0}, 6},
    });
    BasisFunctionArray arr = orbital_description_data::get_basis_function_array(
        {o1, c, o2});
    arr.print();
    int n = arr.get_number_of_basis_functions();
    array_helpers::SquareArray overlap(n);
    array_helpers::SquareArray kinetic(n);
    array_helpers::SquareArray nuclear(n);
    array_helpers::HypercubeArray repulsion_exchange(n);
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            overlap(i, j) = arr.overlap(i, j);
            kinetic(i, j) = arr.kinetic(i, j);
            nuclear(i, j) = arr.nuclear(i, j, nuclear_charges);
            for (int k = 0; k < n; k++) {
                for (int l = k; l < n; l++) {
                    repulsion_exchange(i, j, k, l) 
                        = arr.repulsion_exchange(i, j, k, l);
                    if (l > k) {
                        repulsion_exchange(i, j, l, k)
                            = repulsion_exchange(i, j, k, l);
                    }
                }
            }
            if (j > i) {
                overlap(j, i) = overlap(i, j);
                kinetic(j, i) = kinetic(i, j);
                nuclear(j, i) = nuclear(i, j);
                repulsion_exchange(j, i, repulsion_exchange(i, j));
            }
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("Overlap %d, %d: %g\n", i, j, overlap(i, j));
            printf("Kinetic %d, %d: %g\n", i, j, kinetic(i, j));
            printf("Nuclear %d, %d: %g\n", i, j, nuclear(i, j));
        }
    }
    array_helpers::SquareArray h = kinetic + nuclear;
    array_helpers::Array1D energies(11);
    array_helpers::Array2D orbitals
        = orbital_description_data::get_orbital_basis_function_coefficients(
        11, {o1, c, o2});
    // array_helpers::Array2D orbitals(11, 61);
    // orbitals(0, 0) = 1.0;
    // for (int i = 11; i < overlap.row_size(); i++)
    //     orbitals(10, i) = 0.0;

    for (int i = 0; i < orbitals.col_size(); i++) {
        for (int j = 0; j < orbitals.row_size(); j++) {
            printf("%g ", orbitals(i, j));
        }
        printf("\n");
    }
    for (int i = 0; i < 20; i++) {
        iteration(energies, orbitals, overlap, h, 
            repulsion_exchange, orbitals);
        for (int i = 0; i < energies.size(); i++)
            printf("%g ", energies(i));
        printf("\n");
    }
    double ke = get_kinetic_energy(kinetic, orbitals);
    double pe = get_nuclear_potential_energy(nuclear, orbitals);
    double re = get_repulsion_exchange_energy(repulsion_exchange, orbitals);
    double ne = nuclear_charges.get_energy();
    printf("Total energy: %g\n", ke + pe + re + ne);
    for (int i = 0; i < orbitals.col_size(); i++) {
        for (int j = 0; j < orbitals.row_size(); j++) {
            printf("%g ", orbitals(i, j));

        }
        printf("\n");
    }
}

void o2_example() {
    orbital_description_data::PositionedOrbitalsData o1
        {atomic_data_descriptions::ORB_8P8E_1S6_2S3111_2P3111,
            {.ind{0.0, 0.0, 0.0, 0.0}}};
    orbital_description_data::PositionedOrbitalsData o2
        {atomic_data_descriptions::ORB_8P8E_1S6_2S3111_2P3111,
            {.ind{0.0, 2.31, 0.0, 0.0}}};
    NuclearChargesArray nuclear_charges = NuclearChargesArray({
        {{.t=0.0, 0.0, 0.0,  0.0}, 8},
        {{.t=0.0, 2.31, 0.0,  0.0}, 8},
    });
    BasisFunctionArray arr = orbital_description_data::get_basis_function_array(
        {o1, o2});
    arr.print();
    int n = arr.get_number_of_basis_functions();
    array_helpers::SquareArray overlap(n);
    array_helpers::SquareArray kinetic(n);
    array_helpers::SquareArray nuclear(n);
    array_helpers::HypercubeArray repulsion_exchange(n);
    // # pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            overlap(i, j) = arr.overlap(i, j);
            kinetic(i, j) = arr.kinetic(i, j);
            nuclear(i, j) = arr.nuclear(i, j, nuclear_charges);
            for (int k = 0; k < n; k++) {
                for (int l = k; l < n; l++) {
                    repulsion_exchange(i, j, k, l) 
                        = arr.repulsion_exchange(i, j, k, l);
                    if (l > k) {
                        repulsion_exchange(i, j, l, k)
                            = repulsion_exchange(i, j, k, l);
                    }
                }
            }
            if (j > i) {
                overlap(j, i) = overlap(i, j);
                kinetic(j, i) = kinetic(i, j);
                nuclear(j, i) = nuclear(i, j);
                repulsion_exchange(j, i, repulsion_exchange(i, j));
            }
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("Overlap %d, %d: %g\n", i, j, overlap(i, j));
            printf("Kinetic %d, %d: %g\n", i, j, kinetic(i, j));
            printf("Nuclear %d, %d: %g\n", i, j, nuclear(i, j));
        }
    }
    array_helpers::SquareArray h = kinetic + nuclear;
    array_helpers::Array1D energies(8);
    array_helpers::Array2D orbitals
        = orbital_description_data::get_orbital_basis_function_coefficients(
        8, {o1, o2});
    // array_helpers::Array2D orbitals(11, 61);
    // orbitals(0, 0) = 1.0;
    // for (int i = 11; i < overlap.row_size(); i++)
    //     orbitals(10, i) = 0.0;

    for (int i = 0; i < orbitals.col_size(); i++) {
        for (int j = 0; j < orbitals.row_size(); j++) {
            printf("%g ", orbitals(i, j));
        }
        printf("\n");
    }
    for (int i = 0; i < 10; i++) {
        iteration(energies, orbitals, overlap, h, 
            repulsion_exchange, orbitals);
        for (int i = 0; i < energies.size(); i++)
            printf("%g ", energies(i));
        printf("\n");
    }
    double ke = get_kinetic_energy(kinetic, orbitals);
    double pe = get_nuclear_potential_energy(nuclear, orbitals);
    double re = get_repulsion_exchange_energy(repulsion_exchange, orbitals);
    double ne = nuclear_charges.get_energy();
    printf("Total energy: %g\n", ke + pe + re + ne);
    for (int i = 0; i < orbitals.col_size(); i++) {
        for (int j = 0; j < orbitals.row_size(); j++) {
            printf("%g ", orbitals(i, j));

        }
        printf("\n");
    }
}



int main() {
    // array_helpers::test1();
    // array_helpers::test2();
    // array_helpers::test3();
    // array_helpers::test4();
    // array_helpers::test5();
    // array_helpers::test6();
    // array_helpers::test7();
    // array_helpers::test8();
    // array_helpers::test9();
    // h2_example();
    // h2o_example();
    // co2_example();
    // o2_example();
    si_unrestricted_example();
    return 0;
}