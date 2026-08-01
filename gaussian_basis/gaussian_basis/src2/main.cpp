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
    repulsion = 2.0*repulsion_exchange_tensor.reduce(
        2, 3, orbitals, orbitals
    );
    array_helpers::SquareArray
    exchange = repulsion_exchange_tensor.reduce(
        1, 3, orbitals, orbitals
    );
    array_helpers::SquareArray
    fock = kinetic_nuclear + repulsion - exchange;

    compute_eigenvalues_eigenvectors(
        energies, next_orbitals, overlap, fock);

}

double get_kinetic_energy(
    const array_helpers::SquareArray &kinetic,
    const array_helpers::Array2D &orbitals) {
    return 2.0*kinetic.reduce(orbitals);
}

double get_nuclear_potential_energy(
    const array_helpers::SquareArray &nuclear,
    const array_helpers::Array2D &orbitals) {
    return 2.0*nuclear.reduce(orbitals);
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

void h2_example() {
    orbital_description_data::PositionedOrbitalsData h1
        {atomic_data_descriptions::ORB_1P1E_1S21_2S21_2P21, 
            {.t=0.0, 1.37,  0.0,  0.0}}; 
    orbital_description_data::PositionedOrbitalsData h2
        {atomic_data_descriptions::ORB_1P1E_1S21_2S21_2P21, 
            {.t=0.0, 0.0, 0.0,  0.0}}; 
    NuclearChargesArray nuclear_charges = NuclearChargesArray({
        {{.t=0.0, 1.37,  0.0,  0.0}, 1},
        {{.t=0.0, 0.0, 0.0,  0.0}, 1},
        // {{.t=0.0, .x=0.0, .y=0.0, .z=0.0}, 8},
    });
    BasisFunctionArray arr = orbital_description_data::get_basis_function_array(
        {h1, h2});
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
    for (int i = 0; i < orbitals.col_size(); i++) {
        for (int j = 0; j < orbitals.row_size(); j++) {
            printf("%g ", orbitals(i, j));
        }
        printf("\n");
    }
    // for (int i = 5; i < overlap.row_size(); i++)
    //     orbitals(4, i) = 1.0;
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

void h2o_example() {
    orbital_description_data::PositionedOrbitalsData h1
        {atomic_data_descriptions::ORB_1P1E_1S21_2S21_2P21, 
            {.t=0.0, -1.93044664,  0.82666546,  0.0}}; 
    orbital_description_data::PositionedOrbitalsData h2
        {atomic_data_descriptions::ORB_1P1E_1S21_2S21_2P21, 
            {.t=0.0, 0.82666546, -1.93044664,  0.0}}; 
    orbital_description_data::PositionedOrbitalsData o 
        {atomic_data_descriptions::ORB_10P10E_1S5_2S311_2P311,
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

void co2_example() {
    orbital_description_data::PositionedOrbitalsData o1
        {atomic_data_descriptions::ORB_8P8E_1S5_2S32_2P32,
            {.ind{0.0, -2.2, 0.0, 0.0}}};
    orbital_description_data::PositionedOrbitalsData o2
        {atomic_data_descriptions::ORB_8P8E_1S5_2S32_2P32,
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
    h2_example();
    // h2o_example();
    // co2_example();
    // o2_example();
    return 0;
}