#include "converge.hpp"


void converge::closed_iteration(
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

void converge::open_iteration(
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

void single_electron_iteration(
    array_helpers::Array1D &energies_next,
    array_helpers::Array2D &orbitals_next,
    const array_helpers::SquareArray &overlap,
    const array_helpers::SquareArray &kinetic_nuclear
) {
    compute_eigenvalues_eigenvectors(
        energies_next, orbitals_next, 
        overlap, kinetic_nuclear);
}

struct OperatorArrays {
    array_helpers::SquareArray &overlap;
    array_helpers::SquareArray &kinetic;
    array_helpers::SquareArray &nuclear;
    array_helpers::HypercubeArray &repulsion_exchange;
};

struct ClosedSystem {
    int electron_count;
    OperatorArrays operators;
    BasisFunctionArray basis_functions;
    NuclearChargesArray nuclear_charges;
    array_helpers::Array2D orbitals;
};

struct OpenSystem {
    int up_electron_count;
    int down_electron_count;
    OperatorArrays operators;
    BasisFunctionArray basis_functions;
    NuclearChargesArray nuclear_charges;
    array_helpers::Array2D up_orbitals;
    array_helpers::Array2D down_orbitals;
};

void fill_arrays(
    array_helpers::SquareArray &overlap,
    array_helpers::SquareArray &kinetic,
    array_helpers::SquareArray &nuclear,
    array_helpers::HypercubeArray &repulsion_exchange,
    const BasisFunctionArray &basis_functions,
    const NuclearChargesArray &nuclear_charges
) {
    int n = basis_functions.get_number_of_basis_functions();
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            overlap(i, j) = basis_functions.overlap(i, j);
            kinetic(i, j) = basis_functions.kinetic(i, j);
            nuclear(i, j) = basis_functions.nuclear(
                i, j, nuclear_charges);
            for (int k = 0; k < n; k++) {
                for (int l = k; l < n; l++) {
                    repulsion_exchange(i, j, k, l) 
                        = basis_functions.repulsion_exchange(i, j, k, l);
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
}

void converge::closed(
    array_helpers::Array1D &energies,
    array_helpers::Array2D &orbitals,
    // std::optional<array_helpers::Array2D> &energies_history,
    int occupied_count,
    const array_helpers::SquareArray &overlap,
    const array_helpers::SquareArray &kinetic_nuclear,
    const array_helpers::HypercubeArray &repulsion_exchange,
    int n_iterations, bool verbose) {
    array_helpers::Array1D energies_iter(occupied_count);
    array_helpers::Array2D orbitals_iter(
        occupied_count, orbitals.row_size());
    // for (int i = 0; i < occupied_count; i++) {
    //     energies_iter(i) = energies(i);
    //     for (int j = 0; j < orbitals.row_size(); j++)
    //         orbitals_iter(i, j) = orbitals(i, j);
    // }
    single_electron_iteration(
        energies_iter, orbitals_iter,
        overlap, kinetic_nuclear);
    for (int i = 0; i < n_iterations; i++) {
        if (i == n_iterations - 1) {
            closed_iteration(energies, orbitals, overlap,
                kinetic_nuclear,
                repulsion_exchange,
                orbitals_iter);
        } else {
            closed_iteration(energies_iter, orbitals_iter,
                overlap, kinetic_nuclear, 
                repulsion_exchange, orbitals_iter);
        }
        if (verbose) {
            printf("Iteration: %d\n", i);
            for (int k = 0; k < energies_iter.size(); k++)
                printf("%g\n", energies_iter(k));
            printf(
                "############################################################\n");
        }
    }
}