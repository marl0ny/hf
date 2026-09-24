#include "compute_energies.hpp"


double compute_energies::kinetic(
    const array_helpers::SquareArray &kinetic,
    const array_helpers::Array2D &orbitals) {
    return 2.0*kinetic.reduce(orbitals);
}

double compute_energies::kinetic(
    const array_helpers::SquareArray &kinetic,
    const array_helpers::Array2D &orbitals1,
    const array_helpers::Array2D &orbitals2) {
    return 2.0*kinetic.reduce(orbitals1, orbitals2);
}

double compute_energies::nuclear_potential(
    const array_helpers::SquareArray &nuclear,
    const array_helpers::Array2D &orbitals) {
    return 2.0*nuclear.reduce(orbitals);
}

double compute_energies::nuclear_potential(
    const array_helpers::SquareArray &nuclear,
    const array_helpers::Array2D &orbitals1,
    const array_helpers::Array2D &orbitals2) {
    return 2.0*nuclear.reduce(orbitals1, orbitals2);
}

double compute_energies::repulsion(
    const array_helpers::HypercubeArray &repulsion_exchange,
    const array_helpers::Array2D &orbitals
) {
    return repulsion_exchange.reduce(
        0, 1, orbitals, orbitals, 2, 3, orbitals, orbitals
    );
}

double compute_energies::exchange(
    const array_helpers::HypercubeArray &repulsion_exchange,
    const array_helpers::Array2D &orbitals
) {
    return repulsion_exchange.reduce(
        0, 2, orbitals, orbitals, 1, 3, orbitals, orbitals
    );
}

double compute_energies::repulsion(
    const array_helpers::Symmetric4 &repulsion_exchange,
    const array_helpers::Array2D &orbitals
) {
    return repulsion_exchange.reduce(
        0, 1, orbitals, orbitals, 2, 3, orbitals, orbitals
    );
}

double compute_energies::exchange(
    const array_helpers::Symmetric4 &repulsion_exchange,
    const array_helpers::Array2D &orbitals
) {
    return repulsion_exchange.reduce(
        0, 2, orbitals, orbitals, 1, 3, orbitals, orbitals
    );
}


double compute_energies::repulsion_exchange(
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

double compute_energies::repulsion_exchange(
    const array_helpers::Symmetric4 &repulsion_exchange,
    const array_helpers::Array2D &orbitals) {
    double repulsion = repulsion_exchange.reduce(
        0, 1, orbitals, orbitals, 2, 3, orbitals, orbitals
    );
    double exchange = repulsion_exchange.reduce(
        0, 2, orbitals, orbitals, 1, 3, orbitals, orbitals
    );
    return 2.0*repulsion - exchange;
}

double compute_energies::repulsion_exchange(
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