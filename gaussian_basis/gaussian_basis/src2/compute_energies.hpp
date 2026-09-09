#include "array_helpers.hpp"

#ifndef _COMPUTE_ENERGIES_
#define _COMPUTE_ENERGIES_

namespace compute_energies {

    double kinetic(
        const array_helpers::SquareArray &kinetic,
        const array_helpers::Array2D &orbitals);
    
    double kinetic(
        const array_helpers::SquareArray &kinetic,
        const array_helpers::Array2D &orbitals1,
        const array_helpers::Array2D &orbitals2);
    
    double nuclear_potential(
        const array_helpers::SquareArray &nuclear,
        const array_helpers::Array2D &orbitals);
    
    double nuclear_potential(
        const array_helpers::SquareArray &nuclear,
        const array_helpers::Array2D &orbitals1,
        const array_helpers::Array2D &orbitals2);
    
    double repulsion(
        const array_helpers::HypercubeArray &repulsion_exchange,
        const array_helpers::Array2D &orbitals);
    
    double exchange(
        const array_helpers::HypercubeArray &repulsion_exchange,
        const array_helpers::Array2D &orbitals); 
    
    double repulsion_exchange(
        const array_helpers::HypercubeArray &repulsion_exchange,
        const array_helpers::Array2D &orbitals);
    
    double repulsion_exchange(
        const array_helpers::Symmetric4 &repulsion_exchange,
        const array_helpers::Array2D &orbitals);
    
    double repulsion_exchange(
        const array_helpers::HypercubeArray &repulsion_exchange,
        const array_helpers::Array2D &orbitals1,
        const array_helpers::Array2D &orbitals2);

}

#endif