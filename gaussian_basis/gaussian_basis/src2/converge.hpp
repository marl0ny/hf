#include "basis_function_array.hpp"
#include "orbitals_description.hpp"
#include "molecular_geometry.hpp"
#include "array_helpers.hpp"
#include "eigenvalues_eigenvectors.hpp"


#ifndef _CONVERGE_
#define _CONVERGE_

namespace converge {

    void iteration(
        array_helpers::Array1D &energies,
        array_helpers::Array2D &next_orbitals,
        const array_helpers::SquareArray &overlap,
        const array_helpers::SquareArray &kinetic_nuclear,
        const array_helpers::HypercubeArray &repulsion_exchange_tensor,
        const array_helpers::Array2D &orbitals);
    
    void iteration(
        array_helpers::Array1D &energies,
        array_helpers::Array2D &next_orbitals,
        const array_helpers::SquareArray &overlap,
        const array_helpers::SquareArray &kinetic_nuclear,
        const array_helpers::Symmetric4 &repulsion_exchange_tensor,
        const array_helpers::Array2D &orbitals);

    void closed(
        array_helpers::Array1D &energies,
        array_helpers::Array2D &orbitals,
        int occupied_count,
        const array_helpers::SquareArray &overlap,
        const array_helpers::SquareArray &kinetic_nuclear,
        const array_helpers::HypercubeArray &repulsion_exchange,
        int n_iterations, bool verbose=false);
    
    void closed(
        array_helpers::Array1D &energies,
        array_helpers::Array2D &orbitals,
        int occupied_count,
        const array_helpers::SquareArray &overlap,
        const array_helpers::SquareArray &kinetic_nuclear,
        const array_helpers::Symmetric4 &repulsion_exchange,
        int n_iterations, bool verbose=false);
    
    void open(
        array_helpers::Array1D &energies_up,
        array_helpers::Array2D &orbitals_up, int occupied_up_count,
        array_helpers::Array1D &energies_down, 
        array_helpers::Array2D &orbitals_down, int occupied_down_count,
        const array_helpers::SquareArray &overlap,
        const array_helpers::SquareArray &kinetic_nuclear,
        const array_helpers::HypercubeArray &repulsion_exchange,
        int n_iterations, bool verbose=false);

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
    );

}

#endif