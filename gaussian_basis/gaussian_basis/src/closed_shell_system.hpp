#include "nuclear.hpp"
#include "matrices.hpp"
#include "orbitals.hpp"

class ClosedShellSystem {
    int orbitals_count;
    int basis_func_count;
    std::vector<double> orbitals;
    std::vector<double> kinetic;
    std::vector<double> nuclear;
    std::vector<double> two_electron_integrals;
    std::vector<double> hamiltonian;
    std::vector<double> energies;
    NuclearConfiguration nuclear_configuration;
    public:
    ClosedShellSystem(
        int number_of_electrons,
        const Orbitals &orbitals,
        const NuclearConfiguration &nuclear_config);
    void solve(int iter_count);
    double get_kinetic_energy() const;
    double get_nuclear_potential_energy() const;
    double get_repulsion_exchange_energy() const;
    double get_nuclear_configuration_energy() const;
    double get_total_energy() const;
};
