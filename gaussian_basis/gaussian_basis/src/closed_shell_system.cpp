#include "closed_shell_system.hpp"

#include "Eigen/Core"
#include "Eigen/Sparse"
#include "Eigen/Eigenvalues"

/*
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
*/

static inline double at_ind(const double *m, int n, int i, int j) {
    return m[n*i + j];
}


static inline double at_ind(const double *m, int n, int i, int j, int k) {
    return m[n*n*i + n*j + k];
}


static inline double at_ind(
    const double *m, int n, int i, int j, int k, int l_) {
    return m[n*n*n*i + n*n*j + n*k + l_];
}

static inline void reduce_ijkl_mk_ml_to_ij(
    double *dst,
    const double *tensor_ijkl, 
    const double *tensor_mk, const double *tensor_ml,
    int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int m = 0; m < size; m++) {
                for (int k = 0; k < size; k++) {
                    for (int l_ = 0; l_ < size; l_++) {
                        dst[i*size + j] 
                            += at_ind(tensor_ijkl, size, i, j, k, l_)
                            *at_ind(tensor_mk, size, m, k)
                            *at_ind(tensor_ml, size, m, l_);
                    }
                }
            }
        }
    }
}

static inline void reduce_ijkl_mj_ml_to_ij(
    double *dst,
    const double *tensor_ijkl, 
    const double *tensor_mk, const double *tensor_ml,
    int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int m = 0; m < size; m++) {
                for (int k = 0; k < size; k++) {
                    for (int l_ = 0; l_ < size; l_++) {
                        dst[i*size + j] 
                            += at_ind(tensor_ijkl, size, i, j, k, l_)
                            *at_ind(tensor_mk, size, m, j)
                            *at_ind(tensor_ml, size, m, l_);
                    }
                }
            }
        }
    }
}

static inline void add(
    double *dst, 
    const double *a, const double *b, const double *c, int n) {
    for (int i = 0; i < n; i++)
        dst[i] = a[i] + b[i] + c[i];
}

ClosedShellSystem::
ClosedShellSystem(
    int number_of_electrons, 
    const Orbitals &orbitals,
    const NuclearConfiguration &nuclear_config) {
    this->nuclear_configuration = nuclear_config;
    this->kinetic = orbitals.get_kinetic();
    this->nuclear = orbitals.get_nuclear_potential(nuclear_configuration);
    for (int i = 0; i < this->nuclear.size(); i++)
        this->hamiltonian[i] = this->kinetic[i] + this->nuclear[i];
    this->two_electron_integrals = orbitals.get_two_electron_integrals();
    this->energies = std::vector<double>(
        orbitals.number_of_basis_functions(), 0.0);
    // for (const AtomicOrbitalsParameters &atomic_orbitals_params: orbitals) {

    // }
}

typedef Eigen::MatrixXd Matrix;

void ClosedShellSystem::solve(int iter_count) {
    const double *orbitals = &this->orbitals[0];
    const double *energies = &this->energies[0];
    const double *two_electron_integrals 
        = &this->two_electron_integrals[0];
    const double *hamiltonian = &this->hamiltonian[0];
    int n = this->basis_func_count;
    for (int i = 0; i < iter_count; i++) {
        // std::vector<double> fock(n*n, 0.0);
        Matrix fock = Matrix::Zero(1, n*n);
        std::vector<double> repulsion(n*n, 0.0);
        std::vector<double> exchange(n*n, 0.0);
        reduce_ijkl_mk_ml_to_ij(
            &repulsion[0], 
            two_electron_integrals, 
            orbitals, orbitals, n);
        reduce_ijkl_mj_ml_to_ij(
            &exchange[0], 
            two_electron_integrals, 
            orbitals, orbitals, n);
        add(&fock(0, 0), hamiltonian, &repulsion[0], &exchange[0], n*n);
        fock.resize(n, n);
        Eigen::EigenSolver<Matrix> solver(fock);
        Matrix eigenvectors = solver.eigenvectors().transpose();
        eigenvectors.resize(1, n*n);
        Matrix eigenvalues = solver.eigenvalues();
        for (int i = 0; i < n; i++) {
            this->energies[i] = eigenvalues[i];
            for (int j = 0; j < n; j++)
                this->orbitals[i*n + j] = eigenvectors(i, j);
        }

    }
}

double ClosedShellSystem::get_kinetic_energy() const {

}

double ClosedShellSystem::get_nuclear_potential_energy() const {

}

double ClosedShellSystem::get_repulsion_exchange_energy() const {

}

double ClosedShellSystem::get_nuclear_configuration_energy() const {
    
}

double ClosedShellSystem::get_total_energy() const {
    return this->get_kinetic_energy() + 
        this->get_nuclear_potential_energy() +
        this->get_repulsion_exchange_energy() +
        this->get_nuclear_configuration_energy();
}
