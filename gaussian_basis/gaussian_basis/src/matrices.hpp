#include "gaussian3d.hpp"


#ifndef _MATRICES_
#define _MATRICES_

void set_overlap_elements(double *mat, const Gaussian3D *g, int n);

void set_kinetic_elements(double *mat, const Gaussian3D *g, int n);

void set_nuclear_potential_elements(
    double *mat, const Gaussian3D *g, int n,
    Vec3 *nuc_loc, int *charges, int charge_count
);

void set_two_electron_integrals_elements(
    double *arr, const Gaussian3D *g, int n
);

#endif