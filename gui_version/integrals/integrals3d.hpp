#include "gaussian3d.hpp"

#include <set>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <iostream>

#ifndef _INTEGRALS3D_
#define _INTEGRALS3D_

double overlap(const Gaussian3D &g1, const Gaussian3D &g2);

double kinetic(const Gaussian3D &g1, const Gaussian3D &g2);

double nuclear_single_charge(const Gaussian3D &g, const Gaussian3D &h,
                             struct spatial::Vector r);

double coulomb_coefficient(int i, int j, int k, int n,
                           double orb_exp, const spatial::Vector &r12);

double repulsion(const Gaussian3D &g1,
                 const Gaussian3D &h1,
                 const Gaussian3D &g2,
                 const Gaussian3D &h2);

#endif
