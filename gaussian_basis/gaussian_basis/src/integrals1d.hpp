#include "gaussian1d.hpp"
#include "vec3.hpp"


#ifndef _INTEGRALS1D_
#define _INTEGRALS1D_

double boys_func(double x, int n);

double overlap_coefficient(int n, 
                           Gaussian1D g1, Gaussian1D g2);

double overlap1d(Gaussian1D g1, Gaussian1D g2);

double laplacian1d(Gaussian1D g1, Gaussian1D g2);

double coulomb_coefficient(int i, int j, int k, int n,
                           double orb_exp, const Vec3 &r12);

#endif
