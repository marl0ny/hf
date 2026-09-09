#include "gaussian1d.hpp"
#include "spatial.hpp"


#ifndef _INTEGRALS1D_
#define _INTEGRALS1D_

using fp_type = float;

fp_type boys_func(fp_type x, int n);

fp_type overlap_coefficient(int n, 
                           Gaussian1D g1, Gaussian1D g2);

fp_type overlap1d(Gaussian1D g1, Gaussian1D g2);

fp_type laplacian1d(Gaussian1D g1, Gaussian1D g2);

fp_type coulomb_coefficient(int i, int j, int k, int n,
                           fp_type orb_exp, const spatial::Vector &r12);

#endif
