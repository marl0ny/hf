#include "gaussian1d.hpp"
#include "spatial.hpp"

#include <set>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <iostream>

#ifndef _INTEGRALS1D_
#define _INTEGRALS1D_

using fp_type = float;

fp_type boys_func(fp_type x, int n);

fp_type overlap_coefficient(int n, 
                           Gaussian1D g1, Gaussian1D g2);

fp_type overlap1d(Gaussian1D g1, Gaussian1D g2);

fp_type laplacian1d(Gaussian1D g1, Gaussian1D g2);

fp_type overlap_coefficient_1(
    int n, Gaussian1D g1, Gaussian1D g2);

fp_type overlap_coefficient_2(
    int n, Gaussian1D g1, Gaussian1D g2);

fp_type overlap1d_1(Gaussian1D g1, Gaussian1D g2);

fp_type overlap1d_2(Gaussian1D g1, Gaussian1D g2);

fp_type laplacian1d_1(Gaussian1D g1, Gaussian1D g2);

fp_type laplacian1d_2(Gaussian1D g1, Gaussian1D g2);

#endif
