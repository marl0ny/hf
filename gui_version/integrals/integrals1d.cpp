/* Compute the various integral relations between Gaussian functions
of the form a*exp(-alpha*x^2).

This is indebted to the following article:

    Joshua Goings, 
    A (hopefully) gentle guide to the computer implementation 
    of molecular integrals. 2017.
    https://joshuagoings.com/2017/04/28/integrals/


*/
#include "integrals1d.hpp"
#include "gaussian3d.hpp"
#include "boys_function.hpp"
#include <algorithm>
#include <boost/math/policies/error_handling.hpp>
#include <cmath>
#include <complex.h>
#include <map>
#include <tuple>

#define PI 3.141592653589793

using namespace spatial;

using fp_type = float;

struct OverlapCoeffVals {
    bool is_one;
    fp_type r21; // Distance from the second to the first Gaussian
    fp_type e1, e2; // Orbital exponents on the Gaussians
};

static inline fp_type overlap_coeff_helper(
    int n, int a1, int a2, const struct OverlapCoeffVals &c) {

    if (n < 0 || n > (a1 + a2)) {
        return 0.0;
    } else if (a1 == a2 && a1 == n && n == 0) {
        return exp(-c.r21*c.r21*(c.e1*c.e2/(c.e1 + c.e2)));
    } else if (a1 == 0) {
        return 0.5/(c.e1 + c.e2)*overlap_coeff_helper(n-1, a1, a2-1, c)
            + (c.e1*1.0*c.r21)/(c.e1 + c.e2)
                *overlap_coeff_helper(n, a1, a2-1, c)
            + (n + 1)*overlap_coeff_helper(n+1, a1, a2-1, c);
    } else {
        return 0.5/(c.e1 + c.e2)*overlap_coeff_helper(n-1, a1-1, a2, c)
            - (1.0*c.e2*c.r21)/(c.e1 + c.e2)
                *overlap_coeff_helper(n, a1-1, a2, c)
            + (n + 1)*overlap_coeff_helper(n+1, a1-1, a2, c);
    }
}

static inline fp_type overlap_derivative_coeff_helper(
    int n, int a1, int a2, const struct OverlapCoeffVals &c) {
    // int e1 = c.e1, e2 = c.e2;
    fp_type sign = (c.is_one)? 1.0: -1.0;
    if (n < 0 || n > (a1 + a2)) {
        return 0.0;
    } else if (a1 == a2 && a1 == n && n == 0) {
        return -2.0*sign*c.r21*exp(-c.r21*c.r21*(c.e1*c.e2/(c.e1 + c.e2)));
    } else if (a1 == 0) {
        return 0.5/(c.e1 + c.e2)*overlap_derivative_coeff_helper(
                n-1, a1, a2-1, c)
            + (c.e1*sign)/(c.e1 + c.e2)
                *overlap_coeff_helper(n, a1, a2-1, c)
            + (c.e1*c.r21)/(c.e1 + c.e2)
                *overlap_derivative_coeff_helper(n, a1, a2-1, c)
            + (n + 1)*overlap_derivative_coeff_helper(n+1, a1, a2-1, c);
    } else {
        return 0.5/(c.e1 + c.e2)*overlap_derivative_coeff_helper(
                n-1, a1-1, a2, c)
            - (c.e2*sign)/(c.e1 + c.e2)
                *overlap_coeff_helper(n, a1-1, a2, c)
            - (c.e2*c.r21)/(c.e1 + c.e2)
                *overlap_derivative_coeff_helper(n, a1-1, a2, c)
            + (n + 1)*overlap_derivative_coeff_helper(n+1, a1-1, a2, c);
    } 
}

/* Obtain the overlap coefficients between two 1D Gaussians.

Refer to the section "Overlap Integrals" from Joshua Goings' blog post
here: https://joshuagoings.com/2017/04/28/integrals/.
*/
fp_type overlap_coefficient(int n, 
                           Gaussian1D g1, Gaussian1D g2) {
    fp_type e1 = g1.orbital_exponent();
    fp_type e2 = g2.orbital_exponent();
    fp_type r1 = g1.position();
    fp_type r2 = g2.position();
    fp_type r21 = r1 - r2;
    return overlap_coeff_helper(
        n, g1.angular(), g2.angular(), 
        {.r21=r21, .e1=e1, .e2=e2});
}

fp_type overlap_coefficient_1(int n, Gaussian1D g1, Gaussian1D g2) {
    fp_type e1 = g1.orbital_exponent();
    fp_type e2 = g2.orbital_exponent();
    fp_type r1 = g1.position();
    fp_type r2 = g2.position();
    fp_type r21 = r1 - r2;
    return overlap_derivative_coeff_helper(
        n, g1.angular(), g2.angular(), 
        {.is_one=true, .r21=r21, .e1=e1, .e2=e2});
}

fp_type overlap_coefficient_2(int n, Gaussian1D g1, Gaussian1D g2) {
    fp_type e1 = g1.orbital_exponent();
    fp_type e2 = g2.orbital_exponent();
    fp_type r1 = g1.position();
    fp_type r2 = g2.position();
    fp_type r21 = r1 - r2;
    return overlap_derivative_coeff_helper(
        n, g1.angular(), g2.angular(), 
        {.is_one=false, .r21=r21, .e1=e1, .e2=e2});
}

fp_type overlap1d(Gaussian1D g1, Gaussian1D g2) {
    return overlap_coefficient(0, g1, g2)
        *sqrt(PI/(g1.orbital_exponent() + g2.orbital_exponent()));
}

fp_type overlap1d_1(Gaussian1D g1, Gaussian1D g2) {
    return overlap_coefficient_1(0, g1, g2)
        *sqrt(PI/(g1.orbital_exponent() + g2.orbital_exponent()));
}

fp_type overlap1d_2(Gaussian1D g1, Gaussian1D g2) {
    return overlap_coefficient_2(0, g1, g2)
        *sqrt(PI/(g1.orbital_exponent() + g2.orbital_exponent()));
}

/* Laplacian integral of two 1D Gaussians.

Refer to the section "Kinetic energy integrals" from Joshua Goings'
article: https://joshuagoings.com/2017/04/28/integrals/. 
*/
fp_type laplacian1d(Gaussian1D g1, Gaussian1D g2) {
    long a2 = g2.angular();
    fp_type e2 = g2.orbital_exponent();
    return (a2*(a2-1)*overlap1d(g1, g2-2)
            - 2.0*e2*(2*a2+1)*overlap1d(g1, g2)
            + 4.0*e2*e2*overlap1d(g1, g2+2));
}

fp_type laplacian1d_1(Gaussian1D g1, Gaussian1D g2) {
    long a2 = g2.angular();
    fp_type e2 = g2.orbital_exponent();
    return (a2*(a2-1)*overlap1d_1(g1, g2-2)
            - 2.0*e2*(2*a2+1)*overlap1d_1(g1, g2)
            + 4.0*e2*e2*overlap1d_1(g1, g2+2));
}

fp_type laplacian1d_2(Gaussian1D g1, Gaussian1D g2) {
    long a2 = g2.angular();
    fp_type e2 = g2.orbital_exponent();
    return (a2*(a2-1)*overlap1d_2(g1, g2-2)
            - 2.0*e2*(2*a2+1)*overlap1d_2(g1, g2)
            + 4.0*e2*e2*overlap1d_2(g1, g2+2));
}

static fp_type coulomb_coefficient_helper(
    fp_type *bf,
    int indices[3], int n, fp_type orb_exp, const spatial::Vector &r12);
