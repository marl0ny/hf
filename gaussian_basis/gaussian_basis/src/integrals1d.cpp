#include "integrals1d.hpp"
#include "gaussian3d.hpp"
#include <boost/math/policies/error_handling.hpp>
#include <cmath>
#include <boost/math/special_functions/hypergeometric_1F1.hpp>
#include <boost/math/special_functions/hypergeometric_pFq.hpp>
#include <stdio.h>

#define PI 3.141592653589793

double boys_func(double x, int n) {
    return boost::math::hypergeometric_1F1(
        (long double)n + (long double)0.5, 
        (long double)n + (long double)1.5, 
          (long double)(-x))/(2.0*n + 1.0);
    // return hyp1f1(n + 0.5, n + 1.5, -x)/(2.0*n + 1.0);
}

struct OverlapCoeffVals {
    double r21; // Distance from the second to the first Gaussian
    double e1, e2; // Orbital exponents on the Gaussians
};

static inline double overlap_coeff_helper(
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

double overlap_coefficient(int n, 
                           Gaussian1D g1, Gaussian1D g2) {
    double e1 = g1.orbital_exponent();
    double e2 = g2.orbital_exponent();
    double r1 = g1.position();
    double r2 = g2.position();
    double r21 = r1 - r2;
    return overlap_coeff_helper(
        n, g1.angular(), g2.angular(), 
        {.r21=r21, .e1=e1, .e2=e2});
    // if (n < 0 || n > (g1.angular() + g2.angular())) {
    //     return 0.0;
    // } else if (g1.angular() == g2.angular() 
    //             && g1.angular() == n && n == 0) {
    //     return exp(-(r1 - r2)*(r1 - r2)*(e1*e2/(e1 + e2)));
    // } else if (g1.angular() == 0) {
    //     return 0.5/(e1 + e2)*overlap_coefficient(n-1, g1, g2-1)
    //         + (e1*1.0*r21)/(e1 + e2)*overlap_coefficient(n, g1, g2-1)
    //         + (n + 1)*overlap_coefficient(n+1, g1, g2-1);
    // } else {
    //     return 0.5/(e1 + e2)*overlap_coefficient(n-1, g1-1, g2)
    //         - (1.0*e2*r21)/(e1 + e2)*overlap_coefficient(n, g1-1, g2)
    //         + (n + 1)*overlap_coefficient(n+1, g1-1, g2);
    // }
}

double overlap1d(Gaussian1D g1, Gaussian1D g2) {
    return overlap_coefficient(0, g1, g2)
        *sqrt(PI/(g1.orbital_exponent() + g2.orbital_exponent()));
}

double laplacian1d(Gaussian1D g1, Gaussian1D g2) {
    long a2 = g2.angular();
    double e2 = g2.orbital_exponent();
    return (a2*(a2-1)*overlap1d(g1, g2-2)
            - 2.0*e2*(2*a2+1)*overlap1d(g1, g2)
            + 4.0*e2*e2*overlap1d(g1, g2+2));
}

double coulomb_coefficient(int i, int j, int k, int n,
                           double orb_exp, const Vec3 &r12) {
    if (i == j && j == k && k == 0) {
        return pow((-2*orb_exp), n)*boys_func(orb_exp*(dot(r12, r12)), n);
    } else if (i < 0 ||  j < 0 || k < 0) {
        return 0.0;
    } else if (j == k && k == 0) {
        return (i-1)*coulomb_coefficient(i-2, j, k, n+1, orb_exp, r12)
               + r12[0]*coulomb_coefficient(i-1, j, k, n+1, orb_exp, r12);
    } else if (k == 0) {
        return (j-1)*coulomb_coefficient(i, j-2, k, n+1, orb_exp, r12)
               + r12[1]*coulomb_coefficient(i, j-1, k, n+1, orb_exp, r12);
    } else {
        return (k-1)*coulomb_coefficient(i, j, k-2, n+1, orb_exp, r12)
               + r12[2]*coulomb_coefficient(i, j, k-1, n+1, orb_exp, r12);
    }
}
