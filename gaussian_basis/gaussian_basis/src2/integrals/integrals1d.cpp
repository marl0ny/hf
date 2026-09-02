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

struct OverlapCoeffVals {
    double r21; // Distance from the second to the first Gaussian
    double e1, e2; // Orbital exponents on the Gaussians
};

static double overlap_coeff_helper_helper(
    int n, int a1, int a2, double r, double e1, double e2
);

static inline double overlap_coeff_helper(
    int n, int a1, int a2, const struct OverlapCoeffVals &c) {

    if (a1 <= 4 && a2 <= 4) {
        if (n < 0 || n > (a1 + a2))
            return 0.0;
        return overlap_coeff_helper_helper(
            n, a1, a2, c.r21, c.e1, c.e2);
    }
    puts("Computing overlap coefficient recursively...");

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

/* Obtain the overlap coefficients between two 1D Gaussians.

Refer to the section "Overlap Integrals" from Joshua Goings' blog post
here: https://joshuagoings.com/2017/04/28/integrals/.
*/
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
}

double overlap1d(Gaussian1D g1, Gaussian1D g2) {
    return overlap_coefficient(0, g1, g2)
        *sqrt(PI/(g1.orbital_exponent() + g2.orbital_exponent()));
}

/* Laplacian integral of two 1D Gaussians.

Refer to the section "Kinetic energy integrals" from Joshua Goings'
article: https://joshuagoings.com/2017/04/28/integrals/. 
*/
double laplacian1d(Gaussian1D g1, Gaussian1D g2) {
    long a2 = g2.angular();
    double e2 = g2.orbital_exponent();
    return (a2*(a2-1)*overlap1d(g1, g2-2)
            - 2.0*e2*(2*a2+1)*overlap1d(g1, g2)
            + 4.0*e2*e2*overlap1d(g1, g2+2));
}

static double coulomb_coefficient_helper(
    int indices[3], int n, double orb_exp, const spatial::Vector &r12);

/* Compute the Coulomb coefficients. This is used in integrals
that involve the Coulomb potential, such as the
nuclear and repulsion-exchange integrals. Refer to the section
"Nuclear attraction integrals" from Joshua Goings' blog post:
https://joshuagoings.com/2017/04/28/integrals/.
*/

double coulomb_coefficient(int i, int j, int k, int n,
                           double orb_exp, const Vector &r12) {
    
    if (std::max(i, std::max(j, k)) <= 4) {
        int indices[3] = {i, j, k};
        return coulomb_coefficient_helper(indices, n, orb_exp, r12);
    } else {
        // printf("%d\n", std::max(i, std::max(j, k)));
    }
    // puts("Computing Coulomb coefficient recursively...");

    if (i == 0 && j == 0 && k == 0) {
        return pow((-2*orb_exp), n)
        // *from_boost_library::boys(orb_exp*(dot(r12, r12)), n);
        *beylkin_sharma::boys(orb_exp*(dot(r12, r12)), n);
    } else if (i < 0 ||  j < 0 || k < 0) {
        return 0.0;
    } else if (j == 0 && k == 0) {
        return (i-1)*coulomb_coefficient(i-2, j, k, n+1, orb_exp, r12)
               + r12.x*coulomb_coefficient(i-1, j, k, n+1, orb_exp, r12);
    } else if (k == 0) {
        return (j-1)*coulomb_coefficient(i, j-2, k, n+1, orb_exp, r12)
               + r12.y*coulomb_coefficient(i, j-1, k, n+1, orb_exp, r12);
    } else {
        return (k-1)*coulomb_coefficient(i, j, k-2, n+1, orb_exp, r12)
               + r12.z*coulomb_coefficient(i, j, k-1, n+1, orb_exp, r12);
    }
}
/* 
double coulomb_coefficient(int i, int j, int k, int n,
                           double orb_exp, const Vector &r12) {
    if (i == j && j == k && k == 0) {
        return pow((-2*orb_exp), n)
        // *from_boost_library::boys(orb_exp*(dot(r12, r12)), n);
        *beylkin_sharma::boys(orb_exp*(dot(r12, r12)), n);
    } else if (i < 0 ||  j < 0 || k < 0) {
        return 0.0;
    } else if (j == k && k == 0) {
        return (i-1)*coulomb_coefficient(i-2, j, k, n+1, orb_exp, r12)
               + r12.x*coulomb_coefficient(i-1, j, k, n+1, orb_exp, r12);
    } else if (k == 0) {
        return (j-1)*coulomb_coefficient(i, j-2, k, n+1, orb_exp, r12)
               + r12.y*coulomb_coefficient(i, j-1, k, n+1, orb_exp, r12);
    } else {
        return (k-1)*coulomb_coefficient(i, j, k-2, n+1, orb_exp, r12)
               + r12.z*coulomb_coefficient(i, j, k-1, n+1, orb_exp, r12);
    }
}*/

static double pw(double a, int b) {
    return pow(a, b);
}

std::map<std::tuple<double, int>, double> s_values {};

static inline double bf(double a, int n) {
    // std::pair<double, int> key {a, double(n)};
    // if (s_values.count(key) > 0) {
    //     // printf("Value computed before.\n");
    //     return s_values.at(key);
    // }
    // double val = beylkin_sharma::boys(a, n);
    // s_values.insert({key, val});
    // return val;
    // return from_boost_library::boys(a, n);
    return beylkin_sharma::boys(a, n);
}

#define SWAP(a, b, tp) tp tmp = (a); (a) = (b); (b) = (tmp);

static double coulomb_coefficient_helper(
    int indices[3], int n, double orb_exp, const spatial::Vector &r12) {
    // printf("%d, %d, %d\n", indices[0], indices[1], indices[2]);
    spatial::Vector s12 {r12};
    // 1 2 3
    if (indices[2] > indices[1]) {
        std::swap(indices[2], indices[1]);
        SWAP(s12.z, s12.y, double);
    }
    // 1 3 2
    if (indices[1] > indices[0]) {
        std::swap(indices[0], indices[1]);
        SWAP(s12.y, s12.x, double);
    }
    // 3 1 2
    if (indices[2] > indices[1]) {
        std::swap(indices[2], indices[1]);
        SWAP(s12.z, s12.y, double);
    }
    // printf("%d, %d, %d\n", indices[0], indices[1], indices[2]);
    // printf("%g, %g, %g\n", s12.x, s12.y, s12.z);

    int i = indices[0], j = indices[1], k = indices[2];
    // if (i > 2) {
    //     puts("Coeff i value greater than one.");
    // }

    double e = orb_exp;
    double r2 = dot(r12, r12);
    double val = 0.0;
    double x = s12.x, y = s12.y, z = s12.z;
    double x2 = x*x, y2 = y*y, z2 = z*z;
    double x4 = x2*x2, y4 = y2*y2, z4 = z2*z2;

    int hex_ijk = i*16*16 + j*16 + k;

    switch(hex_ijk) {

        case 0x000:
            return bf(e*r2, n)*pw(-2.0*e, n);
            break;
        case 0x100:
            return x*bf(e*r2, n + 1)*pw(-2.0*e, n + 1);
            break;
        case 0x110:
            return x*y*bf(e*r2, n + 2)*pw(-2.0*e, n + 2);
            break;
        case 0x111:
            return x*y*z*bf(e*r2, n + 3)*pw(-2.0*e, n + 3);
            break;
        case 0x200:
            return x2*bf(e*r2, n + 2)*pw(-2.0*e, n + 2) + 1.0*bf(e*r2, n + 1)*pw(-2.0*e, n + 1);
            break;
        case 0x210:
            return y*(x2*bf(e*r2, n + 3)*pw(-2.0*e, n + 3) + 1.0*bf(e*r2, n + 2)*pw(-2.0*e, n + 2));
            break;
        case 0x211:
            return y*z*(x2*bf(e*r2, n + 4)*pw(-2.0*e, n + 4) + 1.0*bf(e*r2, n + 3)*pw(-2.0*e, n + 3));
            break;
        case 0x220:
            return 1.0*x2*y2*bf(e*r2, n + 4)*pw(-2.0*e, n + 4) + 1.0*x2*bf(e*r2, n + 3)*pw(-2.0*e, n + 3) + 1.0*y2*bf(e*r2, n + 3)*pw(-2.0*e, n + 3) + 1.0*bf(e*r2, n + 2)*pw(-2.0*e, n + 2);
            break;
        case 0x221:
            return 1.0*z*(x2*y2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + x2*bf(e*r2, n + 4)*pw(-2.0*e, n + 4) + y2*bf(e*r2, n + 4)*pw(-2.0*e, n + 4) + bf(e*r2, n + 3)*pw(-2.0*e, n + 3));
            break;
        case 0x222:
            return 1.0*x2*bf(e*r2, n + 4)*pw(-2.0*e, n + 4) + 1.0*y2*(x2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + bf(e*r2, n + 4)*pw(-2.0*e, n + 4)) + z2*(1.0*x2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + y2*(x2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 1.0*bf(e*r2, n + 5)*pw(-2.0*e, n + 5)) + 1.0*bf(e*r2, n + 4)*pw(-2.0*e, n + 4)) + 1.0*bf(e*r2, n + 3)*pw(-2.0*e, n + 3);
            break;
        case 0x300:
            return x*(x2*bf(e*r2, n + 3)*pw(-2.0*e, n + 3) + 3.0*bf(e*r2, n + 2)*pw(-2.0*e, n + 2));
            break;
        case 0x310:
            return x*y*(x2*bf(e*r2, n + 4)*pw(-2.0*e, n + 4) + 3.0*bf(e*r2, n + 3)*pw(-2.0*e, n + 3));
            break;
        case 0x311:
            return x*y*z*(x2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 3.0*bf(e*r2, n + 4)*pw(-2.0*e, n + 4));
            break;
        case 0x320:
            return x*(1.0*x2*bf(e*r2, n + 4)*pw(-2.0*e, n + 4) + y2*(x2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 3.0*bf(e*r2, n + 4)*pw(-2.0*e, n + 4)) + 3.0*bf(e*r2, n + 3)*pw(-2.0*e, n + 3));
            break;
        case 0x321:
            return x*z*(1.0*x2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + y2*(x2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 3.0*bf(e*r2, n + 5)*pw(-2.0*e, n + 5)) + 3.0*bf(e*r2, n + 4)*pw(-2.0*e, n + 4));
            break;
        case 0x322:
            return x*(1.0*x2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + y2*(1.0*x2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 3.0*bf(e*r2, n + 5)*pw(-2.0*e, n + 5)) + z2*(1.0*x2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + y2*(x2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 3.0*bf(e*r2, n + 6)*pw(-2.0*e, n + 6)) + 3.0*bf(e*r2, n + 5)*pw(-2.0*e, n + 5)) + 3.0*bf(e*r2, n + 4)*pw(-2.0*e, n + 4));
            break;
        case 0x330:
            return x*y*(3.0*x2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + y2*(x2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 3.0*bf(e*r2, n + 5)*pw(-2.0*e, n + 5)) + 9.0*bf(e*r2, n + 4)*pw(-2.0*e, n + 4));
            break;
        case 0x331:
            return x*y*z*(3.0*x2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + y2*(x2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 3.0*bf(e*r2, n + 6)*pw(-2.0*e, n + 6)) + 9.0*bf(e*r2, n + 5)*pw(-2.0*e, n + 5));
            break;
        case 0x332:
            return x*y*(3.0*x2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 1.0*y2*(x2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 3.0*bf(e*r2, n + 6)*pw(-2.0*e, n + 6)) + z2*(3.0*x2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + y2*(x2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 3.0*bf(e*r2, n + 7)*pw(-2.0*e, n + 7)) + 9.0*bf(e*r2, n + 6)*pw(-2.0*e, n + 6)) + 9.0*bf(e*r2, n + 5)*pw(-2.0*e, n + 5));
            break;
        case 0x333:
            return x*y*z*(9.0*x2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 3.0*y2*(x2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 3.0*bf(e*r2, n + 7)*pw(-2.0*e, n + 7)) + z2*(3.0*x2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + y2*(x2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 3.0*bf(e*r2, n + 8)*pw(-2.0*e, n + 8)) + 9.0*bf(e*r2, n + 7)*pw(-2.0*e, n + 7)) + 27.0*bf(e*r2, n + 6)*pw(-2.0*e, n + 6));
            break;
        case 0x400:
            return 6.0*x2*bf(e*r2, n + 3)*pw(-2.0*e, n + 3) + 1.0*x4*bf(e*r2, n + 4)*pw(-2.0*e, n + 4) + 3.0*bf(e*r2, n + 2)*pw(-2.0*e, n + 2);
            break;
        case 0x410:
            return y*(6.0*x2*bf(e*r2, n + 4)*pw(-2.0*e, n + 4) + 1.0*x4*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 3.0*bf(e*r2, n + 3)*pw(-2.0*e, n + 3));
            break;
        case 0x411:
            return y*z*(6.0*x2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 1.0*x4*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 3.0*bf(e*r2, n + 4)*pw(-2.0*e, n + 4));
            break;
        case 0x420:
            return 6.0*x2*y2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 6.0*x2*bf(e*r2, n + 4)*pw(-2.0*e, n + 4) + 1.0*x4*y2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 1.0*x4*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 3.0*y2*bf(e*r2, n + 4)*pw(-2.0*e, n + 4) + 3.0*bf(e*r2, n + 3)*pw(-2.0*e, n + 3);
            break;
        case 0x421:
            return z*(6.0*x2*y2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 6.0*x2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 1.0*x4*y2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 1.0*x4*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 3.0*y2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 3.0*bf(e*r2, n + 4)*pw(-2.0*e, n + 4));
            break;
        case 0x422:
            return 6.0*x2*y2*z2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 6.0*x2*y2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 6.0*x2*z2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 6.0*x2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 1.0*x4*y2*z2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 1.0*x4*y2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 1.0*x4*z2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 1.0*x4*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 3.0*y2*z2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 3.0*y2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 3.0*z2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 3.0*bf(e*r2, n + 4)*pw(-2.0*e, n + 4);
            break;
        case 0x430:
            return y*(6.0*x2*y2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 18.0*x2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 1.0*x4*y2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 3.0*x4*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 3.0*y2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 9.0*bf(e*r2, n + 4)*pw(-2.0*e, n + 4));
            break;
        case 0x431:
            return y*z*(6.0*x2*y2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 18.0*x2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 1.0*x4*y2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 3.0*x4*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 3.0*y2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 9.0*bf(e*r2, n + 5)*pw(-2.0*e, n + 5));
            break;
        case 0x432:
            return y*(6.0*x2*y2*z2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 6.0*x2*y2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 18.0*x2*z2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 18.0*x2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 1.0*x4*y2*z2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 1.0*x4*y2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 3.0*x4*z2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 3.0*x4*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 3.0*y2*z2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 3.0*y2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 9.0*z2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 9.0*bf(e*r2, n + 5)*pw(-2.0*e, n + 5));
            break;
        case 0x433:
            return y*z*(6.0*x2*y2*z2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 18.0*x2*y2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 18.0*x2*z2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 54.0*x2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 1.0*x4*y2*z2*bf(e*r2, n + 10)*pw(-2.0*e, n + 10) + 3.0*x4*y2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 3.0*x4*z2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 9.0*x4*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 3.0*y2*z2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 9.0*y2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 9.0*z2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 27.0*bf(e*r2, n + 6)*pw(-2.0*e, n + 6));
            break;
        case 0x440:
            return 36.0*x2*y2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 6.0*x2*y4*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 18.0*x2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 6.0*x4*y2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 1.0*x4*y4*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 3.0*x4*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 18.0*y2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 3.0*y4*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 9.0*bf(e*r2, n + 4)*pw(-2.0*e, n + 4);
            break;
        case 0x441:
            return z*(36.0*x2*y2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 6.0*x2*y4*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 18.0*x2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 6.0*x4*y2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 1.0*x4*y4*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 3.0*x4*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 18.0*y2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 3.0*y4*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 9.0*bf(e*r2, n + 5)*pw(-2.0*e, n + 5));
            break;
        case 0x442:
            return 36.0*x2*y2*z2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 36.0*x2*y2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 6.0*x2*y4*z2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 6.0*x2*y4*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 18.0*x2*z2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 18.0*x2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 6.0*x4*y2*z2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 6.0*x4*y2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 1.0*x4*y4*z2*bf(e*r2, n + 10)*pw(-2.0*e, n + 10) + 1.0*x4*y4*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 3.0*x4*z2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 3.0*x4*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 18.0*y2*z2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 18.0*y2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 3.0*y4*z2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 3.0*y4*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 9.0*z2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 9.0*bf(e*r2, n + 5)*pw(-2.0*e, n + 5);
            break;
        case 0x443:
            return z*(36.0*x2*y2*z2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 108.0*x2*y2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 6.0*x2*y4*z2*bf(e*r2, n + 10)*pw(-2.0*e, n + 10) + 18.0*x2*y4*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 18.0*x2*z2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 54.0*x2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 6.0*x4*y2*z2*bf(e*r2, n + 10)*pw(-2.0*e, n + 10) + 18.0*x4*y2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 1.0*x4*y4*z2*bf(e*r2, n + 11)*pw(-2.0*e, n + 11) + 3.0*x4*y4*bf(e*r2, n + 10)*pw(-2.0*e, n + 10) + 3.0*x4*z2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 9.0*x4*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 18.0*y2*z2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 54.0*y2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 3.0*y4*z2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 9.0*y4*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 9.0*z2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 27.0*bf(e*r2, n + 6)*pw(-2.0*e, n + 6));
            break;
        case 0x444:
            return 216.0*x2*y2*z2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 36.0*x2*y2*z4*bf(e*r2, n + 10)*pw(-2.0*e, n + 10) + 108.0*x2*y2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 36.0*x2*y4*z2*bf(e*r2, n + 10)*pw(-2.0*e, n + 10) + 6.0*x2*y4*z4*bf(e*r2, n + 11)*pw(-2.0*e, n + 11) + 18.0*x2*y4*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 108.0*x2*z2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 18.0*x2*z4*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 54.0*x2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 36.0*x4*y2*z2*bf(e*r2, n + 10)*pw(-2.0*e, n + 10) + 6.0*x4*y2*z4*bf(e*r2, n + 11)*pw(-2.0*e, n + 11) + 18.0*x4*y2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 6.0*x4*y4*z2*bf(e*r2, n + 11)*pw(-2.0*e, n + 11) + 1.0*x4*y4*z4*bf(e*r2, n + 12)*pw(-2.0*e, n + 12) + 3.0*x4*y4*bf(e*r2, n + 10)*pw(-2.0*e, n + 10) + 18.0*x4*z2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 3.0*x4*z4*bf(e*r2, n + 10)*pw(-2.0*e, n + 10) + 9.0*x4*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 108.0*y2*z2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 18.0*y2*z4*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 54.0*y2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 18.0*y4*z2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 3.0*y4*z4*bf(e*r2, n + 10)*pw(-2.0*e, n + 10) + 9.0*y4*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 54.0*z2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 9.0*z4*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 27.0*bf(e*r2, n + 6)*pw(-2.0*e, n + 6);
            break;


        
        default:
            return 0.0;
            break;

    }

    return val;

}

static double overlap_coeff_helper_helper(
    int n, int a1, int a2, double r, double e1, double e2
) {
    double r2 = r*r;
    double r3 = r2*r;
    double r4 = r2*r2;
    double r5 = r4*r;
    double r6 = r3*r3;
    double r7 = r6*r;
    double r8 = r7*r;

    double e1_2 = e1*e1;
    double e1_3 = e1_2*e1;
    double e1_4 = e1_2*e1_2;
    double e1_5 = e1_4*e1;
    double e1_6 = e1_3*e1_3;
    double e1_7 = e1_6*e1;
    double e1_8 = e1_4*e1_4;

    double e2_2 = e2*e2;
    double e2_3 = e2_2*e2;
    double e2_4 = e2_2*e2_2;
    double e2_5 = e2_4*e2;
    double e2_6 = e2_3*e2_3;
    double e2_7 = e2_6*e2;
    double e2_8 = e2_4*e2_4;

    double e1e2_2 = (e1 + e2)*(e1 + e2);
    double e1e2_3 = e1e2_2*(e1 + e2);
    double e1e2_4 = e1e2_2*e1e2_2;
    double e1e2_5 = e1e2_4*(e1 + e2);
    double e1e2_6 = e1e2_3*e1e2_3;
    double e1e2_7 = e1e2_6*(e1 + e2);
    double e1e2_8 = e1e2_4*e1e2_4;

    int hex_n_a1_a2 = 16*16*n + 16*a1 + a2;
    switch(hex_n_a1_a2) {
        case 0x000:
            return  exp(-e1*e2*r2/(e1 + e2));
            break;
        case 0x001:
            return  e1*r*exp(-e1*e2*r2/(e1 + e2))/(e1 + e2);
            break;
        case 0x101:
            return  0.5*exp(-e1*e2*r2/(e1 + e2))/(e1 + e2);
            break;
        case 0x002:
            return  (0.5*e1 + e1_2*r2 + 0.5*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_2;
            break;
        case 0x102:
            return  1.0*e1*r*exp(-e1*e2*r2/(e1 + e2))/e1e2_2;
            break;
        case 0x202:
            return  0.25*exp(-e1*e2*r2/(e1 + e2))/e1e2_2;
            break;
        case 0x003:
            return  e1*r*(1.5*e1 + e1_2*r2 + 1.5*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            break;
        case 0x103:
            return  (0.75*e1 + 1.5*e1_2*r2 + 0.75*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            break;
        case 0x203:
            return  0.75*e1*r*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            break;
        case 0x303:
            return  0.125*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            break;
        case 0x004:
            return  (1.0*e1_2*r2*(e1 + e2) + e1_2*r2*(1.5*e1 + e1_2*r2 + 1.5*e2) + 0.5*e1e2_2 + (e1 + e2)*(0.25*e1 + 0.5*e1_2*r2 + 0.25*e2))*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            break;
        case 0x104:
            return  e1*r*(3.0*e1 + 2.0*e1_2*r2 + 3.0*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            break;
        case 0x204:
            return  (0.75*e1 + 1.5*e1_2*r2 + 0.75*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            break;
        case 0x304:
            return  0.5*e1*r*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            break;
        case 0x404:
            return  0.0625*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            break;
        case 0x010:
            return  -e2*r*exp(-e1*e2*r2/(e1 + e2))/(e1 + e2);
            break;
        case 0x110:
            return  0.5*exp(-e1*e2*r2/(e1 + e2))/(e1 + e2);
            break;
        case 0x011:
            return  (-e1*e2*r2 + 0.5*e1 + 0.5*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_2;
            break;
        case 0x111:
            return  0.5*r*(e1 - e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_2;
            break;
        case 0x211:
            return  0.25*exp(-e1*e2*r2/(e1 + e2))/e1e2_2;
            break;
        case 0x012:
            return  r*(1.0*e1*(e1 + e2) - e2*(0.5*e1 + e1_2*r2 + 0.5*e2))*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            break;
        case 0x112:
            return  (-1.0*e1*e2*r2 + 0.75*e1 + 0.5*e1_2*r2 + 0.75*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            break;
        case 0x212:
            return  r*(0.5*e1 - 0.25*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            break;
        case 0x312:
            return  0.125*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            break;
        case 0x013:
            return  (-e1*e2*r2*(1.5*e1 + e1_2*r2 + 1.5*e2) + 1.0*e1_2*r2*(e1 + e2) + 0.5*e1e2_2 + (e1 + e2)*(0.25*e1 + 0.5*e1_2*r2 + 0.25*e2))*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            break;
        case 0x113:
            return  r*(1.5*e1*(e1 + e2) + 0.5*e1*(1.5*e1 + e1_2*r2 + 1.5*e2) - e2*(0.75*e1 + 1.5*e1_2*r2 + 0.75*e2))*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            break;
        case 0x213:
            return  0.75*(-e1*e2*r2 + e1 + e1_2*r2 + e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            break;
        case 0x313:
            return  r*(0.375*e1 - 0.125*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            break;
        case 0x413:
            return  0.0625*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            break;
        case 0x014:
            return  r*(1.5*e1*e2_2 + 5.25*e1_2*e2 - 3.0*e1_2*e2_2*r2 - 1.0*e1_3*e2*r2 + 3.0*e1_3 - 1.0*e1_4*e2*r4 + 2.0*e1_4*r2 - 0.75*e2_3)*exp(-e1*e2*r2/(e1 + e2))/(5.0*e1*e2_4 + 10.0*e1_2*e2_3 + 10.0*e1_3*e2_2 + 5.0*e1_4*e2 + 1.0*e1_5 + 1.0*e2_5);
            break;
        case 0x114:
            return  (3.75*e1*e2 - 3.0*e1*e2_2*r2 + 1.5*e1_2*e2*r2 + 1.875*e1_2 - 2.0*e1_3*e2*r4 + 4.5*e1_3*r2 + 0.5*e1_4*r4 + 1.875*e2_2)*exp(-e1*e2*r2/(e1 + e2))/(5.0*e1*e2_4 + 10.0*e1_2*e2_3 + 10.0*e1_3*e2_2 + 5.0*e1_4*e2 + 1.0*e1_5 + 1.0*e2_5);
            break;
        case 0x214:
            return  r*(1.5*e1*(e1 + e2) + e1*(1.5*e1 + 1.0*e1_2*r2 + 1.5*e2) - e2*(0.75*e1 + 1.5*e1_2*r2 + 0.75*e2))*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            break;
        case 0x314:
            return  (-0.5*e1*e2*r2 + 0.625*e1 + 0.75*e1_2*r2 + 0.625*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            break;
        case 0x414:
            return  r*(0.25*e1 - 0.0625*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            break;
        case 0x514:
            return  0.03125*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            break;
        case 0x020:
            return  (0.5*e1 + 0.5*e2 + e2_2*r2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_2;
            break;
        case 0x120:
            return  -1.0*e2*r*exp(-e1*e2*r2/(e1 + e2))/e1e2_2;
            break;
        case 0x220:
            return  0.25*exp(-e1*e2*r2/(e1 + e2))/e1e2_2;
            break;
        case 0x021:
            return  r*(e2*(e1*e2*r2 - 0.5*e1 - 0.5*e2) + 0.5*(e1 - e2)*(e1 + e2))*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            break;
        case 0x121:
            return  (-0.5*e1*e2*r2 + 0.75*e1 - 0.5*e2*r2*(e1 - e2) + 0.75*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            break;
        case 0x221:
            return  r*(0.25*e1 - 0.5*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            break;
        case 0x321:
            return  0.125*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            break;
        case 0x022:
            return  (-1.0*e1*e2*r2*(e1 + e2) + 0.5*e1e2_2 - e2*r2*(1.0*e1*(e1 + e2) - e2*(0.5*e1 + e1_2*r2 + 0.5*e2)) + (e1 + e2)*(0.25*e1 + 0.5*e1_2*r2 + 0.25*e2))*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            break;
        case 0x122:
            return  r*(0.5*e1*(e1 + e2) - 0.5*e2*(0.5*e1 + e1_2*r2 + 0.5*e2) - e2*(-1.0*e1*e2*r2 + 0.75*e1 + 0.5*e1_2*r2 + 0.75*e2) + (e1 - 0.5*e2)*(e1 + e2))*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            break;
        case 0x222:
            return  (-0.5*e1*e2*r2 + 0.75*e1 + 0.25*e1_2*r2 - e2*r2*(0.5*e1 - 0.25*e2) + 0.75*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            break;
        case 0x322:
            return  0.25*r*(e1 - e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            break;
        case 0x422:
            return  0.0625*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            break;
        case 0x023:
            return  r*(1.5*e1*e1e2_2 + 0.5*e1*(e1 + e2)*(1.5*e1 + e1_2*r2 + 1.5*e2) - e2*(e1 + e2)*(0.75*e1 + 1.5*e1_2*r2 + 0.75*e2) - e2*(-e1*e2*r2*(1.5*e1 + e1_2*r2 + 1.5*e2) + 1.0*e1_2*r2*(e1 + e2) + 0.5*e1e2_2 + (e1 + e2)*(0.25*e1 + 0.5*e1_2*r2 + 0.25*e2)))*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            break;
        case 0x123:
            return  (3.75*e1*e2 - 3.75*e1*e2_2*r2 - 2.25*e1_2*e2*r2 + 1.5*e1_2*e2_2*r4 + 1.875*e1_2 - 1.0*e1_3*e2*r4 + 2.25*e1_3*r2 + 1.875*e2_2 + 0.75*e2_3*r2)*exp(-e1*e2*r2/(e1 + e2))/(5.0*e1*e2_4 + 10.0*e1_2*e2_3 + 10.0*e1_3*e2_2 + 5.0*e1_4*e2 + 1.0*e1_5 + 1.0*e2_5);
            break;
        case 0x223:
            return  r*(0.75*e1*(e1 + e2) + 0.25*e1*(1.5*e1 + e1_2*r2 + 1.5*e2) - 0.5*e2*(0.75*e1 + 1.5*e1_2*r2 + 0.75*e2) - 0.75*e2*(-e1*e2*r2 + e1 + e1_2*r2 + e2) + (e1 + e2)*(1.125*e1 - 0.375*e2))*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            break;
        case 0x323:
            return  (-0.375*e1*e2*r2 + 0.625*e1 + 0.375*e1_2*r2 - e2*r2*(0.375*e1 - 0.125*e2) + 0.625*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            break;
        case 0x423:
            return  r*(0.1875*e1 - 0.125*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            break;
        case 0x523:
            return  0.03125*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            break;
        case 0x024:
            return  (5.625*e1*e2_2 - 4.5*e1*e2_3*r2 + 5.625*e1_2*e2 - 6.75*e1_2*e2_2*r2 + 3.0*e1_2*e2_3*r4 + 3.0*e1_3*e2*r2 - 1.0*e1_3*e2_2*r4 + 1.875*e1_3 - 3.5*e1_4*e2*r4 + 1.0*e1_4*e2_2*r6 + 4.5*e1_4*r2 + 0.5*e1_5*r4 + 1.875*e2_3 + 0.75*e2_4*r2)*exp(-e1*e2*r2/(e1 + e2))/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
            break;
        case 0x124:
            return  r*(3.0*e1*e2_3*r2 + 11.25*e1_2*e2 - 6.0*e1_2*e2_2*r2 - 6.0*e1_3*e2*r2 + 2.0*e1_3*e2_2*r4 + 7.5*e1_3 - 1.0*e1_4*e2*r4 + 3.0*e1_4*r2 - 3.75*e2_3)*exp(-e1*e2*r2/(e1 + e2))/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
            break;
        case 0x224:
            return  (5.625*e1*e2 - 5.25*e1*e2_2*r2 - 1.5*e1_2*e2*r2 + 1.5*e1_2*e2_2*r4 + 2.8125*e1_2 - 2.0*e1_3*e2*r4 + 4.5*e1_3*r2 + 0.25*e1_4*r4 + 2.8125*e2_2 + 0.75*e2_3*r2)*exp(-e1*e2*r2/(e1 + e2))/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
            break;
        case 0x324:
            return  r*(0.75*e1*(e1 + e2) + e1*(0.75*e1 + 0.5*e1_2*r2 + 0.75*e2) - 0.5*e2*(0.75*e1 + 1.5*e1_2*r2 + 0.75*e2) - e2*(-0.5*e1*e2*r2 + 0.625*e1 + 0.75*e1_2*r2 + 0.625*e2) + (e1 - 0.25*e2)*(e1 + e2))*exp(-e1*e2*r2/(e1 + e2))/e1e2_6;
            break;
        case 0x424:
            return  (-0.25*e1*e2*r2 + 0.46875*e1 + 0.375*e1_2*r2 - e2*r2*(0.25*e1 - 0.0625*e2) + 0.46875*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_6;
            break;
        case 0x524:
            return  r*(0.125*e1 - 0.0625*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_6;
            break;
        case 0x624:
            return  0.015625*exp(-e1*e2*r2/(e1 + e2))/e1e2_6;
            break;
        case 0x030:
            return  e2*r*(-1.5*e1 - 1.5*e2 - e2_2*r2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            break;
        case 0x130:
            return  (0.75*e1 + 0.75*e2 + 1.5*e2_2*r2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            break;
        case 0x230:
            return  -0.75*e2*r*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            break;
        case 0x330:
            return  0.125*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            break;
        case 0x031:
            return  (0.5*e1e2_2 - 0.5*e2*r2*(e1 - e2)*(e1 + e2) - e2*r2*(e2*(e1*e2*r2 - 0.5*e1 - 0.5*e2) + 0.5*(e1 - e2)*(e1 + e2)) + (e1 + e2)*(-0.5*e1*e2*r2 + 0.25*e1 + 0.25*e2))*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            break;
        case 0x131:
            return  r*(-1.5*e1*e2 + 1.5*e1*e2_2*r2 + 0.75*e1_2 - 2.25*e2_2 - 0.5*e2_3*r2)*exp(-e1*e2*r2/(e1 + e2))/(4.0*e1*e2_3 + 6.0*e1_2*e2_2 + 4.0*e1_3*e2 + 1.0*e1_4 + 1.0*e2_4);
            break;
        case 0x231:
            return  (-0.25*e1*e2*r2 + 0.75*e1 + e2*r2*(-0.25*e1 + 0.5*e2) - 0.25*e2*r2*(e1 - e2) + 0.75*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            break;
        case 0x331:
            return  r*(0.125*e1 - 0.375*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            break;
        case 0x431:
            return  0.0625*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            break;
        case 0x032:
            return  r*(-3.0*e1*e2_2 + 2.5*e1*e2_3*r2 + 0.75*e1_2*e2 + 1.5*e1_2*e2_2*r2 - 1.0*e1_2*e2_3*r4 - 1.5*e1_3*e2*r2 + 1.5*e1_3 - 2.25*e2_3 - 0.5*e2_4*r2)*exp(-e1*e2*r2/(e1 + e2))/(5.0*e1*e2_4 + 10.0*e1_2*e2_3 + 10.0*e1_3*e2_2 + 5.0*e1_4*e2 + 1.0*e1_5 + 1.0*e2_5);
            break;
        case 0x132:
            return  (3.75*e1*e2 - 2.25*e1*e2_2*r2 - 1.0*e1*e2_3*r4 - 3.75*e1_2*e2*r2 + 1.5*e1_2*e2_2*r4 + 1.875*e1_2 + 0.75*e1_3*r2 + 1.875*e2_2 + 2.25*e2_3*r2)*exp(-e1*e2*r2/(e1 + e2))/(5.0*e1*e2_4 + 10.0*e1_2*e2_3 + 10.0*e1_3*e2_2 + 5.0*e1_4*e2 + 1.0*e1_5 + 1.0*e2_5);
            break;
        case 0x232:
            return  r*(-0.75*e1*e2 + 1.5*e1*e2_2*r2 - 0.75*e1_2*e2*r2 + 1.5*e1_2 - 2.25*e2_2 - 0.25*e2_3*r2)*exp(-e1*e2*r2/(e1 + e2))/(5.0*e1*e2_4 + 10.0*e1_2*e2_3 + 10.0*e1_3*e2_2 + 5.0*e1_4*e2 + 1.0*e1_5 + 1.0*e2_5);
            break;
        case 0x332:
            return  (-0.25*e1*e2*r2 + 0.625*e1 + 0.125*e1_2*r2 + 0.25*e2*r2*(-e1 + e2) - 0.5*e2*r2*(0.5*e1 - 0.25*e2) + 0.625*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            break;
        case 0x432:
            return  r*(0.125*e1 - 0.1875*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            break;
        case 0x532:
            return  0.03125*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            break;
        case 0x033:
            return  (5.625*e1*e2_2 - 2.25*e1*e2_3*r2 - 1.5*e1*e2_4*r4 + 5.625*e1_2*e2 - 9.0*e1_2*e2_2*r2 + 3.0*e1_2*e2_3*r4 - 2.25*e1_3*e2*r2 + 3.0*e1_3*e2_2*r4 - 1.0*e1_3*e2_3*r6 + 1.875*e1_3 - 1.5*e1_4*e2*r4 + 2.25*e1_4*r2 + 1.875*e2_3 + 2.25*e2_4*r2)*exp(-e1*e2*r2/(e1 + e2))/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
            break;
        case 0x133:
            return  r*(-5.625*e1*e2_2 + 6.0*e1*e2_3*r2 + 5.625*e1_2*e2 - 1.5*e1_2*e2_3*r4 - 6.0*e1_3*e2*r2 + 1.5*e1_3*e2_2*r4 + 5.625*e1_3 + 0.75*e1_4*r2 - 5.625*e2_3 - 0.75*e2_4*r2)*exp(-e1*e2*r2/(e1 + e2))/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
            break;
        case 0x233:
            return  (5.625*e1*e2 - 4.5*e1*e2_2*r2 - 0.75*e1*e2_3*r4 - 4.5*e1_2*e2*r2 + 2.25*e1_2*e2_2*r4 + 2.8125*e1_2 - 0.75*e1_3*e2*r4 + 2.25*e1_3*r2 + 2.8125*e2_2 + 2.25*e2_3*r2)*exp(-e1*e2*r2/(e1 + e2))/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
            break;
        case 0x333:
            return  r*(1.125*e1*e2_2*r2 - 1.125*e1_2*e2*r2 + 1.875*e1_2 + 0.125*e1_3*r2 - 1.875*e2_2 - 0.125*e2_3*r2)*exp(-e1*e2*r2/(e1 + e2))/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
            break;
        case 0x433:
            return  (-0.1875*e1*e2*r2 + 0.46875*e1 + 0.1875*e1_2*r2 + e2*r2*(-0.1875*e1 + 0.125*e2) - 0.5*e2*r2*(0.375*e1 - 0.125*e2) + 0.46875*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_6;
            break;
        case 0x533:
            return  0.09375*r*(e1 - e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_6;
            break;
        case 0x633:
            return  0.015625*exp(-e1*e2*r2/(e1 + e2))/e1e2_6;
            break;
        case 0x034:
            return  r*(-9.375*e1*e2_3 + 7.5*e1*e2_4*r2 + 5.625*e1_2*e2_2 + 3.75*e1_2*e2_3*r2 - 3.0*e1_2*e2_4*r4 + 16.875*e1_3*e2 - 15.0*e1_3*e2_2*r2 + 3.0*e1_3*e2_3*r4 - 7.5*e1_4*e2*r2 + 4.5*e1_4*e2_2*r4 - 1.0*e1_4*e2_3*r6 + 7.5*e1_4 - 1.5*e1_5*e2*r4 + 3.0*e1_5*r2 - 5.625*e2_4 - 0.75*e2_5*r2)*exp(-e1*e2*r2/(e1 + e2))/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
            break;
        case 0x134:
            return  (19.6875*e1*e2_2 - 11.25*e1*e2_3*r2 - 3.0*e1*e2_4*r4 + 19.6875*e1_2*e2 - 28.125*e1_2*e2_2*r2 + 10.5*e1_2*e2_3*r4 + 4.5*e1_3*e2_2*r4 - 2.0*e1_3*e2_3*r6 + 6.5625*e1_3 - 8.25*e1_4*e2*r4 + 1.5*e1_4*e2_2*r6 + 11.25*e1_4*r2 + 0.75*e1_5*r4 + 6.5625*e2_3 + 5.625*e2_4*r2)*exp(-e1*e2*r2/(e1 + e2))/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
            break;
        case 0x234:
            return  r*(-5.625*e1*e2_2 + 8.25*e1*e2_3*r2 + 14.0625*e1_2*e2 - 4.5*e1_2*e2_2*r2 - 1.5*e1_2*e2_3*r4 - 10.5*e1_3*e2*r2 + 3.0*e1_3*e2_2*r4 + 11.25*e1_3 - 0.75*e1_4*e2*r4 + 3.0*e1_4*r2 - 8.4375*e2_3 - 0.75*e2_4*r2)*exp(-e1*e2*r2/(e1 + e2))/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
            break;
        case 0x334:
            return  (6.5625*e1*e2 - 5.625*e1*e2_2*r2 - 0.5*e1*e2_3*r4 - 3.75*e1_2*e2*r2 + 2.25*e1_2*e2_2*r4 + 3.28125*e1_2 - 1.5*e1_3*e2*r4 + 3.75*e1_3*r2 + 0.125*e1_4*r4 + 3.28125*e2_2 + 1.875*e2_3*r2)*exp(-e1*e2*r2/(e1 + e2))/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
            break;
        case 0x434:
            return  r*(0.46875*e1*e2 + 0.75*e1*e2_2*r2 - 1.125*e1_2*e2*r2 + 1.875*e1_2 + 0.25*e1_3*r2 - 1.40625*e2_2 - 0.0625*e2_3*r2)*exp(-e1*e2*r2/(e1 + e2))/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
            break;
        case 0x534:
            return  (-0.125*e1*e2*r2 + 0.328125*e1 + 0.1875*e1_2*r2 + e2*r2*(-0.125*e1 + 0.0625*e2) - 0.5*e2*r2*(0.25*e1 - 0.0625*e2) + 0.328125*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_7;
            break;
        case 0x634:
            return  r*(0.0625*e1 - 0.046875*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_7;
            break;
        case 0x734:
            return  0.0078125*exp(-e1*e2*r2/(e1 + e2))/e1e2_7;
            break;
        case 0x040:
            return  (0.5*e1e2_2 + 1.0*e2_2*r2*(e1 + e2) + e2_2*r2*(1.5*e1 + 1.5*e2 + e2_2*r2) + (e1 + e2)*(0.25*e1 + 0.25*e2 + 0.5*e2_2*r2))*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            break;
        case 0x140:
            return  e2*r*(-3.0*e1 - 3.0*e2 - 2.0*e2_2*r2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            break;
        case 0x240:
            return  (0.75*e1 + 0.75*e2 + 1.5*e2_2*r2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            break;
        case 0x340:
            return  -0.5*e2*r*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            break;
        case 0x440:
            return  0.0625*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            break;
        case 0x041:
            return  r*(-5.25*e1*e2_2 + 1.0*e1*e2_3*r2 + 1.0*e1*e2_4*r4 - 1.5*e1_2*e2 + 3.0*e1_2*e2_2*r2 + 0.75*e1_3 - 3.0*e2_3 - 2.0*e2_4*r2)*exp(-e1*e2*r2/(e1 + e2))/(5.0*e1*e2_4 + 10.0*e1_2*e2_3 + 10.0*e1_3*e2_2 + 5.0*e1_4*e2 + 1.0*e1_5 + 1.0*e2_5);
            break;
        case 0x141:
            return  (3.75*e1*e2 + 1.5*e1*e2_2*r2 - 2.0*e1*e2_3*r4 - 3.0*e1_2*e2*r2 + 1.875*e1_2 + 1.875*e2_2 + 4.5*e2_3*r2 + 0.5*e2_4*r4)*exp(-e1*e2*r2/(e1 + e2))/(5.0*e1*e2_4 + 10.0*e1_2*e2_3 + 10.0*e1_3*e2_2 + 5.0*e1_4*e2 + 1.0*e1_5 + 1.0*e2_5);
            break;
        case 0x241:
            return  r*(-2.25*e1*e2 + 1.5*e1*e2_2*r2 + 0.75*e1_2 - 3.0*e2_2 - 1.0*e2_3*r2)*exp(-e1*e2*r2/(e1 + e2))/(5.0*e1*e2_4 + 10.0*e1_2*e2_3 + 10.0*e1_3*e2_2 + 5.0*e1_4*e2 + 1.0*e1_5 + 1.0*e2_5);
            break;
        case 0x341:
            return  (-0.125*e1*e2*r2 + 0.625*e1 + 0.5*e2*r2*(-0.25*e1 + 0.5*e2) + e2*r2*(-0.125*e1 + 0.375*e2) - 0.125*e2*r2*(e1 - e2) + 0.625*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            break;
        case 0x441:
            return  r*(0.0625*e1 - 0.25*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            break;
        case 0x541:
            return  0.03125*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            break;
        case 0x042:
            return  (5.625*e1*e2_2 + 3.0*e1*e2_3*r2 - 3.5*e1*e2_4*r4 + 5.625*e1_2*e2 - 6.75*e1_2*e2_2*r2 - 1.0*e1_2*e2_3*r4 + 1.0*e1_2*e2_4*r6 - 4.5*e1_3*e2*r2 + 3.0*e1_3*e2_2*r4 + 1.875*e1_3 + 0.75*e1_4*r2 + 1.875*e2_3 + 4.5*e2_4*r2 + 0.5*e2_5*r4)*exp(-e1*e2*r2/(e1 + e2))/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
            break;
        case 0x142:
            return  r*(-11.25*e1*e2_2 + 6.0*e1*e2_3*r2 + 1.0*e1*e2_4*r4 + 6.0*e1_2*e2_2*r2 - 2.0*e1_2*e2_3*r4 - 3.0*e1_3*e2*r2 + 3.75*e1_3 - 7.5*e2_3 - 3.0*e2_4*r2)*exp(-e1*e2*r2/(e1 + e2))/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
            break;
        case 0x242:
            return  (5.625*e1*e2 - 1.5*e1*e2_2*r2 - 2.0*e1*e2_3*r4 - 5.25*e1_2*e2*r2 + 1.5*e1_2*e2_2*r4 + 2.8125*e1_2 + 0.75*e1_3*r2 + 2.8125*e2_2 + 4.5*e2_3*r2 + 0.25*e2_4*r4)*exp(-e1*e2*r2/(e1 + e2))/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
            break;
        case 0x342:
            return  r*(-1.25*e1*e2 + 1.5*e1*e2_2*r2 - 0.5*e1_2*e2*r2 + 1.25*e1_2 - 2.5*e2_2 - 0.5*e2_3*r2)*exp(-e1*e2*r2/(e1 + e2))/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
            break;
        case 0x442:
            return  (-0.125*e1*e2*r2 + 0.46875*e1 + 0.0625*e1_2*r2 + 0.125*e2*r2*(-e1 + e2) + e2*r2*(-0.125*e1 + 0.1875*e2) - 0.25*e2*r2*(0.5*e1 - 0.25*e2) + 0.46875*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_6;
            break;
        case 0x542:
            return  r*(0.0625*e1 - 0.125*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_6;
            break;
        case 0x642:
            return  0.015625*exp(-e1*e2*r2/(e1 + e2))/e1e2_6;
            break;
        case 0x043:
            return  r*(-16.875*e1*e2_3 + 7.5*e1*e2_4*r2 + 1.5*e1*e2_5*r4 - 5.625*e1_2*e2_2 + 15.0*e1_2*e2_3*r2 - 4.5*e1_2*e2_4*r4 + 9.375*e1_3*e2 - 3.75*e1_3*e2_2*r2 - 3.0*e1_3*e2_3*r4 + 1.0*e1_3*e2_4*r6 - 7.5*e1_4*e2*r2 + 3.0*e1_4*e2_2*r4 + 5.625*e1_4 + 0.75*e1_5*r2 - 7.5*e2_4 - 3.0*e2_5*r2)*exp(-e1*e2*r2/(e1 + e2))/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
            break;
        case 0x143:
            return  (19.6875*e1*e2_2 - 8.25*e1*e2_4*r4 + 19.6875*e1_2*e2 - 28.125*e1_2*e2_2*r2 + 4.5*e1_2*e2_3*r4 + 1.5*e1_2*e2_4*r6 - 11.25*e1_3*e2*r2 + 10.5*e1_3*e2_2*r4 - 2.0*e1_3*e2_3*r6 + 6.5625*e1_3 - 3.0*e1_4*e2*r4 + 5.625*e1_4*r2 + 6.5625*e2_3 + 11.25*e2_4*r2 + 0.75*e2_5*r4)*exp(-e1*e2*r2/(e1 + e2))/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
            break;
        case 0x243:
            return  r*(-14.0625*e1*e2_2 + 10.5*e1*e2_3*r2 + 0.75*e1*e2_4*r4 + 5.625*e1_2*e2 + 4.5*e1_2*e2_2*r2 - 3.0*e1_2*e2_3*r4 - 8.25*e1_3*e2*r2 + 1.5*e1_3*e2_2*r4 + 8.4375*e1_3 + 0.75*e1_4*r2 - 11.25*e2_3 - 3.0*e2_4*r2)*exp(-e1*e2*r2/(e1 + e2))/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
            break;
        case 0x343:
            return  (6.5625*e1*e2 - 3.75*e1*e2_2*r2 - 1.5*e1*e2_3*r4 - 5.625*e1_2*e2*r2 + 2.25*e1_2*e2_2*r4 + 3.28125*e1_2 - 0.5*e1_3*e2*r4 + 1.875*e1_3*r2 + 3.28125*e2_2 + 3.75*e2_3*r2 + 0.125*e2_4*r4)*exp(-e1*e2*r2/(e1 + e2))/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
            break;
        case 0x443:
            return  r*(-0.46875*e1*e2 + 1.125*e1*e2_2*r2 - 0.75*e1_2*e2*r2 + 1.40625*e1_2 + 0.0625*e1_3*r2 - 1.875*e2_2 - 0.25*e2_3*r2)*exp(-e1*e2*r2/(e1 + e2))/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
            break;
        case 0x543:
            return  (-0.09375*e1*e2*r2 + 0.328125*e1 + 0.09375*e1_2*r2 + 0.09375*e2*r2*(-e1 + e2) + 0.5*e2*r2*(-0.1875*e1 + 0.125*e2) - 0.25*e2*r2*(0.375*e1 - 0.125*e2) + 0.328125*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_7;
            break;
        case 0x643:
            return  r*(0.046875*e1 - 0.0625*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_7;
            break;
        case 0x743:
            return  0.0078125*exp(-e1*e2*r2/(e1 + e2))/e1e2_7;
            break;
        case 0x044:
            return  (26.25*e1*e2_3 + 3.75*e1*e2_4*r2 - 10.5*e1*e2_5*r4 + 39.375*e1_2*e2_2 - 45.0*e1_2*e2_3*r2 + 3.75*e1_2*e2_4*r4 + 3.0*e1_2*e2_5*r6 + 26.25*e1_3*e2 - 45.0*e1_3*e2_2*r2 + 30.0*e1_3*e2_3*r4 - 5.0*e1_3*e2_4*r6 + 3.75*e1_4*e2*r2 + 3.75*e1_4*e2_2*r4 - 5.0*e1_4*e2_3*r6 + 1.0*e1_4*e2_4*r8 + 6.5625*e1_4 - 10.5*e1_5*e2*r4 + 3.0*e1_5*e2_2*r6 + 11.25*e1_5*r2 + 0.75*e1_6*r4 + 6.5625*e2_4 + 11.25*e2_5*r2 + 0.75*e2_6*r4)*exp(-e1*e2*r2/(e1 + e2))/(8.0*e1*e2_7 + 28.0*e1_2*e2_6 + 56.0*e1_3*e2_5 + 70.0*e1_4*e2_4 + 56.0*e1_5*e2_3 + 28.0*e1_6*e2_2 + 8.0*e1_7*e2 + 1.0*e1_8 + 1.0*e2_8);
            break;
        case 0x144:
            return  r*(-52.5*e1*e2_3 + 30.0*e1*e2_4*r2 + 3.0*e1*e2_5*r4 + 37.5*e1_2*e2_3*r2 - 15.0*e1_2*e2_4*r4 + 52.5*e1_3*e2 - 37.5*e1_3*e2_2*r2 + 2.0*e1_3*e2_4*r6 - 30.0*e1_4*e2*r2 + 15.0*e1_4*e2_2*r4 - 2.0*e1_4*e2_3*r6 + 26.25*e1_4 - 3.0*e1_5*e2*r4 + 7.5*e1_5*r2 - 26.25*e2_4 - 7.5*e2_5*r2)*exp(-e1*e2*r2/(e1 + e2))/(8.0*e1*e2_7 + 28.0*e1_2*e2_6 + 56.0*e1_3*e2_5 + 70.0*e1_4*e2_4 + 56.0*e1_5*e2_3 + 28.0*e1_6*e2_2 + 8.0*e1_7*e2 + 1.0*e1_8 + 1.0*e2_8);
            break;
        case 0x244:
            return  (39.375*e1*e2_2 - 11.25*e1*e2_3*r2 - 11.25*e1*e2_4*r4 + 39.375*e1_2*e2 - 56.25*e1_2*e2_2*r2 + 15.0*e1_2*e2_3*r4 + 1.5*e1_2*e2_4*r6 - 11.25*e1_3*e2*r2 + 15.0*e1_3*e2_2*r4 - 4.0*e1_3*e2_3*r6 + 13.125*e1_3 - 11.25*e1_4*e2*r4 + 1.5*e1_4*e2_2*r6 + 16.875*e1_4*r2 + 0.75*e1_5*r4 + 13.125*e2_3 + 16.875*e2_4*r2 + 0.75*e2_5*r4)*exp(-e1*e2*r2/(e1 + e2))/(8.0*e1*e2_7 + 28.0*e1_2*e2_6 + 56.0*e1_3*e2_5 + 70.0*e1_4*e2_4 + 56.0*e1_5*e2_3 + 28.0*e1_6*e2_2 + 8.0*e1_7*e2 + 1.0*e1_8 + 1.0*e2_8);
            break;
        case 0x344:
            return  r*(-13.125*e1*e2_2 + 12.5*e1*e2_3*r2 + 0.5*e1*e2_4*r4 + 13.125*e1_2*e2 - 3.0*e1_2*e2_3*r4 - 12.5*e1_3*e2*r2 + 3.0*e1_3*e2_2*r4 + 13.125*e1_3 - 0.5*e1_4*e2*r4 + 2.5*e1_4*r2 - 13.125*e2_3 - 2.5*e2_4*r2)*exp(-e1*e2*r2/(e1 + e2))/(8.0*e1*e2_7 + 28.0*e1_2*e2_6 + 56.0*e1_3*e2_5 + 70.0*e1_4*e2_4 + 56.0*e1_5*e2_3 + 28.0*e1_6*e2_2 + 8.0*e1_7*e2 + 1.0*e1_8 + 1.0*e2_8);
            break;
        case 0x444:
            return  (6.5625*e1*e2 - 4.6875*e1*e2_2*r2 - 1.0*e1*e2_3*r4 - 4.6875*e1_2*e2*r2 + 2.25*e1_2*e2_2*r4 + 3.28125*e1_2 - 1.0*e1_3*e2*r4 + 2.8125*e1_3*r2 + 0.0625*e1_4*r4 + 3.28125*e2_2 + 2.8125*e2_3*r2 + 0.0625*e2_4*r4)*exp(-e1*e2*r2/(e1 + e2))/(8.0*e1*e2_7 + 28.0*e1_2*e2_6 + 56.0*e1_3*e2_5 + 70.0*e1_4*e2_4 + 56.0*e1_5*e2_3 + 28.0*e1_6*e2_2 + 8.0*e1_7*e2 + 1.0*e1_8 + 1.0*e2_8);
            break;
        case 0x544:
            return  r*(0.75*e1*e2_2*r2 - 0.75*e1_2*e2*r2 + 1.3125*e1_2 + 0.125*e1_3*r2 - 1.3125*e2_2 - 0.125*e2_3*r2)*exp(-e1*e2*r2/(e1 + e2))/(8.0*e1*e2_7 + 28.0*e1_2*e2_6 + 56.0*e1_3*e2_5 + 70.0*e1_4*e2_4 + 56.0*e1_5*e2_3 + 28.0*e1_6*e2_2 + 8.0*e1_7*e2 + 1.0*e1_8 + 1.0*e2_8);
            break;
        case 0x644:
            return  (-0.0625*e1*e2*r2 + 0.21875*e1 + 0.09375*e1_2*r2 + 0.5*e2*r2*(-0.125*e1 + 0.0625*e2) + e2*r2*(-0.0625*e1 + 0.046875*e2) - 0.25*e2*r2*(0.25*e1 - 0.0625*e2) + 0.21875*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_8;
            break;
        case 0x744:
            return  0.03125*r*(e1 - e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_8;
            break;
        case 0x844:
            return  0.00390625*exp(-e1*e2*r2/(e1 + e2))/e1e2_8;
            break;


        default:
            break;
    }
    return 0.0;
}