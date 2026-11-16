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

fp_type overlap1d(Gaussian1D g1, Gaussian1D g2) {
    return overlap_coefficient(0, g1, g2)
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

static fp_type coulomb_coefficient_helper(
    fp_type *bf,
    int indices[3], int n, fp_type orb_exp, const spatial::Vector &r12);

/* Compute the Coulomb coefficients. This is used in integrals
that involve the Coulomb potential, such as the
nuclear and repulsion-exchange integrals. Refer to the section
"Nuclear attraction integrals" from Joshua Goings' blog post:
https://joshuagoings.com/2017/04/28/integrals/.
*/

fp_type coulomb_coefficient(fp_type *bf, int i, int j, int k, int n,
                           fp_type orb_exp, const Vector &r12) {
    if (std::max(i, std::max(j, k)) <= 4) {
        int indices[3] = {i, j, k};
        return coulomb_coefficient_helper(bf, indices, n, orb_exp, r12);
    }
    return 0.0;
}

static fp_type pw(fp_type a, int b) {
    return pow(a, b);
}

// std::map<std::tuple<fp_type, int>, fp_type> s_values {};

static inline fp_type bf(fp_type a, int n) {
    return beylkin_sharma::boys(a, n);
}

#define SWAP(a, b, tp) tp tmp = (a); (a) = (b); (b) = (tmp);

static fp_type coulomb_coefficient_helper(
    fp_type *bf,
    int indices[3], int n, fp_type orb_exp, const spatial::Vector &r12) {
    // printf("%d, %d, %d\n", indices[0], indices[1], indices[2]);
    spatial::Vector s12 {r12};
    // 1 2 3
    if (indices[2] > indices[1]) {
        std::swap(indices[2], indices[1]);
        SWAP(s12.z, s12.y, fp_type);
    }
    // 1 3 2
    if (indices[1] > indices[0]) {
        std::swap(indices[0], indices[1]);
        SWAP(s12.y, s12.x, fp_type);
    }
    // 3 1 2
    if (indices[2] > indices[1]) {
        std::swap(indices[2], indices[1]);
        SWAP(s12.z, s12.y, fp_type);
    }
    // printf("%d, %d, %d\n", indices[0], indices[1], indices[2]);
    // printf("%g, %g, %g\n", s12.x, s12.y, s12.z);

    int i = indices[0], j = indices[1], k = indices[2];
    // if (i > 2) {
    //     puts("Coeff i value greater than one.");
    // }

    fp_type e = orb_exp;
    fp_type r2 = dot(r12, r12);
    fp_type val = 0.0;
    fp_type x = s12.x, y = s12.y, z = s12.z;
    fp_type x2 = x*x, y2 = y*y, z2 = z*z;
    fp_type x4 = x2*x2, y4 = y2*y2, z4 = z2*z2;
    fp_type x6 = x4*x2;
    fp_type y6 = y4*y2;
    fp_type z6 = z4*z2;
    fp_type ep1 = e;
    fp_type ep2 = e*e;
    fp_type ep3 = e*e*e;
    fp_type ep4 = e*e*e*e;
    fp_type ep5 = e*e*e*e*e;
    fp_type ep6 = e*e*e*e*e*e;
    fp_type ep7 = e*e*e*e*e*e*e;

    int hex_ijk = i*16*16 + j*16 + k;
    int hexIJK = hex_ijk;

    if (hexIJK == 0x000) {
        return bf[0];
    }
    else if (hexIJK == 0x100) {
        return -2*bf[1]*ep1*x;
    }
    else if (hexIJK == 0x110) {
        return 4*bf[2]*ep2*x*y;
    }
    else if (hexIJK == 0x111) {
        return -8*bf[3]*ep3*x*y*z;
    }
    else if (hexIJK == 0x200) {
        return -2.0*bf[1]*ep1 + 4.0*bf[2]*ep2*x2;
    }
    else if (hexIJK == 0x210) {
        return 4.0*bf[2]*ep2*y - 8.0*bf[3]*ep3*x2*y;
    }
    else if (hexIJK == 0x211) {
        return -8.0*bf[3]*ep3*y*z + 16.0*bf[4]*ep4*x2*y*z;
    }
    else if (hexIJK == 0x220) {
        return 4.0*bf[2]*ep2 - 8.0*bf[3]*ep3*x2 - 8.0*bf[3]*ep3*y2 + 16.0*bf[4]*ep4*x2*y2;
    }
    else if (hexIJK == 0x221) {
        return -8.0*bf[3]*ep3*z + 16.0*bf[4]*ep4*x2*z + 16.0*bf[4]*ep4*y2*z - 32.0*bf[5]*ep5*x2*y2*z;
    }
    else if (hexIJK == 0x222) {
        return -8.0*bf[3]*ep3 + 16.0*bf[4]*ep4*x2 + 16.0*bf[4]*ep4*y2 + 16.0*bf[4]*ep4*z2 - 32.0*bf[5]*ep5*x2*y2 - 32.0*bf[5]*ep5*x2*z2 - 32.0*bf[5]*ep5*y2*z2 + 64.0*bf[6]*ep6*x2*y2*z2;
    }
    else if (hexIJK == 0x300) {
        return 12.0*bf[2]*ep2*x - 8.0*bf[3]*ep3*x*x2;
    }
    else if (hexIJK == 0x310) {
        return -24.0*bf[3]*ep3*x*y + 16.0*bf[4]*ep4*x*x2*y;
    }
    else if (hexIJK == 0x311) {
        return 48.0*bf[4]*ep4*x*y*z - 32.0*bf[5]*ep5*x*x2*y*z;
    }
    else if (hexIJK == 0x320) {
        return -24.0*bf[3]*ep3*x + 16.0*bf[4]*ep4*x*x2 + 48.0*bf[4]*ep4*x*y2 - 32.0*bf[5]*ep5*x*x2*y2;
    }
    else if (hexIJK == 0x321) {
        return 48.0*bf[4]*ep4*x*z - 32.0*bf[5]*ep5*x*x2*z - 96.0*bf[5]*ep5*x*y2*z + 64.0*bf[6]*ep6*x*x2*y2*z;
    }
    else if (hexIJK == 0x330) {
        return 144.0*bf[4]*ep4*x*y - 96.0*bf[5]*ep5*x*x2*y - 96.0*bf[5]*ep5*x*y*y2 + 64.0*bf[6]*ep6*x*x2*y*y2;
    }
    else if (hexIJK == 0x400) {
        return 12.0*bf[2]*ep2 - 48.0*bf[3]*ep3*x2 + 16.0*bf[4]*ep4*x4;
    }
    else if (hexIJK == 0x410) {
        return -24.0*bf[3]*ep3*y + 96.0*bf[4]*ep4*x2*y - 32.0*bf[5]*ep5*x4*y;
    }
    else if (hexIJK == 0x411) {
        return 48.0*bf[4]*ep4*y*z - 192.0*bf[5]*ep5*x2*y*z + 64.0*bf[6]*ep6*x4*y*z;
    }
    else if (hexIJK == 0x420) {
        return -24.0*bf[3]*ep3 + 96.0*bf[4]*ep4*x2 + 48.0*bf[4]*ep4*y2 - 192.0*bf[5]*ep5*x2*y2 - 32.0*bf[5]*ep5*x4 + 64.0*bf[6]*ep6*x4*y2;
    }
    else if (hexIJK == 0x500) {
        return -120.0*bf[3]*ep3*x + 160.0*bf[4]*ep4*x*x2 - 32.0*bf[5]*ep5*x*x4;
    }
    else if (hexIJK == 0x510) {
        return 240.0*bf[4]*ep4*x*y - 320.0*bf[5]*ep5*x*x2*y + 64.0*bf[6]*ep6*x*x4*y;
    }
    else if (hexIJK == 0x600) {
        return -120.0*bf[3]*ep3 + 720.0*bf[4]*ep4*x2 - 480.0*bf[5]*ep5*x4 + 64.0*bf[6]*ep6*x6;
    }

    return val;

}

static fp_type coulomb_coefficient_helper(
    int indices[3], int n, fp_type orb_exp, const spatial::Vector &r12) {
    // printf("%d, %d, %d\n", indices[0], indices[1], indices[2]);
    spatial::Vector s12 {r12};
    // 1 2 3
    if (indices[2] > indices[1]) {
        std::swap(indices[2], indices[1]);
        SWAP(s12.z, s12.y, fp_type);
    }
    // 1 3 2
    if (indices[1] > indices[0]) {
        std::swap(indices[0], indices[1]);
        SWAP(s12.y, s12.x, fp_type);
    }
    // 3 1 2
    if (indices[2] > indices[1]) {
        std::swap(indices[2], indices[1]);
        SWAP(s12.z, s12.y, fp_type);
    }
    // printf("%d, %d, %d\n", indices[0], indices[1], indices[2]);
    // printf("%g, %g, %g\n", s12.x, s12.y, s12.z);

    int i = indices[0], j = indices[1], k = indices[2];
    // if (i > 2) {
    //     puts("Coeff i value greater than one.");
    // }

    fp_type e = orb_exp;
    fp_type r2 = dot(r12, r12);
    fp_type val = 0.0;
    fp_type x = s12.x, y = s12.y, z = s12.z;
    fp_type x2 = x*x, y2 = y*y, z2 = z*z;
    fp_type x4 = x2*x2, y4 = y2*y2, z4 = z2*z2;
    fp_type x6 = x4*x2;
    fp_type y6 = y4*y2;
    fp_type z6 = z4*z2;
    fp_type ep1 = e;
    fp_type ep2 = e*e;
    fp_type ep3 = e*e*e;
    fp_type ep4 = e*e*e*e;
    fp_type ep5 = e*e*e*e*e;
    fp_type ep6 = e*e*e*e*e*e;
    fp_type ep7 = e*e*e*e*e*e*e;

    int hex_ijk = i*16*16 + j*16 + k;
    int hexIJK = hex_ijk;

    if (hexIJK == 0x000) {
        return bf(e*r2, 0)*pw(e, 0);
    }
    else if (hexIJK == 0x100) {
        return -2*ep1*x*bf(e*r2, 1);
    }
    else if (hexIJK == 0x110) {
        return 4*ep2*x*y*bf(e*r2, 2);
    }
    else if (hexIJK == 0x111) {
        return -8*ep3*x*y*z*bf(e*r2, 3);
    }
    else if (hexIJK == 0x200) {
        return -2.0*ep1*bf(e*r2, 1) + 4.0*ep2*x2*bf(e*r2, 2);
    }
    else if (hexIJK == 0x210) {
        return 4.0*ep2*y*bf(e*r2, 2) - 8.0*ep3*x2*y*bf(e*r2, 3);
    }
    else if (hexIJK == 0x211) {
        return -8.0*ep3*y*z*bf(e*r2, 3) + 16.0*ep4*x2*y*z*bf(e*r2, 4);
    }
    else if (hexIJK == 0x220) {
        return 4.0*ep2*bf(e*r2, 2) - 8.0*ep3*x2*bf(e*r2, 3) - 8.0*ep3*y2*bf(e*r2, 3) + 16.0*ep4*x2*y2*bf(e*r2, 4);
    }
    else if (hexIJK == 0x221) {
        return -8.0*ep3*z*bf(e*r2, 3) + 16.0*ep4*x2*z*bf(e*r2, 4) + 16.0*ep4*y2*z*bf(e*r2, 4) - 32.0*ep5*x2*y2*z*bf(e*r2, 5);
    }
    else if (hexIJK == 0x222) {
        return -8.0*ep3*bf(e*r2, 3) + 16.0*ep4*x2*bf(e*r2, 4) + 16.0*ep4*y2*bf(e*r2, 4) + 16.0*ep4*z2*bf(e*r2, 4) - 32.0*ep5*x2*y2*bf(e*r2, 5) - 32.0*ep5*x2*z2*bf(e*r2, 5) - 32.0*ep5*y2*z2*bf(e*r2, 5) + 64.0*ep6*x2*y2*z2*bf(e*r2, 6);
    }
    else if (hexIJK == 0x300) {
        return 12.0*ep2*x*bf(e*r2, 2) - 8.0*ep3*x*x2*bf(e*r2, 3);
    }
    else if (hexIJK == 0x310) {
        return -24.0*ep3*x*y*bf(e*r2, 3) + 16.0*ep4*x*x2*y*bf(e*r2, 4);
    }
    else if (hexIJK == 0x311) {
        return 48.0*ep4*x*y*z*bf(e*r2, 4) - 32.0*ep5*x*x2*y*z*bf(e*r2, 5);
    }
    else if (hexIJK == 0x320) {
        return -24.0*ep3*x*bf(e*r2, 3) + 16.0*ep4*x*x2*bf(e*r2, 4) + 48.0*ep4*x*y2*bf(e*r2, 4) - 32.0*ep5*x*x2*y2*bf(e*r2, 5);
    }
    else if (hexIJK == 0x321) {
        return 48.0*ep4*x*z*bf(e*r2, 4) - 32.0*ep5*x*x2*z*bf(e*r2, 5) - 96.0*ep5*x*y2*z*bf(e*r2, 5) + 64.0*ep6*x*x2*y2*z*bf(e*r2, 6);
    }
    else if (hexIJK == 0x330) {
        return 144.0*ep4*x*y*bf(e*r2, 4) - 96.0*ep5*x*x2*y*bf(e*r2, 5) - 96.0*ep5*x*y*y2*bf(e*r2, 5) + 64.0*ep6*x*x2*y*y2*bf(e*r2, 6);
    }
    else if (hexIJK == 0x400) {
        return 12.0*ep2*bf(e*r2, 2) - 48.0*ep3*x2*bf(e*r2, 3) + 16.0*ep4*x4*bf(e*r2, 4);
    }
    else if (hexIJK == 0x410) {
        return -24.0*ep3*y*bf(e*r2, 3) + 96.0*ep4*x2*y*bf(e*r2, 4) - 32.0*ep5*x4*y*bf(e*r2, 5);
    }
    else if (hexIJK == 0x411) {
        return 48.0*ep4*y*z*bf(e*r2, 4) - 192.0*ep5*x2*y*z*bf(e*r2, 5) + 64.0*ep6*x4*y*z*bf(e*r2, 6);
    }
    else if (hexIJK == 0x420) {
        return -24.0*ep3*bf(e*r2, 3) + 96.0*ep4*x2*bf(e*r2, 4) + 48.0*ep4*y2*bf(e*r2, 4) - 192.0*ep5*x2*y2*bf(e*r2, 5) - 32.0*ep5*x4*bf(e*r2, 5) + 64.0*ep6*x4*y2*bf(e*r2, 6);
    }
    else if (hexIJK == 0x500) {
        return -120.0*ep3*x*bf(e*r2, 3) + 160.0*ep4*x*x2*bf(e*r2, 4) - 32.0*ep5*x*x4*bf(e*r2, 5);
    }
    else if (hexIJK == 0x510) {
        return 240.0*ep4*x*y*bf(e*r2, 4) - 320.0*ep5*x*x2*y*bf(e*r2, 5) + 64.0*ep6*x*x4*y*bf(e*r2, 6);
    }
    else if (hexIJK == 0x600) {
        return -120.0*ep3*bf(e*r2, 3) + 720.0*ep4*x2*bf(e*r2, 4) - 480.0*ep5*x4*bf(e*r2, 5) + 64.0*ep6*x6*bf(e*r2, 6);
    }

    return val;

}

fp_type coulomb_coefficient(int i, int j, int k, int n,
                           fp_type orb_exp, const Vector &r12) {
    if (std::max(i, std::max(j, k)) <= 4) {
        int indices[3] = {i, j, k};
        return coulomb_coefficient_helper(indices, n, orb_exp, r12);
    }
    return 0.0;
}