/* The Boys function is used to find the Coulomb coefficients, which
is in turn used to compute integrals involving the Coulomb potential.
See the section "Nuclear attraction integrals" from this article
by Joshua Goings:
    https://joshuagoings.com/2017/04/28/integrals/.
*/
#include "boys_function.hpp"
#include <boost/math/special_functions/hypergeometric_1F1.hpp>
#include <boost/math/special_functions/hypergeometric_pFq.hpp>

#define PI 3.141592653589793

/*
This uses Hypergeometric 1F1 as implemented by the Boost math library:
https://live.boost.org/doc/libs/master/libs/\
math/doc/html/math_toolkit/hypergeometric/hypergeometric_1f1.html.
*/
double from_boost_library::boys(double x, int n) {
    return boost::math::hypergeometric_1F1(
        (long double)n + (long double)0.5, 
        (long double)n + (long double)1.5, 
          (long double)(-x))/(2.0*n + 1.0);
    // return hyp1f1(n + 0.5, n + 1.5, -x)/(2.0*n + 1.0);
}

/* Please look at Table II from "A fast algorithm for computing the Boys function"
by Gregory Beylkin and Sandeep Sharma. */
static const std::complex<double> WEIGHTS[26] = {
    std::complex(7.071943132057001, 16.487291250752115), std::complex(3.64436324028985e-11, 2.6411751072107504e-11),
    std::complex(7.071943132057001, -16.487291250752115), std::complex(3.64436324028985e-11, -2.6411751072107504e-11),
    std::complex(-0.5714327171519163, 13.278579453233633), std::complex(1.8185250346753633e-07, -2.186045897139935e-06),
    std::complex(-0.5714327171519163, -13.278579453233633), std::complex(1.8185250346753633e-07, 2.186045897139935e-06),
    std::complex(-4.719302133039251, 9.983525711237103), std::complex(-0.0009948916927205575, -0.00023049079105203073),
    std::complex(-4.719302133039251, -9.983525711237103), std::complex(-0.0009948916927205575, 0.00023049079105203073),
    std::complex(-7.170466277289509, 6.671236083982077), std::complex(-0.025625216985879006, 0.03581833527487698),
    std::complex(-7.170466277289509, -6.671236083982077), std::complex(-0.025625216985879006, -0.03581833527487698),
    std::complex(-8.48997470547247, 3.343480416846749), std::complex(0.16506801544880723, 0.32273964471776045),
    std::complex(-8.48997470547247, -3.343480416846749), std::complex(0.16506801544880723, -0.32273964471776045),
    std::complex(36.56441436315097, 0.0), std::complex(-2.0104641661565164e-26, 0.0),
    std::complex(-3.2424239255921954, 0.0), std::complex(-0.0003956353695504208, 0.0),
    std::complex(-8.906604773310075, 0.0), std::complex(0.7234994580508529, 0.0),
};

static double boys_func_recurse_upwards(double x, int n);

static double boys_func_recurse_to_zero(double x, int n);

/* Please look at "A fast algorithm for computing the Boys function"
by Gregory Beylkin and Sandeep Sharma, which is where this
implementation is taken from. This function is WIP.

The Boys function boys(n, x) is computed recursively.
Section I of this article contains a short formula for deciding when to
recurse to n=0, and when to go upwards to the maximum value of n. */
double beylkin_sharma::boys(double x, int n) {
    if (x == 0)
        return 1.0/(2.0*n + 1.0);
    double y = 1.0;
    int n_max = 12;
    for (int j = 1; j <= n_max; j++) // See I.
        y *= (j - 0.5);
    double z = std::pow(y, 1.0/n_max);
    if (abs(x) < z)
        return boys_func_recurse_upwards(x, n);
    else
        return boys_func_recurse_to_zero(x, n);
}

/* See "A fast algorithm for computing the Boys function"
by Gregory Beylkin and Sandeep Sharma, particularly the section containing
equation (3) for recursing to the maximum n value. */
static double boys_func_recurse_upwards(double x, int n) {
    std::complex c_val = 0.0;
    for (int i = 0; i < 13; i++) {
        std::complex exp_val = WEIGHTS[2*i];
        std::complex weight = WEIGHTS[2*i + 1];
        c_val += (
            weight 
            *exp(exp_val)/(-x - exp_val)
            *(exp(-(x + exp_val)) - 1.0));
    }
    double val = std::real(0.5*c_val);
    for (int n_iter = 12; n_iter > n; n_iter--) {
        val = x/(n_iter - 0.5)*val + exp(-x)/(2.0*(n_iter - 0.5));
    }
    return val;
    
    /* if (n == 12) {
        std::complex val = 0.0;
        for (int i = 0; i < 13; i++) {
            std::complex exp_val = WEIGHTS[2*i];
            std::complex weight = WEIGHTS[2*i + 1];
            val += (
                weight 
                *exp(exp_val)/(-x - exp_val)
                *(exp(-(x + exp_val)) - 1.0));
        }
        return std::real(0.5*val);
    } else if (n < 12) { // (3)
        int n2 = n + 1;
        return x/(n2 - 0.5)*boys_func_recurse_upwards(x, n2) 
            + exp(-x)/(2.0*(n2 - 0.5));
    } */

    /* else {
        return ((n - 0.5)/x)*boys_func_recurse_high(x, n-1) - 0.5*exp(-x)/x;
    }*/
    // return boys_func(x, n);
}

/* See "A fast algorithm for computing the Boys function"
by Gregory Beylkin and Sandeep Sharma, particularly the section containing
equation (4) for the n = 0 base case, and (2) for the recursion relation
that goes to zero. */
static double boys_func_recurse_to_zero(double x, int n) {
    double val = sqrt(PI)*std::erf(sqrt(x))/(2.0*sqrt(x));
    for (int n_iter = 1; n_iter <= n; n_iter++)
        val = ((n_iter - 0.5)/x)*val - 0.5*exp(-x)/x;
    return val;

    /* if (n == 0) // See (4)
        return sqrt(PI)*std::erf(sqrt(x))/(2.0*sqrt(x));
    // See (2)
    return ((n - 0.5)/x)*boys_func_recurse_to_zero(x, n-1) - 0.5*exp(-x)/x;*/
}
