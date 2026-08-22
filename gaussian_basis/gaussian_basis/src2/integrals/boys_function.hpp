#ifndef _BOYS_FUNC_
#define _BOYS_FUNC_

namespace from_boost_library {
    // Compute the boys function using the 1F1 functions
    // from the Boost library.
    double boys(double x, int n);
}

namespace beylkin_sharma {
    /* Implementation from "A fast algorithm for computing the Boys function"
    by Gregory Beylkin and Sandeep Sharma. */
    double boys(double x, int n);

};

#endif