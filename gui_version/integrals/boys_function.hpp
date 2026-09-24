#ifndef _BOYS_FUNC_
#define _BOYS_FUNC_


namespace from_boost_library {
    // Compute the boys function using the 1F1 functions
    // from the Boost library.
    using fp_type = float;
    fp_type boys(fp_type x, int n);
}

namespace beylkin_sharma {
    /* Implementation from "A fast algorithm for computing the Boys function"
    by Gregory Beylkin and Sandeep Sharma. */
    using fp_type = float;
    fp_type boys(fp_type x, int n);

};

#endif