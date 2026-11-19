#ifndef _BOYS_FUNC_
#define _BOYS_FUNC_

#include <set>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <iostream>

class BFTable {
    const static int MAX_SIZE = 16;
    using fp_type = float;
    fp_type ind[MAX_SIZE];
    public:
    fp_type &operator[](unsigned int index);
    fp_type operator[](unsigned int index) const;
};

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
    void compute_bf_table(BFTable &bf, fp_type x, int n_max);

};

#endif