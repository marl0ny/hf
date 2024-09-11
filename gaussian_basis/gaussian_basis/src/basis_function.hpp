#include "gaussian3d.hpp"

#ifndef _BASIS_FUNCTION_
#define _BASIS_FUNCTION_

struct BasisFunction {
    long count;
    Gaussian3D *primitives;
    const Gaussian3D &operator[] (int index) const;
    // void normalize();
    bool operator==(const BasisFunction &w) const;
};

#endif