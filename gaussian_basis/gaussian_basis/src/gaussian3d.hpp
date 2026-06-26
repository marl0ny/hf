#include "gaussian1d.hpp"
#include "vec3.hpp"

#ifndef _GAUSSIAN3D_
#define _GAUSSIAN3D_

class Gaussian3D {
    double orb_exp;
    double amp;
    short ang[4];
    struct spatial::Vec3 r0;
    public:
    Gaussian3D(
        double orb_exponent,
        double amplitude, 
        short ang1, short ang2, short ang3,
        const spatial::Vec3 &position);
    Gaussian1D get_gaussian1d(int index) const;
    spatial::Vec3 position() const;
    double orbital_exponent() const;
    spatial::Vec3 angular() const;
    double amplitude() const;
};

struct spatial::Vec3 product_center(const Gaussian3D &g, const Gaussian3D &h);

#endif
