#include "gaussian1d.hpp"
#include "spatial.hpp"

#ifndef _GAUSSIAN3D_
#define _GAUSSIAN3D_

class Gaussian3D {
    double orb_exp;
    double amp;
    short ang[4];
    struct spatial::Vector r0;
    public:
    Gaussian3D(
        double orb_exponent,
        double amplitude, 
        short ang1, short ang2, short ang3,
        const spatial::Vector &position);
    Gaussian1D get_x() const;
    Gaussian1D get_y() const;
    Gaussian1D get_z() const;
    Gaussian1D get_gaussian1d(int index) const;
    spatial::Vector position() const;
    double orbital_exponent() const;
    spatial::Vector angular() const;
    double amplitude() const;
};

struct spatial::Vector product_center(const Gaussian3D &g, const Gaussian3D &h);

#endif
