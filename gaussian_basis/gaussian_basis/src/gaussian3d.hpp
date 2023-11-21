#include "gaussian1d.hpp"
#include "vec3.hpp"

#ifndef _GAUSSIAN3D_
#define _GAUSSIAN3D_

class Gaussian3D {
    double orb_exp;
    double amp;
    short ang[4];
    struct Vec3 r0;
    // Gaussian1D gaussian0;
    // Gaussian1D gaussian1;
    // Gaussian1D gaussian2;
    public:
    Gaussian3D(double amp, double orb_exp,
               struct Vec3 pos, struct Vec3 ang);
    Gaussian3D(double amp, double orb_exp, 
               double pos_x, double pos_y, double pos_z,
               double ang_x, double ang_y, double ang_z);
    Gaussian1D get_gaussian1d(int index) const;
    Vec3 position() const;
    double orbital_exponent() const;
    Vec3 angular() const;
    double amplitude() const;
};

struct Vec3 product_center(const Gaussian3D &g, const Gaussian3D &h);

#endif
