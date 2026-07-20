#include "gaussian3d.hpp"
#include "gaussian1d.hpp"


Gaussian3D::Gaussian3D(
    double orb_exponent,
    double amplitude, 
    short ang_x, short ang_y, short ang_z,
    const spatial::Vector &position) {
    this->orb_exp = orb_exponent;
    this->amp = amplitude;
    this->ang[0] = ang_x;
    this->ang[1] = ang_y;
    this->ang[2] = ang_z;
    this->r0 = position;
}

Gaussian1D Gaussian3D::get_gaussian1d(int index) const {
    if (index == 0) {
        return Gaussian1D(orb_exp, this->r0.x, (long)this->ang[0]);
        // return gaussian0;
    } else if (index == 1) {
        return Gaussian1D(orb_exp, this->r0.y, (long)this->ang[1]);
        // return gaussian1;
    } else if (index == 2) {
        return Gaussian1D(orb_exp, this->r0.z, (long)this->ang[2]);
        // return gaussian2;
    } else {
        return {};
    }
}

spatial::Vector Gaussian3D::position() const {
    return this->r0;
    // return {{{this->gaussian0.position(),
    //         this->gaussian1.position(),
    //         this->gaussian2.position()}}};
}

double Gaussian3D::orbital_exponent() const {
    return this->orb_exp;
}

spatial::Vector Gaussian3D::angular() const {
    return {{{
        .t=0.0,
        .x=(double)this->ang[0],
        .y=(double)this->ang[1], .z=(double)this->ang[2]}}};
    // return {{{(double)this->gaussian0.angular(),
    //         (double)this->gaussian1.angular(),
    //         (double)this->gaussian2.angular()}}};
}

double Gaussian3D::amplitude() const {
    return this->amp;
}

struct spatial::Vector product_center(const Gaussian3D &g, const Gaussian3D &h) {
    return ((g.orbital_exponent()*g.position())
            + (h.orbital_exponent()*h.position())) / 
           (g.orbital_exponent() + h.orbital_exponent());
}
