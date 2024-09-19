#include <cmath>
#include "gaussian3d.hpp"


Gaussian1D Gaussian3D::get_gaussian1d(int index) const {
    if (index == 0) {
        return Gaussian1D(orb_exp, this->r0[0], (long)this->ang[0]);
        // return gaussian0;
    } else if (index == 1) {
        return Gaussian1D(orb_exp, this->r0[1], (long)this->ang[1]);
        // return gaussian1;
    } else if (index == 2) {
        return Gaussian1D(orb_exp, this->r0[2], (long)this->ang[2]);
        // return gaussian2;
    } else {
        return {};
    }
}

Vec3 Gaussian3D::position() const {
    return this->r0;
    // return {{{this->gaussian0.position(),
    //         this->gaussian1.position(),
    //         this->gaussian2.position()}}};
}

double Gaussian3D::orbital_exponent() const {
    return this->orb_exp;
}

Vec3 Gaussian3D::angular() const {
    return {{{
        (double)this->ang[0], (double)this->ang[1], (double)this->ang[2]}}};
    // return {{{(double)this->gaussian0.angular(),
    //         (double)this->gaussian1.angular(),
    //         (double)this->gaussian2.angular()}}};
}

double Gaussian3D::amplitude() const {
    return this->amp;
}

struct Vec3 product_center(const Gaussian3D &g, const Gaussian3D &h) {
    return ((g.orbital_exponent()*g.position())
            + (h.orbital_exponent()*h.position())) / 
           (g.orbital_exponent() + h.orbital_exponent());
}
