#include <cmath>
#include "gaussian3d.hpp"

/* Gaussian3D::Gaussian3D(double amp, double orb_exp, 
                       double pos_x, double pos_y, double pos_z,
                       double ang_x, double ang_y, double ang_z) {
    this->amp = amp;
    this->orb_exp = orb_exp;
    this->r0.x = pos_x;
    this->r0.y = pos_y;
    this->r0.z = pos_z;
    this->ang[0] = (short)ang_x;
    this->ang[1] = (short)ang_y;
    this->ang[2] = (short)ang_z;
    // gaussian0 = Gaussian1D(orb_exp, pos_x, ang_x);
    // gaussian1 = Gaussian1D(orb_exp, pos_y, ang_y);
    // gaussian1 = Gaussian1D(orb_exp, pos_z, ang_z);
}

Gaussian3D::Gaussian3D(double amp, double orb_exp, 
                       struct Vec3 pos, struct Vec3 ang) {
    Gaussian3D(amp, orb_exp, pos.x, pos.y, pos.z, ang.x, ang.y, ang.z);
}*/

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
