#include <cmath>
#include "gaussian1d.hpp"


Gaussian1D::Gaussian1D() {
    this->orb_exp = 0.0;
    this->r0 = 0.0;
    this->ang = 0;
}

Gaussian1D::Gaussian1D(double orb_exp, double r0, long ang) {
    this->orb_exp = orb_exp;
    this->r0 = r0;
    this->ang = ang;
}

double Gaussian1D::operator()(double x) const {
    return pow(x - r0, ang)*exp(-orb_exp*(x - r0)*(x - r0));
}

double Gaussian1D::position() const {
    return this->r0;
}
double Gaussian1D::orbital_exponent() const {
    return this->orb_exp;
}
long Gaussian1D::angular() const {
    return this->ang;
}

Gaussian1D Gaussian1D::operator+(int n) {
    return Gaussian1D(orb_exp, r0, ang + n);
}

Gaussian1D Gaussian1D::operator-(int n) {
    return Gaussian1D(orb_exp, r0, ang - n);
}
