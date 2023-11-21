#ifndef _GAUSSIAN1D_
#define _GAUSSIAN1D_


class Gaussian1D {
    double r0;
    long ang;
    double orb_exp;
    public:
    Gaussian1D();
    Gaussian1D(double orb_exp, double r0, long ang);
    double operator()(double x) const;
    double position() const;
    double orbital_exponent() const;
    long angular() const;
    Gaussian1D operator+(int n);
    Gaussian1D operator-(int n);
};

#endif
