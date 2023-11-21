#include "gaussian1d.hpp"
#include "integrals1d.hpp"
#include "gaussian3d.hpp"
#include <cmath>

#define PI 3.141592653589793


double overlap(const Gaussian3D &g1, const Gaussian3D &g2) {
    return g1.amplitude()*g2.amplitude() *
        overlap1d(g1.get_gaussian1d(0), g2.get_gaussian1d(0))
        * overlap1d(g1.get_gaussian1d(1), g2.get_gaussian1d(1))
        * overlap1d(g1.get_gaussian1d(2), g2.get_gaussian1d(2));
}

double kinetic(const Gaussian3D &g1, const Gaussian3D &g2) {
    double amplitude1 = g1.amplitude();
    double amplitude2 = g2.amplitude();
    Gaussian1D g1x = g1.get_gaussian1d(0);
    Gaussian1D g1y = g1.get_gaussian1d(1);
    Gaussian1D g1z = g1.get_gaussian1d(2);
    Gaussian1D g2x = g2.get_gaussian1d(0);
    Gaussian1D g2y = g2.get_gaussian1d(1);
    Gaussian1D g2z = g2.get_gaussian1d(2);
    return -0.5*(
            laplacian1d(g1x, g2x)*overlap1d(g1y, g2y)*overlap1d(g1z, g2z)
            + overlap1d(g1x, g2x)*laplacian1d(g1y, g2y)*overlap1d(g1z, g2z)
            + overlap1d(g1x, g2x)*overlap1d(g1y, g2y)*laplacian1d(g1z, g2z)
            )*amplitude1*amplitude2;
}

double nuclear_single_charge(const Gaussian3D &g, const Gaussian3D &h,
                             struct Vec3 r) {
    Gaussian1D gx = g.get_gaussian1d(0);
    Gaussian1D gy = g.get_gaussian1d(1);
    Gaussian1D gz = g.get_gaussian1d(2);
    Gaussian1D hx = h.get_gaussian1d(0);
    Gaussian1D hy = h.get_gaussian1d(1);
    Gaussian1D hz = h.get_gaussian1d(2);
    Vec3 r2 = product_center(g, h);
    Vec3 r12 = r2 - r;
    double orb_exp = g.orbital_exponent() + h.orbital_exponent();
    double val = 0.0;
    for (int i = 0; i < (gx.angular() + hx.angular() + 1); i++) {
        for (int j = 0; j < (gy.angular() + hy.angular() + 1); j++) {
            for (int k = 0; k < (gz.angular() + hz.angular() + 1); k++) {
                val += overlap_coefficient(i, gx, hx)
                       * overlap_coefficient(j, gy, hy)
                       * overlap_coefficient(k, gz, hz)
                       * coulomb_coefficient(i, j, k, 0,
                                             orb_exp, r12);
            }
        }
    }
    return -g.amplitude()*h.amplitude()*2.0*PI*val/orb_exp;

}

static double repulsion_inner(const Gaussian3D &g2, const Gaussian3D &h2,
                              int ix, int iy, int iz,
                              double orb_exp, const Vec3 &r12) {
    double val = 0.0;
    Gaussian1D g2x = g2.get_gaussian1d(0);
    Gaussian1D g2y = g2.get_gaussian1d(1);
    Gaussian1D g2z = g2.get_gaussian1d(2);
    Gaussian1D h2x = h2.get_gaussian1d(0);
    Gaussian1D h2y = h2.get_gaussian1d(1);
    Gaussian1D h2z = h2.get_gaussian1d(2);
    for (int jx = 0; jx < (g2x.angular() + h2x.angular() + 1); jx++) {
        // double overlap_x = overlap_coefficient(jx, g2x, h2x);
        for (int jy = 0; jy < (g2y.angular() + h2y.angular() + 1); jy++) {
            // double overlap_y = overlap_coefficient(jy, g2y, h2y);
            for (int jz = 0; jz < (g2z.angular() + h2z.angular() + 1); jz++) {
                val += pow(-1.0, jx + jy + jz) *
                       overlap_coefficient(jx, g2x, h2x)
                       * overlap_coefficient(jy, g2y, h2y)
                       * overlap_coefficient(jz, g2z, h2z)
                       * coulomb_coefficient(ix+jx, iy+jy, iz+jz, 0,
                                             orb_exp, r12);
            }
        }
    }
    return val;
}

double repulsion(const Gaussian3D &g1,
                 const Gaussian3D &h1,
                 const Gaussian3D &g2,
                 const Gaussian3D &h2) {
    double amplitude = g1.amplitude()*g2.amplitude()
                        *h1.amplitude()*h2.amplitude();
    if (amplitude == 0.0)
        return 0.0;
    double val = 0.0;
    Gaussian1D g1x = g1.get_gaussian1d(0);
    Gaussian1D g1y = g1.get_gaussian1d(1);
    Gaussian1D g1z = g1.get_gaussian1d(2);
    Gaussian1D h1x = h1.get_gaussian1d(0);
    Gaussian1D h1y = h1.get_gaussian1d(1);
    Gaussian1D h1z = h1.get_gaussian1d(2);
    double orb_exp1 = g1.orbital_exponent() + h1.orbital_exponent();
    double orb_exp2 = g2.orbital_exponent() + h2.orbital_exponent();
    double orb_exp = orb_exp1*orb_exp2/(orb_exp1 + orb_exp2);
    struct Vec3 r12 = product_center(g1, h1) - product_center(g2, h2);
    for (int ix = 0; ix < (g1x.angular() + h1x.angular() + 1); ix++) {
        // double overlap_x = overlap_coefficient(ix, g1x, h1x);
        for (int iy = 0; iy < (g1y.angular() + h1y.angular() + 1); iy++) {
            // double overlap_y = overlap_coefficient(iy, g1y, h1y); 
            for (int iz = 0; iz < (g1z.angular() + h1z.angular() + 1); iz++) {
                val += 2.0*pow(PI, (5.0/2.0))
                       / (orb_exp1*orb_exp2*sqrt(orb_exp1 + orb_exp2))
                       * overlap_coefficient(ix, g1x, h1x)
                       * overlap_coefficient(iy, g1y, h1y)
                       * overlap_coefficient(iz, g1z, h1z) 
                       * repulsion_inner(g2, h2,
                                         ix, iy, iz,
                                         orb_exp, r12);
            }
        }
    }
    return val*amplitude;
}
