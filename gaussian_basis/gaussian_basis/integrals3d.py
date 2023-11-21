import numpy as np
from typing import List, Union
from .gaussian3d import Gaussian3D, product_center
from .integrals1d import *


def overlap(g1: Gaussian3D, g2: Gaussian3D) -> float:
    return g1.amplitude()*g2.amplitude() * \
           overlap1d(g1.get_gaussian1d(0), g2.get_gaussian1d(0)) \
           * overlap1d(g1.get_gaussian1d(1), g2.get_gaussian1d(1)) \
           * overlap1d(g1.get_gaussian1d(2), g2.get_gaussian1d(2))


def kinetic(g1: Gaussian3D, g2: Gaussian3D) -> float:
    amplitude1 = g1.amplitude()
    amplitude2 = g2.amplitude()
    g1x = g1.get_gaussian1d(0)
    g1y = g1.get_gaussian1d(1)
    g1z = g1.get_gaussian1d(2)
    g2x = g2.get_gaussian1d(0)
    g2y = g2.get_gaussian1d(1)
    g2z = g2.get_gaussian1d(2)
    return -0.5*(
            laplacian1d(g1x, g2x)*overlap1d(g1y, g2y)*overlap1d(g1z, g2z)
            + overlap1d(g1x, g2x)*laplacian1d(g1y, g2y)*overlap1d(g1z, g2z)
            + overlap1d(g1x, g2x)*overlap1d(g1y, g2y)*laplacian1d(g1z, g2z)
            )*amplitude1*amplitude2


def nuclear_single_charge(g: Gaussian3D, h: Gaussian3D, r) -> float:
    if r.shape != (3,):
        raise Exception
    gx, gy, gz = g.get_gaussian1d(0), g.get_gaussian1d(1), g.get_gaussian1d(2)
    hx, hy, hz = h.get_gaussian1d(0), h.get_gaussian1d(1), h.get_gaussian1d(2)
    r2 = product_center(g, h)
    r12 = r2 - r
    orb_exp = g.orbital_exponent() + h.orbital_exponent()
    val = 0.0
    for i in range(gx.angular() + hx.angular() + 1):
        for j in range(gy.angular() + hy.angular() + 1):
            for k in range(gz.angular() + hz.angular() + 1):
                val += overlap_coefficient(i, gx, hx) \
                       * overlap_coefficient(j, gy, hy) \
                       * overlap_coefficient(k, gz, hz) \
                       * coulomb_coefficient(i=i, j=j, k=k, n=0,
                                             orb_exp=orb_exp, r12=r12)
    return -g.amplitude()*h.amplitude()*2.0*np.pi*val/orb_exp


def nuclear(g1: Gaussian3D, g2: Gaussian3D,
            nuc_loc_charge_lst: List[List[Union[np.ndarray, int]]]) -> float:
    return sum([loc_charge[1]*nuclear_single_charge(g1, g2, loc_charge[0])
                for loc_charge in nuc_loc_charge_lst])


def repulsion_inner(g2: Gaussian3D, h2: Gaussian3D,
                    ix: int, iy: int, iz: int,
                    orb_exp: float, r12: np.ndarray) -> float:
    val = 0.0
    g2x = g2.get_gaussian1d(0)
    g2y = g2.get_gaussian1d(1)
    g2z = g2.get_gaussian1d(2)
    h2x = h2.get_gaussian1d(0)
    h2y = h2.get_gaussian1d(1)
    h2z = h2.get_gaussian1d(2)
    for jx in range(g2x.angular() + h2x.angular() + 1):
        for jy in range(g2y.angular() + h2y.angular() + 1):
            for jz in range(g2z.angular() + h2z.angular() + 1):
                val += (-1.0)**(jx + jy + jz) * \
                       overlap_coefficient(jx, g2x, h2x) \
                       * overlap_coefficient(jy, g2y, h2y) \
                       * overlap_coefficient(jz, g2z, h2z) \
                       * coulomb_coefficient(i=ix+jx, j=iy+jy, k=iz+jz, n=0,
                                             orb_exp=orb_exp, r12=r12)
    return val


def repulsion(g1: Gaussian3D, h1: Gaussian3D,
              g2: Gaussian3D, h2: Gaussian3D) -> float:
    amplitude = g1.amplitude()*g2.amplitude()*h1.amplitude()*h2.amplitude()
    if amplitude == 0.0:
        return 0.0
    val = 0.0
    g1x = g1.get_gaussian1d(0)
    g1y = g1.get_gaussian1d(1)
    g1z = g1.get_gaussian1d(2)
    h1x = h1.get_gaussian1d(0)
    h1y = h1.get_gaussian1d(1)
    h1z = h1.get_gaussian1d(2)
    orb_exp1 = g1.orbital_exponent() + h1.orbital_exponent()
    orb_exp2 = g2.orbital_exponent() + h2.orbital_exponent()
    orb_exp = orb_exp1*orb_exp2/(orb_exp1 + orb_exp2)
    r12 = product_center(g1, h1) - product_center(g2, h2)
    for ix in range(g1x.angular() + h1x.angular() + 1):
        for iy in range(g1y.angular() + h1y.angular() + 1):
            for iz in range(g1z.angular() + h1z.angular() + 1):
                val += 2.0*np.pi**(5.0/2.0) \
                       / (orb_exp1*orb_exp2*np.sqrt(orb_exp1 + orb_exp2)) \
                       * overlap_coefficient(ix, g1x, h1x) \
                       * overlap_coefficient(iy, g1y, h1y)\
                       * overlap_coefficient(iz, g1z, h1z) \
                       * repulsion_inner(g2=g2, h2=h2,
                                         ix=ix, iy=iy, iz=iz,
                                         orb_exp=orb_exp, r12=r12)
    return val*amplitude
