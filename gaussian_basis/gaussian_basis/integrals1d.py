"""Compute the various integral relations between Gaussian functions
of the form a*exp(-alpha*x^2).

This is indebted to the following article:

    Joshua Goings,
    A (hopefully) gentle guide to the computer implementation 
    of molecular integrals. 2017.
    https://joshuagoings.com/2017/04/28/integrals/

"""
import numpy as np
from scipy.special import hyp1f1
from .gaussian1d import Gaussian1D


def boys_func(x: float, n: int) -> float:
    """The Boys function is used to find the Coulomb coefficients, which
    is in turn used to compute integrals involving the Coulomb potential.
    See the section "Nuclear attraction integrals" from this article
    by Joshua Goings:
        https://joshuagoings.com/2017/04/28/integrals/.
    """
    return hyp1f1(n + 0.5, n + 1.5, -x)/(2.0*n + 1.0)


def overlap_coefficient(n: int, g1: Gaussian1D, g2: Gaussian1D) -> float:
    """Obtain the overlap coefficients between two 1D Gaussians.

    Refer to the section "Overlap Integrals" from Joshua Goings' blog post
    here: https://joshuagoings.com/2017/04/28/integrals/.
    """
    e1 = g1.orbital_exponent()
    e2 = g2.orbital_exponent()
    r1 = g1.position()
    r2 = g2.position()
    r21 = r1 - r2
    if n < 0 or n > (g1.angular() + g2.angular()):
        return 0.0
    elif g1.angular() == g2.angular() == n == 0:
        return np.exp(-(r1 - r2)**2*(e1*e2/(e1 + e2)))
    elif g1.angular() == 0:
        return (
            0.5/(e1 + e2)*overlap_coefficient(n-1, g1, g2-1)
            + (e1*1.0*r21)/(e1 + e2)*overlap_coefficient(n, g1, g2-1)
            + (n + 1)*overlap_coefficient(n+1, g1, g2-1)
        )
    else:
        return (
            0.5/(e1 + e2)*overlap_coefficient(n-1, g1-1, g2)
            - (1.0*e2*r21)/(e1 + e2)*overlap_coefficient(n, g1-1, g2)
            + (n + 1)*overlap_coefficient(n+1, g1-1, g2)
        )


def overlap1d(g1: Gaussian1D, g2: Gaussian1D) -> float:
    return overlap_coefficient(0, g1, g2) * \
           np.sqrt(np.pi/(g1.orbital_exponent() + g2.orbital_exponent()))


def laplacian1d(g1: Gaussian1D, g2: Gaussian1D) -> float:
    """Laplacian integral of two 1D Gaussians.

    Refer to the section "Kinetic energy integrals" from Joshua Goings'
    article: https://joshuagoings.com/2017/04/28/integrals/.
    """
    a2 = g2.angular()
    e2 = g2.orbital_exponent()
    return (a2*(a2-1)*overlap1d(g1, g2-2)
            - 2.0*e2*(2*a2+1)*overlap1d(g1, g2)
            + 4.0*e2**2*overlap1d(g1, g2+2))


def coulomb_coefficient(i: int, j: int, k: int, n: int,
                        orb_exp: float, r12: np.ndarray):
    """Compute the Coulomb coefficients. This is used in integrals
    that involve the Coulomb potential, such as the
    nuclear and repulsion-exchange integrals. Refer to the section
    "Nuclear attraction integrals" from Joshua Goings' blog post:
    https://joshuagoings.com/2017/04/28/integrals/.
    """
    if i == j == k == 0:
        return (-2*orb_exp)**n*boys_func(orb_exp*(r12 @ r12), n)
    elif i < 0 or j < 0 or k < 0:
        return 0.0
    elif j == k == 0:
        return (i-1)*coulomb_coefficient(i-2, j, k, n+1, orb_exp, r12) \
               + r12[0]*coulomb_coefficient(i-1, j, k, n+1, orb_exp, r12)
    elif k == 0:
        return (j-1)*coulomb_coefficient(i, j-2, k, n+1, orb_exp, r12) \
               + r12[1]*coulomb_coefficient(i, j-1, k, n+1, orb_exp, r12)
    else:
        return (k-1)*coulomb_coefficient(i, j, k-2, n+1, orb_exp, r12) \
               + r12[2]*coulomb_coefficient(i, j, k-1, n+1, orb_exp, r12)


