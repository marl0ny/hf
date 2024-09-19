"""Use Sympy to compute the repulsion-exchange integrals for spherically
symmetric wave functions. Here closed shell is assumed.

Obtaining an expression for these repulsion-exchange integrals requires the
spherical harmonics expansion of the Coulomb potential, which can be found
in [1]. One then obtains integrals over the product of four spherical
harmonic functions, where Sympy's Gaunt function [2] is used to solve for
them, using Sympy's spherical harmonics functions [3].

    1.  Boudreau J., Swanson E. Quantum Mechanics II-many body systems,
        pg 813-814. In Applied Computational Physics. Oxford University Press.

    2.  Gaunt,
        https://docs.sympy.org/latest/modules/physics/wigner.html\\
            #sympy.physics.wigner.gaunt

    3.  Spherical Harmonics,
        https://docs.sympy.org/latest/modules/functions/\\
            special.html#spherical-harmonics
"""
import sympy
from sympy.physics.wigner import gaunt
from sympy import Symbol, Function
from sympy import pi, integrate


# Spherical harmonics function
def Y(L, m, theta, phi):
    return (sympy.functions
            .special
            .spherical_harmonics.Ynm(L, m, theta, phi).expand(func=True))


# r< and r> expressions that come from
# the expansion of the Coulomb potential.
r_min = Function('r_min')
r_max = Function('r_max')

# Radial function of the orbital
radial_1s = Function('radial_1s')
radial_2s = Function('radial_2s')
radial_2p = Function('radial_2p')

def radial(n: int, L: int, r):
    if n == 1:
        return radial_1s(r)
    elif n == 2:
        if L == 0:
            return radial_2s(r)
        elif L == 1:
            return radial_2p(r)
    else:
        raise NotImplementedError

# r2 is the variable for the radial
# part that is being integrated over in the
# exchange integral
r, r2 = Symbol('r'), Symbol('r2')
phi = Symbol('phi')
theta = Symbol('theta')

# n_j: the energy level for the orbital
# that "feels" the exchange force from
# the other orbitals.
# L_j and m_j are its angular and magnetic
# quantum numbers respectively.
n_j, L_j, m_j = 2, 0, 0

expr = 0

# n_i: the energy level for the orbital
# that is "giving off" the exchange force
for n_i in [1, 2]:
    for L_i in range(n_i):
        for m_i in range(-L_i, L_i+1):
            # print(L_i, m_i)
            for lambda_ in range(0, 10):
                for mu in range(-lambda_, lambda_+1):
                    f1 = -1 if abs(mu) % 2 == 1 else 1
                    f2 = -1 if abs(m_i) % 2 == 1 else 1
                    harmonic3_int = \
                        f1*f2*gaunt(lambda_, L_i, L_j, -mu, -m_i, m_j)
                    if harmonic3_int != 0:
                        expr += (r_min(r, r2)**(lambda_)
                                 /r_max(r, r2)**(lambda_ + 1)
                                 *radial(n_i, L_i, r2)*radial(n_j, L_j, r2)
                                 *4*pi/(2*lambda_ + 1)
                                 *harmonic3_int
                                 *Y(L_i, m_i, theta, phi)
                                 *Y(lambda_, mu, theta, phi)
                                 *radial(n_i, L_i, r)
                                 /Y(L_j, m_j, theta, phi))


print(f'For orbital n = {n_j}, L = {L_j}, m = {m_j}')
expr = integrate(expr.simplify().expand(), r2).expand()
# print(expr)

from sympy import init_printing
init_printing()

try:
    from IPython import display
    display.display(expr)
except ImportError:
    print(expr)