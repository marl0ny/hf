import sympy
from sympy.physics.wigner import gaunt
from sympy import Symbol, Function
from sympy import pi, integrate


CLOSED_SHELL = False


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
radial_3s = Function('radial_3s')
radial_3p = Function('radial_3p')
radial_4s = Function('radial_4s')
radial_4p = Function('radial_4p')
radial_4d = Function('radial_4d')


def radial(n, L, r):
    if n == 1:
        return radial_1s(r)
    elif n == 2:
        if L == 0:
            return radial_2s(r)
        elif L == 1:
            return radial_2p(r)
    elif n == 3:
        if L == 0:
            return radial_3s(r)
        elif L == 1:
            return radial_3p(r)
    elif n == 4:
        if L == 0:
            return radial_4s(r)
        elif L == 1:
            return radial_4p(r)
        elif L == 2:
            return radial_4d(r)
    else:
        raise NotImplementedError


# r2 is the variable for the radial
# part that is being integrated over in the
# exchange integral
r, r2 = Symbol('r'), Symbol('r2')
phi = Symbol('phi')
theta = Symbol('theta')


# Nested dictionary for keeping track of the
# principle, angular, magnetic, and spin quantum numbers
# for each of the orbitals.
# The top level keys are the principle quantum numbers.
# The keys for the second level of nesting
# are for the angular quantum numbers.
# The keys for the third and final level of the
# nested dictionary are for the magnetic quantum numbers,
# whose values are a list of the possible spin orbitals.
principle_angular_magnetic = {1: {0: {0: [-1, 1]}
                                 }, 
                              2: {0: {0: [-1, 1]}, 
                                  1: {-1: [-1, 1], 0: [-1, 1], 1: [-1, 1]}
                                 },
                              3: {0: {0: [-1, 1]}, 
                                  1: {-1: [-1, 1], 0: [-1, 1], 1: [-1, 1]}
                                 },
                              4: {0: {0: [-1, 1]}, 
                                  1: {-1: [-1, 1], 0: [-1, 1], 1: [-1, 1]},
                                  2: {-2: [-1, 1], -1: [-1, 1], 
                                      0: [-1, 1], 
                                      1: [-1, 1], 2: [-1, 1]}
                                 },
                             }


def get_repulsion_formula(n_i, L_i, m_i, s_i, lev_ang_mag_spin):
    """
    n_i: the energy level for the orbital
    that "feels" the repulsion
    force from the other orbitals.
    L_i, m_i, and s_i are its angular, magnetic, and spin
    quantum numbers respectively.
    lev_ang_mag_spin: nested dictionary containing
    the energy level, angular, magnetic, and spin quantum
    numbers of the other orbitals. 
    """
    expr = 0
    for n_j in lev_ang_mag_spin.keys():
        for L_j in lev_ang_mag_spin[n_j].keys():
            for m_j in lev_ang_mag_spin[n_j][L_j].keys():
                for s_j in lev_ang_mag_spin[n_j][L_j][m_j]:
                    for lambda_ in range(0, 5):
                        for mu in range(-lambda_, lambda_+1):
                            f1 = -1 if abs(mu) % 2 == 1 else 1
                            f2 = -1 if abs(m_j) % 2 == 1 else 1
                            harmonic3_int = \
                                f1*f2*gaunt(lambda_, L_j, L_j, 
                                            -mu, -m_j, m_j)
                            if harmonic3_int != 0:
                                term = (r_min(r, r2)**(lambda_)
                                        /r_max(r, r2)**(lambda_ + 1)
                                        *radial(n_j, L_j, r2)**2
                                        *4*pi/(2*lambda_ + 1)
                                        *harmonic3_int
                                        *Y(lambda_, mu, theta, phi)
                                        # *Y(L_i, m_i, theta, phi)
                                        *radial(n_i, L_i, r))
                                expr += term
    return integrate(expr.simplify().expand(), r2).expand()


def get_exchange_formula(n_i, L_i, m_i, s_i, lev_ang_mag_spin):
    """
    n_i: the energy level for the orbital
    that "feels" the exchange
    force from the other orbitals.
    L_i, m_i, and s_i are its angular, magnetic, and spin
    quantum numbers respectively.
    lev_ang_mag_spin: nested dictionary containing
    the energy level, angular, magnetic, and spin quantum
    numbers of the other orbitals.
    """
    expr = 0
    for n_j in lev_ang_mag_spin.keys():
        for L_j in lev_ang_mag_spin[n_j].keys():
            for m_j in lev_ang_mag_spin[n_j][L_j].keys():
                for s_j in lev_ang_mag_spin[n_j][L_j][m_j]:
                    for lambda_ in range(0, 5):
                        for mu in range(-lambda_, lambda_+1):
                            if s_i == s_j: 
                                f1 = -1 if abs(mu) % 2 == 1 else 1
                                f2 = -1 if abs(m_j) % 2 == 1 else 1
                                harmonic3_int = \
                                    f1*f2*gaunt(lambda_, L_j, L_i,
                                                -mu, -m_j, m_i)
                                if harmonic3_int != 0:
                                    term = (r_min(r, r2)**(lambda_)
                                            /r_max(r, r2)**(lambda_ + 1)
                                            *radial(n_j, L_j, r2)
                                            *radial(n_i, L_i, r2)
                                            *4*pi/(2*lambda_ + 1)
                                            *harmonic3_int
                                            *Y(lambda_, mu, theta, phi)
                                            *Y(L_j, m_j, theta, phi)
                                            *radial(n_j, L_j, r)
                                            /Y(L_i, m_i, theta, phi)
                                            )
                                    expr += term


    return integrate(expr.simplify().expand(), r2).expand()


from sympy import init_printing
init_printing()

try:
    from IPython import display
    print_func = display.display
except ImportError:
    print_func = print


for n_i in principle_angular_magnetic.keys():
        for L_i in principle_angular_magnetic[n_i].keys():
            for m_i in principle_angular_magnetic[n_i][L_i].keys():
                for s_i in principle_angular_magnetic[n_i][L_i][m_i]:
                # for s_i in [1]:
                    print(f'For orbital n = {n_i}, '
                          f'L = {L_i}, m = {m_i}, s = {s_i}:')
                    expr1 = get_repulsion_formula(
                        n_i, L_i, m_i, s_i, principle_angular_magnetic)
                    expr2 = get_exchange_formula(
                        n_i, L_i, m_i, s_i,
                        principle_angular_magnetic)
                    print('Repulsion: ')
                    print_func(expr1)
                    print('Exchange: ')
                    print_func(expr2)
                    print('')