import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


SQRT_2 = np.sqrt(2.0)
SQRT_PI = np.sqrt(np.pi)
SQRT_2PI = np.sqrt(2.0*np.pi)
SQRT_3PI = np.sqrt(3.0*np.pi)
SQRT_6PI = np.sqrt(6.0*np.pi)


def gaussian(x: np.ndarray, amp: float, orb_exp: float) -> np.ndarray:
    return amp*np.exp(-orb_exp*x**2)


def get_angular_number(orbital_name: str) -> int:
    letter = orbital_name[1]
    if letter == 's':
        return 0
    if letter == 'p':
        return 1
    if letter == 'd':
        return 2
    if letter == 'f':
        return 3


def get_slater_orbital(which: str, atomic_number: int,
                       r2: np.ndarray) -> np.ndarray:
    # Reference for the spherical harmonics
    # https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
    # http://hyperphysics.phy-astr.gsu.edu/hbase/quantum/hydwf.html
    slater_orbitals = {
        '1s': lambda z_n, r: z_n**(3.0/2.0)*np.exp(-z_n*r)/SQRT_PI,
        '2s': lambda z_n, r: z_n**(3.0/2.0
                                   )*np.exp(-0.5*z_n*r
                                            )*(2.0 - z_n*r)/(4.0*SQRT_2PI),
        '2p': lambda z_n, r: z_n**(5.0/2.0
                                   )*r*np.exp(-0.5*z_n*r)/(4.0*SQRT_2PI),
        '3s': lambda z_n, r: z_n**(3.0/2.0
                                   )*np.exp(-z_n*r/3.0
                                            )*(27.0 - 18.0*z_n*r
                                               + 2.0*z_n**2*r**2
                                               )/(81.0*SQRT_3PI),
        '3p': lambda z_n, r: z_n**(5.0/2.0
                                   )*SQRT_2*r*np.exp(-z_n*r/3.0
                                                     )*(6.0 - z_n*r
                                                        )/(81.0*SQRT_PI),
        '3d': lambda z_n, r: z_n**(5.0/2.0
                                   )*np.exp(-z_n*r/3.0)*(3.0*z_n*r**2
                                                         - z_n*r**2
                                                         )/(81.0*SQRT_6PI)
    }
    return slater_orbitals[which](atomic_number, r2)


def gaussian_fit_slater_orbital(which: str, number_of: int,
                                atomic_number: int,
                                r: np.ndarray, params=None
                               ) -> dict:
    slater_orbital = get_slater_orbital(which, atomic_number, r)
    an = get_angular_number(which)

    def fit_function(parameters: list) -> np.ndarray:
        gauss_list = []
        for j in range(len(parameters)//2):
            co1, ex1 = parameters[j*2], parameters[j*2 + 1]
            gauss_list.append(r**(an + 0)*gaussian(r, co1, ex1))
        return (slater_orbital - sum(gauss_list)).flatten()

    if params is None:
        params = []
        for _ in range(number_of):
            params.extend([1.0, 1.0])
    data = least_squares(fit_function, params,
                         ftol=1e-12, xtol=1e-12, gtol=1e-12)
    print(data['optimality'])
    ret_val = {"coefficients": [], "exponents": []}
    for k in range(len(data['x']) // 2):
        co2 = data['x'][2*k]
        ex2 = data['x'][2*k + 1]
        ret_val["coefficients"].append(co2)
        ret_val["exponents"].append(ex2)
    return ret_val


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    orbital_letters = ['1s', '2s', '2p', '3s', '3p', '3d']
    orbitals_data = {}
    r1 = np.linspace(0.0, 20.0, 500)
    r3 = np.linspace(0.0, 40.0, 500)
    gaussian_count = 3
    nuc_charge = 1
    for letter in orbital_letters:
        r = r1 if int(letter[0]) <= 2 else r3
        orbital_data = gaussian_fit_slater_orbital(letter, gaussian_count,
                                                   nuc_charge, r)
        orbitals_data[letter] = orbital_data
        gauss_sum = np.zeros([len(r)])
        for c, e in zip(orbital_data['coefficients'],
                        orbital_data['exponents']):
            gauss_sum += r**get_angular_number(letter)*gaussian(r, c, e)
        plt.title(f'Orbital {letter}')
        plt.plot(r, get_slater_orbital(letter, nuc_charge, r),
                                       label='Actual')
        plt.plot(r, gauss_sum, label='Gaussian fit')
        plt.legend()
        plt.show()
        plt.close()
    import json
    with open(f'../data/{nuc_charge}p1e-'
              f'{gaussian_count}gaussians.json', 'w') as f:
        json.dump(orbitals_data, f)


