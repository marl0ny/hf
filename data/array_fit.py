import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


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


def fit_to_orbital(which: str, number_of: int,
                   r: np.ndarray, u: np.ndarray, params=None):
    an = get_angular_number(which)

    def fit_function(parameters: list) -> np.ndarray:
        gauss_list = []
        for j in range(len(parameters)//2):
            coefficient, exp_orb = parameters[j*2], parameters[j*2 + 1]
            gauss_list.append(r**(an+0)*gaussian(r, coefficient, exp_orb))
        return (u/r - sum(gauss_list)).flatten()

    if params is None:
        params = []
        for _ in range(number_of):
            params.extend([1.0, 1.0])

    data = least_squares(fit_function, params)
    print(data['optimality'])
    ret_val = {"coefficients": [], "exponents": []}
    for k in range(len(data['x']) // 2):
        c2 = data['x'][2*k]
        exp_orb2 = data['x'][2*k + 1]
        ret_val["coefficients"].append(c2)
        ret_val["exponents"].append(exp_orb2)
    return ret_val


if __name__ == '__main__':

    import sys
    import json
    import re

    filename = '10p10e_fd.json'
    number_of_gaussians = 4

    print(sys.argv)
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    if len(sys.argv) > 2:
        number_of_gaussians = int(sys.argv[2])

    string = ''
    with open(filename, 'r') as f:
        for line in f:
            string += line
        data = json.loads(string)
    e_count = int(re.search(r'[0-9]+e', filename).group(0)[:-1])
    p_count = int(re.search(r'[0-9]+p', filename).group(0)[:-1])
    gauss_data = {}
    show_r_scaled_plots = True
    for o in data.keys():
        r_ = np.array(data[o]['r'])
        values = np.array(data[o]['values'])
        gauss_data[o] = fit_to_orbital(o, number_of_gaussians, r_, values)
        gauss_sum = np.zeros([len(r_)])
        for c, e in zip(gauss_data[o]['coefficients'],
                        gauss_data[o]['exponents']):
            gauss_sum += r_**get_angular_number(o)*gaussian(r_, c, e)
        if show_r_scaled_plots:
            plt.plot(r_, r_*values / r_, label=f'Original: {o}')
            plt.plot(r_, r_*gauss_sum, label=f'Gaussian fit: {o}')
        else:
            plt.plot(r_, values/r_, label=f'Original: {o}')
            plt.plot(r_, gauss_sum, label=f'Gaussian fit: {o}')
    plt.legend()
    plt.show()
    plt.close()
    import json
    with open(f'../data/{p_count}p{e_count}e-'
              f'{number_of_gaussians}gaussians.json',
              'w') as f:
        json.dump(gauss_data, f)

