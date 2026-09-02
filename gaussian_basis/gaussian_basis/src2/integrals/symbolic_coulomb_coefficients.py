from sympy import Symbol, Function
import re

boys_func = Function('bf')
power = Function('pw')

def coulomb_coefficient(i, j, k, n, e, x, y, z):
    if i == 0 and j == 0 and k == 0:
        return pow((-2.0 * e), n) * boys_func(e * (x**2 + y**2 + z**2), n)
    elif i < 0 or j < 0 or k < 0:
        return 0
    elif j == 0 and k == 0:
        return (i - 1.0) * coulomb_coefficient(i-2, j, k, n+1, e, x, y, z)\
                 + x*coulomb_coefficient(i-1, j, k, n+1, e, x, y, z)
    elif k == 0:
        return (j - 1.0) * coulomb_coefficient(i, j-2, k, n+1, e, x, y, z) \
                + y*coulomb_coefficient(i, j-1, k, n+1, e, x, y, z)
    else:
        return (k - 1.0) * coulomb_coefficient(i, j, k-2, n+1, e, x, y, z) \
                + z*coulomb_coefficient(i, j, k-1, n+1, e, x, y, z)

x = Symbol('x')
y = Symbol('y')
z = Symbol('z')
n = Symbol('n', integer=True)
e = Symbol('e')
r2 = Symbol('r2')
x2, y2, z2 = Symbol('x2'), Symbol('y2'), Symbol('z2')
x4, y4, z4 = Symbol('x4'), Symbol('y4'), Symbol('z4')

from time import perf_counter
t1 = perf_counter()
for i in range(5):
    for j in range(0, i+1):
        for k in range(0, j+1):
            val = coulomb_coefficient(
                i, j, k, n, e, x, y, z).subs(
                    x**2 + y**2 + z**2, r2
                    ).simplify().factor().subs(
                    # ).factor().subs(
                        x**4, x4
                    ).subs(
                        y**4, y4
                    ).subs(
                        z**4, z4
                    ).subs(
                        x**2, x2
                    ).subs(
                        y**2, y2
                    ).subs(
                        z**2, z2
                    )  # .simplify()
            val = val.subs((-2.0)**n, 'neg2_pow_n')
            for pow_val in range(10, 0, -1):
                val = val.subs(e**(n + int(pow_val)), f'e_pow_n_plus_{pow_val}')
                val = val.subs((-2.0*e)**(n + int(pow_val)), f'neg2_e_pow_n_plus_{pow_val}')
                if pow_val >= 2:
                    val = val.subs(e**int(pow_val), f'e_pow_{pow_val}')
            # print(f'    if (i == {i} && j == {j} && k == {k})')
            print(f'        case 0x{i}{j}{k}:')
            if (val != 0):

                str_expr = str(val)
                # str_expr = re.sub('\*\*\(', '_pow(', str_expr)
                # str_expr = re.sub('\*\*', '_pow', str_expr)
                print('            return ' + str_expr + ';')
                print('            break;')
            else:
                print('            return 0.0;')
t2 = perf_counter()
print('Time taken: ', t2 - t1)