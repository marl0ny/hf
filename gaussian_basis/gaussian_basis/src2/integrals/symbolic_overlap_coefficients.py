from sympy import Symbol, symbols, Function
from sympy import exp
import re


def overlap_coefficient(n: int, a1: int, a2: int, r, e1, e2):
    if n < 0 or n > (a1 + a2):
        return 0
    elif a1 == a2 == n == 0:
        return exp(-r**2*(e1*e2/(e1 + e2)))
    elif a1 == 0:
        return 0.5/(e1 + e2)*overlap_coefficient(n-1, a1, a2-1, r, e1, e2) \
            + (e1*r)/(e1 + e2) \
                *overlap_coefficient(n, a1, a2-1, r, e1, e2) \
            + (n + 1)*overlap_coefficient(n+1, a1, a2-1, r, e1, e2)
    else:
        return 0.5/(e1 + e2)*overlap_coefficient(n-1, a1-1, a2, r, e1, e2) \
            - (e2*r)/(e1 + e2) \
                *overlap_coefficient(n, a1-1, a2, r, e1, e2) \
            + (n + 1)*overlap_coefficient(n+1, a1-1, a2, r, e1, e2)


if __name__ == '__main__':

    e1 = Symbol('e1')
    e2 = Symbol('e2')
    r = Symbol('r')
    for a1 in range(5):
        for a2 in range(5):
            for n in range(a1 + a2 + 1):
                print(f'        case 0x{n}{a1}{a2}:')
                expr = overlap_coefficient(n, a1, a2, r, e1, e2
                                        ).simplify()
                for i in range(9, 1, -1):
                    expr = expr.subs(e1**i, f'e1_{i}')
                    expr = expr.subs(e2**i, f'e2_{i}')
                    expr = expr.subs((e1 + e2)**i, f'e1e2_{i}')
                    expr = expr.subs(r**i, f'r{i}')
                print('            return ', str(expr) + ';')
                print('            break;')

