from sympy import Symbol, Function
import re
from time import perf_counter


boys_func = Function('bf')
power = Function('pw')


GLSL_BF_RECURSE2ZERO_STRING = """
void bFValuesRecurseTo0(
{}    float x, 
{}) {{
    vec4 res;
    float val = sqrt(PI)*erf(sqrt(x))/(2.0*sqrt(x));
    float bx = 0.5*exp(-x)/x;
    for (int nIter = 1; nIter <= n{}; nIter++) {{
        val = ((float(nIter) - 0.5)/x)*val - bx;
{}
    }}
}}
"""

GLSL_BF_RECURSE_UP_STRING = """
void bFValuesRecurseUp(
{}    float x,
{}) {{
    complex cVal = computeBFRecurseUpwardsIntermediary(x);
    float val = (mul(complex(0.5, 0.0), cVal)).r;
    vec4 res;
    for (int nIter = 12; nIter > n0; nIter--) {{
        val = x/(float(nIter) - 0.5)*val 
            + exp(-x)/(2.0*(float(nIter) - 0.5));
{}
    }}
}}
"""

GLSL_BF_VALUES_STRING = """
void bFValues(
{}    float x, 
{}) {{
    if (abs(x) < 1e-60) {{
{}
    }}
    if (abs(x) < RECURSE_DIR_TURN_POINT)
        bFValuesRecurseUp({}, x, {});
    else
        bFValuesRecurseTo0({}, x, {});
}}
"""


def coulomb_coefficient(i, j, k, n, e, x, y, z):
    if i == 0 and j == 0 and k == 0:
        return (-2)**n*power(e, n) * boys_func(e * (x**2 + y**2 + z**2), n)
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
x6, y6, z6 = Symbol('x6'), Symbol('y6'), Symbol('z6')


def print_cpp_implementation(min_angular, max_angular):
    t1 = perf_counter()
    for i in range(max_angular + 1):
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
                    val = val.subs(e**(n + int(pow_val)), 
                                   f'e_pow_n_plus_{pow_val}')
                    val = val.subs((-2.0*e)**(n + int(pow_val)), 
                                   f'neg2_e_pow_n_plus_{pow_val}')
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


def print_glsl_implementation(min_angular, max_angular):
    t1 = perf_counter()
    for i in range(min_angular, max_angular + 1):
        for j in range(0, i+1):
            for k in range(0, j+1):
                val = coulomb_coefficient(
                    i, j, k, 0, e, x, y, z).subs(
                        x**2 + y**2 + z**2, r2
                        ).simplify().factor().subs(
                            x**6, x6
                        ).subs(
                            y**6, y6
                        ).subs(
                            z**6, z6
                        ).subs(
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
                        ).simplify()
                # val = val.subs((-2.0)**n, 'neg2_pow_n')
                rep_val = val
                for pow_val in range(5*max_angular, 0, -1):
                    rep_val = rep_val.subs(boys_func(e*r2, pow_val), f'bF{pow_val}')
                    rep_val = rep_val.subs(power(e, pow_val), f'ep{pow_val}')
                bf_set = set(re.findall('bF[0-9]+', str(rep_val)))
                bf_int_vals = [int(e[2:]) for e in bf_set]
                bf_int_vals.sort()
                if len(bf_set) > 1:
                    val = rep_val
                if len(bf_set) == 2:
                    a, b = bf_int_vals
                    vec_line = f'vec2 vecBF = vecBFValues(e*r2, ivec2({a}, {b}));'
                    def_line = f'float bF{a} = vecBF[0], bF{b} = vecBF[1];'
                elif len(bf_set) == 3:
                    a, b, c = bf_int_vals
                    vec_line = f'vec3 vecBF = vecBFValues(e*r2, ivec3({a}, {b}, {c}));'
                    def_line = f'float bF{a} = vecBF[0], bF{b} = vecBF[1], bF{c} = vecBF[2];'
                elif len(bf_set) == 4:
                    a, b, c, d = bf_int_vals
                    vec_line = f'vec4 vecBF = vecBFValues(e*r2, ivec4({a}, {b}, {c}, {d}));'
                    def_line = f'float bF{a} = vecBF[0], bF{b} = vecBF[1], bF{c} = vecBF[2], bF{d} = vecBF[3];'
                elif len(bf_set) >= 5:
                    def_line = ''
                    for iter, value in enumerate(bf_int_vals):
                        if iter == 0:
                            def_line += f'float bF{value} = 0.0, '
                        elif iter < (len(bf_int_vals) - 1):
                            def_line += f'bF{value} = 0.0, '
                        else:
                            def_line += f'bF{value} = 0.0;'
                    eval_line = 'bFValues('
                    for value in bf_int_vals:
                        eval_line += f'bF{value}, '
                    eval_line += 'e*r2, '
                    for iter, value in enumerate(bf_int_vals):
                        eval_line += f'{value}' \
                            + (', ' if iter < len(bf_int_vals) - 1 else ');')
                # print(bf_int_vals)

                print(f'        if (hexIJK == 0x{i}{j}{k})' + ' {')
                if (val != 0):

                    str_expr = str(val)
                    if len(bf_set) >= 2 and len(bf_set) <= 4:
                        tmp = 12*' ' + vec_line + '\n'
                        tmp += 12*' ' + def_line + '\n'
                        str_expr = tmp + '            return ' + str_expr + ';'
                        print(str_expr)
                    elif len(bf_set) >= 5:
                        tmp = 12*' ' + def_line + '\n'
                        tmp += 12*' ' + eval_line + '\n'
                        str_expr = tmp + '            return ' + str_expr + ';'
                        print(str_expr)
                    else:
                        print('            return ' + str_expr + ';')
                    print('        }')
                else:
                    print('            return 0.0;')
    t2 = perf_counter()
    print('Time taken: ', t2 - t1)



def get_glsl_helper_functions(amount):
    y_args = ''
    n_args = ''
    set_vals = ''
    for i in range(amount):
        y_args += f'    inout float y{i},\n'
        n_args += f'    int n{i}' + (',\n' if i < amount - 1 else '')
        set_vals += ' '*8 + f'if (nIter == n{i})\n'
        set_vals += ' '*12 + f'y{i} = val;' \
            + ('\n' if i < amount - 1 else '')
    print(GLSL_BF_RECURSE2ZERO_STRING.format(
        y_args, n_args, amount - 1, set_vals))
    print()
    set_vals2 = ''
    for i in range(amount):
        set_vals2 += ' '*8 + f'if (nIter == (n{i} + 1))\n'
        set_vals2 += ' '*12 + f'y{i} = val;' \
            + ('\n' if i < amount - 1 else '')
    print(GLSL_BF_RECURSE_UP_STRING.format(y_args, n_args, set_vals2))
    print()
    y_args2 = ''
    n_args2 = ''
    set_vals3 = ''
    for i in range(amount):
        y_args2 += f'y{i}' + (', ' if i < amount - 1 else '')
        n_args2 += f'n{i}' + (', ' if i < amount - 1 else '')
        set_vals3 += ' '*8 + f'y{i} = 1.0/(2.0*float(n{i}) + 1.0);\n'
    print(GLSL_BF_VALUES_STRING.format(
        y_args, n_args, set_vals3, y_args2, n_args2, y_args2, n_args2))


print_glsl_implementation(0, 6)
