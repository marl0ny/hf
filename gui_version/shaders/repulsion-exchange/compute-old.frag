// #version 120

#if (__VERSION__ >= 330) || (defined(GL_ES) && __VERSION__ >= 300)
#define texture2D texture
#else
#define texture texture2D
#endif

#if (__VERSION__ > 120) || defined(GL_ES)
precision highp float;
#endif
 
#if __VERSION__ <= 120
varying vec2 UV;
#define fragColor gl_FragColor
#else
in vec2 UV;
out vec4 fragColor;
#endif


#define complex vec2

#define PI 3.141592653589793

#define RECURSE_DIR_TURN_POINT 4.5425955121971775

#define SWAP_F(a, b) float tmp_f = (a); (a) = (b); (b) = (tmp_f)
#define SWAP_I(a, b) int tmp_i = (a); (a) = (b); (b) = (tmp_i)

uniform sampler2D basisFunctionSpec1Tex;
uniform sampler2D basisFunctionSpec2Tex;
uniform sampler2D primitivesTex;
uniform sampler2D indicesTex;
uniform sampler2D previousSumTex;
uniform int numberOfBasisFunctions;
uniform int maxNumberOfPrimitives;
uniform ivec4 primitiveIndices;
uniform int maxAngularIter;

const int DEBUG_COULOMB = 1;
const int DEBUG_OVERLAP = 2;
uniform int debug;
uniform sampler2D debugIndTex;
uniform int debugCoulombFactor;
uniform float debugCoulombOrbExp;
uniform vec3 debugCoulombR;
uniform vec2 debugOverlapPositions;
uniform vec2 debugOverlapExponents;


complex WEIGHTS[26] = complex[](
    complex(7.071943132057001, 16.487291250752115), complex(3.64436324028985e-11, 2.6411751072107504e-11),
    complex(7.071943132057001, -16.487291250752115), complex(3.64436324028985e-11, -2.6411751072107504e-11),
    complex(-0.5714327171519163, 13.278579453233633), complex(1.8185250346753633e-07, -2.186045897139935e-06),
    complex(-0.5714327171519163, -13.278579453233633), complex(1.8185250346753633e-07, 2.186045897139935e-06),
    complex(-4.719302133039251, 9.983525711237103), complex(-0.0009948916927205575, -0.00023049079105203073),
    complex(-4.719302133039251, -9.983525711237103), complex(-0.0009948916927205575, 0.00023049079105203073),
    complex(-7.170466277289509, 6.671236083982077), complex(-0.025625216985879006, 0.03581833527487698),
    complex(-7.170466277289509, -6.671236083982077), complex(-0.025625216985879006, -0.03581833527487698),
    complex(-8.48997470547247, 3.343480416846749), complex(0.16506801544880723, 0.32273964471776045),
    complex(-8.48997470547247, -3.343480416846749), complex(0.16506801544880723, -0.32273964471776045),
    complex(36.56441436315097, 0.0), complex(-2.0104641661565164e-26, 0.0),
    complex(-3.2424239255921954, 0.0), complex(-0.0003956353695504208, 0.0),
    complex(-8.906604773310075, 0.0), complex(0.7234994580508529, 0.0)
);

complex cExp(complex z) {
    return exp(z.x)*complex(cos(z.y), sin(z.y));
}

complex mul(complex w, complex z) {
    return complex(w[0]*z[0] - w[1]*z[1], w[0]*z[1] + w[1]*z[0]);
}

complex inv(complex w) {
    return complex(w.x, -w.y)/(w.x*w.x + w.y*w.y);
}

/* Please read
"Simple approximations for the error function and its inverse"
by Vedder J.
Am. J. Phys. 55, 762–763 (1987)
*/
float erf(float x) {
    #if __VERSION__ > 150
    // float arg = (167.0/148.0)*x + (11.0/109.0)*(x*x*x);
    float a = (167.0/148.0), b = (11.0/109.0), c = 0.0;
    // float a = 1.12860381e+00, b= 1.02606209e-01, c = -1.08902527e-03;
    float arg = a*x + b*(x*x*x) + c*(x*x*x*x*x);
    if (arg > 10.0)
        return 1.0;
    else if (arg < -10.0)
        return -1.0;
    // s = exp(ix) - exp(-ix)/(2i)
    // c = (exp(ix) + exp(-ix))/2
    return tanh(arg);
    #else
    // TODO!!!
    return 0.0;
    #endif
}

complex computeBFRecurseUpwardsIntermediary(float x) {
    complex cVal = complex(0.0, 0.0);
    complex cX = complex(x, 0.0);
    complex one = complex(1.0, 0.0);
    complex oneHalf = complex(0.5, 0.0);
    for (int i = 0; i < 13; i++) {
        complex expVal = WEIGHTS[2*i];
        complex weight = WEIGHTS[2*i + 1];
        cVal += mul(weight,
            mul(-mul(cExp(expVal), inv(cX + expVal)),
                (cExp(-(cX + expVal)) - one)));
    }
    return cVal;
}

/* See "A fast algorithm for computing the Boys function"
by Gregory Beylkin and Sandeep Sharma, particularly the section containing
equation (3) for recursing to the maximum n value. */
float boysFuncRecurseUpwards(float x, int n) {
    complex cVal = computeBFRecurseUpwardsIntermediary(x);
    float val = (mul(complex(0.5, 0.0), cVal)).r;
    for (int n_iter = 12; n_iter > n; n_iter--) {
        val = x/(float(n_iter) - 0.5)*val 
            + exp(-x)/(2.0*(float(n_iter) - 0.5));
    }
    return val;
}

/* See "A fast algorithm for computing the Boys function"
by Gregory Beylkin and Sandeep Sharma, particularly the section containing
equation (4) for the n = 0 base if (hex_n_a1_a2 == , and (2) for the recursion relation
that goes to zero. */
float boysFuncRecurseToZero(float x, int n) {
    float val = sqrt(PI)*erf(sqrt(x))/(2.0*sqrt(x));
    float bx = 0.5*exp(-x)/x;
    for (int n_iter = 1; n_iter <= n; n_iter++)
        val = ((float(n_iter) - 0.5)/x)*val - bx;
    return val;
}

vec2 vecBFValuesRecurseTo0(float x, ivec2 nValues) {
    vec2 res;
    float val = sqrt(PI)*erf(sqrt(x))/(2.0*sqrt(x));
    float bx = 0.5*exp(-x)/x;
    for (int nIter = 1; nIter <= nValues[1]; nIter++) {
        val = ((float(nIter) - 0.5)/x)*val - bx;
        if (nIter == nValues[0])
            res[0] = val;
        if (nIter == nValues[1])
            res[1] = val;
    }
    return res;
}

vec2 vecBFValuesRecurseUp(float x, ivec2 nValues) {
    complex cVal = computeBFRecurseUpwardsIntermediary(x);
    float val = (mul(complex(0.5, 0.0), cVal)).r;
    vec2 res;
    for (int nIter = 12; nIter > nValues[0]; nIter--) {
        val = x/(float(nIter) - 0.5)*val 
            + exp(-x)/(2.0*(float(nIter) - 0.5));
        if (nIter == (nValues[0] + 1))
            res[0] = val;
        if (nIter == (nValues[1] + 1))
            res[1] = val;
    }
    return res;
}

vec2 vecBFValues(float x, ivec2 nValues) {
    if (abs(x) < 1e-60)
        return vec2(
            1.0/(2.0*float(nValues[0]) + 1.0),
            1.0/(2.0*float(nValues[1]) + 1.0));
    if (abs(x) < RECURSE_DIR_TURN_POINT) {
        return vecBFValuesRecurseUp(x, nValues);
    } else {
        return vecBFValuesRecurseTo0(x, nValues);
    }

}

vec3 vecBFValuesRecurseTo0(float x, ivec3 nValues) {
    vec3 res;
    float val = sqrt(PI)*erf(sqrt(x))/(2.0*sqrt(x));
    float bx = 0.5*exp(-x)/x;
    for (int nIter = 1; nIter <= nValues[2]; nIter++) {
        val = ((float(nIter) - 0.5)/x)*val - bx;
        if (nIter == nValues[0])
            res[0] = val;
        if (nIter == nValues[1])
            res[1] = val;
        if (nIter == nValues[2])
            res[2] = val;
    }
    return res;
}

vec3 vecBFValuesRecurseUp(float x, ivec3 nValues) {
    complex cVal = computeBFRecurseUpwardsIntermediary(x);
    float val = (mul(complex(0.5, 0.0), cVal)).r;
    vec3 res;
    for (int nIter = 12; nIter > nValues[0]; nIter--) {
        val = x/(float(nIter) - 0.5)*val 
            + exp(-x)/(2.0*(float(nIter) - 0.5));
        if (nIter == (nValues[0] + 1))
            res[0] = val;
        if (nIter == (nValues[1] + 1))
            res[1] = val;
        if (nIter == (nValues[2] + 1))
            res[2] = val;
    }
    return res;
}

vec3 vecBFValues(float x, ivec3 nValues) {
    if (abs(x) < 1e-60)
        return vec3(
            1.0/(2.0*float(nValues[0]) + 1.0),
            1.0/(2.0*float(nValues[1]) + 1.0),
            1.0/(2.0*float(nValues[2]) + 1.0));
    if (abs(x) < RECURSE_DIR_TURN_POINT) {
        return vecBFValuesRecurseUp(x, nValues);
    } else {
        return vecBFValuesRecurseTo0(x, nValues);
    }
}

vec4 vecBFValuesRecurseTo0(float x, ivec4 nValues) {
    vec4 res;
    float val = sqrt(PI)*erf(sqrt(x))/(2.0*sqrt(x));
    float bx = 0.5*exp(-x)/x;
    for (int nIter = 1; nIter <= nValues[3]; nIter++) {
        val = ((float(nIter) - 0.5)/x)*val - bx;
        if (nIter == nValues[0])
            res[0] = val;
        if (nIter == nValues[1])
            res[1] = val;
        if (nIter == nValues[2])
            res[2] = val;
        if (nIter == nValues[3])
            res[3] = val;
    }
    return res;
}

vec4 vecBFValuesRecurseUp(float x, ivec4 nValues) {
    complex cVal = computeBFRecurseUpwardsIntermediary(x);
    float val = (mul(complex(0.5, 0.0), cVal)).r;
    vec4 res;
    for (int nIter = 12; nIter > nValues[0]; nIter--) {
        val = x/(float(nIter) - 0.5)*val 
            + exp(-x)/(2.0*(float(nIter) - 0.5));
        if (nIter == (nValues[0] + 1))
            res[0] = val;
        if (nIter == (nValues[1] + 1))
            res[1] = val;
        if (nIter == (nValues[2] + 1))
            res[2] = val;
        if (nIter == (nValues[3] + 1))
            res[3] = val;
    }
    return res;
}

vec4 vecBFValues(float x, ivec4 nValues) {
    if (abs(x) < 1e-60)
        return vec4(
            1.0/(2.0*float(nValues[0]) + 1.0),
            1.0/(2.0*float(nValues[1]) + 1.0),
            1.0/(2.0*float(nValues[2]) + 1.0),
            1.0/(2.0*float(nValues[3]) + 1.0));
    if (abs(x) < RECURSE_DIR_TURN_POINT) {
        return vecBFValuesRecurseUp(x, nValues);
    } else {
        return vecBFValuesRecurseTo0(x, nValues);
    }

}

/* void vecBFValuesRecurseTo0(
    inout float y0, inout float y1, inout float y2, 
    inout float y3, inout float y4, float x, 
    int n0, int n1, int n2, int n3, int n4) {
    vec4 res;
    float val = sqrt(PI)*erf(sqrt(x))/(2.0*sqrt(x));
    float bx = 0.5*exp(-x)/x;
    for (int nIter = 1; nIter <= y4; nIter++) {
        val = ((float(nIter) - 0.5)/x)*val - bx;
        if (nIter == nValues[0])
            res[0] = val;
        if (nIter == nValues[1])
            res[1] = val;
        if (nIter == nValues[2])
            res[2] = val;
        if (nIter == nValues[3])
            res[3] = val;
    }
    return res;
}

vec4 vecBFValuesRecurseUp(float x, ivec4 nValues) {
    complex cVal = computeBFRecurseUpwardsIntermediary(x);
    float val = (mul(complex(0.5, 0.0), cVal)).r;
    vec4 res;
    for (int nIter = 12; nIter > nValues[0]; nIter--) {
        val = x/(float(nIter) - 0.5)*val 
            + exp(-x)/(2.0*(float(nIter) - 0.5));
        if (nIter == (nValues[0] + 1))
            res[0] = val;
        if (nIter == (nValues[1] + 1))
            res[1] = val;
        if (nIter == (nValues[2] + 1))
            res[2] = val;
        if (nIter == (nValues[3] + 1))
            res[3] = val;
    }
    return res;
}

vec4 vecBFValues(float x, ivec4 nValues) {
    if (abs(x) < 1e-60)
        return vec4(
            1.0/(2.0*float(nValues[0]) + 1.0),
            1.0/(2.0*float(nValues[1]) + 1.0),
            1.0/(2.0*float(nValues[2]) + 1.0),
            1.0/(2.0*float(nValues[3]) + 1.0));
    if (abs(x) < RECURSE_DIR_TURN_POINT) {
        return vecBFValuesRecurseUp(x, nValues);
    } else {
        return vecBFValuesRecurseTo0(x, nValues);
    }

}*/



float bf(float x, int n) {
    if (abs(x) < 1e-60)
        return 1.0/(2.0*float(n) + 1.0);
    // float y = 1.0;
    // int nMax = 12;
    // for (int j = 1; j <= nMax; j++) // See I.
    //     y *= (j - 0.5);
    // float z = pow(y, 1.0/float(nMax));
    if (abs(x) < RECURSE_DIR_TURN_POINT)
        return boysFuncRecurseUpwards(x, n);
    else
        return boysFuncRecurseToZero(x, n);
}

float pw(float a, int b) {
    // if (a < 0.0)
    //     return ((mod(b, 2) == 1)? -1.0: 1.0)*pow(abs(a), float(b));
    // return pow(a, float(b));
    float sgn = 1.0;
    if (a < 0.0 && abs(mod(float(b), float(2))) > 1e-60)
        sgn = -1.0;
    return sgn*pow(abs(a), float(b));
}


float coulombCoefficient(
    ivec3 indices, int n, float orbExp, vec3 r12) {
    // printf("%d, %d, %d\n", indices[0], indices[1], indices[2]);
    vec3 s12 = r12;
    // 1 2 3
    if (indices[2] > indices[1]) {
        SWAP_I(indices[2], indices[1]);
        SWAP_F(s12.z, s12.y);
    }
    // 1 3 2
    if (indices[1] > indices[0]) {
        SWAP_I(indices[0], indices[1]);
        SWAP_F(s12.y, s12.x);
    }
    // 3 1 2
    if (indices[2] > indices[1]) {
        SWAP_I(indices[2], indices[1]);
        SWAP_F(s12.z, s12.y);
    }
    int i = indices[0], j = indices[1], k = indices[2];

    float e = orbExp;
    float r2 = dot(r12, r12);
    float val = 0.0;
    float x = s12.x, y = s12.y, z = s12.z;
    float x2 = x*x, y2 = y*y, z2 = z*z;
    float x4 = x2*x2, y4 = y2*y2, z4 = z2*z2;

    int hexIJK = i*16*16 + j*16 + k;

    {
        if (hexIJK ==  0x000)
            return bf(e*r2, n)*pw(-2.0*e, n);
            
        if (hexIJK ==  0x100)
            return x*bf(e*r2, n + 1)*pw(-2.0*e, n + 1);
            
        if (hexIJK ==  0x110)
            return x*y*bf(e*r2, n + 2)*pw(-2.0*e, n + 2);
            
        if (hexIJK ==  0x111)
            return x*y*z*bf(e*r2, n + 3)*pw(-2.0*e, n + 3);
            
        if (hexIJK ==  0x200)
            return x2*bf(e*r2, n + 2)*pw(-2.0*e, n + 2) + 1.0*bf(e*r2, n + 1)*pw(-2.0*e, n + 1);
            
        if (hexIJK ==  0x210)
            return y*(x2*bf(e*r2, n + 3)*pw(-2.0*e, n + 3) + 1.0*bf(e*r2, n + 2)*pw(-2.0*e, n + 2));
            
        if (hexIJK ==  0x211)
            return y*z*(x2*bf(e*r2, n + 4)*pw(-2.0*e, n + 4) + 1.0*bf(e*r2, n + 3)*pw(-2.0*e, n + 3));
            
        if (hexIJK ==  0x220)
            return 1.0*x2*y2*bf(e*r2, n + 4)*pw(-2.0*e, n + 4) + 1.0*x2*bf(e*r2, n + 3)*pw(-2.0*e, n + 3) + 1.0*y2*bf(e*r2, n + 3)*pw(-2.0*e, n + 3) + 1.0*bf(e*r2, n + 2)*pw(-2.0*e, n + 2);
            
        if (hexIJK ==  0x221)
            return 1.0*z*(x2*y2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + x2*bf(e*r2, n + 4)*pw(-2.0*e, n + 4) + y2*bf(e*r2, n + 4)*pw(-2.0*e, n + 4) + bf(e*r2, n + 3)*pw(-2.0*e, n + 3));
            
        if (hexIJK ==  0x222)
            return 1.0*x2*bf(e*r2, n + 4)*pw(-2.0*e, n + 4) + 1.0*y2*(x2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + bf(e*r2, n + 4)*pw(-2.0*e, n + 4)) + z2*(1.0*x2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + y2*(x2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 1.0*bf(e*r2, n + 5)*pw(-2.0*e, n + 5)) + 1.0*bf(e*r2, n + 4)*pw(-2.0*e, n + 4)) + 1.0*bf(e*r2, n + 3)*pw(-2.0*e, n + 3);
            
        if (hexIJK ==  0x300)
            return x*(x2*bf(e*r2, n + 3)*pw(-2.0*e, n + 3) + 3.0*bf(e*r2, n + 2)*pw(-2.0*e, n + 2));
            
        if (hexIJK ==  0x310)
            return x*y*(x2*bf(e*r2, n + 4)*pw(-2.0*e, n + 4) + 3.0*bf(e*r2, n + 3)*pw(-2.0*e, n + 3));
            
        if (hexIJK ==  0x311)
            return x*y*z*(x2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 3.0*bf(e*r2, n + 4)*pw(-2.0*e, n + 4));
            
        if (hexIJK ==  0x320)
            return x*(1.0*x2*bf(e*r2, n + 4)*pw(-2.0*e, n + 4) + y2*(x2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 3.0*bf(e*r2, n + 4)*pw(-2.0*e, n + 4)) + 3.0*bf(e*r2, n + 3)*pw(-2.0*e, n + 3));
            
        if (hexIJK ==  0x321)
            return x*z*(1.0*x2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + y2*(x2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 3.0*bf(e*r2, n + 5)*pw(-2.0*e, n + 5)) + 3.0*bf(e*r2, n + 4)*pw(-2.0*e, n + 4));
            
        if (hexIJK ==  0x322)
            return x*(1.0*x2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + y2*(1.0*x2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 3.0*bf(e*r2, n + 5)*pw(-2.0*e, n + 5)) + z2*(1.0*x2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + y2*(x2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 3.0*bf(e*r2, n + 6)*pw(-2.0*e, n + 6)) + 3.0*bf(e*r2, n + 5)*pw(-2.0*e, n + 5)) + 3.0*bf(e*r2, n + 4)*pw(-2.0*e, n + 4));
            
        if (hexIJK ==  0x330)
            return x*y*(3.0*x2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + y2*(x2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 3.0*bf(e*r2, n + 5)*pw(-2.0*e, n + 5)) + 9.0*bf(e*r2, n + 4)*pw(-2.0*e, n + 4));
            
        if (hexIJK ==  0x331)
            return x*y*z*(3.0*x2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + y2*(x2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 3.0*bf(e*r2, n + 6)*pw(-2.0*e, n + 6)) + 9.0*bf(e*r2, n + 5)*pw(-2.0*e, n + 5));
            
        if (hexIJK ==  0x332)
            return x*y*(3.0*x2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 1.0*y2*(x2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 3.0*bf(e*r2, n + 6)*pw(-2.0*e, n + 6)) + z2*(3.0*x2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + y2*(x2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 3.0*bf(e*r2, n + 7)*pw(-2.0*e, n + 7)) + 9.0*bf(e*r2, n + 6)*pw(-2.0*e, n + 6)) + 9.0*bf(e*r2, n + 5)*pw(-2.0*e, n + 5));
            
        if (hexIJK ==  0x333)
            return x*y*z*(9.0*x2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 3.0*y2*(x2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 3.0*bf(e*r2, n + 7)*pw(-2.0*e, n + 7)) + z2*(3.0*x2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + y2*(x2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 3.0*bf(e*r2, n + 8)*pw(-2.0*e, n + 8)) + 9.0*bf(e*r2, n + 7)*pw(-2.0*e, n + 7)) + 27.0*bf(e*r2, n + 6)*pw(-2.0*e, n + 6));
            
        if (hexIJK ==  0x400)
            return 6.0*x2*bf(e*r2, n + 3)*pw(-2.0*e, n + 3) + 1.0*x4*bf(e*r2, n + 4)*pw(-2.0*e, n + 4) + 3.0*bf(e*r2, n + 2)*pw(-2.0*e, n + 2);
            
        if (hexIJK ==  0x410)
            return y*(6.0*x2*bf(e*r2, n + 4)*pw(-2.0*e, n + 4) + 1.0*x4*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 3.0*bf(e*r2, n + 3)*pw(-2.0*e, n + 3));
            
        if (hexIJK ==  0x411)
            return y*z*(6.0*x2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 1.0*x4*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 3.0*bf(e*r2, n + 4)*pw(-2.0*e, n + 4));
            
        if (hexIJK ==  0x420)
            return 6.0*x2*y2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 6.0*x2*bf(e*r2, n + 4)*pw(-2.0*e, n + 4) + 1.0*x4*y2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 1.0*x4*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 3.0*y2*bf(e*r2, n + 4)*pw(-2.0*e, n + 4) + 3.0*bf(e*r2, n + 3)*pw(-2.0*e, n + 3);
            
        if (hexIJK ==  0x421)
            return z*(6.0*x2*y2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 6.0*x2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 1.0*x4*y2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 1.0*x4*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 3.0*y2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 3.0*bf(e*r2, n + 4)*pw(-2.0*e, n + 4));
            
        if (hexIJK ==  0x422)
            return 6.0*x2*y2*z2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 6.0*x2*y2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 6.0*x2*z2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 6.0*x2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 1.0*x4*y2*z2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 1.0*x4*y2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 1.0*x4*z2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 1.0*x4*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 3.0*y2*z2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 3.0*y2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 3.0*z2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 3.0*bf(e*r2, n + 4)*pw(-2.0*e, n + 4);
            
        if (hexIJK ==  0x430)
            return y*(6.0*x2*y2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 18.0*x2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 1.0*x4*y2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 3.0*x4*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 3.0*y2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 9.0*bf(e*r2, n + 4)*pw(-2.0*e, n + 4));
            
        if (hexIJK ==  0x431)
            return y*z*(6.0*x2*y2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 18.0*x2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 1.0*x4*y2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 3.0*x4*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 3.0*y2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 9.0*bf(e*r2, n + 5)*pw(-2.0*e, n + 5));
            
        if (hexIJK ==  0x432)
            return y*(6.0*x2*y2*z2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 6.0*x2*y2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 18.0*x2*z2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 18.0*x2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 1.0*x4*y2*z2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 1.0*x4*y2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 3.0*x4*z2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 3.0*x4*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 3.0*y2*z2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 3.0*y2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 9.0*z2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 9.0*bf(e*r2, n + 5)*pw(-2.0*e, n + 5));
            
        if (hexIJK ==  0x433)
            return y*z*(6.0*x2*y2*z2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 18.0*x2*y2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 18.0*x2*z2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 54.0*x2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 1.0*x4*y2*z2*bf(e*r2, n + 10)*pw(-2.0*e, n + 10) + 3.0*x4*y2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 3.0*x4*z2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 9.0*x4*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 3.0*y2*z2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 9.0*y2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 9.0*z2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 27.0*bf(e*r2, n + 6)*pw(-2.0*e, n + 6));
            
        if (hexIJK ==  0x440)
            return 36.0*x2*y2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 6.0*x2*y4*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 18.0*x2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 6.0*x4*y2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 1.0*x4*y4*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 3.0*x4*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 18.0*y2*bf(e*r2, n + 5)*pw(-2.0*e, n + 5) + 3.0*y4*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 9.0*bf(e*r2, n + 4)*pw(-2.0*e, n + 4);
            
        if (hexIJK ==  0x441)
            return z*(36.0*x2*y2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 6.0*x2*y4*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 18.0*x2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 6.0*x4*y2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 1.0*x4*y4*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 3.0*x4*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 18.0*y2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 3.0*y4*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 9.0*bf(e*r2, n + 5)*pw(-2.0*e, n + 5));
            
        if (hexIJK ==  0x442)
            return 36.0*x2*y2*z2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 36.0*x2*y2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 6.0*x2*y4*z2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 6.0*x2*y4*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 18.0*x2*z2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 18.0*x2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 6.0*x4*y2*z2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 6.0*x4*y2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 1.0*x4*y4*z2*bf(e*r2, n + 10)*pw(-2.0*e, n + 10) + 1.0*x4*y4*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 3.0*x4*z2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 3.0*x4*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 18.0*y2*z2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 18.0*y2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 3.0*y4*z2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 3.0*y4*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 9.0*z2*bf(e*r2, n + 6)*pw(-2.0*e, n + 6) + 9.0*bf(e*r2, n + 5)*pw(-2.0*e, n + 5);
            
        if (hexIJK ==  0x443)
            return z*(36.0*x2*y2*z2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 108.0*x2*y2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 6.0*x2*y4*z2*bf(e*r2, n + 10)*pw(-2.0*e, n + 10) + 18.0*x2*y4*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 18.0*x2*z2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 54.0*x2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 6.0*x4*y2*z2*bf(e*r2, n + 10)*pw(-2.0*e, n + 10) + 18.0*x4*y2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 1.0*x4*y4*z2*bf(e*r2, n + 11)*pw(-2.0*e, n + 11) + 3.0*x4*y4*bf(e*r2, n + 10)*pw(-2.0*e, n + 10) + 3.0*x4*z2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 9.0*x4*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 18.0*y2*z2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 54.0*y2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 3.0*y4*z2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 9.0*y4*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 9.0*z2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 27.0*bf(e*r2, n + 6)*pw(-2.0*e, n + 6));
            
        if (hexIJK ==  0x444)
            return 216.0*x2*y2*z2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 36.0*x2*y2*z4*bf(e*r2, n + 10)*pw(-2.0*e, n + 10) + 108.0*x2*y2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 36.0*x2*y4*z2*bf(e*r2, n + 10)*pw(-2.0*e, n + 10) + 6.0*x2*y4*z4*bf(e*r2, n + 11)*pw(-2.0*e, n + 11) + 18.0*x2*y4*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 108.0*x2*z2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 18.0*x2*z4*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 54.0*x2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 36.0*x4*y2*z2*bf(e*r2, n + 10)*pw(-2.0*e, n + 10) + 6.0*x4*y2*z4*bf(e*r2, n + 11)*pw(-2.0*e, n + 11) + 18.0*x4*y2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 6.0*x4*y4*z2*bf(e*r2, n + 11)*pw(-2.0*e, n + 11) + 1.0*x4*y4*z4*bf(e*r2, n + 12)*pw(-2.0*e, n + 12) + 3.0*x4*y4*bf(e*r2, n + 10)*pw(-2.0*e, n + 10) + 18.0*x4*z2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 3.0*x4*z4*bf(e*r2, n + 10)*pw(-2.0*e, n + 10) + 9.0*x4*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 108.0*y2*z2*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 18.0*y2*z4*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 54.0*y2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 18.0*y4*z2*bf(e*r2, n + 9)*pw(-2.0*e, n + 9) + 3.0*y4*z4*bf(e*r2, n + 10)*pw(-2.0*e, n + 10) + 9.0*y4*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 54.0*z2*bf(e*r2, n + 7)*pw(-2.0*e, n + 7) + 9.0*z4*bf(e*r2, n + 8)*pw(-2.0*e, n + 8) + 27.0*bf(e*r2, n + 6)*pw(-2.0*e, n + 6);

    }

    return val;

}

float overlapCoeffHelperHelper(
    int n, int a1, int a2, float r, float e1, float e2
) {
    float r2 = r*r;
    float r3 = r2*r;
    float r4 = r2*r2;
    float r5 = r4*r;
    float r6 = r3*r3;
    float r7 = r6*r;
    float r8 = r7*r;

    float e1_2 = e1*e1;
    float e1_3 = e1_2*e1;
    float e1_4 = e1_2*e1_2;
    float e1_5 = e1_4*e1;
    float e1_6 = e1_3*e1_3;
    float e1_7 = e1_6*e1;
    float e1_8 = e1_4*e1_4;

    float e2_2 = e2*e2;
    float e2_3 = e2_2*e2;
    float e2_4 = e2_2*e2_2;
    float e2_5 = e2_4*e2;
    float e2_6 = e2_3*e2_3;
    float e2_7 = e2_6*e2;
    float e2_8 = e2_4*e2_4;

    float e1e2_2 = (e1 + e2)*(e1 + e2);
    float e1e2_3 = e1e2_2*(e1 + e2);
    float e1e2_4 = e1e2_2*e1e2_2;
    float e1e2_5 = e1e2_4*(e1 + e2);
    float e1e2_6 = e1e2_3*e1e2_3;
    float e1e2_7 = e1e2_6*(e1 + e2);
    float e1e2_8 = e1e2_4*e1e2_4;

    int hex_n_a1_a2 = 16*16*n + 16*a1 + a2;
    {
        if (hex_n_a1_a2 ==  0x000)
            return  exp(-e1*e2*r2/(e1 + e2));
            // break;
        if (hex_n_a1_a2 ==  0x001)
            return  e1*r*exp(-e1*e2*r2/(e1 + e2))/(e1 + e2);
            // break;
        if (hex_n_a1_a2 ==  0x101)
            return  0.5*exp(-e1*e2*r2/(e1 + e2))/(e1 + e2);
            // break;
        if (hex_n_a1_a2 ==  0x002)
            return  (0.5*e1 + e1_2*r2 + 0.5*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_2;
            // break;
        if (hex_n_a1_a2 ==  0x102)
            return  1.0*e1*r*exp(-e1*e2*r2/(e1 + e2))/e1e2_2;
            // break;
        if (hex_n_a1_a2 ==  0x202)
            return  0.25*exp(-e1*e2*r2/(e1 + e2))/e1e2_2;
            // break;
        if (hex_n_a1_a2 ==  0x003)
            return  e1*r*(1.5*e1 + e1_2*r2 + 1.5*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            // break;
        if (hex_n_a1_a2 ==  0x103)
            return  (0.75*e1 + 1.5*e1_2*r2 + 0.75*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            // break;
        if (hex_n_a1_a2 ==  0x203)
            return  0.75*e1*r*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            // break;
        if (hex_n_a1_a2 ==  0x303)
            return  0.125*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            // break;
        if (hex_n_a1_a2 ==  0x004)
            return  (1.0*e1_2*r2*(e1 + e2) + e1_2*r2*(1.5*e1 + e1_2*r2 + 1.5*e2) + 0.5*e1e2_2 + (e1 + e2)*(0.25*e1 + 0.5*e1_2*r2 + 0.25*e2))*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            // break;
        if (hex_n_a1_a2 ==  0x104)
            return  e1*r*(3.0*e1 + 2.0*e1_2*r2 + 3.0*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            // break;
        if (hex_n_a1_a2 ==  0x204)
            return  (0.75*e1 + 1.5*e1_2*r2 + 0.75*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            // break;
        if (hex_n_a1_a2 ==  0x304)
            return  0.5*e1*r*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            // break;
        if (hex_n_a1_a2 ==  0x404)
            return  0.0625*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            // break;
        if (hex_n_a1_a2 ==  0x010)
            return  -e2*r*exp(-e1*e2*r2/(e1 + e2))/(e1 + e2);
            // break;
        if (hex_n_a1_a2 ==  0x110)
            return  0.5*exp(-e1*e2*r2/(e1 + e2))/(e1 + e2);
            // break;
        if (hex_n_a1_a2 ==  0x011)
            return  (-e1*e2*r2 + 0.5*e1 + 0.5*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_2;
            // break;
        if (hex_n_a1_a2 ==  0x111)
            return  0.5*r*(e1 - e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_2;
            // break;
        if (hex_n_a1_a2 ==  0x211)
            return  0.25*exp(-e1*e2*r2/(e1 + e2))/e1e2_2;
            // break;
        if (hex_n_a1_a2 ==  0x012)
            return  r*(1.0*e1*(e1 + e2) - e2*(0.5*e1 + e1_2*r2 + 0.5*e2))*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            // break;
        if (hex_n_a1_a2 ==  0x112)
            return  (-1.0*e1*e2*r2 + 0.75*e1 + 0.5*e1_2*r2 + 0.75*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            // break;
        if (hex_n_a1_a2 ==  0x212)
            return  r*(0.5*e1 - 0.25*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            // break;
        if (hex_n_a1_a2 ==  0x312)
            return  0.125*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            // break;
        if (hex_n_a1_a2 ==  0x013)
            return  (-e1*e2*r2*(1.5*e1 + e1_2*r2 + 1.5*e2) + 1.0*e1_2*r2*(e1 + e2) + 0.5*e1e2_2 + (e1 + e2)*(0.25*e1 + 0.5*e1_2*r2 + 0.25*e2))*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            // break;
        if (hex_n_a1_a2 ==  0x113)
            return  r*(1.5*e1*(e1 + e2) + 0.5*e1*(1.5*e1 + e1_2*r2 + 1.5*e2) - e2*(0.75*e1 + 1.5*e1_2*r2 + 0.75*e2))*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            // break;
        if (hex_n_a1_a2 ==  0x213)
            return  0.75*(-e1*e2*r2 + e1 + e1_2*r2 + e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            // break;
        if (hex_n_a1_a2 ==  0x313)
            return  r*(0.375*e1 - 0.125*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            // break;
        if (hex_n_a1_a2 ==  0x413)
            return  0.0625*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            // break;
        if (hex_n_a1_a2 ==  0x014)
            return  r*(1.5*e1*e2_2 + 5.25*e1_2*e2 - 3.0*e1_2*e2_2*r2 - 1.0*e1_3*e2*r2 + 3.0*e1_3 - 1.0*e1_4*e2*r4 + 2.0*e1_4*r2 - 0.75*e2_3)*exp(-e1*e2*r2/(e1 + e2))/(5.0*e1*e2_4 + 10.0*e1_2*e2_3 + 10.0*e1_3*e2_2 + 5.0*e1_4*e2 + 1.0*e1_5 + 1.0*e2_5);
            // break;
        if (hex_n_a1_a2 ==  0x114)
            return  (3.75*e1*e2 - 3.0*e1*e2_2*r2 + 1.5*e1_2*e2*r2 + 1.875*e1_2 - 2.0*e1_3*e2*r4 + 4.5*e1_3*r2 + 0.5*e1_4*r4 + 1.875*e2_2)*exp(-e1*e2*r2/(e1 + e2))/(5.0*e1*e2_4 + 10.0*e1_2*e2_3 + 10.0*e1_3*e2_2 + 5.0*e1_4*e2 + 1.0*e1_5 + 1.0*e2_5);
            // break;
        if (hex_n_a1_a2 ==  0x214)
            return  r*(1.5*e1*(e1 + e2) + e1*(1.5*e1 + 1.0*e1_2*r2 + 1.5*e2) - e2*(0.75*e1 + 1.5*e1_2*r2 + 0.75*e2))*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            // break;
        if (hex_n_a1_a2 ==  0x314)
            return  (-0.5*e1*e2*r2 + 0.625*e1 + 0.75*e1_2*r2 + 0.625*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            // break;
        if (hex_n_a1_a2 ==  0x414)
            return  r*(0.25*e1 - 0.0625*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            // break;
        if (hex_n_a1_a2 ==  0x514)
            return  0.03125*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            // break;
        if (hex_n_a1_a2 ==  0x020)
            return  (0.5*e1 + 0.5*e2 + e2_2*r2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_2;
            // break;
        if (hex_n_a1_a2 ==  0x120)
            return  -1.0*e2*r*exp(-e1*e2*r2/(e1 + e2))/e1e2_2;
            // break;
        if (hex_n_a1_a2 ==  0x220)
            return  0.25*exp(-e1*e2*r2/(e1 + e2))/e1e2_2;
            // break;
        if (hex_n_a1_a2 ==  0x021)
            return  r*(e2*(e1*e2*r2 - 0.5*e1 - 0.5*e2) + 0.5*(e1 - e2)*(e1 + e2))*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            // break;
        if (hex_n_a1_a2 ==  0x121)
            return  (-0.5*e1*e2*r2 + 0.75*e1 - 0.5*e2*r2*(e1 - e2) + 0.75*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            // break;
        if (hex_n_a1_a2 ==  0x221)
            return  r*(0.25*e1 - 0.5*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            // break;
        if (hex_n_a1_a2 ==  0x321)
            return  0.125*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            // break;
        if (hex_n_a1_a2 ==  0x022)
            return  (-1.0*e1*e2*r2*(e1 + e2) + 0.5*e1e2_2 - e2*r2*(1.0*e1*(e1 + e2) - e2*(0.5*e1 + e1_2*r2 + 0.5*e2)) + (e1 + e2)*(0.25*e1 + 0.5*e1_2*r2 + 0.25*e2))*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            // break;
        if (hex_n_a1_a2 ==  0x122)
            return  r*(0.5*e1*(e1 + e2) - 0.5*e2*(0.5*e1 + e1_2*r2 + 0.5*e2) - e2*(-1.0*e1*e2*r2 + 0.75*e1 + 0.5*e1_2*r2 + 0.75*e2) + (e1 - 0.5*e2)*(e1 + e2))*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            // break;
        if (hex_n_a1_a2 ==  0x222)
            return  (-0.5*e1*e2*r2 + 0.75*e1 + 0.25*e1_2*r2 - e2*r2*(0.5*e1 - 0.25*e2) + 0.75*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            // break;
        if (hex_n_a1_a2 ==  0x322)
            return  0.25*r*(e1 - e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            // break;
        if (hex_n_a1_a2 ==  0x422)
            return  0.0625*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            // break;
        if (hex_n_a1_a2 ==  0x023)
            return  r*(1.5*e1*e1e2_2 + 0.5*e1*(e1 + e2)*(1.5*e1 + e1_2*r2 + 1.5*e2) - e2*(e1 + e2)*(0.75*e1 + 1.5*e1_2*r2 + 0.75*e2) - e2*(-e1*e2*r2*(1.5*e1 + e1_2*r2 + 1.5*e2) + 1.0*e1_2*r2*(e1 + e2) + 0.5*e1e2_2 + (e1 + e2)*(0.25*e1 + 0.5*e1_2*r2 + 0.25*e2)))*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            // break;
        if (hex_n_a1_a2 ==  0x123)
            return  (3.75*e1*e2 - 3.75*e1*e2_2*r2 - 2.25*e1_2*e2*r2 + 1.5*e1_2*e2_2*r4 + 1.875*e1_2 - 1.0*e1_3*e2*r4 + 2.25*e1_3*r2 + 1.875*e2_2 + 0.75*e2_3*r2)*exp(-e1*e2*r2/(e1 + e2))/(5.0*e1*e2_4 + 10.0*e1_2*e2_3 + 10.0*e1_3*e2_2 + 5.0*e1_4*e2 + 1.0*e1_5 + 1.0*e2_5);
            // break;
        if (hex_n_a1_a2 ==  0x223)
            return  r*(0.75*e1*(e1 + e2) + 0.25*e1*(1.5*e1 + e1_2*r2 + 1.5*e2) - 0.5*e2*(0.75*e1 + 1.5*e1_2*r2 + 0.75*e2) - 0.75*e2*(-e1*e2*r2 + e1 + e1_2*r2 + e2) + (e1 + e2)*(1.125*e1 - 0.375*e2))*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            // break;
        if (hex_n_a1_a2 ==  0x323)
            return  (-0.375*e1*e2*r2 + 0.625*e1 + 0.375*e1_2*r2 - e2*r2*(0.375*e1 - 0.125*e2) + 0.625*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            // break;
        if (hex_n_a1_a2 ==  0x423)
            return  r*(0.1875*e1 - 0.125*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            // break;
        if (hex_n_a1_a2 ==  0x523)
            return  0.03125*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            // break;
        if (hex_n_a1_a2 ==  0x024)
            return  (5.625*e1*e2_2 - 4.5*e1*e2_3*r2 + 5.625*e1_2*e2 - 6.75*e1_2*e2_2*r2 + 3.0*e1_2*e2_3*r4 + 3.0*e1_3*e2*r2 - 1.0*e1_3*e2_2*r4 + 1.875*e1_3 - 3.5*e1_4*e2*r4 + 1.0*e1_4*e2_2*r6 + 4.5*e1_4*r2 + 0.5*e1_5*r4 + 1.875*e2_3 + 0.75*e2_4*r2)*exp(-e1*e2*r2/(e1 + e2))/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
            // break;
        if (hex_n_a1_a2 ==  0x124)
            return  r*(3.0*e1*e2_3*r2 + 11.25*e1_2*e2 - 6.0*e1_2*e2_2*r2 - 6.0*e1_3*e2*r2 + 2.0*e1_3*e2_2*r4 + 7.5*e1_3 - 1.0*e1_4*e2*r4 + 3.0*e1_4*r2 - 3.75*e2_3)*exp(-e1*e2*r2/(e1 + e2))/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
            // break;
        if (hex_n_a1_a2 ==  0x224)
            return  (5.625*e1*e2 - 5.25*e1*e2_2*r2 - 1.5*e1_2*e2*r2 + 1.5*e1_2*e2_2*r4 + 2.8125*e1_2 - 2.0*e1_3*e2*r4 + 4.5*e1_3*r2 + 0.25*e1_4*r4 + 2.8125*e2_2 + 0.75*e2_3*r2)*exp(-e1*e2*r2/(e1 + e2))/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
            // break;
        if (hex_n_a1_a2 ==  0x324)
            return  r*(0.75*e1*(e1 + e2) + e1*(0.75*e1 + 0.5*e1_2*r2 + 0.75*e2) - 0.5*e2*(0.75*e1 + 1.5*e1_2*r2 + 0.75*e2) - e2*(-0.5*e1*e2*r2 + 0.625*e1 + 0.75*e1_2*r2 + 0.625*e2) + (e1 - 0.25*e2)*(e1 + e2))*exp(-e1*e2*r2/(e1 + e2))/e1e2_6;
            // break;
        if (hex_n_a1_a2 ==  0x424)
            return  (-0.25*e1*e2*r2 + 0.46875*e1 + 0.375*e1_2*r2 - e2*r2*(0.25*e1 - 0.0625*e2) + 0.46875*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_6;
            // break;
        if (hex_n_a1_a2 ==  0x524)
            return  r*(0.125*e1 - 0.0625*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_6;
            // break;
        if (hex_n_a1_a2 ==  0x624)
            return  0.015625*exp(-e1*e2*r2/(e1 + e2))/e1e2_6;
            // break;
        if (hex_n_a1_a2 ==  0x030)
            return  e2*r*(-1.5*e1 - 1.5*e2 - e2_2*r2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            // break;
        if (hex_n_a1_a2 ==  0x130)
            return  (0.75*e1 + 0.75*e2 + 1.5*e2_2*r2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            // break;
        if (hex_n_a1_a2 ==  0x230)
            return  -0.75*e2*r*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            // break;
        if (hex_n_a1_a2 ==  0x330)
            return  0.125*exp(-e1*e2*r2/(e1 + e2))/e1e2_3;
            // break;
        if (hex_n_a1_a2 ==  0x031)
            return  (0.5*e1e2_2 - 0.5*e2*r2*(e1 - e2)*(e1 + e2) - e2*r2*(e2*(e1*e2*r2 - 0.5*e1 - 0.5*e2) + 0.5*(e1 - e2)*(e1 + e2)) + (e1 + e2)*(-0.5*e1*e2*r2 + 0.25*e1 + 0.25*e2))*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            // break;
        if (hex_n_a1_a2 ==  0x131)
            return  r*(-1.5*e1*e2 + 1.5*e1*e2_2*r2 + 0.75*e1_2 - 2.25*e2_2 - 0.5*e2_3*r2)*exp(-e1*e2*r2/(e1 + e2))/(4.0*e1*e2_3 + 6.0*e1_2*e2_2 + 4.0*e1_3*e2 + 1.0*e1_4 + 1.0*e2_4);
            // break;
        if (hex_n_a1_a2 ==  0x231)
            return  (-0.25*e1*e2*r2 + 0.75*e1 + e2*r2*(-0.25*e1 + 0.5*e2) - 0.25*e2*r2*(e1 - e2) + 0.75*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            // break;
        if (hex_n_a1_a2 ==  0x331)
            return  r*(0.125*e1 - 0.375*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            // break;
        if (hex_n_a1_a2 ==  0x431)
            return  0.0625*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            // break;
        if (hex_n_a1_a2 ==  0x032)
            return  r*(-3.0*e1*e2_2 + 2.5*e1*e2_3*r2 + 0.75*e1_2*e2 + 1.5*e1_2*e2_2*r2 - 1.0*e1_2*e2_3*r4 - 1.5*e1_3*e2*r2 + 1.5*e1_3 - 2.25*e2_3 - 0.5*e2_4*r2)*exp(-e1*e2*r2/(e1 + e2))/(5.0*e1*e2_4 + 10.0*e1_2*e2_3 + 10.0*e1_3*e2_2 + 5.0*e1_4*e2 + 1.0*e1_5 + 1.0*e2_5);
            // break;
        if (hex_n_a1_a2 ==  0x132)
            return  (3.75*e1*e2 - 2.25*e1*e2_2*r2 - 1.0*e1*e2_3*r4 - 3.75*e1_2*e2*r2 + 1.5*e1_2*e2_2*r4 + 1.875*e1_2 + 0.75*e1_3*r2 + 1.875*e2_2 + 2.25*e2_3*r2)*exp(-e1*e2*r2/(e1 + e2))/(5.0*e1*e2_4 + 10.0*e1_2*e2_3 + 10.0*e1_3*e2_2 + 5.0*e1_4*e2 + 1.0*e1_5 + 1.0*e2_5);
            // break;
        if (hex_n_a1_a2 ==  0x232)
            return  r*(-0.75*e1*e2 + 1.5*e1*e2_2*r2 - 0.75*e1_2*e2*r2 + 1.5*e1_2 - 2.25*e2_2 - 0.25*e2_3*r2)*exp(-e1*e2*r2/(e1 + e2))/(5.0*e1*e2_4 + 10.0*e1_2*e2_3 + 10.0*e1_3*e2_2 + 5.0*e1_4*e2 + 1.0*e1_5 + 1.0*e2_5);
            // break;
        if (hex_n_a1_a2 ==  0x332)
            return  (-0.25*e1*e2*r2 + 0.625*e1 + 0.125*e1_2*r2 + 0.25*e2*r2*(-e1 + e2) - 0.5*e2*r2*(0.5*e1 - 0.25*e2) + 0.625*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            // break;
        if (hex_n_a1_a2 ==  0x432)
            return  r*(0.125*e1 - 0.1875*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            // break;
        if (hex_n_a1_a2 ==  0x532)
            return  0.03125*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            // break;
        if (hex_n_a1_a2 ==  0x033)
            return  (5.625*e1*e2_2 - 2.25*e1*e2_3*r2 - 1.5*e1*e2_4*r4 + 5.625*e1_2*e2 - 9.0*e1_2*e2_2*r2 + 3.0*e1_2*e2_3*r4 - 2.25*e1_3*e2*r2 + 3.0*e1_3*e2_2*r4 - 1.0*e1_3*e2_3*r6 + 1.875*e1_3 - 1.5*e1_4*e2*r4 + 2.25*e1_4*r2 + 1.875*e2_3 + 2.25*e2_4*r2)*exp(-e1*e2*r2/(e1 + e2))/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
            // break;
        if (hex_n_a1_a2 ==  0x133)
            return  r*(-5.625*e1*e2_2 + 6.0*e1*e2_3*r2 + 5.625*e1_2*e2 - 1.5*e1_2*e2_3*r4 - 6.0*e1_3*e2*r2 + 1.5*e1_3*e2_2*r4 + 5.625*e1_3 + 0.75*e1_4*r2 - 5.625*e2_3 - 0.75*e2_4*r2)*exp(-e1*e2*r2/(e1 + e2))/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
            // break;
        if (hex_n_a1_a2 ==  0x233)
            return  (5.625*e1*e2 - 4.5*e1*e2_2*r2 - 0.75*e1*e2_3*r4 - 4.5*e1_2*e2*r2 + 2.25*e1_2*e2_2*r4 + 2.8125*e1_2 - 0.75*e1_3*e2*r4 + 2.25*e1_3*r2 + 2.8125*e2_2 + 2.25*e2_3*r2)*exp(-e1*e2*r2/(e1 + e2))/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
            // break;
        if (hex_n_a1_a2 ==  0x333)
            return  r*(1.125*e1*e2_2*r2 - 1.125*e1_2*e2*r2 + 1.875*e1_2 + 0.125*e1_3*r2 - 1.875*e2_2 - 0.125*e2_3*r2)*exp(-e1*e2*r2/(e1 + e2))/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
            // break;
        if (hex_n_a1_a2 ==  0x433)
            return  (-0.1875*e1*e2*r2 + 0.46875*e1 + 0.1875*e1_2*r2 + e2*r2*(-0.1875*e1 + 0.125*e2) - 0.5*e2*r2*(0.375*e1 - 0.125*e2) + 0.46875*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_6;
            // break;
        if (hex_n_a1_a2 ==  0x533)
            return  0.09375*r*(e1 - e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_6;
            // break;
        if (hex_n_a1_a2 ==  0x633)
            return  0.015625*exp(-e1*e2*r2/(e1 + e2))/e1e2_6;
            // break;
        if (hex_n_a1_a2 ==  0x034)
            return  r*(-9.375*e1*e2_3 + 7.5*e1*e2_4*r2 + 5.625*e1_2*e2_2 + 3.75*e1_2*e2_3*r2 - 3.0*e1_2*e2_4*r4 + 16.875*e1_3*e2 - 15.0*e1_3*e2_2*r2 + 3.0*e1_3*e2_3*r4 - 7.5*e1_4*e2*r2 + 4.5*e1_4*e2_2*r4 - 1.0*e1_4*e2_3*r6 + 7.5*e1_4 - 1.5*e1_5*e2*r4 + 3.0*e1_5*r2 - 5.625*e2_4 - 0.75*e2_5*r2)*exp(-e1*e2*r2/(e1 + e2))/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
            // break;
        if (hex_n_a1_a2 ==  0x134)
            return  (19.6875*e1*e2_2 - 11.25*e1*e2_3*r2 - 3.0*e1*e2_4*r4 + 19.6875*e1_2*e2 - 28.125*e1_2*e2_2*r2 + 10.5*e1_2*e2_3*r4 + 4.5*e1_3*e2_2*r4 - 2.0*e1_3*e2_3*r6 + 6.5625*e1_3 - 8.25*e1_4*e2*r4 + 1.5*e1_4*e2_2*r6 + 11.25*e1_4*r2 + 0.75*e1_5*r4 + 6.5625*e2_3 + 5.625*e2_4*r2)*exp(-e1*e2*r2/(e1 + e2))/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
            // break;
        if (hex_n_a1_a2 ==  0x234)
            return  r*(-5.625*e1*e2_2 + 8.25*e1*e2_3*r2 + 14.0625*e1_2*e2 - 4.5*e1_2*e2_2*r2 - 1.5*e1_2*e2_3*r4 - 10.5*e1_3*e2*r2 + 3.0*e1_3*e2_2*r4 + 11.25*e1_3 - 0.75*e1_4*e2*r4 + 3.0*e1_4*r2 - 8.4375*e2_3 - 0.75*e2_4*r2)*exp(-e1*e2*r2/(e1 + e2))/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
            // break;
        if (hex_n_a1_a2 ==  0x334)
            return  (6.5625*e1*e2 - 5.625*e1*e2_2*r2 - 0.5*e1*e2_3*r4 - 3.75*e1_2*e2*r2 + 2.25*e1_2*e2_2*r4 + 3.28125*e1_2 - 1.5*e1_3*e2*r4 + 3.75*e1_3*r2 + 0.125*e1_4*r4 + 3.28125*e2_2 + 1.875*e2_3*r2)*exp(-e1*e2*r2/(e1 + e2))/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
            // break;
        if (hex_n_a1_a2 ==  0x434)
            return  r*(0.46875*e1*e2 + 0.75*e1*e2_2*r2 - 1.125*e1_2*e2*r2 + 1.875*e1_2 + 0.25*e1_3*r2 - 1.40625*e2_2 - 0.0625*e2_3*r2)*exp(-e1*e2*r2/(e1 + e2))/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
            // break;
        if (hex_n_a1_a2 ==  0x534)
            return  (-0.125*e1*e2*r2 + 0.328125*e1 + 0.1875*e1_2*r2 + e2*r2*(-0.125*e1 + 0.0625*e2) - 0.5*e2*r2*(0.25*e1 - 0.0625*e2) + 0.328125*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_7;
            // break;
        if (hex_n_a1_a2 ==  0x634)
            return  r*(0.0625*e1 - 0.046875*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_7;
            // break;
        if (hex_n_a1_a2 ==  0x734)
            return  0.0078125*exp(-e1*e2*r2/(e1 + e2))/e1e2_7;
            // break;
        if (hex_n_a1_a2 ==  0x040)
            return  (0.5*e1e2_2 + 1.0*e2_2*r2*(e1 + e2) + e2_2*r2*(1.5*e1 + 1.5*e2 + e2_2*r2) + (e1 + e2)*(0.25*e1 + 0.25*e2 + 0.5*e2_2*r2))*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            // break;
        if (hex_n_a1_a2 ==  0x140)
            return  e2*r*(-3.0*e1 - 3.0*e2 - 2.0*e2_2*r2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            // break;
        if (hex_n_a1_a2 ==  0x240)
            return  (0.75*e1 + 0.75*e2 + 1.5*e2_2*r2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            // break;
        if (hex_n_a1_a2 ==  0x340)
            return  -0.5*e2*r*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            // break;
        if (hex_n_a1_a2 ==  0x440)
            return  0.0625*exp(-e1*e2*r2/(e1 + e2))/e1e2_4;
            // break;
        if (hex_n_a1_a2 ==  0x041)
            return  r*(-5.25*e1*e2_2 + 1.0*e1*e2_3*r2 + 1.0*e1*e2_4*r4 - 1.5*e1_2*e2 + 3.0*e1_2*e2_2*r2 + 0.75*e1_3 - 3.0*e2_3 - 2.0*e2_4*r2)*exp(-e1*e2*r2/(e1 + e2))/(5.0*e1*e2_4 + 10.0*e1_2*e2_3 + 10.0*e1_3*e2_2 + 5.0*e1_4*e2 + 1.0*e1_5 + 1.0*e2_5);
            // break;
        if (hex_n_a1_a2 ==  0x141)
            return  (3.75*e1*e2 + 1.5*e1*e2_2*r2 - 2.0*e1*e2_3*r4 - 3.0*e1_2*e2*r2 + 1.875*e1_2 + 1.875*e2_2 + 4.5*e2_3*r2 + 0.5*e2_4*r4)*exp(-e1*e2*r2/(e1 + e2))/(5.0*e1*e2_4 + 10.0*e1_2*e2_3 + 10.0*e1_3*e2_2 + 5.0*e1_4*e2 + 1.0*e1_5 + 1.0*e2_5);
            // break;
        if (hex_n_a1_a2 ==  0x241)
            return  r*(-2.25*e1*e2 + 1.5*e1*e2_2*r2 + 0.75*e1_2 - 3.0*e2_2 - 1.0*e2_3*r2)*exp(-e1*e2*r2/(e1 + e2))/(5.0*e1*e2_4 + 10.0*e1_2*e2_3 + 10.0*e1_3*e2_2 + 5.0*e1_4*e2 + 1.0*e1_5 + 1.0*e2_5);
            // break;
        if (hex_n_a1_a2 ==  0x341)
            return  (-0.125*e1*e2*r2 + 0.625*e1 + 0.5*e2*r2*(-0.25*e1 + 0.5*e2) + e2*r2*(-0.125*e1 + 0.375*e2) - 0.125*e2*r2*(e1 - e2) + 0.625*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            // break;
        if (hex_n_a1_a2 ==  0x441)
            return  r*(0.0625*e1 - 0.25*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            // break;
        if (hex_n_a1_a2 ==  0x541)
            return  0.03125*exp(-e1*e2*r2/(e1 + e2))/e1e2_5;
            // break;
        if (hex_n_a1_a2 ==  0x042)
            return  (5.625*e1*e2_2 + 3.0*e1*e2_3*r2 - 3.5*e1*e2_4*r4 + 5.625*e1_2*e2 - 6.75*e1_2*e2_2*r2 - 1.0*e1_2*e2_3*r4 + 1.0*e1_2*e2_4*r6 - 4.5*e1_3*e2*r2 + 3.0*e1_3*e2_2*r4 + 1.875*e1_3 + 0.75*e1_4*r2 + 1.875*e2_3 + 4.5*e2_4*r2 + 0.5*e2_5*r4)*exp(-e1*e2*r2/(e1 + e2))/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
            // break;
        if (hex_n_a1_a2 ==  0x142)
            return  r*(-11.25*e1*e2_2 + 6.0*e1*e2_3*r2 + 1.0*e1*e2_4*r4 + 6.0*e1_2*e2_2*r2 - 2.0*e1_2*e2_3*r4 - 3.0*e1_3*e2*r2 + 3.75*e1_3 - 7.5*e2_3 - 3.0*e2_4*r2)*exp(-e1*e2*r2/(e1 + e2))/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
            // break;
        if (hex_n_a1_a2 ==  0x242)
            return  (5.625*e1*e2 - 1.5*e1*e2_2*r2 - 2.0*e1*e2_3*r4 - 5.25*e1_2*e2*r2 + 1.5*e1_2*e2_2*r4 + 2.8125*e1_2 + 0.75*e1_3*r2 + 2.8125*e2_2 + 4.5*e2_3*r2 + 0.25*e2_4*r4)*exp(-e1*e2*r2/(e1 + e2))/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
            // break;
        if (hex_n_a1_a2 ==  0x342)
            return  r*(-1.25*e1*e2 + 1.5*e1*e2_2*r2 - 0.5*e1_2*e2*r2 + 1.25*e1_2 - 2.5*e2_2 - 0.5*e2_3*r2)*exp(-e1*e2*r2/(e1 + e2))/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
            // break;
        if (hex_n_a1_a2 ==  0x442)
            return  (-0.125*e1*e2*r2 + 0.46875*e1 + 0.0625*e1_2*r2 + 0.125*e2*r2*(-e1 + e2) + e2*r2*(-0.125*e1 + 0.1875*e2) - 0.25*e2*r2*(0.5*e1 - 0.25*e2) + 0.46875*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_6;
            // break;
        if (hex_n_a1_a2 ==  0x542)
            return  r*(0.0625*e1 - 0.125*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_6;
            // break;
        if (hex_n_a1_a2 ==  0x642)
            return  0.015625*exp(-e1*e2*r2/(e1 + e2))/e1e2_6;
            // break;
        if (hex_n_a1_a2 ==  0x043)
            return  r*(-16.875*e1*e2_3 + 7.5*e1*e2_4*r2 + 1.5*e1*e2_5*r4 - 5.625*e1_2*e2_2 + 15.0*e1_2*e2_3*r2 - 4.5*e1_2*e2_4*r4 + 9.375*e1_3*e2 - 3.75*e1_3*e2_2*r2 - 3.0*e1_3*e2_3*r4 + 1.0*e1_3*e2_4*r6 - 7.5*e1_4*e2*r2 + 3.0*e1_4*e2_2*r4 + 5.625*e1_4 + 0.75*e1_5*r2 - 7.5*e2_4 - 3.0*e2_5*r2)*exp(-e1*e2*r2/(e1 + e2))/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
            // break;
        if (hex_n_a1_a2 ==  0x143)
            return  (19.6875*e1*e2_2 - 8.25*e1*e2_4*r4 + 19.6875*e1_2*e2 - 28.125*e1_2*e2_2*r2 + 4.5*e1_2*e2_3*r4 + 1.5*e1_2*e2_4*r6 - 11.25*e1_3*e2*r2 + 10.5*e1_3*e2_2*r4 - 2.0*e1_3*e2_3*r6 + 6.5625*e1_3 - 3.0*e1_4*e2*r4 + 5.625*e1_4*r2 + 6.5625*e2_3 + 11.25*e2_4*r2 + 0.75*e2_5*r4)*exp(-e1*e2*r2/(e1 + e2))/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
            // break;
        if (hex_n_a1_a2 ==  0x243)
            return  r*(-14.0625*e1*e2_2 + 10.5*e1*e2_3*r2 + 0.75*e1*e2_4*r4 + 5.625*e1_2*e2 + 4.5*e1_2*e2_2*r2 - 3.0*e1_2*e2_3*r4 - 8.25*e1_3*e2*r2 + 1.5*e1_3*e2_2*r4 + 8.4375*e1_3 + 0.75*e1_4*r2 - 11.25*e2_3 - 3.0*e2_4*r2)*exp(-e1*e2*r2/(e1 + e2))/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
            // break;
        if (hex_n_a1_a2 ==  0x343)
            return  (6.5625*e1*e2 - 3.75*e1*e2_2*r2 - 1.5*e1*e2_3*r4 - 5.625*e1_2*e2*r2 + 2.25*e1_2*e2_2*r4 + 3.28125*e1_2 - 0.5*e1_3*e2*r4 + 1.875*e1_3*r2 + 3.28125*e2_2 + 3.75*e2_3*r2 + 0.125*e2_4*r4)*exp(-e1*e2*r2/(e1 + e2))/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
            // break;
        if (hex_n_a1_a2 ==  0x443)
            return  r*(-0.46875*e1*e2 + 1.125*e1*e2_2*r2 - 0.75*e1_2*e2*r2 + 1.40625*e1_2 + 0.0625*e1_3*r2 - 1.875*e2_2 - 0.25*e2_3*r2)*exp(-e1*e2*r2/(e1 + e2))/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
            // break;
        if (hex_n_a1_a2 ==  0x543)
            return  (-0.09375*e1*e2*r2 + 0.328125*e1 + 0.09375*e1_2*r2 + 0.09375*e2*r2*(-e1 + e2) + 0.5*e2*r2*(-0.1875*e1 + 0.125*e2) - 0.25*e2*r2*(0.375*e1 - 0.125*e2) + 0.328125*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_7;
            // break;
        if (hex_n_a1_a2 ==  0x643)
            return  r*(0.046875*e1 - 0.0625*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_7;
            // break;
        if (hex_n_a1_a2 ==  0x743)
            return  0.0078125*exp(-e1*e2*r2/(e1 + e2))/e1e2_7;
            // break;
        if (hex_n_a1_a2 ==  0x044)
            return  (26.25*e1*e2_3 + 3.75*e1*e2_4*r2 - 10.5*e1*e2_5*r4 + 39.375*e1_2*e2_2 - 45.0*e1_2*e2_3*r2 + 3.75*e1_2*e2_4*r4 + 3.0*e1_2*e2_5*r6 + 26.25*e1_3*e2 - 45.0*e1_3*e2_2*r2 + 30.0*e1_3*e2_3*r4 - 5.0*e1_3*e2_4*r6 + 3.75*e1_4*e2*r2 + 3.75*e1_4*e2_2*r4 - 5.0*e1_4*e2_3*r6 + 1.0*e1_4*e2_4*r8 + 6.5625*e1_4 - 10.5*e1_5*e2*r4 + 3.0*e1_5*e2_2*r6 + 11.25*e1_5*r2 + 0.75*e1_6*r4 + 6.5625*e2_4 + 11.25*e2_5*r2 + 0.75*e2_6*r4)*exp(-e1*e2*r2/(e1 + e2))/(8.0*e1*e2_7 + 28.0*e1_2*e2_6 + 56.0*e1_3*e2_5 + 70.0*e1_4*e2_4 + 56.0*e1_5*e2_3 + 28.0*e1_6*e2_2 + 8.0*e1_7*e2 + 1.0*e1_8 + 1.0*e2_8);
            // break;
        if (hex_n_a1_a2 ==  0x144)
            return  r*(-52.5*e1*e2_3 + 30.0*e1*e2_4*r2 + 3.0*e1*e2_5*r4 + 37.5*e1_2*e2_3*r2 - 15.0*e1_2*e2_4*r4 + 52.5*e1_3*e2 - 37.5*e1_3*e2_2*r2 + 2.0*e1_3*e2_4*r6 - 30.0*e1_4*e2*r2 + 15.0*e1_4*e2_2*r4 - 2.0*e1_4*e2_3*r6 + 26.25*e1_4 - 3.0*e1_5*e2*r4 + 7.5*e1_5*r2 - 26.25*e2_4 - 7.5*e2_5*r2)*exp(-e1*e2*r2/(e1 + e2))/(8.0*e1*e2_7 + 28.0*e1_2*e2_6 + 56.0*e1_3*e2_5 + 70.0*e1_4*e2_4 + 56.0*e1_5*e2_3 + 28.0*e1_6*e2_2 + 8.0*e1_7*e2 + 1.0*e1_8 + 1.0*e2_8);
            // break;
        if (hex_n_a1_a2 ==  0x244)
            return  (39.375*e1*e2_2 - 11.25*e1*e2_3*r2 - 11.25*e1*e2_4*r4 + 39.375*e1_2*e2 - 56.25*e1_2*e2_2*r2 + 15.0*e1_2*e2_3*r4 + 1.5*e1_2*e2_4*r6 - 11.25*e1_3*e2*r2 + 15.0*e1_3*e2_2*r4 - 4.0*e1_3*e2_3*r6 + 13.125*e1_3 - 11.25*e1_4*e2*r4 + 1.5*e1_4*e2_2*r6 + 16.875*e1_4*r2 + 0.75*e1_5*r4 + 13.125*e2_3 + 16.875*e2_4*r2 + 0.75*e2_5*r4)*exp(-e1*e2*r2/(e1 + e2))/(8.0*e1*e2_7 + 28.0*e1_2*e2_6 + 56.0*e1_3*e2_5 + 70.0*e1_4*e2_4 + 56.0*e1_5*e2_3 + 28.0*e1_6*e2_2 + 8.0*e1_7*e2 + 1.0*e1_8 + 1.0*e2_8);
            // break;
        if (hex_n_a1_a2 ==  0x344)
            return  r*(-13.125*e1*e2_2 + 12.5*e1*e2_3*r2 + 0.5*e1*e2_4*r4 + 13.125*e1_2*e2 - 3.0*e1_2*e2_3*r4 - 12.5*e1_3*e2*r2 + 3.0*e1_3*e2_2*r4 + 13.125*e1_3 - 0.5*e1_4*e2*r4 + 2.5*e1_4*r2 - 13.125*e2_3 - 2.5*e2_4*r2)*exp(-e1*e2*r2/(e1 + e2))/(8.0*e1*e2_7 + 28.0*e1_2*e2_6 + 56.0*e1_3*e2_5 + 70.0*e1_4*e2_4 + 56.0*e1_5*e2_3 + 28.0*e1_6*e2_2 + 8.0*e1_7*e2 + 1.0*e1_8 + 1.0*e2_8);
            // break;
        if (hex_n_a1_a2 ==  0x444)
            return  (6.5625*e1*e2 - 4.6875*e1*e2_2*r2 - 1.0*e1*e2_3*r4 - 4.6875*e1_2*e2*r2 + 2.25*e1_2*e2_2*r4 + 3.28125*e1_2 - 1.0*e1_3*e2*r4 + 2.8125*e1_3*r2 + 0.0625*e1_4*r4 + 3.28125*e2_2 + 2.8125*e2_3*r2 + 0.0625*e2_4*r4)*exp(-e1*e2*r2/(e1 + e2))/(8.0*e1*e2_7 + 28.0*e1_2*e2_6 + 56.0*e1_3*e2_5 + 70.0*e1_4*e2_4 + 56.0*e1_5*e2_3 + 28.0*e1_6*e2_2 + 8.0*e1_7*e2 + 1.0*e1_8 + 1.0*e2_8);
            // break;
        if (hex_n_a1_a2 ==  0x544)
            return  r*(0.75*e1*e2_2*r2 - 0.75*e1_2*e2*r2 + 1.3125*e1_2 + 0.125*e1_3*r2 - 1.3125*e2_2 - 0.125*e2_3*r2)*exp(-e1*e2*r2/(e1 + e2))/(8.0*e1*e2_7 + 28.0*e1_2*e2_6 + 56.0*e1_3*e2_5 + 70.0*e1_4*e2_4 + 56.0*e1_5*e2_3 + 28.0*e1_6*e2_2 + 8.0*e1_7*e2 + 1.0*e1_8 + 1.0*e2_8);
            // break;
        if (hex_n_a1_a2 ==  0x644)
            return  (-0.0625*e1*e2*r2 + 0.21875*e1 + 0.09375*e1_2*r2 + 0.5*e2*r2*(-0.125*e1 + 0.0625*e2) + e2*r2*(-0.0625*e1 + 0.046875*e2) - 0.25*e2*r2*(0.25*e1 - 0.0625*e2) + 0.21875*e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_8;
            // break;
        if (hex_n_a1_a2 ==  0x744)
            return  0.03125*r*(e1 - e2)*exp(-e1*e2*r2/(e1 + e2))/e1e2_8;
            // break;
        if (hex_n_a1_a2 ==  0x844)
            return  0.00390625*exp(-e1*e2*r2/(e1 + e2))/e1e2_8;
            // break;


        // default)
            // break;
    }
    return 0.0;
}

float overlapCoefficient(
    int n, 
    float x1, int a1, float e1,
    float x2, int a2, float e2) {
    float r = x1 - x2;
    // return 1.0;
    return overlapCoeffHelperHelper(
        n, a1, a2, r, e1, e2);
}

vec3 productCenter(
    vec3 gPosition, float gExp, vec3 hPosition, float hExp) {
    return ((gExp*gPosition) + (hExp*hPosition)) / (gExp + hExp);
}

float repulsionInner(
    vec3 r3, ivec3 a3, float e3,
    vec3 r4, ivec3 a4, float e4,
    int ix, int iy, int iz,
    float orbExp, vec3 rLR
) {
    float eps1 = 1e-40;
    float val = 0.0;

    for (int jx = 0; jx < maxAngularIter && jx < (a3.x + a4.x + 1); jx++) {
        
        float overlapX 
            = overlapCoefficient(
                jx, 
                r3.x, a3.x, e3, 
                r4.x, a4.x, e4);
        
        for (int jy = 0; 
             jy < maxAngularIter && 
             abs(overlapX) > eps1 && jy < (a3.y + a4.y + 1); jy++) {
            
            float overlapY 
                = overlapCoefficient(
                    jy, 
                    r3.y, a3.y, e3, 
                    r4.y, a4.y, e4);
            
            for (int jz = 0;
                 jz < maxAngularIter &&
                 abs(overlapY) > eps1 && jz < (a3.z + a4.z + 1); jz++) {
                
                float overlapZ 
                    = overlapCoefficient(
                        jz,
                        r3.z, a3.z, e3,
                        r4.z, a4.z, e4);
                float overlaps = overlapX*overlapY*overlapZ;
                // ivec3 indices, int n, float orbExp, vec4 r12)

                // float factor = (mod(jx + jy + jz, 2) == 1)? -1.0: 1.0;
                float factor = (mod(float(jx + jy + jz), 2.0) > 1e-30)? 
                    -1.0: 1.0;
                if (abs(overlaps) > eps1) {
                    val += factor * overlaps * coulombCoefficient(
                            ivec3(ix + jx, iy + jy, iz + jz), 0,
                            orbExp, rLR);
                }

            }
        }
    }
    return val;

}

float repulsion(
    float amplitude1, vec3 r1, ivec3 a1, float e1,
    float amplitude2, vec3 r2, ivec3 a2, float e2,
    float amplitude3, vec3 r3, ivec3 a3, float e3,
    float amplitude4, vec3 r4, ivec3 a4, float e4
) {
    float amplitude = amplitude1*amplitude2*amplitude3*amplitude4;
    if (amplitude == 0.0)
        return 0.0;
    float orbExpL = e1 + e2;
    float orbExpR = e3 + e4;
    float orbExp = orbExpL*orbExpR/(orbExpL + orbExpR);
    vec3 rLR = productCenter(r1, e1, r2, e2) - productCenter(r3, e3, r4, e4);
    float eps1 = 1e-40;
    float val = 0.0;

    for (int ix = 0;
         ix < maxAngularIter && ix < (a1.x + a2.x + 1); ix++) {
        
        float overlapX 
            = overlapCoefficient(
                ix, 
                r1.x, a1.x, e1,
                r2.x, a2.x, e2);
        
        for (int iy = 0; 
             iy < maxAngularIter
             && abs(overlapX) > eps1 && iy < (a1.y + a2.y + 1); iy++) {
            
            float overlapY 
                = overlapCoefficient(
                    iy, 
                    r1.y, a1.y, e1, 
                    r2.y, a2.y, e2);
            
            for (int iz = 0;
                 iz < maxAngularIter
                 && abs(overlapY) > eps1 && iz < (a1.z + a2.z + 1); iz++) {
                
                float overlapZ 
                    = overlapCoefficient(
                        iz, 
                        r1.z, a1.z, e1, 
                        r2.z, a2.z, e2);
                float overlaps = overlapX*overlapY*overlapZ;

                if (abs(overlaps) > eps1)
                    val += 
                        overlaps
                        * repulsionInner(
                            r3, a3, e3, r4, a4, e4,
                            ix, iy, iz, orbExp, rLR
                        );

            }
        }
    }
    return amplitude*val*(2.0*pow(PI, (5.0/2.0))) 
        / (orbExpL*orbExpR*sqrt(orbExpL + orbExpR));

}

void main () {

    vec4 indices = texture2D(indicesTex, UV);
    vec2 indI = vec2((indices[0] + 0.5)/float(numberOfBasisFunctions), 0.5);
    vec2 indJ = vec2((indices[1] + 0.5)/float(numberOfBasisFunctions), 0.5);
    vec2 indK = vec2((indices[2] + 0.5)/float(numberOfBasisFunctions), 0.5);
    vec2 indL = vec2((indices[3] + 0.5)/float(numberOfBasisFunctions), 0.5);
    vec3 rI = texture2D(basisFunctionSpec1Tex, indI).xyz;
    vec3 rJ = texture2D(basisFunctionSpec1Tex, indJ).xyz;
    vec3 rK = texture2D(basisFunctionSpec1Tex, indK).xyz;
    vec3 rL = texture2D(basisFunctionSpec1Tex, indL).xyz;
    int countI = int(texture2D(basisFunctionSpec1Tex, indI).w);
    int countJ = int(texture2D(basisFunctionSpec1Tex, indJ).w);
    int countK = int(texture2D(basisFunctionSpec1Tex, indK).w);
    int countL = int(texture2D(basisFunctionSpec1Tex, indL).w);
    if (primitiveIndices[0] >= countI ||
        primitiveIndices[1] >= countJ ||
        primitiveIndices[2] >= countK ||
        primitiveIndices[3] >= countL) {
        fragColor = texture2D(previousSumTex, UV);
    } else {
        vec3 angI = texture2D(basisFunctionSpec2Tex, indI).xyz;
        ivec3 angularI = ivec3(int(angI.x), int(angI.y), int(angI.z));
        vec3 angJ = texture2D(basisFunctionSpec2Tex, indJ).xyz;
        ivec3 angularJ = ivec3(int(angJ.x), int(angJ.y), int(angJ.z));
        vec3 angK = texture2D(basisFunctionSpec2Tex, indK).xyz;
        ivec3 angularK = ivec3(int(angK.x), int(angK.y), int(angK.z));
        vec3 angL = texture2D(basisFunctionSpec2Tex, indL).xyz;
        ivec3 angularL = ivec3(int(angL.x), int(angL.y), int(angL.z));
        vec2 primIndI = vec2(
            (float(primitiveIndices[0]) + 0.5)/float(maxNumberOfPrimitives),
            indI[0]);
        vec2 primIndJ = vec2(
            (float(primitiveIndices[1]) + 0.5)/float(maxNumberOfPrimitives),
            indJ[0]);
        vec2 primIndK = vec2(
            (float(primitiveIndices[2]) + 0.5)/float(maxNumberOfPrimitives),
            indK[0]);
        vec2 primIndL = vec2(
            (float(primitiveIndices[3]) + 0.5)/float(maxNumberOfPrimitives),
            indL[0]);
        float ampI = texture2D(primitivesTex, primIndI)[0];
        float eI = texture2D(primitivesTex, primIndI)[1];
        float ampJ = texture2D(primitivesTex, primIndJ)[0];
        float eJ = texture2D(primitivesTex, primIndJ)[1];
        float ampK = texture2D(primitivesTex, primIndK)[0];
        float eK = texture2D(primitivesTex, primIndK)[1];
        float ampL = texture2D(primitivesTex, primIndL)[0];
        float eL = texture2D(primitivesTex, primIndL)[1];
        float sumTerm = repulsion(
            ampI, rI, angularI, eI,
            ampJ, rJ, angularJ, eJ,
            ampK, rK, angularK, eK,
            ampL, rL, angularL, eL
        );
        fragColor = vec4(sumTerm) + texture2D(previousSumTex, UV);
    }

    if (debug > 0) {
    
        vec3 fDebugInd = texture2D(debugIndTex, UV).xyz;
        ivec3 debugInd = ivec3(
            int(fDebugInd.x), int(fDebugInd.y), int(fDebugInd.z)
        );


        // fragColor = vec4(fDebugInd.x + fDebugInd.y + fDebugInd.z);
        if (debug == DEBUG_COULOMB) { 
            // float coulombCoefficient(
            // ivec3 indices, int n, float orbExp, vec3 r12)
            float val = coulombCoefficient(
                debugInd, debugCoulombFactor, debugCoulombOrbExp,
                debugCoulombR);
            fragColor = vec4(val);
        } 
        else if (debug == DEBUG_OVERLAP) {
            int n = debugInd[0];
            int a1 = debugInd[1];
            int a2 = debugInd[2];
            float e1 = debugOverlapExponents[0];
            float e2 = debugOverlapExponents[1];
            float x1 = debugOverlapPositions[0];
            float x2 = debugOverlapPositions[1];
            float val = overlapCoefficient(
                n, x1, a1, e1, x2, a2, e2);
            fragColor = vec4(val);
        }
    }
}