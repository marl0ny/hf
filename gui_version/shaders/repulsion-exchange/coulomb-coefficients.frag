#version 120

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

void bFValuesRecurseTo0(
    inout float y0,
    inout float y1,
    inout float y2,
    inout float y3,
    inout float y4,
    float x, 
    int n0,
    int n1,
    int n2,
    int n3,
    int n4) {
    vec4 res;
    float val = sqrt(PI)*erf(sqrt(x))/(2.0*sqrt(x));
    float bx = 0.5*exp(-x)/x;
    for (int nIter = 1; nIter <= n4; nIter++) {
        val = ((float(nIter) - 0.5)/x)*val - bx;
        if (nIter == n0)
            y0 = val;
        if (nIter == n1)
            y1 = val;
        if (nIter == n2)
            y2 = val;
        if (nIter == n3)
            y3 = val;
        if (nIter == n4)
            y4 = val;
    }
}



void bFValuesRecurseUp(
    inout float y0,
    inout float y1,
    inout float y2,
    inout float y3,
    inout float y4,
    float x,
    int n0,
    int n1,
    int n2,
    int n3,
    int n4) {
    complex cVal = computeBFRecurseUpwardsIntermediary(x);
    float val = (mul(complex(0.5, 0.0), cVal)).r;
    vec4 res;
    for (int nIter = 12; nIter > n0; nIter--) {
        val = x/(float(nIter) - 0.5)*val 
            + exp(-x)/(2.0*(float(nIter) - 0.5));
        if (nIter == (n0 + 1))
            y0 = val;
        if (nIter == (n1 + 1))
            y1 = val;
        if (nIter == (n2 + 1))
            y2 = val;
        if (nIter == (n3 + 1))
            y3 = val;
        if (nIter == (n4 + 1))
            y4 = val;
    }
}



void bFValues(
    inout float y0,
    inout float y1,
    inout float y2,
    inout float y3,
    inout float y4,
    float x, 
    int n0,
    int n1,
    int n2,
    int n3,
    int n4) {
    if (abs(x) < 1e-60) {
        y0 = 1.0/(2.0*float(n0) + 1.0);
        y1 = 1.0/(2.0*float(n1) + 1.0);
        y2 = 1.0/(2.0*float(n2) + 1.0);
        y3 = 1.0/(2.0*float(n3) + 1.0);
        y4 = 1.0/(2.0*float(n4) + 1.0);

    }
    if (abs(x) < RECURSE_DIR_TURN_POINT)
        bFValuesRecurseUp(y0, y1, y2, y3, y4, x, n0, n1, n2, n3, n4);
    else
        bFValuesRecurseTo0(y0, y1, y2, y3, y4, x, n0, n1, n2, n3, n4);
}


void bFValuesRecurseTo0(
    inout float y0,
    inout float y1,
    inout float y2,
    inout float y3,
    inout float y4,
    inout float y5,
    float x, 
    int n0,
    int n1,
    int n2,
    int n3,
    int n4,
    int n5) {
    vec4 res;
    float val = sqrt(PI)*erf(sqrt(x))/(2.0*sqrt(x));
    float bx = 0.5*exp(-x)/x;
    for (int nIter = 1; nIter <= n5; nIter++) {
        val = ((float(nIter) - 0.5)/x)*val - bx;
        if (nIter == n0)
            y0 = val;
        if (nIter == n1)
            y1 = val;
        if (nIter == n2)
            y2 = val;
        if (nIter == n3)
            y3 = val;
        if (nIter == n4)
            y4 = val;
        if (nIter == n5)
            y5 = val;
    }
}



void bFValuesRecurseUp(
    inout float y0,
    inout float y1,
    inout float y2,
    inout float y3,
    inout float y4,
    inout float y5,
    float x,
    int n0,
    int n1,
    int n2,
    int n3,
    int n4,
    int n5) {
    complex cVal = computeBFRecurseUpwardsIntermediary(x);
    float val = (mul(complex(0.5, 0.0), cVal)).r;
    vec4 res;
    for (int nIter = 12; nIter > n0; nIter--) {
        val = x/(float(nIter) - 0.5)*val 
            + exp(-x)/(2.0*(float(nIter) - 0.5));
        if (nIter == (n0 + 1))
            y0 = val;
        if (nIter == (n1 + 1))
            y1 = val;
        if (nIter == (n2 + 1))
            y2 = val;
        if (nIter == (n3 + 1))
            y3 = val;
        if (nIter == (n4 + 1))
            y4 = val;
        if (nIter == (n5 + 1))
            y5 = val;
    }
}



void bFValues(
    inout float y0,
    inout float y1,
    inout float y2,
    inout float y3,
    inout float y4,
    inout float y5,
    float x, 
    int n0,
    int n1,
    int n2,
    int n3,
    int n4,
    int n5) {
    if (abs(x) < 1e-60) {
        y0 = 1.0/(2.0*float(n0) + 1.0);
        y1 = 1.0/(2.0*float(n1) + 1.0);
        y2 = 1.0/(2.0*float(n2) + 1.0);
        y3 = 1.0/(2.0*float(n3) + 1.0);
        y4 = 1.0/(2.0*float(n4) + 1.0);
        y5 = 1.0/(2.0*float(n5) + 1.0);

    }
    if (abs(x) < RECURSE_DIR_TURN_POINT)
        bFValuesRecurseUp(y0, y1, y2, y3, y4, y5, x, n0, n1, n2, n3, n4, n5);
    else
        bFValuesRecurseTo0(y0, y1, y2, y3, y4, y5, x, n0, n1, n2, n3, n4, n5);
}


void bFValuesRecurseTo0(
    inout float y0,
    inout float y1,
    inout float y2,
    inout float y3,
    inout float y4,
    inout float y5,
    inout float y6,
    float x, 
    int n0,
    int n1,
    int n2,
    int n3,
    int n4,
    int n5,
    int n6) {
    vec4 res;
    float val = sqrt(PI)*erf(sqrt(x))/(2.0*sqrt(x));
    float bx = 0.5*exp(-x)/x;
    for (int nIter = 1; nIter <= n6; nIter++) {
        val = ((float(nIter) - 0.5)/x)*val - bx;
        if (nIter == n0)
            y0 = val;
        if (nIter == n1)
            y1 = val;
        if (nIter == n2)
            y2 = val;
        if (nIter == n3)
            y3 = val;
        if (nIter == n4)
            y4 = val;
        if (nIter == n5)
            y5 = val;
        if (nIter == n6)
            y6 = val;
    }
}



void bFValuesRecurseUp(
    inout float y0,
    inout float y1,
    inout float y2,
    inout float y3,
    inout float y4,
    inout float y5,
    inout float y6,
    float x,
    int n0,
    int n1,
    int n2,
    int n3,
    int n4,
    int n5,
    int n6) {
    complex cVal = computeBFRecurseUpwardsIntermediary(x);
    float val = (mul(complex(0.5, 0.0), cVal)).r;
    vec4 res;
    for (int nIter = 12; nIter > n0; nIter--) {
        val = x/(float(nIter) - 0.5)*val 
            + exp(-x)/(2.0*(float(nIter) - 0.5));
        if (nIter == (n0 + 1))
            y0 = val;
        if (nIter == (n1 + 1))
            y1 = val;
        if (nIter == (n2 + 1))
            y2 = val;
        if (nIter == (n3 + 1))
            y3 = val;
        if (nIter == (n4 + 1))
            y4 = val;
        if (nIter == (n5 + 1))
            y5 = val;
        if (nIter == (n6 + 1))
            y6 = val;
    }
}



void bFValues(
    inout float y0,
    inout float y1,
    inout float y2,
    inout float y3,
    inout float y4,
    inout float y5,
    inout float y6,
    float x, 
    int n0,
    int n1,
    int n2,
    int n3,
    int n4,
    int n5,
    int n6) {
    if (abs(x) < 1e-60) {
        y0 = 1.0/(2.0*float(n0) + 1.0);
        y1 = 1.0/(2.0*float(n1) + 1.0);
        y2 = 1.0/(2.0*float(n2) + 1.0);
        y3 = 1.0/(2.0*float(n3) + 1.0);
        y4 = 1.0/(2.0*float(n4) + 1.0);
        y5 = 1.0/(2.0*float(n5) + 1.0);
        y6 = 1.0/(2.0*float(n6) + 1.0);

    }
    if (abs(x) < RECURSE_DIR_TURN_POINT)
        bFValuesRecurseUp(y0, y1, y2, y3, y4, y5, y6, x, n0, n1, n2, n3, n4, n5, n6);
    else
        bFValuesRecurseTo0(y0, y1, y2, y3, y4, y5, y6, x, n0, n1, n2, n3, n4, n5, n6);
}


void bFValuesRecurseTo0(
    inout float y0,
    inout float y1,
    inout float y2,
    inout float y3,
    inout float y4,
    inout float y5,
    inout float y6,
    inout float y7,
    float x, 
    int n0,
    int n1,
    int n2,
    int n3,
    int n4,
    int n5,
    int n6,
    int n7) {
    vec4 res;
    float val = sqrt(PI)*erf(sqrt(x))/(2.0*sqrt(x));
    float bx = 0.5*exp(-x)/x;
    for (int nIter = 1; nIter <= n7; nIter++) {
        val = ((float(nIter) - 0.5)/x)*val - bx;
        if (nIter == n0)
            y0 = val;
        if (nIter == n1)
            y1 = val;
        if (nIter == n2)
            y2 = val;
        if (nIter == n3)
            y3 = val;
        if (nIter == n4)
            y4 = val;
        if (nIter == n5)
            y5 = val;
        if (nIter == n6)
            y6 = val;
        if (nIter == n7)
            y7 = val;
    }
}



void bFValuesRecurseUp(
    inout float y0,
    inout float y1,
    inout float y2,
    inout float y3,
    inout float y4,
    inout float y5,
    inout float y6,
    inout float y7,
    float x,
    int n0,
    int n1,
    int n2,
    int n3,
    int n4,
    int n5,
    int n6,
    int n7) {
    complex cVal = computeBFRecurseUpwardsIntermediary(x);
    float val = (mul(complex(0.5, 0.0), cVal)).r;
    vec4 res;
    for (int nIter = 12; nIter > n0; nIter--) {
        val = x/(float(nIter) - 0.5)*val 
            + exp(-x)/(2.0*(float(nIter) - 0.5));
        if (nIter == (n0 + 1))
            y0 = val;
        if (nIter == (n1 + 1))
            y1 = val;
        if (nIter == (n2 + 1))
            y2 = val;
        if (nIter == (n3 + 1))
            y3 = val;
        if (nIter == (n4 + 1))
            y4 = val;
        if (nIter == (n5 + 1))
            y5 = val;
        if (nIter == (n6 + 1))
            y6 = val;
        if (nIter == (n7 + 1))
            y7 = val;
    }
}



void bFValues(
    inout float y0,
    inout float y1,
    inout float y2,
    inout float y3,
    inout float y4,
    inout float y5,
    inout float y6,
    inout float y7,
    float x, 
    int n0,
    int n1,
    int n2,
    int n3,
    int n4,
    int n5,
    int n6,
    int n7) {
    if (abs(x) < 1e-60) {
        y0 = 1.0/(2.0*float(n0) + 1.0);
        y1 = 1.0/(2.0*float(n1) + 1.0);
        y2 = 1.0/(2.0*float(n2) + 1.0);
        y3 = 1.0/(2.0*float(n3) + 1.0);
        y4 = 1.0/(2.0*float(n4) + 1.0);
        y5 = 1.0/(2.0*float(n5) + 1.0);
        y6 = 1.0/(2.0*float(n6) + 1.0);
        y7 = 1.0/(2.0*float(n7) + 1.0);

    }
    if (abs(x) < RECURSE_DIR_TURN_POINT)
        bFValuesRecurseUp(y0, y1, y2, y3, y4, y5, y6, y7, x, n0, n1, n2, n3, n4, n5, n6, n7);
    else
        bFValuesRecurseTo0(y0, y1, y2, y3, y4, y5, y6, y7, x, n0, n1, n2, n3, n4, n5, n6, n7);
}


void bFValuesRecurseTo0(
    inout float y0,
    inout float y1,
    inout float y2,
    inout float y3,
    inout float y4,
    inout float y5,
    inout float y6,
    inout float y7,
    inout float y8,
    float x, 
    int n0,
    int n1,
    int n2,
    int n3,
    int n4,
    int n5,
    int n6,
    int n7,
    int n8) {
    vec4 res;
    float val = sqrt(PI)*erf(sqrt(x))/(2.0*sqrt(x));
    float bx = 0.5*exp(-x)/x;
    for (int nIter = 1; nIter <= n8; nIter++) {
        val = ((float(nIter) - 0.5)/x)*val - bx;
        if (nIter == n0)
            y0 = val;
        if (nIter == n1)
            y1 = val;
        if (nIter == n2)
            y2 = val;
        if (nIter == n3)
            y3 = val;
        if (nIter == n4)
            y4 = val;
        if (nIter == n5)
            y5 = val;
        if (nIter == n6)
            y6 = val;
        if (nIter == n7)
            y7 = val;
        if (nIter == n8)
            y8 = val;
    }
}



void bFValuesRecurseUp(
    inout float y0,
    inout float y1,
    inout float y2,
    inout float y3,
    inout float y4,
    inout float y5,
    inout float y6,
    inout float y7,
    inout float y8,
    float x,
    int n0,
    int n1,
    int n2,
    int n3,
    int n4,
    int n5,
    int n6,
    int n7,
    int n8) {
    complex cVal = computeBFRecurseUpwardsIntermediary(x);
    float val = (mul(complex(0.5, 0.0), cVal)).r;
    vec4 res;
    for (int nIter = 12; nIter > n0; nIter--) {
        val = x/(float(nIter) - 0.5)*val 
            + exp(-x)/(2.0*(float(nIter) - 0.5));
        if (nIter == (n0 + 1))
            y0 = val;
        if (nIter == (n1 + 1))
            y1 = val;
        if (nIter == (n2 + 1))
            y2 = val;
        if (nIter == (n3 + 1))
            y3 = val;
        if (nIter == (n4 + 1))
            y4 = val;
        if (nIter == (n5 + 1))
            y5 = val;
        if (nIter == (n6 + 1))
            y6 = val;
        if (nIter == (n7 + 1))
            y7 = val;
        if (nIter == (n8 + 1))
            y8 = val;
    }
}



void bFValues(
    inout float y0,
    inout float y1,
    inout float y2,
    inout float y3,
    inout float y4,
    inout float y5,
    inout float y6,
    inout float y7,
    inout float y8,
    float x, 
    int n0,
    int n1,
    int n2,
    int n3,
    int n4,
    int n5,
    int n6,
    int n7,
    int n8) {
    if (abs(x) < 1e-60) {
        y0 = 1.0/(2.0*float(n0) + 1.0);
        y1 = 1.0/(2.0*float(n1) + 1.0);
        y2 = 1.0/(2.0*float(n2) + 1.0);
        y3 = 1.0/(2.0*float(n3) + 1.0);
        y4 = 1.0/(2.0*float(n4) + 1.0);
        y5 = 1.0/(2.0*float(n5) + 1.0);
        y6 = 1.0/(2.0*float(n6) + 1.0);
        y7 = 1.0/(2.0*float(n7) + 1.0);
        y8 = 1.0/(2.0*float(n8) + 1.0);

    }
    if (abs(x) < RECURSE_DIR_TURN_POINT)
        bFValuesRecurseUp(y0, y1, y2, y3, y4, y5, y6, y7, y8, x, n0, n1, n2, n3, n4, n5, n6, n7, n8);
    else
        bFValuesRecurseTo0(y0, y1, y2, y3, y4, y5, y6, y7, y8, x, n0, n1, n2, n3, n4, n5, n6, n7, n8);
}


void bFValuesRecurseTo0(
    inout float y0,
    inout float y1,
    inout float y2,
    inout float y3,
    inout float y4,
    inout float y5,
    inout float y6,
    inout float y7,
    inout float y8,
    inout float y9,
    float x, 
    int n0,
    int n1,
    int n2,
    int n3,
    int n4,
    int n5,
    int n6,
    int n7,
    int n8,
    int n9) {
    vec4 res;
    float val = sqrt(PI)*erf(sqrt(x))/(2.0*sqrt(x));
    float bx = 0.5*exp(-x)/x;
    for (int nIter = 1; nIter <= n9; nIter++) {
        val = ((float(nIter) - 0.5)/x)*val - bx;
        if (nIter == n0)
            y0 = val;
        if (nIter == n1)
            y1 = val;
        if (nIter == n2)
            y2 = val;
        if (nIter == n3)
            y3 = val;
        if (nIter == n4)
            y4 = val;
        if (nIter == n5)
            y5 = val;
        if (nIter == n6)
            y6 = val;
        if (nIter == n7)
            y7 = val;
        if (nIter == n8)
            y8 = val;
        if (nIter == n9)
            y9 = val;
    }
}



void bFValuesRecurseUp(
    inout float y0,
    inout float y1,
    inout float y2,
    inout float y3,
    inout float y4,
    inout float y5,
    inout float y6,
    inout float y7,
    inout float y8,
    inout float y9,
    float x,
    int n0,
    int n1,
    int n2,
    int n3,
    int n4,
    int n5,
    int n6,
    int n7,
    int n8,
    int n9) {
    complex cVal = computeBFRecurseUpwardsIntermediary(x);
    float val = (mul(complex(0.5, 0.0), cVal)).r;
    vec4 res;
    for (int nIter = 12; nIter > n0; nIter--) {
        val = x/(float(nIter) - 0.5)*val 
            + exp(-x)/(2.0*(float(nIter) - 0.5));
        if (nIter == (n0 + 1))
            y0 = val;
        if (nIter == (n1 + 1))
            y1 = val;
        if (nIter == (n2 + 1))
            y2 = val;
        if (nIter == (n3 + 1))
            y3 = val;
        if (nIter == (n4 + 1))
            y4 = val;
        if (nIter == (n5 + 1))
            y5 = val;
        if (nIter == (n6 + 1))
            y6 = val;
        if (nIter == (n7 + 1))
            y7 = val;
        if (nIter == (n8 + 1))
            y8 = val;
        if (nIter == (n9 + 1))
            y9 = val;
    }
}



void bFValues(
    inout float y0,
    inout float y1,
    inout float y2,
    inout float y3,
    inout float y4,
    inout float y5,
    inout float y6,
    inout float y7,
    inout float y8,
    inout float y9,
    float x, 
    int n0,
    int n1,
    int n2,
    int n3,
    int n4,
    int n5,
    int n6,
    int n7,
    int n8,
    int n9) {
    if (abs(x) < 1e-60) {
        y0 = 1.0/(2.0*float(n0) + 1.0);
        y1 = 1.0/(2.0*float(n1) + 1.0);
        y2 = 1.0/(2.0*float(n2) + 1.0);
        y3 = 1.0/(2.0*float(n3) + 1.0);
        y4 = 1.0/(2.0*float(n4) + 1.0);
        y5 = 1.0/(2.0*float(n5) + 1.0);
        y6 = 1.0/(2.0*float(n6) + 1.0);
        y7 = 1.0/(2.0*float(n7) + 1.0);
        y8 = 1.0/(2.0*float(n8) + 1.0);
        y9 = 1.0/(2.0*float(n9) + 1.0);

    }
    if (abs(x) < RECURSE_DIR_TURN_POINT)
        bFValuesRecurseUp(y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, x, n0, n1, n2, n3, n4, n5, n6, n7, n8, n9);
    else
        bFValuesRecurseTo0(y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, x, n0, n1, n2, n3, n4, n5, n6, n7, n8, n9);
}

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

float coulombCoefficientHelper(
    ivec3 indices, int n, float orbExp, vec3 r12) {
    vec3 s12 = r12;
    // if (indices[2] > indices[1]) {
    //     SWAP_I(indices[2], indices[1]);
    //     SWAP_F(s12.z, s12.y);
    // }
    // // 1 3 2
    // if (indices[1] > indices[0]) {
    //     SWAP_I(indices[0], indices[1]);
    //     SWAP_F(s12.y, s12.x);
    // }
    // // 3 1 2
    // if (indices[2] > indices[1]) {
    //     SWAP_I(indices[2], indices[1]);
    //     SWAP_F(s12.z, s12.y);
    // }
    int i = indices[0], j = indices[1], k = indices[2];
    int hexIJK = i*16*16 + j*16 + k;
    float e = orbExp;
    float r2 = dot(r12, r12);
    float val = 0.0;
    float x = s12.x, y = s12.y, z = s12.z;
    float x2 = x*x, y2 = y*y, z2 = z*z;
    float x4 = x2*x2, y4 = y2*y2, z4 = z2*z2;
    if (i < 2) {
        if (hexIJK == 0x000) {
            return bf(e*r2, n)*pw(-2.0*e, n);
        }
        if (hexIJK == 0x100) {
            return x*bf(e*r2, n + 1)*pw(-2.0*e, n + 1);
        }
        if (hexIJK == 0x110) {
            return x*y*bf(e*r2, n + 2)*pw(-2.0*e, n + 2);
        }
        if (hexIJK == 0x111) {
            return x*y*z*bf(e*r2, n + 3)*pw(-2.0*e, n + 3);
        }
    } else if (i == 2) {
        if (hexIJK == 0x200) {
            vec2 vecBF = vecBFValues(e*r2, ivec2(n + 1, n + 2));
            float bF1 = vecBF[0], bF2 = vecBF[1];
            return 1.0*bF1*pw(-2.0*e, n + 1) + 1.0*bF2*x2*pw(-2.0*e, n + 2);
        }
        if (hexIJK == 0x210) {
            vec2 vecBF = vecBFValues(e*r2, ivec2(n + 2, n + 3));
            float bF2 = vecBF[0], bF3 = vecBF[1];
            return 1.0*y*(bF2*pw(-2.0*e, n + 2) + bF3*x2*pw(-2.0*e, n + 3));
        }
        if (hexIJK == 0x211) {
            vec2 vecBF = vecBFValues(e*r2, ivec2(n + 3, n + 4));
            float bF3 = vecBF[0], bF4 = vecBF[1];
            return 1.0*y*z*(bF3*pw(-2.0*e, n + 3) + bF4*x2*pw(-2.0*e, n + 4));
        }
        if (hexIJK == 0x220) {
            vec3 vecBF = vecBFValues(e*r2, ivec3(n + 2, n + 3, n + 4));
            float bF2 = vecBF[0], bF3 = vecBF[1], bF4 = vecBF[2];
            return 1.0*bF2*pw(-2.0*e, n + 2) + 1.0*bF3*x2*pw(-2.0*e, n + 3) + 1.0*bF3*y2*pw(-2.0*e, n + 3) + 1.0*bF4*x2*y2*pw(-2.0*e, n + 4);
        }
        if (hexIJK == 0x221) {
            vec3 vecBF = vecBFValues(e*r2, ivec3(n + 3, n + 4, n + 5));
            float bF3 = vecBF[0], bF4 = vecBF[1], bF5 = vecBF[2];
            return 1.0*z*(bF3*pw(-2.0*e, n + 3) + bF4*x2*pw(-2.0*e, n + 4) + bF4*y2*pw(-2.0*e, n + 4) + bF5*x2*y2*pw(-2.0*e, n + 5));
        }
        if (hexIJK == 0x222) {
            vec4 vecBF = vecBFValues(e*r2, ivec4(n + 3, n + 4, n + 5, n + 6));
            float bF3 = vecBF[0], bF4 = vecBF[1], bF5 = vecBF[2], bF6 = vecBF[3];
            return 1.0*bF3*pw(-2.0*e, n + 3) + 1.0*bF4*x2*pw(-2.0*e, n + 4) + 1.0*bF4*y2*pw(-2.0*e, n + 4) + 1.0*bF4*z2*pw(-2.0*e, n + 4) + 1.0*bF5*x2*y2*pw(-2.0*e, n + 5) + 1.0*bF5*x2*z2*pw(-2.0*e, n + 5) + 1.0*bF5*y2*z2*pw(-2.0*e, n + 5) + 1.0*bF6*x2*y2*z2*pw(-2.0*e, n + 6);
        }
    } else if (i == 3) {
        if (hexIJK == 0x300) {
            vec2 vecBF = vecBFValues(e*r2, ivec2(n + 2, n + 3));
            float bF2 = vecBF[0], bF3 = vecBF[1];
            return x*(3.0*bF2*pw(-2.0*e, n + 2) + 1.0*bF3*x2*pw(-2.0*e, n + 3));
        }
        if (hexIJK == 0x310) {
            vec2 vecBF = vecBFValues(e*r2, ivec2(n + 3, n + 4));
            float bF3 = vecBF[0], bF4 = vecBF[1];
            return x*y*(3.0*bF3*pw(-2.0*e, n + 3) + 1.0*bF4*x2*pw(-2.0*e, n + 4));
        }
        if (hexIJK == 0x311) {
            vec2 vecBF = vecBFValues(e*r2, ivec2(n + 4, n + 5));
            float bF4 = vecBF[0], bF5 = vecBF[1];
            return x*y*z*(3.0*bF4*pw(-2.0*e, n + 4) + 1.0*bF5*x2*pw(-2.0*e, n + 5));
        }
        if (hexIJK == 0x320) {
            vec3 vecBF = vecBFValues(e*r2, ivec3(n + 3, n + 4, n + 5));
            float bF3 = vecBF[0], bF4 = vecBF[1], bF5 = vecBF[2];
            return x*(3.0*bF3*pw(-2.0*e, n + 3) + 1.0*bF4*x2*pw(-2.0*e, n + 4) + 3.0*bF4*y2*pw(-2.0*e, n + 4) + 1.0*bF5*x2*y2*pw(-2.0*e, n + 5));
        }
        if (hexIJK == 0x321) {
            vec3 vecBF = vecBFValues(e*r2, ivec3(n + 4, n + 5, n + 6));
            float bF4 = vecBF[0], bF5 = vecBF[1], bF6 = vecBF[2];
            return x*z*(3.0*bF4*pw(-2.0*e, n + 4) + 1.0*bF5*x2*pw(-2.0*e, n + 5) + 3.0*bF5*y2*pw(-2.0*e, n + 5) + 1.0*bF6*x2*y2*pw(-2.0*e, n + 6));
        }
        if (hexIJK == 0x322) {
            vec4 vecBF = vecBFValues(e*r2, ivec4(n + 4, n + 5, n + 6, n + 7));
            float bF4 = vecBF[0], bF5 = vecBF[1], bF6 = vecBF[2], bF7 = vecBF[3];
            return x*(3.0*bF4*pw(-2.0*e, n + 4) + 1.0*bF5*x2*pw(-2.0*e, n + 5) + 3.0*bF5*y2*pw(-2.0*e, n + 5) + 3.0*bF5*z2*pw(-2.0*e, n + 5) + 1.0*bF6*x2*y2*pw(-2.0*e, n + 6) + 1.0*bF6*x2*z2*pw(-2.0*e, n + 6) + 3.0*bF6*y2*z2*pw(-2.0*e, n + 6) + 1.0*bF7*x2*y2*z2*pw(-2.0*e, n + 7));
        }
        if (hexIJK == 0x330) {
            vec3 vecBF = vecBFValues(e*r2, ivec3(n + 4, n + 5, n + 6));
            float bF4 = vecBF[0], bF5 = vecBF[1], bF6 = vecBF[2];
            return x*y*(9.0*bF4*pw(-2.0*e, n + 4) + 3.0*bF5*x2*pw(-2.0*e, n + 5) + 3.0*bF5*y2*pw(-2.0*e, n + 5) + 1.0*bF6*x2*y2*pw(-2.0*e, n + 6));
        }
        if (hexIJK == 0x331) {
            vec3 vecBF = vecBFValues(e*r2, ivec3(n + 5, n + 6, n + 7));
            float bF5 = vecBF[0], bF6 = vecBF[1], bF7 = vecBF[2];
            return x*y*z*(9.0*bF5*pw(-2.0*e, n + 5) + 3.0*bF6*x2*pw(-2.0*e, n + 6) + 3.0*bF6*y2*pw(-2.0*e, n + 6) + 1.0*bF7*x2*y2*pw(-2.0*e, n + 7));
        }
        if (hexIJK == 0x332) {
            vec4 vecBF = vecBFValues(e*r2, ivec4(n + 5, n + 6, n + 7, n + 8));
            float bF5 = vecBF[0], bF6 = vecBF[1], bF7 = vecBF[2], bF8 = vecBF[3];
            return x*y*(9.0*bF5*pw(-2.0*e, n + 5) + 3.0*bF6*x2*pw(-2.0*e, n + 6) + 3.0*bF6*y2*pw(-2.0*e, n + 6) + 9.0*bF6*z2*pw(-2.0*e, n + 6) + 1.0*bF7*x2*y2*pw(-2.0*e, n + 7) + 3.0*bF7*x2*z2*pw(-2.0*e, n + 7) + 3.0*bF7*y2*z2*pw(-2.0*e, n + 7) + 1.0*bF8*x2*y2*z2*pw(-2.0*e, n + 8));
        }
        if (hexIJK == 0x333) {
            vec4 vecBF = vecBFValues(e*r2, ivec4(n + 6, n + 7, n + 8, n + 9));
            float bF6 = vecBF[0], bF7 = vecBF[1], bF8 = vecBF[2], bF9 = vecBF[3];
            return x*y*z*(27.0*bF6*pw(-2.0*e, n + 6) + 9.0*bF7*x2*pw(-2.0*e, n + 7) + 9.0*bF7*y2*pw(-2.0*e, n + 7) + 9.0*bF7*z2*pw(-2.0*e, n + 7) + 3.0*bF8*x2*y2*pw(-2.0*e, n + 8) + 3.0*bF8*x2*z2*pw(-2.0*e, n + 8) + 3.0*bF8*y2*z2*pw(-2.0*e, n + 8) + 1.0*bF9*x2*y2*z2*pw(-2.0*e, n + 9));
        }
    } else if (i == 4) {
        if (hexIJK == 0x400) {
            vec3 vecBF = vecBFValues(e*r2, ivec3(n + 2, n + 3, n + 4));
            float bF2 = vecBF[0], bF3 = vecBF[1], bF4 = vecBF[2];
            return 3.0*bF2*pw(-2.0*e, n + 2) + 6.0*bF3*x2*pw(-2.0*e, n + 3) + 1.0*bF4*x4*pw(-2.0*e, n + 4);
        }
        if (hexIJK == 0x410) {
            vec3 vecBF = vecBFValues(e*r2, ivec3(n + 3, n + 4, n + 5));
            float bF3 = vecBF[0], bF4 = vecBF[1], bF5 = vecBF[2];
            return y*(3.0*bF3*pw(-2.0*e, n + 3) + 6.0*bF4*x2*pw(-2.0*e, n + 4) + 1.0*bF5*x4*pw(-2.0*e, n + 5));
        }
        if (hexIJK == 0x411) {
            vec3 vecBF = vecBFValues(e*r2, ivec3(n + 4, n + 5, n + 6));
            float bF4 = vecBF[0], bF5 = vecBF[1], bF6 = vecBF[2];
            return y*z*(3.0*bF4*pw(-2.0*e, n + 4) + 6.0*bF5*x2*pw(-2.0*e, n + 5) + 1.0*bF6*x4*pw(-2.0*e, n + 6));
        }
        if (hexIJK == 0x420) {
            vec4 vecBF = vecBFValues(e*r2, ivec4(n + 3, n + 4, n + 5, n + 6));
            float bF3 = vecBF[0], bF4 = vecBF[1], bF5 = vecBF[2], bF6 = vecBF[3];
            return 3.0*bF3*pw(-2.0*e, n + 3) + 6.0*bF4*x2*pw(-2.0*e, n + 4) + 3.0*bF4*y2*pw(-2.0*e, n + 4) + 6.0*bF5*x2*y2*pw(-2.0*e, n + 5) + 1.0*bF5*x4*pw(-2.0*e, n + 5) + 1.0*bF6*x4*y2*pw(-2.0*e, n + 6);
        }
        if (hexIJK == 0x421) {
            vec4 vecBF = vecBFValues(e*r2, ivec4(n + 4, n + 5, n + 6, n + 7));
            float bF4 = vecBF[0], bF5 = vecBF[1], bF6 = vecBF[2], bF7 = vecBF[3];
            return z*(3.0*bF4*pw(-2.0*e, n + 4) + 6.0*bF5*x2*pw(-2.0*e, n + 5) + 3.0*bF5*y2*pw(-2.0*e, n + 5) + 6.0*bF6*x2*y2*pw(-2.0*e, n + 6) + 1.0*bF6*x4*pw(-2.0*e, n + 6) + 1.0*bF7*x4*y2*pw(-2.0*e, n + 7));
        }
        if (hexIJK == 0x422) {
            float bF4 = 0.0, bF5 = 0.0, bF6 = 0.0, bF7 = 0.0, bF8 = 0.0;
            bFValues(bF4, bF5, bF6, bF7, bF8, e*r2, n + 4, n + 5, n + 6, n + 7, n + 8);
            return 3.0*bF4*pw(-2.0*e, n + 4) + 6.0*bF5*x2*pw(-2.0*e, n + 5) + 3.0*bF5*y2*pw(-2.0*e, n + 5) + 3.0*bF5*z2*pw(-2.0*e, n + 5) + 6.0*bF6*x2*y2*pw(-2.0*e, n + 6) + 6.0*bF6*x2*z2*pw(-2.0*e, n + 6) + 1.0*bF6*x4*pw(-2.0*e, n + 6) + 3.0*bF6*y2*z2*pw(-2.0*e, n + 6) + 6.0*bF7*x2*y2*z2*pw(-2.0*e, n + 7) + 1.0*bF7*x4*y2*pw(-2.0*e, n + 7) + 1.0*bF7*x4*z2*pw(-2.0*e, n + 7) + 1.0*bF8*x4*y2*z2*pw(-2.0*e, n + 8);
        }
        if (hexIJK == 0x430) {
            vec4 vecBF = vecBFValues(e*r2, ivec4(n + 4, n + 5, n + 6, n + 7));
            float bF4 = vecBF[0], bF5 = vecBF[1], bF6 = vecBF[2], bF7 = vecBF[3];
            return y*(9.0*bF4*pw(-2.0*e, n + 4) + 18.0*bF5*x2*pw(-2.0*e, n + 5) + 3.0*bF5*y2*pw(-2.0*e, n + 5) + 6.0*bF6*x2*y2*pw(-2.0*e, n + 6) + 3.0*bF6*x4*pw(-2.0*e, n + 6) + 1.0*bF7*x4*y2*pw(-2.0*e, n + 7));
        }
        if (hexIJK == 0x431) {
            vec4 vecBF = vecBFValues(e*r2, ivec4(n + 5, n + 6, n + 7, n + 8));
            float bF5 = vecBF[0], bF6 = vecBF[1], bF7 = vecBF[2], bF8 = vecBF[3];
            return y*z*(9.0*bF5*pw(-2.0*e, n + 5) + 18.0*bF6*x2*pw(-2.0*e, n + 6) + 3.0*bF6*y2*pw(-2.0*e, n + 6) + 6.0*bF7*x2*y2*pw(-2.0*e, n + 7) + 3.0*bF7*x4*pw(-2.0*e, n + 7) + 1.0*bF8*x4*y2*pw(-2.0*e, n + 8));
        }
        if (hexIJK == 0x432) {
            float bF5 = 0.0, bF6 = 0.0, bF7 = 0.0, bF8 = 0.0, bF9 = 0.0;
            bFValues(bF5, bF6, bF7, bF8, bF9, e*r2, n + 5, n + 6, n + 7, n + 8, n + 9);
            return y*(9.0*bF5*pw(-2.0*e, n + 5) + 18.0*bF6*x2*pw(-2.0*e, n + 6) + 3.0*bF6*y2*pw(-2.0*e, n + 6) + 9.0*bF6*z2*pw(-2.0*e, n + 6) + 6.0*bF7*x2*y2*pw(-2.0*e, n + 7) + 18.0*bF7*x2*z2*pw(-2.0*e, n + 7) + 3.0*bF7*x4*pw(-2.0*e, n + 7) + 3.0*bF7*y2*z2*pw(-2.0*e, n + 7) + 6.0*bF8*x2*y2*z2*pw(-2.0*e, n + 8) + 1.0*bF8*x4*y2*pw(-2.0*e, n + 8) + 3.0*bF8*x4*z2*pw(-2.0*e, n + 8) + 1.0*bF9*x4*y2*z2*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x433) {
            float bF6 = 0.0, bF7 = 0.0, bF8 = 0.0, bF9 = 0.0, bF10 = 0.0;
            bFValues(bF6, bF7, bF8, bF9, bF10, e*r2, n + 6, n + 7, n + 8, n + 9, n + 10);
            return y*z*(1.0*bF10*x4*y2*z2*pw(-2.0*e, n + 10) + 27.0*bF6*pw(-2.0*e, n + 6) + 54.0*bF7*x2*pw(-2.0*e, n + 7) + 9.0*bF7*y2*pw(-2.0*e, n + 7) + 9.0*bF7*z2*pw(-2.0*e, n + 7) + 18.0*bF8*x2*y2*pw(-2.0*e, n + 8) + 18.0*bF8*x2*z2*pw(-2.0*e, n + 8) + 9.0*bF8*x4*pw(-2.0*e, n + 8) + 3.0*bF8*y2*z2*pw(-2.0*e, n + 8) + 6.0*bF9*x2*y2*z2*pw(-2.0*e, n + 9) + 3.0*bF9*x4*y2*pw(-2.0*e, n + 9) + 3.0*bF9*x4*z2*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x440) {
            float bF4 = 0.0, bF5 = 0.0, bF6 = 0.0, bF7 = 0.0, bF8 = 0.0;
            bFValues(bF4, bF5, bF6, bF7, bF8, e*r2, n + 4, n + 5, n + 6, n + 7, n + 8);
            return 9.0*bF4*pw(-2.0*e, n + 4) + 18.0*bF5*x2*pw(-2.0*e, n + 5) + 18.0*bF5*y2*pw(-2.0*e, n + 5) + 36.0*bF6*x2*y2*pw(-2.0*e, n + 6) + 3.0*bF6*x4*pw(-2.0*e, n + 6) + 3.0*bF6*y4*pw(-2.0*e, n + 6) + 6.0*bF7*x2*y4*pw(-2.0*e, n + 7) + 6.0*bF7*x4*y2*pw(-2.0*e, n + 7) + 1.0*bF8*x4*y4*pw(-2.0*e, n + 8);
        }
        if (hexIJK == 0x441) {
            float bF5 = 0.0, bF6 = 0.0, bF7 = 0.0, bF8 = 0.0, bF9 = 0.0;
            bFValues(bF5, bF6, bF7, bF8, bF9, e*r2, n + 5, n + 6, n + 7, n + 8, n + 9);
            return z*(9.0*bF5*pw(-2.0*e, n + 5) + 18.0*bF6*x2*pw(-2.0*e, n + 6) + 18.0*bF6*y2*pw(-2.0*e, n + 6) + 36.0*bF7*x2*y2*pw(-2.0*e, n + 7) + 3.0*bF7*x4*pw(-2.0*e, n + 7) + 3.0*bF7*y4*pw(-2.0*e, n + 7) + 6.0*bF8*x2*y4*pw(-2.0*e, n + 8) + 6.0*bF8*x4*y2*pw(-2.0*e, n + 8) + 1.0*bF9*x4*y4*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x442) {
            float bF5 = 0.0, bF6 = 0.0, bF7 = 0.0, bF8 = 0.0, bF9 = 0.0, bF10 = 0.0;
            bFValues(bF5, bF6, bF7, bF8, bF9, bF10, e*r2, n + 5, n + 6, n + 7, n + 8, n + 9, n + 10);
            return 1.0*bF10*x4*y4*z2*pw(-2.0*e, n + 10) + 9.0*bF5*pw(-2.0*e, n + 5) + 18.0*bF6*x2*pw(-2.0*e, n + 6) + 18.0*bF6*y2*pw(-2.0*e, n + 6) + 9.0*bF6*z2*pw(-2.0*e, n + 6) + 36.0*bF7*x2*y2*pw(-2.0*e, n + 7) + 18.0*bF7*x2*z2*pw(-2.0*e, n + 7) + 3.0*bF7*x4*pw(-2.0*e, n + 7) + 18.0*bF7*y2*z2*pw(-2.0*e, n + 7) + 3.0*bF7*y4*pw(-2.0*e, n + 7) + 36.0*bF8*x2*y2*z2*pw(-2.0*e, n + 8) + 6.0*bF8*x2*y4*pw(-2.0*e, n + 8) + 6.0*bF8*x4*y2*pw(-2.0*e, n + 8) + 3.0*bF8*x4*z2*pw(-2.0*e, n + 8) + 3.0*bF8*y4*z2*pw(-2.0*e, n + 8) + 6.0*bF9*x2*y4*z2*pw(-2.0*e, n + 9) + 6.0*bF9*x4*y2*z2*pw(-2.0*e, n + 9) + 1.0*bF9*x4*y4*pw(-2.0*e, n + 9);
        }
        if (hexIJK == 0x443) {
            float bF6 = 0.0, bF7 = 0.0, bF8 = 0.0, bF9 = 0.0, bF10 = 0.0, bF11 = 0.0;
            bFValues(bF6, bF7, bF8, bF9, bF10, bF11, e*r2, n + 6, n + 7, n + 8, n + 9, n + 10, n + 11);
            return z*(6.0*bF10*x2*y4*z2*pw(-2.0*e, n + 10) + 6.0*bF10*x4*y2*z2*pw(-2.0*e, n + 10) + 3.0*bF10*x4*y4*pw(-2.0*e, n + 10) + 1.0*bF11*x4*y4*z2*pw(-2.0*e, n + 11) + 27.0*bF6*pw(-2.0*e, n + 6) + 54.0*bF7*x2*pw(-2.0*e, n + 7) + 54.0*bF7*y2*pw(-2.0*e, n + 7) + 9.0*bF7*z2*pw(-2.0*e, n + 7) + 108.0*bF8*x2*y2*pw(-2.0*e, n + 8) + 18.0*bF8*x2*z2*pw(-2.0*e, n + 8) + 9.0*bF8*x4*pw(-2.0*e, n + 8) + 18.0*bF8*y2*z2*pw(-2.0*e, n + 8) + 9.0*bF8*y4*pw(-2.0*e, n + 8) + 36.0*bF9*x2*y2*z2*pw(-2.0*e, n + 9) + 18.0*bF9*x2*y4*pw(-2.0*e, n + 9) + 18.0*bF9*x4*y2*pw(-2.0*e, n + 9) + 3.0*bF9*x4*z2*pw(-2.0*e, n + 9) + 3.0*bF9*y4*z2*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x444) {
            float bF6 = 0.0, bF7 = 0.0, bF8 = 0.0, bF9 = 0.0, bF10 = 0.0, bF11 = 0.0, bF12 = 0.0;
            bFValues(bF6, bF7, bF8, bF9, bF10, bF11, bF12, e*r2, n + 6, n + 7, n + 8, n + 9, n + 10, n + 11, n + 12);
            return 36.0*bF10*x2*y2*z4*pw(-2.0*e, n + 10) + 36.0*bF10*x2*y4*z2*pw(-2.0*e, n + 10) + 36.0*bF10*x4*y2*z2*pw(-2.0*e, n + 10) + 3.0*bF10*x4*y4*pw(-2.0*e, n + 10) + 3.0*bF10*x4*z4*pw(-2.0*e, n + 10) + 3.0*bF10*y4*z4*pw(-2.0*e, n + 10) + 6.0*bF11*x2*y4*z4*pw(-2.0*e, n + 11) + 6.0*bF11*x4*y2*z4*pw(-2.0*e, n + 11) + 6.0*bF11*x4*y4*z2*pw(-2.0*e, n + 11) + 1.0*bF12*x4*y4*z4*pw(-2.0*e, n + 12) + 27.0*bF6*pw(-2.0*e, n + 6) + 54.0*bF7*x2*pw(-2.0*e, n + 7) + 54.0*bF7*y2*pw(-2.0*e, n + 7) + 54.0*bF7*z2*pw(-2.0*e, n + 7) + 108.0*bF8*x2*y2*pw(-2.0*e, n + 8) + 108.0*bF8*x2*z2*pw(-2.0*e, n + 8) + 9.0*bF8*x4*pw(-2.0*e, n + 8) + 108.0*bF8*y2*z2*pw(-2.0*e, n + 8) + 9.0*bF8*y4*pw(-2.0*e, n + 8) + 9.0*bF8*z4*pw(-2.0*e, n + 8) + 216.0*bF9*x2*y2*z2*pw(-2.0*e, n + 9) + 18.0*bF9*x2*y4*pw(-2.0*e, n + 9) + 18.0*bF9*x2*z4*pw(-2.0*e, n + 9) + 18.0*bF9*x4*y2*pw(-2.0*e, n + 9) + 18.0*bF9*x4*z2*pw(-2.0*e, n + 9) + 18.0*bF9*y2*z4*pw(-2.0*e, n + 9) + 18.0*bF9*y4*z2*pw(-2.0*e, n + 9);
        }
    } else if (i == 5) {
        if (hexIJK == 0x500) {
            vec3 vecBF = vecBFValues(e*r2, ivec3(n + 3, n + 4, n + 5));
            float bF3 = vecBF[0], bF4 = vecBF[1], bF5 = vecBF[2];
            return x*(15.0*bF3*pw(-2.0*e, n + 3) + 10.0*bF4*x2*pw(-2.0*e, n + 4) + 1.0*bF5*x4*pw(-2.0*e, n + 5));
        }
        if (hexIJK == 0x510) {
            vec3 vecBF = vecBFValues(e*r2, ivec3(n + 4, n + 5, n + 6));
            float bF4 = vecBF[0], bF5 = vecBF[1], bF6 = vecBF[2];
            return x*y*(15.0*bF4*pw(-2.0*e, n + 4) + 10.0*bF5*x2*pw(-2.0*e, n + 5) + 1.0*bF6*x4*pw(-2.0*e, n + 6));
        }
        if (hexIJK == 0x511) {
            vec3 vecBF = vecBFValues(e*r2, ivec3(n + 5, n + 6, n + 7));
            float bF5 = vecBF[0], bF6 = vecBF[1], bF7 = vecBF[2];
            return x*y*z*(15.0*bF5*pw(-2.0*e, n + 5) + 10.0*bF6*x2*pw(-2.0*e, n + 6) + 1.0*bF7*x4*pw(-2.0*e, n + 7));
        }
        if (hexIJK == 0x520) {
            vec4 vecBF = vecBFValues(e*r2, ivec4(n + 4, n + 5, n + 6, n + 7));
            float bF4 = vecBF[0], bF5 = vecBF[1], bF6 = vecBF[2], bF7 = vecBF[3];
            return x*(15.0*bF4*pw(-2.0*e, n + 4) + 10.0*bF5*x2*pw(-2.0*e, n + 5) + 15.0*bF5*y2*pw(-2.0*e, n + 5) + 10.0*bF6*x2*y2*pw(-2.0*e, n + 6) + 1.0*bF6*x4*pw(-2.0*e, n + 6) + 1.0*bF7*x4*y2*pw(-2.0*e, n + 7));
        }
        if (hexIJK == 0x521) {
            vec4 vecBF = vecBFValues(e*r2, ivec4(n + 5, n + 6, n + 7, n + 8));
            float bF5 = vecBF[0], bF6 = vecBF[1], bF7 = vecBF[2], bF8 = vecBF[3];
            return x*z*(15.0*bF5*pw(-2.0*e, n + 5) + 10.0*bF6*x2*pw(-2.0*e, n + 6) + 15.0*bF6*y2*pw(-2.0*e, n + 6) + 10.0*bF7*x2*y2*pw(-2.0*e, n + 7) + 1.0*bF7*x4*pw(-2.0*e, n + 7) + 1.0*bF8*x4*y2*pw(-2.0*e, n + 8));
        }
        if (hexIJK == 0x522) {
            float bF5 = 0.0, bF6 = 0.0, bF7 = 0.0, bF8 = 0.0, bF9 = 0.0;
            bFValues(bF5, bF6, bF7, bF8, bF9, e*r2, n + 5, n + 6, n + 7, n + 8, n + 9);
            return x*(15.0*bF5*pw(-2.0*e, n + 5) + 10.0*bF6*x2*pw(-2.0*e, n + 6) + 15.0*bF6*y2*pw(-2.0*e, n + 6) + 15.0*bF6*z2*pw(-2.0*e, n + 6) + 10.0*bF7*x2*y2*pw(-2.0*e, n + 7) + 10.0*bF7*x2*z2*pw(-2.0*e, n + 7) + 1.0*bF7*x4*pw(-2.0*e, n + 7) + 15.0*bF7*y2*z2*pw(-2.0*e, n + 7) + 10.0*bF8*x2*y2*z2*pw(-2.0*e, n + 8) + 1.0*bF8*x4*y2*pw(-2.0*e, n + 8) + 1.0*bF8*x4*z2*pw(-2.0*e, n + 8) + 1.0*bF9*x4*y2*z2*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x530) {
            vec4 vecBF = vecBFValues(e*r2, ivec4(n + 5, n + 6, n + 7, n + 8));
            float bF5 = vecBF[0], bF6 = vecBF[1], bF7 = vecBF[2], bF8 = vecBF[3];
            return x*y*(45.0*bF5*pw(-2.0*e, n + 5) + 30.0*bF6*x2*pw(-2.0*e, n + 6) + 15.0*bF6*y2*pw(-2.0*e, n + 6) + 10.0*bF7*x2*y2*pw(-2.0*e, n + 7) + 3.0*bF7*x4*pw(-2.0*e, n + 7) + 1.0*bF8*x4*y2*pw(-2.0*e, n + 8));
        }
        if (hexIJK == 0x531) {
            vec4 vecBF = vecBFValues(e*r2, ivec4(n + 6, n + 7, n + 8, n + 9));
            float bF6 = vecBF[0], bF7 = vecBF[1], bF8 = vecBF[2], bF9 = vecBF[3];
            return x*y*z*(45.0*bF6*pw(-2.0*e, n + 6) + 30.0*bF7*x2*pw(-2.0*e, n + 7) + 15.0*bF7*y2*pw(-2.0*e, n + 7) + 10.0*bF8*x2*y2*pw(-2.0*e, n + 8) + 3.0*bF8*x4*pw(-2.0*e, n + 8) + 1.0*bF9*x4*y2*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x532) {
            float bF6 = 0.0, bF7 = 0.0, bF8 = 0.0, bF9 = 0.0, bF10 = 0.0;
            bFValues(bF6, bF7, bF8, bF9, bF10, e*r2, n + 6, n + 7, n + 8, n + 9, n + 10);
            return x*y*(1.0*bF10*x4*y2*z2*pw(-2.0*e, n + 10) + 45.0*bF6*pw(-2.0*e, n + 6) + 30.0*bF7*x2*pw(-2.0*e, n + 7) + 15.0*bF7*y2*pw(-2.0*e, n + 7) + 45.0*bF7*z2*pw(-2.0*e, n + 7) + 10.0*bF8*x2*y2*pw(-2.0*e, n + 8) + 30.0*bF8*x2*z2*pw(-2.0*e, n + 8) + 3.0*bF8*x4*pw(-2.0*e, n + 8) + 15.0*bF8*y2*z2*pw(-2.0*e, n + 8) + 10.0*bF9*x2*y2*z2*pw(-2.0*e, n + 9) + 1.0*bF9*x4*y2*pw(-2.0*e, n + 9) + 3.0*bF9*x4*z2*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x533) {
            float bF7 = 0.0, bF8 = 0.0, bF9 = 0.0, bF10 = 0.0, bF11 = 0.0;
            bFValues(bF7, bF8, bF9, bF10, bF11, e*r2, n + 7, n + 8, n + 9, n + 10, n + 11);
            return x*y*z*(10.0*bF10*x2*y2*z2*pw(-2.0*e, n + 10) + 3.0*bF10*x4*y2*pw(-2.0*e, n + 10) + 3.0*bF10*x4*z2*pw(-2.0*e, n + 10) + 1.0*bF11*x4*y2*z2*pw(-2.0*e, n + 11) + 135.0*bF7*pw(-2.0*e, n + 7) + 90.0*bF8*x2*pw(-2.0*e, n + 8) + 45.0*bF8*y2*pw(-2.0*e, n + 8) + 45.0*bF8*z2*pw(-2.0*e, n + 8) + 30.0*bF9*x2*y2*pw(-2.0*e, n + 9) + 30.0*bF9*x2*z2*pw(-2.0*e, n + 9) + 9.0*bF9*x4*pw(-2.0*e, n + 9) + 15.0*bF9*y2*z2*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x540) {
            float bF5 = 0.0, bF6 = 0.0, bF7 = 0.0, bF8 = 0.0, bF9 = 0.0;
            bFValues(bF5, bF6, bF7, bF8, bF9, e*r2, n + 5, n + 6, n + 7, n + 8, n + 9);
            return x*(45.0*bF5*pw(-2.0*e, n + 5) + 30.0*bF6*x2*pw(-2.0*e, n + 6) + 90.0*bF6*y2*pw(-2.0*e, n + 6) + 60.0*bF7*x2*y2*pw(-2.0*e, n + 7) + 3.0*bF7*x4*pw(-2.0*e, n + 7) + 15.0*bF7*y4*pw(-2.0*e, n + 7) + 10.0*bF8*x2*y4*pw(-2.0*e, n + 8) + 6.0*bF8*x4*y2*pw(-2.0*e, n + 8) + 1.0*bF9*x4*y4*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x541) {
            float bF6 = 0.0, bF7 = 0.0, bF8 = 0.0, bF9 = 0.0, bF10 = 0.0;
            bFValues(bF6, bF7, bF8, bF9, bF10, e*r2, n + 6, n + 7, n + 8, n + 9, n + 10);
            return x*z*(1.0*bF10*x4*y4*pw(-2.0*e, n + 10) + 45.0*bF6*pw(-2.0*e, n + 6) + 30.0*bF7*x2*pw(-2.0*e, n + 7) + 90.0*bF7*y2*pw(-2.0*e, n + 7) + 60.0*bF8*x2*y2*pw(-2.0*e, n + 8) + 3.0*bF8*x4*pw(-2.0*e, n + 8) + 15.0*bF8*y4*pw(-2.0*e, n + 8) + 10.0*bF9*x2*y4*pw(-2.0*e, n + 9) + 6.0*bF9*x4*y2*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x542) {
            float bF6 = 0.0, bF7 = 0.0, bF8 = 0.0, bF9 = 0.0, bF10 = 0.0, bF11 = 0.0;
            bFValues(bF6, bF7, bF8, bF9, bF10, bF11, e*r2, n + 6, n + 7, n + 8, n + 9, n + 10, n + 11);
            return x*(10.0*bF10*x2*y4*z2*pw(-2.0*e, n + 10) + 6.0*bF10*x4*y2*z2*pw(-2.0*e, n + 10) + 1.0*bF10*x4*y4*pw(-2.0*e, n + 10) + 1.0*bF11*x4*y4*z2*pw(-2.0*e, n + 11) + 45.0*bF6*pw(-2.0*e, n + 6) + 30.0*bF7*x2*pw(-2.0*e, n + 7) + 90.0*bF7*y2*pw(-2.0*e, n + 7) + 45.0*bF7*z2*pw(-2.0*e, n + 7) + 60.0*bF8*x2*y2*pw(-2.0*e, n + 8) + 30.0*bF8*x2*z2*pw(-2.0*e, n + 8) + 3.0*bF8*x4*pw(-2.0*e, n + 8) + 90.0*bF8*y2*z2*pw(-2.0*e, n + 8) + 15.0*bF8*y4*pw(-2.0*e, n + 8) + 60.0*bF9*x2*y2*z2*pw(-2.0*e, n + 9) + 10.0*bF9*x2*y4*pw(-2.0*e, n + 9) + 6.0*bF9*x4*y2*pw(-2.0*e, n + 9) + 3.0*bF9*x4*z2*pw(-2.0*e, n + 9) + 15.0*bF9*y4*z2*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x543) {
            float bF7 = 0.0, bF8 = 0.0, bF9 = 0.0, bF10 = 0.0, bF11 = 0.0, bF12 = 0.0;
            bFValues(bF7, bF8, bF9, bF10, bF11, bF12, e*r2, n + 7, n + 8, n + 9, n + 10, n + 11, n + 12);
            return x*z*(60.0*bF10*x2*y2*z2*pw(-2.0*e, n + 10) + 30.0*bF10*x2*y4*pw(-2.0*e, n + 10) + 18.0*bF10*x4*y2*pw(-2.0*e, n + 10) + 3.0*bF10*x4*z2*pw(-2.0*e, n + 10) + 15.0*bF10*y4*z2*pw(-2.0*e, n + 10) + 10.0*bF11*x2*y4*z2*pw(-2.0*e, n + 11) + 6.0*bF11*x4*y2*z2*pw(-2.0*e, n + 11) + 3.0*bF11*x4*y4*pw(-2.0*e, n + 11) + 1.0*bF12*x4*y4*z2*pw(-2.0*e, n + 12) + 135.0*bF7*pw(-2.0*e, n + 7) + 90.0*bF8*x2*pw(-2.0*e, n + 8) + 270.0*bF8*y2*pw(-2.0*e, n + 8) + 45.0*bF8*z2*pw(-2.0*e, n + 8) + 180.0*bF9*x2*y2*pw(-2.0*e, n + 9) + 30.0*bF9*x2*z2*pw(-2.0*e, n + 9) + 9.0*bF9*x4*pw(-2.0*e, n + 9) + 90.0*bF9*y2*z2*pw(-2.0*e, n + 9) + 45.0*bF9*y4*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x544) {
            float bF7 = 0.0, bF8 = 0.0, bF9 = 0.0, bF10 = 0.0, bF11 = 0.0, bF12 = 0.0, bF13 = 0.0;
            bFValues(bF7, bF8, bF9, bF10, bF11, bF12, bF13, e*r2, n + 7, n + 8, n + 9, n + 10, n + 11, n + 12, n + 13);
            return x*(360.0*bF10*x2*y2*z2*pw(-2.0*e, n + 10) + 30.0*bF10*x2*y4*pw(-2.0*e, n + 10) + 30.0*bF10*x2*z4*pw(-2.0*e, n + 10) + 18.0*bF10*x4*y2*pw(-2.0*e, n + 10) + 18.0*bF10*x4*z2*pw(-2.0*e, n + 10) + 90.0*bF10*y2*z4*pw(-2.0*e, n + 10) + 90.0*bF10*y4*z2*pw(-2.0*e, n + 10) + 60.0*bF11*x2*y2*z4*pw(-2.0*e, n + 11) + 60.0*bF11*x2*y4*z2*pw(-2.0*e, n + 11) + 36.0*bF11*x4*y2*z2*pw(-2.0*e, n + 11) + 3.0*bF11*x4*y4*pw(-2.0*e, n + 11) + 3.0*bF11*x4*z4*pw(-2.0*e, n + 11) + 15.0*bF11*y4*z4*pw(-2.0*e, n + 11) + 10.0*bF12*x2*y4*z4*pw(-2.0*e, n + 12) + 6.0*bF12*x4*y2*z4*pw(-2.0*e, n + 12) + 6.0*bF12*x4*y4*z2*pw(-2.0*e, n + 12) + 1.0*bF13*x4*y4*z4*pw(-2.0*e, n + 13) + 135.0*bF7*pw(-2.0*e, n + 7) + 90.0*bF8*x2*pw(-2.0*e, n + 8) + 270.0*bF8*y2*pw(-2.0*e, n + 8) + 270.0*bF8*z2*pw(-2.0*e, n + 8) + 180.0*bF9*x2*y2*pw(-2.0*e, n + 9) + 180.0*bF9*x2*z2*pw(-2.0*e, n + 9) + 9.0*bF9*x4*pw(-2.0*e, n + 9) + 540.0*bF9*y2*z2*pw(-2.0*e, n + 9) + 45.0*bF9*y4*pw(-2.0*e, n + 9) + 45.0*bF9*z4*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x550) {
            float bF6 = 0.0, bF7 = 0.0, bF8 = 0.0, bF9 = 0.0, bF10 = 0.0;
            bFValues(bF6, bF7, bF8, bF9, bF10, e*r2, n + 6, n + 7, n + 8, n + 9, n + 10);
            return x*y*(1.0*bF10*x4*y4*pw(-2.0*e, n + 10) + 225.0*bF6*pw(-2.0*e, n + 6) + 150.0*bF7*x2*pw(-2.0*e, n + 7) + 150.0*bF7*y2*pw(-2.0*e, n + 7) + 100.0*bF8*x2*y2*pw(-2.0*e, n + 8) + 15.0*bF8*x4*pw(-2.0*e, n + 8) + 15.0*bF8*y4*pw(-2.0*e, n + 8) + 10.0*bF9*x2*y4*pw(-2.0*e, n + 9) + 10.0*bF9*x4*y2*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x551) {
            float bF7 = 0.0, bF8 = 0.0, bF9 = 0.0, bF10 = 0.0, bF11 = 0.0;
            bFValues(bF7, bF8, bF9, bF10, bF11, e*r2, n + 7, n + 8, n + 9, n + 10, n + 11);
            return x*y*z*(10.0*bF10*x2*y4*pw(-2.0*e, n + 10) + 10.0*bF10*x4*y2*pw(-2.0*e, n + 10) + 1.0*bF11*x4*y4*pw(-2.0*e, n + 11) + 225.0*bF7*pw(-2.0*e, n + 7) + 150.0*bF8*x2*pw(-2.0*e, n + 8) + 150.0*bF8*y2*pw(-2.0*e, n + 8) + 100.0*bF9*x2*y2*pw(-2.0*e, n + 9) + 15.0*bF9*x4*pw(-2.0*e, n + 9) + 15.0*bF9*y4*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x552) {
            float bF7 = 0.0, bF8 = 0.0, bF9 = 0.0, bF10 = 0.0, bF11 = 0.0, bF12 = 0.0;
            bFValues(bF7, bF8, bF9, bF10, bF11, bF12, e*r2, n + 7, n + 8, n + 9, n + 10, n + 11, n + 12);
            return x*y*(100.0*bF10*x2*y2*z2*pw(-2.0*e, n + 10) + 10.0*bF10*x2*y4*pw(-2.0*e, n + 10) + 10.0*bF10*x4*y2*pw(-2.0*e, n + 10) + 15.0*bF10*x4*z2*pw(-2.0*e, n + 10) + 15.0*bF10*y4*z2*pw(-2.0*e, n + 10) + 10.0*bF11*x2*y4*z2*pw(-2.0*e, n + 11) + 10.0*bF11*x4*y2*z2*pw(-2.0*e, n + 11) + 1.0*bF11*x4*y4*pw(-2.0*e, n + 11) + 1.0*bF12*x4*y4*z2*pw(-2.0*e, n + 12) + 225.0*bF7*pw(-2.0*e, n + 7) + 150.0*bF8*x2*pw(-2.0*e, n + 8) + 150.0*bF8*y2*pw(-2.0*e, n + 8) + 225.0*bF8*z2*pw(-2.0*e, n + 8) + 100.0*bF9*x2*y2*pw(-2.0*e, n + 9) + 150.0*bF9*x2*z2*pw(-2.0*e, n + 9) + 15.0*bF9*x4*pw(-2.0*e, n + 9) + 150.0*bF9*y2*z2*pw(-2.0*e, n + 9) + 15.0*bF9*y4*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x553) {
            float bF8 = 0.0, bF9 = 0.0, bF10 = 0.0, bF11 = 0.0, bF12 = 0.0, bF13 = 0.0;
            bFValues(bF8, bF9, bF10, bF11, bF12, bF13, e*r2, n + 8, n + 9, n + 10, n + 11, n + 12, n + 13);
            return x*y*z*(300.0*bF10*x2*y2*pw(-2.0*e, n + 10) + 150.0*bF10*x2*z2*pw(-2.0*e, n + 10) + 45.0*bF10*x4*pw(-2.0*e, n + 10) + 150.0*bF10*y2*z2*pw(-2.0*e, n + 10) + 45.0*bF10*y4*pw(-2.0*e, n + 10) + 100.0*bF11*x2*y2*z2*pw(-2.0*e, n + 11) + 30.0*bF11*x2*y4*pw(-2.0*e, n + 11) + 30.0*bF11*x4*y2*pw(-2.0*e, n + 11) + 15.0*bF11*x4*z2*pw(-2.0*e, n + 11) + 15.0*bF11*y4*z2*pw(-2.0*e, n + 11) + 10.0*bF12*x2*y4*z2*pw(-2.0*e, n + 12) + 10.0*bF12*x4*y2*z2*pw(-2.0*e, n + 12) + 3.0*bF12*x4*y4*pw(-2.0*e, n + 12) + 1.0*bF13*x4*y4*z2*pw(-2.0*e, n + 13) + 675.0*bF8*pw(-2.0*e, n + 8) + 450.0*bF9*x2*pw(-2.0*e, n + 9) + 450.0*bF9*y2*pw(-2.0*e, n + 9) + 225.0*bF9*z2*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x554) {
            float bF8 = 0.0, bF9 = 0.0, bF10 = 0.0, bF11 = 0.0, bF12 = 0.0, bF13 = 0.0, bF14 = 0.0;
            bFValues(bF8, bF9, bF10, bF11, bF12, bF13, bF14, e*r2, n + 8, n + 9, n + 10, n + 11, n + 12, n + 13, n + 14);
            return x*y*(300.0*bF10*x2*y2*pw(-2.0*e, n + 10) + 900.0*bF10*x2*z2*pw(-2.0*e, n + 10) + 45.0*bF10*x4*pw(-2.0*e, n + 10) + 900.0*bF10*y2*z2*pw(-2.0*e, n + 10) + 45.0*bF10*y4*pw(-2.0*e, n + 10) + 225.0*bF10*z4*pw(-2.0*e, n + 10) + 600.0*bF11*x2*y2*z2*pw(-2.0*e, n + 11) + 30.0*bF11*x2*y4*pw(-2.0*e, n + 11) + 150.0*bF11*x2*z4*pw(-2.0*e, n + 11) + 30.0*bF11*x4*y2*pw(-2.0*e, n + 11) + 90.0*bF11*x4*z2*pw(-2.0*e, n + 11) + 150.0*bF11*y2*z4*pw(-2.0*e, n + 11) + 90.0*bF11*y4*z2*pw(-2.0*e, n + 11) + 100.0*bF12*x2*y2*z4*pw(-2.0*e, n + 12) + 60.0*bF12*x2*y4*z2*pw(-2.0*e, n + 12) + 60.0*bF12*x4*y2*z2*pw(-2.0*e, n + 12) + 3.0*bF12*x4*y4*pw(-2.0*e, n + 12) + 15.0*bF12*x4*z4*pw(-2.0*e, n + 12) + 15.0*bF12*y4*z4*pw(-2.0*e, n + 12) + 10.0*bF13*x2*y4*z4*pw(-2.0*e, n + 13) + 10.0*bF13*x4*y2*z4*pw(-2.0*e, n + 13) + 6.0*bF13*x4*y4*z2*pw(-2.0*e, n + 13) + 1.0*bF14*x4*y4*z4*pw(-2.0*e, n + 14) + 675.0*bF8*pw(-2.0*e, n + 8) + 450.0*bF9*x2*pw(-2.0*e, n + 9) + 450.0*bF9*y2*pw(-2.0*e, n + 9) + 1350.0*bF9*z2*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x555) {
            float bF9 = 0.0, bF10 = 0.0, bF11 = 0.0, bF12 = 0.0, bF13 = 0.0, bF14 = 0.0, bF15 = 0.0;
            bFValues(bF9, bF10, bF11, bF12, bF13, bF14, bF15, e*r2, n + 9, n + 10, n + 11, n + 12, n + 13, n + 14, n + 15);
            return x*y*z*(2250.0*bF10*x2*pw(-2.0*e, n + 10) + 2250.0*bF10*y2*pw(-2.0*e, n + 10) + 2250.0*bF10*z2*pw(-2.0*e, n + 10) + 1500.0*bF11*x2*y2*pw(-2.0*e, n + 11) + 1500.0*bF11*x2*z2*pw(-2.0*e, n + 11) + 225.0*bF11*x4*pw(-2.0*e, n + 11) + 1500.0*bF11*y2*z2*pw(-2.0*e, n + 11) + 225.0*bF11*y4*pw(-2.0*e, n + 11) + 225.0*bF11*z4*pw(-2.0*e, n + 11) + 1000.0*bF12*x2*y2*z2*pw(-2.0*e, n + 12) + 150.0*bF12*x2*y4*pw(-2.0*e, n + 12) + 150.0*bF12*x2*z4*pw(-2.0*e, n + 12) + 150.0*bF12*x4*y2*pw(-2.0*e, n + 12) + 150.0*bF12*x4*z2*pw(-2.0*e, n + 12) + 150.0*bF12*y2*z4*pw(-2.0*e, n + 12) + 150.0*bF12*y4*z2*pw(-2.0*e, n + 12) + 100.0*bF13*x2*y2*z4*pw(-2.0*e, n + 13) + 100.0*bF13*x2*y4*z2*pw(-2.0*e, n + 13) + 100.0*bF13*x4*y2*z2*pw(-2.0*e, n + 13) + 15.0*bF13*x4*y4*pw(-2.0*e, n + 13) + 15.0*bF13*x4*z4*pw(-2.0*e, n + 13) + 15.0*bF13*y4*z4*pw(-2.0*e, n + 13) + 10.0*bF14*x2*y4*z4*pw(-2.0*e, n + 14) + 10.0*bF14*x4*y2*z4*pw(-2.0*e, n + 14) + 10.0*bF14*x4*y4*z2*pw(-2.0*e, n + 14) + 1.0*bF15*x4*y4*z4*pw(-2.0*e, n + 15) + 3375.0*bF9*pw(-2.0*e, n + 9));
        }
    } else if (i == 6) {
        if (hexIJK == 0x600) {
            vec4 vecBF = vecBFValues(e*r2, ivec4(n + 3, n + 4, n + 5, n + 6));
            float bF3 = vecBF[0], bF4 = vecBF[1], bF5 = vecBF[2], bF6 = vecBF[3];
            return 15.0*bF3*pw(-2.0*e, n + 3) + 45.0*bF4*x2*pw(-2.0*e, n + 4) + 15.0*bF5*x4*pw(-2.0*e, n + 5) + 1.0*bF6*x2*x2*x2*pw(-2.0*e, n + 6);
        }
        if (hexIJK == 0x610) {
            vec4 vecBF = vecBFValues(e*r2, ivec4(n + 4, n + 5, n + 6, n + 7));
            float bF4 = vecBF[0], bF5 = vecBF[1], bF6 = vecBF[2], bF7 = vecBF[3];
            return y*(15.0*bF4*pw(-2.0*e, n + 4) + 45.0*bF5*x2*pw(-2.0*e, n + 5) + 15.0*bF6*x4*pw(-2.0*e, n + 6) + 1.0*bF7*x2*x2*x2*pw(-2.0*e, n + 7));
        }
        if (hexIJK == 0x611) {
            vec4 vecBF = vecBFValues(e*r2, ivec4(n + 5, n + 6, n + 7, n + 8));
            float bF5 = vecBF[0], bF6 = vecBF[1], bF7 = vecBF[2], bF8 = vecBF[3];
            return y*z*(15.0*bF5*pw(-2.0*e, n + 5) + 45.0*bF6*x2*pw(-2.0*e, n + 6) + 15.0*bF7*x4*pw(-2.0*e, n + 7) + 1.0*bF8*x2*x2*x2*pw(-2.0*e, n + 8));
        }
        if (hexIJK == 0x620) {
            float bF4 = 0.0, bF5 = 0.0, bF6 = 0.0, bF7 = 0.0, bF8 = 0.0;
            bFValues(bF4, bF5, bF6, bF7, bF8, e*r2, n + 4, n + 5, n + 6, n + 7, n + 8);
            return 15.0*bF4*pw(-2.0*e, n + 4) + 45.0*bF5*x2*pw(-2.0*e, n + 5) + 15.0*bF5*y2*pw(-2.0*e, n + 5) + 45.0*bF6*x2*y2*pw(-2.0*e, n + 6) + 15.0*bF6*x4*pw(-2.0*e, n + 6) + 1.0*bF7*x2*x2*x2*pw(-2.0*e, n + 7) + 15.0*bF7*x4*y2*pw(-2.0*e, n + 7) + 1.0*bF8*x2*x2*x2*y2*pw(-2.0*e, n + 8);
        }
        if (hexIJK == 0x621) {
            float bF5 = 0.0, bF6 = 0.0, bF7 = 0.0, bF8 = 0.0, bF9 = 0.0;
            bFValues(bF5, bF6, bF7, bF8, bF9, e*r2, n + 5, n + 6, n + 7, n + 8, n + 9);
            return z*(15.0*bF5*pw(-2.0*e, n + 5) + 45.0*bF6*x2*pw(-2.0*e, n + 6) + 15.0*bF6*y2*pw(-2.0*e, n + 6) + 45.0*bF7*x2*y2*pw(-2.0*e, n + 7) + 15.0*bF7*x4*pw(-2.0*e, n + 7) + 1.0*bF8*x2*x2*x2*pw(-2.0*e, n + 8) + 15.0*bF8*x4*y2*pw(-2.0*e, n + 8) + 1.0*bF9*x2*x2*x2*y2*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x622) {
            float bF5 = 0.0, bF6 = 0.0, bF7 = 0.0, bF8 = 0.0, bF9 = 0.0, bF10 = 0.0;
            bFValues(bF5, bF6, bF7, bF8, bF9, bF10, e*r2, n + 5, n + 6, n + 7, n + 8, n + 9, n + 10);
            return 1.0*bF10*x2*x2*x2*y2*z2*pw(-2.0*e, n + 10) + 15.0*bF5*pw(-2.0*e, n + 5) + 45.0*bF6*x2*pw(-2.0*e, n + 6) + 15.0*bF6*y2*pw(-2.0*e, n + 6) + 15.0*bF6*z2*pw(-2.0*e, n + 6) + 45.0*bF7*x2*y2*pw(-2.0*e, n + 7) + 45.0*bF7*x2*z2*pw(-2.0*e, n + 7) + 15.0*bF7*x4*pw(-2.0*e, n + 7) + 15.0*bF7*y2*z2*pw(-2.0*e, n + 7) + 1.0*bF8*x2*x2*x2*pw(-2.0*e, n + 8) + 45.0*bF8*x2*y2*z2*pw(-2.0*e, n + 8) + 15.0*bF8*x4*y2*pw(-2.0*e, n + 8) + 15.0*bF8*x4*z2*pw(-2.0*e, n + 8) + 1.0*bF9*x2*x2*x2*y2*pw(-2.0*e, n + 9) + 1.0*bF9*x2*x2*x2*z2*pw(-2.0*e, n + 9) + 15.0*bF9*x4*y2*z2*pw(-2.0*e, n + 9);
        }
        if (hexIJK == 0x630) {
            float bF5 = 0.0, bF6 = 0.0, bF7 = 0.0, bF8 = 0.0, bF9 = 0.0;
            bFValues(bF5, bF6, bF7, bF8, bF9, e*r2, n + 5, n + 6, n + 7, n + 8, n + 9);
            return y*(45.0*bF5*pw(-2.0*e, n + 5) + 135.0*bF6*x2*pw(-2.0*e, n + 6) + 15.0*bF6*y2*pw(-2.0*e, n + 6) + 45.0*bF7*x2*y2*pw(-2.0*e, n + 7) + 45.0*bF7*x4*pw(-2.0*e, n + 7) + 3.0*bF8*x2*x2*x2*pw(-2.0*e, n + 8) + 15.0*bF8*x4*y2*pw(-2.0*e, n + 8) + 1.0*bF9*x2*x2*x2*y2*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x631) {
            float bF6 = 0.0, bF7 = 0.0, bF8 = 0.0, bF9 = 0.0, bF10 = 0.0;
            bFValues(bF6, bF7, bF8, bF9, bF10, e*r2, n + 6, n + 7, n + 8, n + 9, n + 10);
            return y*z*(1.0*bF10*x2*x2*x2*y2*pw(-2.0*e, n + 10) + 45.0*bF6*pw(-2.0*e, n + 6) + 135.0*bF7*x2*pw(-2.0*e, n + 7) + 15.0*bF7*y2*pw(-2.0*e, n + 7) + 45.0*bF8*x2*y2*pw(-2.0*e, n + 8) + 45.0*bF8*x4*pw(-2.0*e, n + 8) + 3.0*bF9*x2*x2*x2*pw(-2.0*e, n + 9) + 15.0*bF9*x4*y2*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x632) {
            float bF6 = 0.0, bF7 = 0.0, bF8 = 0.0, bF9 = 0.0, bF10 = 0.0, bF11 = 0.0;
            bFValues(bF6, bF7, bF8, bF9, bF10, bF11, e*r2, n + 6, n + 7, n + 8, n + 9, n + 10, n + 11);
            return y*(1.0*bF10*x2*x2*x2*y2*pw(-2.0*e, n + 10) + 3.0*bF10*x2*x2*x2*z2*pw(-2.0*e, n + 10) + 15.0*bF10*x4*y2*z2*pw(-2.0*e, n + 10) + 1.0*bF11*x2*x2*x2*y2*z2*pw(-2.0*e, n + 11) + 45.0*bF6*pw(-2.0*e, n + 6) + 135.0*bF7*x2*pw(-2.0*e, n + 7) + 15.0*bF7*y2*pw(-2.0*e, n + 7) + 45.0*bF7*z2*pw(-2.0*e, n + 7) + 45.0*bF8*x2*y2*pw(-2.0*e, n + 8) + 135.0*bF8*x2*z2*pw(-2.0*e, n + 8) + 45.0*bF8*x4*pw(-2.0*e, n + 8) + 15.0*bF8*y2*z2*pw(-2.0*e, n + 8) + 3.0*bF9*x2*x2*x2*pw(-2.0*e, n + 9) + 45.0*bF9*x2*y2*z2*pw(-2.0*e, n + 9) + 15.0*bF9*x4*y2*pw(-2.0*e, n + 9) + 45.0*bF9*x4*z2*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x633) {
            float bF7 = 0.0, bF8 = 0.0, bF9 = 0.0, bF10 = 0.0, bF11 = 0.0, bF12 = 0.0;
            bFValues(bF7, bF8, bF9, bF10, bF11, bF12, e*r2, n + 7, n + 8, n + 9, n + 10, n + 11, n + 12);
            return y*z*(9.0*bF10*x2*x2*x2*pw(-2.0*e, n + 10) + 45.0*bF10*x2*y2*z2*pw(-2.0*e, n + 10) + 45.0*bF10*x4*y2*pw(-2.0*e, n + 10) + 45.0*bF10*x4*z2*pw(-2.0*e, n + 10) + 3.0*bF11*x2*x2*x2*y2*pw(-2.0*e, n + 11) + 3.0*bF11*x2*x2*x2*z2*pw(-2.0*e, n + 11) + 15.0*bF11*x4*y2*z2*pw(-2.0*e, n + 11) + 1.0*bF12*x2*x2*x2*y2*z2*pw(-2.0*e, n + 12) + 135.0*bF7*pw(-2.0*e, n + 7) + 405.0*bF8*x2*pw(-2.0*e, n + 8) + 45.0*bF8*y2*pw(-2.0*e, n + 8) + 45.0*bF8*z2*pw(-2.0*e, n + 8) + 135.0*bF9*x2*y2*pw(-2.0*e, n + 9) + 135.0*bF9*x2*z2*pw(-2.0*e, n + 9) + 135.0*bF9*x4*pw(-2.0*e, n + 9) + 15.0*bF9*y2*z2*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x640) {
            float bF5 = 0.0, bF6 = 0.0, bF7 = 0.0, bF8 = 0.0, bF9 = 0.0, bF10 = 0.0;
            bFValues(bF5, bF6, bF7, bF8, bF9, bF10, e*r2, n + 5, n + 6, n + 7, n + 8, n + 9, n + 10);
            return 1.0*bF10*x2*x2*x2*y4*pw(-2.0*e, n + 10) + 45.0*bF5*pw(-2.0*e, n + 5) + 135.0*bF6*x2*pw(-2.0*e, n + 6) + 90.0*bF6*y2*pw(-2.0*e, n + 6) + 270.0*bF7*x2*y2*pw(-2.0*e, n + 7) + 45.0*bF7*x4*pw(-2.0*e, n + 7) + 15.0*bF7*y4*pw(-2.0*e, n + 7) + 3.0*bF8*x2*x2*x2*pw(-2.0*e, n + 8) + 45.0*bF8*x2*y4*pw(-2.0*e, n + 8) + 90.0*bF8*x4*y2*pw(-2.0*e, n + 8) + 6.0*bF9*x2*x2*x2*y2*pw(-2.0*e, n + 9) + 15.0*bF9*x4*y4*pw(-2.0*e, n + 9);
        }
        if (hexIJK == 0x641) {
            float bF6 = 0.0, bF7 = 0.0, bF8 = 0.0, bF9 = 0.0, bF10 = 0.0, bF11 = 0.0;
            bFValues(bF6, bF7, bF8, bF9, bF10, bF11, e*r2, n + 6, n + 7, n + 8, n + 9, n + 10, n + 11);
            return z*(6.0*bF10*x2*x2*x2*y2*pw(-2.0*e, n + 10) + 15.0*bF10*x4*y4*pw(-2.0*e, n + 10) + 1.0*bF11*x2*x2*x2*y4*pw(-2.0*e, n + 11) + 45.0*bF6*pw(-2.0*e, n + 6) + 135.0*bF7*x2*pw(-2.0*e, n + 7) + 90.0*bF7*y2*pw(-2.0*e, n + 7) + 270.0*bF8*x2*y2*pw(-2.0*e, n + 8) + 45.0*bF8*x4*pw(-2.0*e, n + 8) + 15.0*bF8*y4*pw(-2.0*e, n + 8) + 3.0*bF9*x2*x2*x2*pw(-2.0*e, n + 9) + 45.0*bF9*x2*y4*pw(-2.0*e, n + 9) + 90.0*bF9*x4*y2*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x642) {
            float bF6 = 0.0, bF7 = 0.0, bF8 = 0.0, bF9 = 0.0, bF10 = 0.0, bF11 = 0.0, bF12 = 0.0;
            bFValues(bF6, bF7, bF8, bF9, bF10, bF11, bF12, e*r2, n + 6, n + 7, n + 8, n + 9, n + 10, n + 11, n + 12);
            return 6.0*bF10*x2*x2*x2*y2*pw(-2.0*e, n + 10) + 3.0*bF10*x2*x2*x2*z2*pw(-2.0*e, n + 10) + 45.0*bF10*x2*y4*z2*pw(-2.0*e, n + 10) + 90.0*bF10*x4*y2*z2*pw(-2.0*e, n + 10) + 15.0*bF10*x4*y4*pw(-2.0*e, n + 10) + 6.0*bF11*x2*x2*x2*y2*z2*pw(-2.0*e, n + 11) + 1.0*bF11*x2*x2*x2*y4*pw(-2.0*e, n + 11) + 15.0*bF11*x4*y4*z2*pw(-2.0*e, n + 11) + 1.0*bF12*x2*x2*x2*y4*z2*pw(-2.0*e, n + 12) + 45.0*bF6*pw(-2.0*e, n + 6) + 135.0*bF7*x2*pw(-2.0*e, n + 7) + 90.0*bF7*y2*pw(-2.0*e, n + 7) + 45.0*bF7*z2*pw(-2.0*e, n + 7) + 270.0*bF8*x2*y2*pw(-2.0*e, n + 8) + 135.0*bF8*x2*z2*pw(-2.0*e, n + 8) + 45.0*bF8*x4*pw(-2.0*e, n + 8) + 90.0*bF8*y2*z2*pw(-2.0*e, n + 8) + 15.0*bF8*y4*pw(-2.0*e, n + 8) + 3.0*bF9*x2*x2*x2*pw(-2.0*e, n + 9) + 270.0*bF9*x2*y2*z2*pw(-2.0*e, n + 9) + 45.0*bF9*x2*y4*pw(-2.0*e, n + 9) + 90.0*bF9*x4*y2*pw(-2.0*e, n + 9) + 45.0*bF9*x4*z2*pw(-2.0*e, n + 9) + 15.0*bF9*y4*z2*pw(-2.0*e, n + 9);
        }
        if (hexIJK == 0x643) {
            float bF7 = 0.0, bF8 = 0.0, bF9 = 0.0, bF10 = 0.0, bF11 = 0.0, bF12 = 0.0, bF13 = 0.0;
            bFValues(bF7, bF8, bF9, bF10, bF11, bF12, bF13, e*r2, n + 7, n + 8, n + 9, n + 10, n + 11, n + 12, n + 13);
            return z*(9.0*bF10*x2*x2*x2*pw(-2.0*e, n + 10) + 270.0*bF10*x2*y2*z2*pw(-2.0*e, n + 10) + 135.0*bF10*x2*y4*pw(-2.0*e, n + 10) + 270.0*bF10*x4*y2*pw(-2.0*e, n + 10) + 45.0*bF10*x4*z2*pw(-2.0*e, n + 10) + 15.0*bF10*y4*z2*pw(-2.0*e, n + 10) + 18.0*bF11*x2*x2*x2*y2*pw(-2.0*e, n + 11) + 3.0*bF11*x2*x2*x2*z2*pw(-2.0*e, n + 11) + 45.0*bF11*x2*y4*z2*pw(-2.0*e, n + 11) + 90.0*bF11*x4*y2*z2*pw(-2.0*e, n + 11) + 45.0*bF11*x4*y4*pw(-2.0*e, n + 11) + 6.0*bF12*x2*x2*x2*y2*z2*pw(-2.0*e, n + 12) + 3.0*bF12*x2*x2*x2*y4*pw(-2.0*e, n + 12) + 15.0*bF12*x4*y4*z2*pw(-2.0*e, n + 12) + 1.0*bF13*x2*x2*x2*y4*z2*pw(-2.0*e, n + 13) + 135.0*bF7*pw(-2.0*e, n + 7) + 405.0*bF8*x2*pw(-2.0*e, n + 8) + 270.0*bF8*y2*pw(-2.0*e, n + 8) + 45.0*bF8*z2*pw(-2.0*e, n + 8) + 810.0*bF9*x2*y2*pw(-2.0*e, n + 9) + 135.0*bF9*x2*z2*pw(-2.0*e, n + 9) + 135.0*bF9*x4*pw(-2.0*e, n + 9) + 90.0*bF9*y2*z2*pw(-2.0*e, n + 9) + 45.0*bF9*y4*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x644) {
            float bF7 = 0.0, bF8 = 0.0, bF9 = 0.0, bF10 = 0.0, bF11 = 0.0, bF12 = 0.0, bF13 = 0.0, bF14 = 0.0;
            bFValues(bF7, bF8, bF9, bF10, bF11, bF12, bF13, bF14, e*r2, n + 7, n + 8, n + 9, n + 10, n + 11, n + 12, n + 13, n + 14);
            return 9.0*bF10*x2*x2*x2*pw(-2.0*e, n + 10) + 1620.0*bF10*x2*y2*z2*pw(-2.0*e, n + 10) + 135.0*bF10*x2*y4*pw(-2.0*e, n + 10) + 135.0*bF10*x2*z4*pw(-2.0*e, n + 10) + 270.0*bF10*x4*y2*pw(-2.0*e, n + 10) + 270.0*bF10*x4*z2*pw(-2.0*e, n + 10) + 90.0*bF10*y2*z4*pw(-2.0*e, n + 10) + 90.0*bF10*y4*z2*pw(-2.0*e, n + 10) + 18.0*bF11*x2*x2*x2*y2*pw(-2.0*e, n + 11) + 18.0*bF11*x2*x2*x2*z2*pw(-2.0*e, n + 11) + 270.0*bF11*x2*y2*z4*pw(-2.0*e, n + 11) + 270.0*bF11*x2*y4*z2*pw(-2.0*e, n + 11) + 540.0*bF11*x4*y2*z2*pw(-2.0*e, n + 11) + 45.0*bF11*x4*y4*pw(-2.0*e, n + 11) + 45.0*bF11*x4*z4*pw(-2.0*e, n + 11) + 15.0*bF11*y4*z4*pw(-2.0*e, n + 11) + 36.0*bF12*x2*x2*x2*y2*z2*pw(-2.0*e, n + 12) + 3.0*bF12*x2*x2*x2*y4*pw(-2.0*e, n + 12) + 3.0*bF12*x2*x2*x2*z4*pw(-2.0*e, n + 12) + 45.0*bF12*x2*y4*z4*pw(-2.0*e, n + 12) + 90.0*bF12*x4*y2*z4*pw(-2.0*e, n + 12) + 90.0*bF12*x4*y4*z2*pw(-2.0*e, n + 12) + 6.0*bF13*x2*x2*x2*y2*z4*pw(-2.0*e, n + 13) + 6.0*bF13*x2*x2*x2*y4*z2*pw(-2.0*e, n + 13) + 15.0*bF13*x4*y4*z4*pw(-2.0*e, n + 13) + 1.0*bF14*x2*x2*x2*y4*z4*pw(-2.0*e, n + 14) + 135.0*bF7*pw(-2.0*e, n + 7) + 405.0*bF8*x2*pw(-2.0*e, n + 8) + 270.0*bF8*y2*pw(-2.0*e, n + 8) + 270.0*bF8*z2*pw(-2.0*e, n + 8) + 810.0*bF9*x2*y2*pw(-2.0*e, n + 9) + 810.0*bF9*x2*z2*pw(-2.0*e, n + 9) + 135.0*bF9*x4*pw(-2.0*e, n + 9) + 540.0*bF9*y2*z2*pw(-2.0*e, n + 9) + 45.0*bF9*y4*pw(-2.0*e, n + 9) + 45.0*bF9*z4*pw(-2.0*e, n + 9);
        }
        if (hexIJK == 0x650) {
            float bF6 = 0.0, bF7 = 0.0, bF8 = 0.0, bF9 = 0.0, bF10 = 0.0, bF11 = 0.0;
            bFValues(bF6, bF7, bF8, bF9, bF10, bF11, e*r2, n + 6, n + 7, n + 8, n + 9, n + 10, n + 11);
            return y*(10.0*bF10*x2*x2*x2*y2*pw(-2.0*e, n + 10) + 15.0*bF10*x4*y4*pw(-2.0*e, n + 10) + 1.0*bF11*x2*x2*x2*y4*pw(-2.0*e, n + 11) + 225.0*bF6*pw(-2.0*e, n + 6) + 675.0*bF7*x2*pw(-2.0*e, n + 7) + 150.0*bF7*y2*pw(-2.0*e, n + 7) + 450.0*bF8*x2*y2*pw(-2.0*e, n + 8) + 225.0*bF8*x4*pw(-2.0*e, n + 8) + 15.0*bF8*y4*pw(-2.0*e, n + 8) + 15.0*bF9*x2*x2*x2*pw(-2.0*e, n + 9) + 45.0*bF9*x2*y4*pw(-2.0*e, n + 9) + 150.0*bF9*x4*y2*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x651) {
            float bF7 = 0.0, bF8 = 0.0, bF9 = 0.0, bF10 = 0.0, bF11 = 0.0, bF12 = 0.0;
            bFValues(bF7, bF8, bF9, bF10, bF11, bF12, e*r2, n + 7, n + 8, n + 9, n + 10, n + 11, n + 12);
            return y*z*(15.0*bF10*x2*x2*x2*pw(-2.0*e, n + 10) + 45.0*bF10*x2*y4*pw(-2.0*e, n + 10) + 150.0*bF10*x4*y2*pw(-2.0*e, n + 10) + 10.0*bF11*x2*x2*x2*y2*pw(-2.0*e, n + 11) + 15.0*bF11*x4*y4*pw(-2.0*e, n + 11) + 1.0*bF12*x2*x2*x2*y4*pw(-2.0*e, n + 12) + 225.0*bF7*pw(-2.0*e, n + 7) + 675.0*bF8*x2*pw(-2.0*e, n + 8) + 150.0*bF8*y2*pw(-2.0*e, n + 8) + 450.0*bF9*x2*y2*pw(-2.0*e, n + 9) + 225.0*bF9*x4*pw(-2.0*e, n + 9) + 15.0*bF9*y4*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x652) {
            float bF7 = 0.0, bF8 = 0.0, bF9 = 0.0, bF10 = 0.0, bF11 = 0.0, bF12 = 0.0, bF13 = 0.0;
            bFValues(bF7, bF8, bF9, bF10, bF11, bF12, bF13, e*r2, n + 7, n + 8, n + 9, n + 10, n + 11, n + 12, n + 13);
            return y*(15.0*bF10*x2*x2*x2*pw(-2.0*e, n + 10) + 450.0*bF10*x2*y2*z2*pw(-2.0*e, n + 10) + 45.0*bF10*x2*y4*pw(-2.0*e, n + 10) + 150.0*bF10*x4*y2*pw(-2.0*e, n + 10) + 225.0*bF10*x4*z2*pw(-2.0*e, n + 10) + 15.0*bF10*y4*z2*pw(-2.0*e, n + 10) + 10.0*bF11*x2*x2*x2*y2*pw(-2.0*e, n + 11) + 15.0*bF11*x2*x2*x2*z2*pw(-2.0*e, n + 11) + 45.0*bF11*x2*y4*z2*pw(-2.0*e, n + 11) + 150.0*bF11*x4*y2*z2*pw(-2.0*e, n + 11) + 15.0*bF11*x4*y4*pw(-2.0*e, n + 11) + 10.0*bF12*x2*x2*x2*y2*z2*pw(-2.0*e, n + 12) + 1.0*bF12*x2*x2*x2*y4*pw(-2.0*e, n + 12) + 15.0*bF12*x4*y4*z2*pw(-2.0*e, n + 12) + 1.0*bF13*x2*x2*x2*y4*z2*pw(-2.0*e, n + 13) + 225.0*bF7*pw(-2.0*e, n + 7) + 675.0*bF8*x2*pw(-2.0*e, n + 8) + 150.0*bF8*y2*pw(-2.0*e, n + 8) + 225.0*bF8*z2*pw(-2.0*e, n + 8) + 450.0*bF9*x2*y2*pw(-2.0*e, n + 9) + 675.0*bF9*x2*z2*pw(-2.0*e, n + 9) + 225.0*bF9*x4*pw(-2.0*e, n + 9) + 150.0*bF9*y2*z2*pw(-2.0*e, n + 9) + 15.0*bF9*y4*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x653) {
            float bF8 = 0.0, bF9 = 0.0, bF10 = 0.0, bF11 = 0.0, bF12 = 0.0, bF13 = 0.0, bF14 = 0.0;
            bFValues(bF8, bF9, bF10, bF11, bF12, bF13, bF14, e*r2, n + 8, n + 9, n + 10, n + 11, n + 12, n + 13, n + 14);
            return y*z*(1350.0*bF10*x2*y2*pw(-2.0*e, n + 10) + 675.0*bF10*x2*z2*pw(-2.0*e, n + 10) + 675.0*bF10*x4*pw(-2.0*e, n + 10) + 150.0*bF10*y2*z2*pw(-2.0*e, n + 10) + 45.0*bF10*y4*pw(-2.0*e, n + 10) + 45.0*bF11*x2*x2*x2*pw(-2.0*e, n + 11) + 450.0*bF11*x2*y2*z2*pw(-2.0*e, n + 11) + 135.0*bF11*x2*y4*pw(-2.0*e, n + 11) + 450.0*bF11*x4*y2*pw(-2.0*e, n + 11) + 225.0*bF11*x4*z2*pw(-2.0*e, n + 11) + 15.0*bF11*y4*z2*pw(-2.0*e, n + 11) + 30.0*bF12*x2*x2*x2*y2*pw(-2.0*e, n + 12) + 15.0*bF12*x2*x2*x2*z2*pw(-2.0*e, n + 12) + 45.0*bF12*x2*y4*z2*pw(-2.0*e, n + 12) + 150.0*bF12*x4*y2*z2*pw(-2.0*e, n + 12) + 45.0*bF12*x4*y4*pw(-2.0*e, n + 12) + 10.0*bF13*x2*x2*x2*y2*z2*pw(-2.0*e, n + 13) + 3.0*bF13*x2*x2*x2*y4*pw(-2.0*e, n + 13) + 15.0*bF13*x4*y4*z2*pw(-2.0*e, n + 13) + 1.0*bF14*x2*x2*x2*y4*z2*pw(-2.0*e, n + 14) + 675.0*bF8*pw(-2.0*e, n + 8) + 2025.0*bF9*x2*pw(-2.0*e, n + 9) + 450.0*bF9*y2*pw(-2.0*e, n + 9) + 225.0*bF9*z2*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x654) {
            float bF8 = 0.0, bF9 = 0.0, bF10 = 0.0, bF11 = 0.0, bF12 = 0.0, bF13 = 0.0, bF14 = 0.0, bF15 = 0.0;
            bFValues(bF8, bF9, bF10, bF11, bF12, bF13, bF14, bF15, e*r2, n + 8, n + 9, n + 10, n + 11, n + 12, n + 13, n + 14, n + 15);
            return y*(1350.0*bF10*x2*y2*pw(-2.0*e, n + 10) + 4050.0*bF10*x2*z2*pw(-2.0*e, n + 10) + 675.0*bF10*x4*pw(-2.0*e, n + 10) + 900.0*bF10*y2*z2*pw(-2.0*e, n + 10) + 45.0*bF10*y4*pw(-2.0*e, n + 10) + 225.0*bF10*z4*pw(-2.0*e, n + 10) + 45.0*bF11*x2*x2*x2*pw(-2.0*e, n + 11) + 2700.0*bF11*x2*y2*z2*pw(-2.0*e, n + 11) + 135.0*bF11*x2*y4*pw(-2.0*e, n + 11) + 675.0*bF11*x2*z4*pw(-2.0*e, n + 11) + 450.0*bF11*x4*y2*pw(-2.0*e, n + 11) + 1350.0*bF11*x4*z2*pw(-2.0*e, n + 11) + 150.0*bF11*y2*z4*pw(-2.0*e, n + 11) + 90.0*bF11*y4*z2*pw(-2.0*e, n + 11) + 30.0*bF12*x2*x2*x2*y2*pw(-2.0*e, n + 12) + 90.0*bF12*x2*x2*x2*z2*pw(-2.0*e, n + 12) + 450.0*bF12*x2*y2*z4*pw(-2.0*e, n + 12) + 270.0*bF12*x2*y4*z2*pw(-2.0*e, n + 12) + 900.0*bF12*x4*y2*z2*pw(-2.0*e, n + 12) + 45.0*bF12*x4*y4*pw(-2.0*e, n + 12) + 225.0*bF12*x4*z4*pw(-2.0*e, n + 12) + 15.0*bF12*y4*z4*pw(-2.0*e, n + 12) + 60.0*bF13*x2*x2*x2*y2*z2*pw(-2.0*e, n + 13) + 3.0*bF13*x2*x2*x2*y4*pw(-2.0*e, n + 13) + 15.0*bF13*x2*x2*x2*z4*pw(-2.0*e, n + 13) + 45.0*bF13*x2*y4*z4*pw(-2.0*e, n + 13) + 150.0*bF13*x4*y2*z4*pw(-2.0*e, n + 13) + 90.0*bF13*x4*y4*z2*pw(-2.0*e, n + 13) + 10.0*bF14*x2*x2*x2*y2*z4*pw(-2.0*e, n + 14) + 6.0*bF14*x2*x2*x2*y4*z2*pw(-2.0*e, n + 14) + 15.0*bF14*x4*y4*z4*pw(-2.0*e, n + 14) + 1.0*bF15*x2*x2*x2*y4*z4*pw(-2.0*e, n + 15) + 675.0*bF8*pw(-2.0*e, n + 8) + 2025.0*bF9*x2*pw(-2.0*e, n + 9) + 450.0*bF9*y2*pw(-2.0*e, n + 9) + 1350.0*bF9*z2*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x655) {
            float bF9 = 0.0, bF10 = 0.0, bF11 = 0.0, bF12 = 0.0, bF13 = 0.0, bF14 = 0.0, bF15 = 0.0, bF16 = 0.0;
            bFValues(bF9, bF10, bF11, bF12, bF13, bF14, bF15, bF16, e*r2, n + 9, n + 10, n + 11, n + 12, n + 13, n + 14, n + 15, n + 16);
            return y*z*(10125.0*bF10*x2*pw(-2.0*e, n + 10) + 2250.0*bF10*y2*pw(-2.0*e, n + 10) + 2250.0*bF10*z2*pw(-2.0*e, n + 10) + 6750.0*bF11*x2*y2*pw(-2.0*e, n + 11) + 6750.0*bF11*x2*z2*pw(-2.0*e, n + 11) + 3375.0*bF11*x4*pw(-2.0*e, n + 11) + 1500.0*bF11*y2*z2*pw(-2.0*e, n + 11) + 225.0*bF11*y4*pw(-2.0*e, n + 11) + 225.0*bF11*z4*pw(-2.0*e, n + 11) + 225.0*bF12*x2*x2*x2*pw(-2.0*e, n + 12) + 4500.0*bF12*x2*y2*z2*pw(-2.0*e, n + 12) + 675.0*bF12*x2*y4*pw(-2.0*e, n + 12) + 675.0*bF12*x2*z4*pw(-2.0*e, n + 12) + 2250.0*bF12*x4*y2*pw(-2.0*e, n + 12) + 2250.0*bF12*x4*z2*pw(-2.0*e, n + 12) + 150.0*bF12*y2*z4*pw(-2.0*e, n + 12) + 150.0*bF12*y4*z2*pw(-2.0*e, n + 12) + 150.0*bF13*x2*x2*x2*y2*pw(-2.0*e, n + 13) + 150.0*bF13*x2*x2*x2*z2*pw(-2.0*e, n + 13) + 450.0*bF13*x2*y2*z4*pw(-2.0*e, n + 13) + 450.0*bF13*x2*y4*z2*pw(-2.0*e, n + 13) + 1500.0*bF13*x4*y2*z2*pw(-2.0*e, n + 13) + 225.0*bF13*x4*y4*pw(-2.0*e, n + 13) + 225.0*bF13*x4*z4*pw(-2.0*e, n + 13) + 15.0*bF13*y4*z4*pw(-2.0*e, n + 13) + 100.0*bF14*x2*x2*x2*y2*z2*pw(-2.0*e, n + 14) + 15.0*bF14*x2*x2*x2*y4*pw(-2.0*e, n + 14) + 15.0*bF14*x2*x2*x2*z4*pw(-2.0*e, n + 14) + 45.0*bF14*x2*y4*z4*pw(-2.0*e, n + 14) + 150.0*bF14*x4*y2*z4*pw(-2.0*e, n + 14) + 150.0*bF14*x4*y4*z2*pw(-2.0*e, n + 14) + 10.0*bF15*x2*x2*x2*y2*z4*pw(-2.0*e, n + 15) + 10.0*bF15*x2*x2*x2*y4*z2*pw(-2.0*e, n + 15) + 15.0*bF15*x4*y4*z4*pw(-2.0*e, n + 15) + 1.0*bF16*x2*x2*x2*y4*z4*pw(-2.0*e, n + 16) + 3375.0*bF9*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x660) {
            float bF6 = 0.0, bF7 = 0.0, bF8 = 0.0, bF9 = 0.0, bF10 = 0.0, bF11 = 0.0, bF12 = 0.0;
            bFValues(bF6, bF7, bF8, bF9, bF10, bF11, bF12, e*r2, n + 6, n + 7, n + 8, n + 9, n + 10, n + 11, n + 12);
            return 45.0*bF10*x2*x2*x2*y2*pw(-2.0*e, n + 10) + 45.0*bF10*x2*y2*y2*y2*pw(-2.0*e, n + 10) + 225.0*bF10*x4*y4*pw(-2.0*e, n + 10) + 15.0*bF11*x2*x2*x2*y4*pw(-2.0*e, n + 11) + 15.0*bF11*x4*y2*y2*y2*pw(-2.0*e, n + 11) + 1.0*bF12*x2*x2*x2*y2*y2*y2*pw(-2.0*e, n + 12) + 225.0*bF6*pw(-2.0*e, n + 6) + 675.0*bF7*x2*pw(-2.0*e, n + 7) + 675.0*bF7*y2*pw(-2.0*e, n + 7) + 2025.0*bF8*x2*y2*pw(-2.0*e, n + 8) + 225.0*bF8*x4*pw(-2.0*e, n + 8) + 225.0*bF8*y4*pw(-2.0*e, n + 8) + 15.0*bF9*x2*x2*x2*pw(-2.0*e, n + 9) + 675.0*bF9*x2*y4*pw(-2.0*e, n + 9) + 675.0*bF9*x4*y2*pw(-2.0*e, n + 9) + 15.0*bF9*y2*y2*y2*pw(-2.0*e, n + 9);
        }
        if (hexIJK == 0x661) {
            float bF7 = 0.0, bF8 = 0.0, bF9 = 0.0, bF10 = 0.0, bF11 = 0.0, bF12 = 0.0, bF13 = 0.0;
            bFValues(bF7, bF8, bF9, bF10, bF11, bF12, bF13, e*r2, n + 7, n + 8, n + 9, n + 10, n + 11, n + 12, n + 13);
            return z*(15.0*bF10*x2*x2*x2*pw(-2.0*e, n + 10) + 675.0*bF10*x2*y4*pw(-2.0*e, n + 10) + 675.0*bF10*x4*y2*pw(-2.0*e, n + 10) + 15.0*bF10*y2*y2*y2*pw(-2.0*e, n + 10) + 45.0*bF11*x2*x2*x2*y2*pw(-2.0*e, n + 11) + 45.0*bF11*x2*y2*y2*y2*pw(-2.0*e, n + 11) + 225.0*bF11*x4*y4*pw(-2.0*e, n + 11) + 15.0*bF12*x2*x2*x2*y4*pw(-2.0*e, n + 12) + 15.0*bF12*x4*y2*y2*y2*pw(-2.0*e, n + 12) + 1.0*bF13*x2*x2*x2*y2*y2*y2*pw(-2.0*e, n + 13) + 225.0*bF7*pw(-2.0*e, n + 7) + 675.0*bF8*x2*pw(-2.0*e, n + 8) + 675.0*bF8*y2*pw(-2.0*e, n + 8) + 2025.0*bF9*x2*y2*pw(-2.0*e, n + 9) + 225.0*bF9*x4*pw(-2.0*e, n + 9) + 225.0*bF9*y4*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x662) {
            float bF7 = 0.0, bF8 = 0.0, bF9 = 0.0, bF10 = 0.0, bF11 = 0.0, bF12 = 0.0, bF13 = 0.0, bF14 = 0.0;
            bFValues(bF7, bF8, bF9, bF10, bF11, bF12, bF13, bF14, e*r2, n + 7, n + 8, n + 9, n + 10, n + 11, n + 12, n + 13, n + 14);
            return 15.0*bF10*x2*x2*x2*pw(-2.0*e, n + 10) + 2025.0*bF10*x2*y2*z2*pw(-2.0*e, n + 10) + 675.0*bF10*x2*y4*pw(-2.0*e, n + 10) + 675.0*bF10*x4*y2*pw(-2.0*e, n + 10) + 225.0*bF10*x4*z2*pw(-2.0*e, n + 10) + 15.0*bF10*y2*y2*y2*pw(-2.0*e, n + 10) + 225.0*bF10*y4*z2*pw(-2.0*e, n + 10) + 45.0*bF11*x2*x2*x2*y2*pw(-2.0*e, n + 11) + 15.0*bF11*x2*x2*x2*z2*pw(-2.0*e, n + 11) + 45.0*bF11*x2*y2*y2*y2*pw(-2.0*e, n + 11) + 675.0*bF11*x2*y4*z2*pw(-2.0*e, n + 11) + 675.0*bF11*x4*y2*z2*pw(-2.0*e, n + 11) + 225.0*bF11*x4*y4*pw(-2.0*e, n + 11) + 15.0*bF11*y2*y2*y2*z2*pw(-2.0*e, n + 11) + 45.0*bF12*x2*x2*x2*y2*z2*pw(-2.0*e, n + 12) + 15.0*bF12*x2*x2*x2*y4*pw(-2.0*e, n + 12) + 45.0*bF12*x2*y2*y2*y2*z2*pw(-2.0*e, n + 12) + 15.0*bF12*x4*y2*y2*y2*pw(-2.0*e, n + 12) + 225.0*bF12*x4*y4*z2*pw(-2.0*e, n + 12) + 1.0*bF13*x2*x2*x2*y2*y2*y2*pw(-2.0*e, n + 13) + 15.0*bF13*x2*x2*x2*y4*z2*pw(-2.0*e, n + 13) + 15.0*bF13*x4*y2*y2*y2*z2*pw(-2.0*e, n + 13) + 1.0*bF14*x2*x2*x2*y2*y2*y2*z2*pw(-2.0*e, n + 14) + 225.0*bF7*pw(-2.0*e, n + 7) + 675.0*bF8*x2*pw(-2.0*e, n + 8) + 675.0*bF8*y2*pw(-2.0*e, n + 8) + 225.0*bF8*z2*pw(-2.0*e, n + 8) + 2025.0*bF9*x2*y2*pw(-2.0*e, n + 9) + 675.0*bF9*x2*z2*pw(-2.0*e, n + 9) + 225.0*bF9*x4*pw(-2.0*e, n + 9) + 675.0*bF9*y2*z2*pw(-2.0*e, n + 9) + 225.0*bF9*y4*pw(-2.0*e, n + 9);
        }
        if (hexIJK == 0x663) {
            float bF8 = 0.0, bF9 = 0.0, bF10 = 0.0, bF11 = 0.0, bF12 = 0.0, bF13 = 0.0, bF14 = 0.0, bF15 = 0.0;
            bFValues(bF8, bF9, bF10, bF11, bF12, bF13, bF14, bF15, e*r2, n + 8, n + 9, n + 10, n + 11, n + 12, n + 13, n + 14, n + 15);
            return z*(6075.0*bF10*x2*y2*pw(-2.0*e, n + 10) + 675.0*bF10*x2*z2*pw(-2.0*e, n + 10) + 675.0*bF10*x4*pw(-2.0*e, n + 10) + 675.0*bF10*y2*z2*pw(-2.0*e, n + 10) + 675.0*bF10*y4*pw(-2.0*e, n + 10) + 45.0*bF11*x2*x2*x2*pw(-2.0*e, n + 11) + 2025.0*bF11*x2*y2*z2*pw(-2.0*e, n + 11) + 2025.0*bF11*x2*y4*pw(-2.0*e, n + 11) + 2025.0*bF11*x4*y2*pw(-2.0*e, n + 11) + 225.0*bF11*x4*z2*pw(-2.0*e, n + 11) + 45.0*bF11*y2*y2*y2*pw(-2.0*e, n + 11) + 225.0*bF11*y4*z2*pw(-2.0*e, n + 11) + 135.0*bF12*x2*x2*x2*y2*pw(-2.0*e, n + 12) + 15.0*bF12*x2*x2*x2*z2*pw(-2.0*e, n + 12) + 135.0*bF12*x2*y2*y2*y2*pw(-2.0*e, n + 12) + 675.0*bF12*x2*y4*z2*pw(-2.0*e, n + 12) + 675.0*bF12*x4*y2*z2*pw(-2.0*e, n + 12) + 675.0*bF12*x4*y4*pw(-2.0*e, n + 12) + 15.0*bF12*y2*y2*y2*z2*pw(-2.0*e, n + 12) + 45.0*bF13*x2*x2*x2*y2*z2*pw(-2.0*e, n + 13) + 45.0*bF13*x2*x2*x2*y4*pw(-2.0*e, n + 13) + 45.0*bF13*x2*y2*y2*y2*z2*pw(-2.0*e, n + 13) + 45.0*bF13*x4*y2*y2*y2*pw(-2.0*e, n + 13) + 225.0*bF13*x4*y4*z2*pw(-2.0*e, n + 13) + 3.0*bF14*x2*x2*x2*y2*y2*y2*pw(-2.0*e, n + 14) + 15.0*bF14*x2*x2*x2*y4*z2*pw(-2.0*e, n + 14) + 15.0*bF14*x4*y2*y2*y2*z2*pw(-2.0*e, n + 14) + 1.0*bF15*x2*x2*x2*y2*y2*y2*z2*pw(-2.0*e, n + 15) + 675.0*bF8*pw(-2.0*e, n + 8) + 2025.0*bF9*x2*pw(-2.0*e, n + 9) + 2025.0*bF9*y2*pw(-2.0*e, n + 9) + 225.0*bF9*z2*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x664) {
            float bF8 = 0.0, bF9 = 0.0, bF10 = 0.0, bF11 = 0.0, bF12 = 0.0, bF13 = 0.0, bF14 = 0.0, bF15 = 0.0, bF16 = 0.0;
            bFValues(bF8, bF9, bF10, bF11, bF12, bF13, bF14, bF15, bF16, e*r2, n + 8, n + 9, n + 10, n + 11, n + 12, n + 13, n + 14, n + 15, n + 16);
            return 6075.0*bF10*x2*y2*pw(-2.0*e, n + 10) + 4050.0*bF10*x2*z2*pw(-2.0*e, n + 10) + 675.0*bF10*x4*pw(-2.0*e, n + 10) + 4050.0*bF10*y2*z2*pw(-2.0*e, n + 10) + 675.0*bF10*y4*pw(-2.0*e, n + 10) + 225.0*bF10*z4*pw(-2.0*e, n + 10) + 45.0*bF11*x2*x2*x2*pw(-2.0*e, n + 11) + 12150.0*bF11*x2*y2*z2*pw(-2.0*e, n + 11) + 2025.0*bF11*x2*y4*pw(-2.0*e, n + 11) + 675.0*bF11*x2*z4*pw(-2.0*e, n + 11) + 2025.0*bF11*x4*y2*pw(-2.0*e, n + 11) + 1350.0*bF11*x4*z2*pw(-2.0*e, n + 11) + 45.0*bF11*y2*y2*y2*pw(-2.0*e, n + 11) + 675.0*bF11*y2*z4*pw(-2.0*e, n + 11) + 1350.0*bF11*y4*z2*pw(-2.0*e, n + 11) + 135.0*bF12*x2*x2*x2*y2*pw(-2.0*e, n + 12) + 90.0*bF12*x2*x2*x2*z2*pw(-2.0*e, n + 12) + 135.0*bF12*x2*y2*y2*y2*pw(-2.0*e, n + 12) + 2025.0*bF12*x2*y2*z4*pw(-2.0*e, n + 12) + 4050.0*bF12*x2*y4*z2*pw(-2.0*e, n + 12) + 4050.0*bF12*x4*y2*z2*pw(-2.0*e, n + 12) + 675.0*bF12*x4*y4*pw(-2.0*e, n + 12) + 225.0*bF12*x4*z4*pw(-2.0*e, n + 12) + 90.0*bF12*y2*y2*y2*z2*pw(-2.0*e, n + 12) + 225.0*bF12*y4*z4*pw(-2.0*e, n + 12) + 270.0*bF13*x2*x2*x2*y2*z2*pw(-2.0*e, n + 13) + 45.0*bF13*x2*x2*x2*y4*pw(-2.0*e, n + 13) + 15.0*bF13*x2*x2*x2*z4*pw(-2.0*e, n + 13) + 270.0*bF13*x2*y2*y2*y2*z2*pw(-2.0*e, n + 13) + 675.0*bF13*x2*y4*z4*pw(-2.0*e, n + 13) + 45.0*bF13*x4*y2*y2*y2*pw(-2.0*e, n + 13) + 675.0*bF13*x4*y2*z4*pw(-2.0*e, n + 13) + 1350.0*bF13*x4*y4*z2*pw(-2.0*e, n + 13) + 15.0*bF13*y2*y2*y2*z4*pw(-2.0*e, n + 13) + 3.0*bF14*x2*x2*x2*y2*y2*y2*pw(-2.0*e, n + 14) + 45.0*bF14*x2*x2*x2*y2*z4*pw(-2.0*e, n + 14) + 90.0*bF14*x2*x2*x2*y4*z2*pw(-2.0*e, n + 14) + 45.0*bF14*x2*y2*y2*y2*z4*pw(-2.0*e, n + 14) + 90.0*bF14*x4*y2*y2*y2*z2*pw(-2.0*e, n + 14) + 225.0*bF14*x4*y4*z4*pw(-2.0*e, n + 14) + 6.0*bF15*x2*x2*x2*y2*y2*y2*z2*pw(-2.0*e, n + 15) + 15.0*bF15*x2*x2*x2*y4*z4*pw(-2.0*e, n + 15) + 15.0*bF15*x4*y2*y2*y2*z4*pw(-2.0*e, n + 15) + 1.0*bF16*x2*x2*x2*y2*y2*y2*z4*pw(-2.0*e, n + 16) + 675.0*bF8*pw(-2.0*e, n + 8) + 2025.0*bF9*x2*pw(-2.0*e, n + 9) + 2025.0*bF9*y2*pw(-2.0*e, n + 9) + 1350.0*bF9*z2*pw(-2.0*e, n + 9);
        }
        if (hexIJK == 0x665) {
            float bF9 = 0.0, bF10 = 0.0, bF11 = 0.0, bF12 = 0.0, bF13 = 0.0, bF14 = 0.0, bF15 = 0.0, bF16 = 0.0, bF17 = 0.0;
            bFValues(bF9, bF10, bF11, bF12, bF13, bF14, bF15, bF16, bF17, e*r2, n + 9, n + 10, n + 11, n + 12, n + 13, n + 14, n + 15, n + 16, n + 17);
            return z*(10125.0*bF10*x2*pw(-2.0*e, n + 10) + 10125.0*bF10*y2*pw(-2.0*e, n + 10) + 2250.0*bF10*z2*pw(-2.0*e, n + 10) + 30375.0*bF11*x2*y2*pw(-2.0*e, n + 11) + 6750.0*bF11*x2*z2*pw(-2.0*e, n + 11) + 3375.0*bF11*x4*pw(-2.0*e, n + 11) + 6750.0*bF11*y2*z2*pw(-2.0*e, n + 11) + 3375.0*bF11*y4*pw(-2.0*e, n + 11) + 225.0*bF11*z4*pw(-2.0*e, n + 11) + 225.0*bF12*x2*x2*x2*pw(-2.0*e, n + 12) + 20250.0*bF12*x2*y2*z2*pw(-2.0*e, n + 12) + 10125.0*bF12*x2*y4*pw(-2.0*e, n + 12) + 675.0*bF12*x2*z4*pw(-2.0*e, n + 12) + 10125.0*bF12*x4*y2*pw(-2.0*e, n + 12) + 2250.0*bF12*x4*z2*pw(-2.0*e, n + 12) + 225.0*bF12*y2*y2*y2*pw(-2.0*e, n + 12) + 675.0*bF12*y2*z4*pw(-2.0*e, n + 12) + 2250.0*bF12*y4*z2*pw(-2.0*e, n + 12) + 675.0*bF13*x2*x2*x2*y2*pw(-2.0*e, n + 13) + 150.0*bF13*x2*x2*x2*z2*pw(-2.0*e, n + 13) + 675.0*bF13*x2*y2*y2*y2*pw(-2.0*e, n + 13) + 2025.0*bF13*x2*y2*z4*pw(-2.0*e, n + 13) + 6750.0*bF13*x2*y4*z2*pw(-2.0*e, n + 13) + 6750.0*bF13*x4*y2*z2*pw(-2.0*e, n + 13) + 3375.0*bF13*x4*y4*pw(-2.0*e, n + 13) + 225.0*bF13*x4*z4*pw(-2.0*e, n + 13) + 150.0*bF13*y2*y2*y2*z2*pw(-2.0*e, n + 13) + 225.0*bF13*y4*z4*pw(-2.0*e, n + 13) + 450.0*bF14*x2*x2*x2*y2*z2*pw(-2.0*e, n + 14) + 225.0*bF14*x2*x2*x2*y4*pw(-2.0*e, n + 14) + 15.0*bF14*x2*x2*x2*z4*pw(-2.0*e, n + 14) + 450.0*bF14*x2*y2*y2*y2*z2*pw(-2.0*e, n + 14) + 675.0*bF14*x2*y4*z4*pw(-2.0*e, n + 14) + 225.0*bF14*x4*y2*y2*y2*pw(-2.0*e, n + 14) + 675.0*bF14*x4*y2*z4*pw(-2.0*e, n + 14) + 2250.0*bF14*x4*y4*z2*pw(-2.0*e, n + 14) + 15.0*bF14*y2*y2*y2*z4*pw(-2.0*e, n + 14) + 15.0*bF15*x2*x2*x2*y2*y2*y2*pw(-2.0*e, n + 15) + 45.0*bF15*x2*x2*x2*y2*z4*pw(-2.0*e, n + 15) + 150.0*bF15*x2*x2*x2*y4*z2*pw(-2.0*e, n + 15) + 45.0*bF15*x2*y2*y2*y2*z4*pw(-2.0*e, n + 15) + 150.0*bF15*x4*y2*y2*y2*z2*pw(-2.0*e, n + 15) + 225.0*bF15*x4*y4*z4*pw(-2.0*e, n + 15) + 10.0*bF16*x2*x2*x2*y2*y2*y2*z2*pw(-2.0*e, n + 16) + 15.0*bF16*x2*x2*x2*y4*z4*pw(-2.0*e, n + 16) + 15.0*bF16*x4*y2*y2*y2*z4*pw(-2.0*e, n + 16) + 1.0*bF17*x2*x2*x2*y2*y2*y2*z4*pw(-2.0*e, n + 17) + 3375.0*bF9*pw(-2.0*e, n + 9));
        }
        if (hexIJK == 0x666) {
            float bF9 = 0.0, bF10 = 0.0, bF11 = 0.0, bF12 = 0.0, bF13 = 0.0, bF14 = 0.0, bF15 = 0.0, bF16 = 0.0, bF17 = 0.0, bF18 = 0.0;
            bFValues(bF9, bF10, bF11, bF12, bF13, bF14, bF15, bF16, bF17, bF18, e*r2, n + 9, n + 10, n + 11, n + 12, n + 13, n + 14, n + 15, n + 16, n + 17, n + 18);
            return 10125.0*bF10*x2*pw(-2.0*e, n + 10) + 10125.0*bF10*y2*pw(-2.0*e, n + 10) + 10125.0*bF10*z2*pw(-2.0*e, n + 10) + 30375.0*bF11*x2*y2*pw(-2.0*e, n + 11) + 30375.0*bF11*x2*z2*pw(-2.0*e, n + 11) + 3375.0*bF11*x4*pw(-2.0*e, n + 11) + 30375.0*bF11*y2*z2*pw(-2.0*e, n + 11) + 3375.0*bF11*y4*pw(-2.0*e, n + 11) + 3375.0*bF11*z4*pw(-2.0*e, n + 11) + 225.0*bF12*x2*x2*x2*pw(-2.0*e, n + 12) + 91125.0*bF12*x2*y2*z2*pw(-2.0*e, n + 12) + 10125.0*bF12*x2*y4*pw(-2.0*e, n + 12) + 10125.0*bF12*x2*z4*pw(-2.0*e, n + 12) + 10125.0*bF12*x4*y2*pw(-2.0*e, n + 12) + 10125.0*bF12*x4*z2*pw(-2.0*e, n + 12) + 225.0*bF12*y2*y2*y2*pw(-2.0*e, n + 12) + 10125.0*bF12*y2*z4*pw(-2.0*e, n + 12) + 10125.0*bF12*y4*z2*pw(-2.0*e, n + 12) + 225.0*bF12*z2*z2*z2*pw(-2.0*e, n + 12) + 675.0*bF13*x2*x2*x2*y2*pw(-2.0*e, n + 13) + 675.0*bF13*x2*x2*x2*z2*pw(-2.0*e, n + 13) + 675.0*bF13*x2*y2*y2*y2*pw(-2.0*e, n + 13) + 30375.0*bF13*x2*y2*z4*pw(-2.0*e, n + 13) + 30375.0*bF13*x2*y4*z2*pw(-2.0*e, n + 13) + 675.0*bF13*x2*z2*z2*z2*pw(-2.0*e, n + 13) + 30375.0*bF13*x4*y2*z2*pw(-2.0*e, n + 13) + 3375.0*bF13*x4*y4*pw(-2.0*e, n + 13) + 3375.0*bF13*x4*z4*pw(-2.0*e, n + 13) + 675.0*bF13*y2*y2*y2*z2*pw(-2.0*e, n + 13) + 675.0*bF13*y2*z2*z2*z2*pw(-2.0*e, n + 13) + 3375.0*bF13*y4*z4*pw(-2.0*e, n + 13) + 2025.0*bF14*x2*x2*x2*y2*z2*pw(-2.0*e, n + 14) + 225.0*bF14*x2*x2*x2*y4*pw(-2.0*e, n + 14) + 225.0*bF14*x2*x2*x2*z4*pw(-2.0*e, n + 14) + 2025.0*bF14*x2*y2*y2*y2*z2*pw(-2.0*e, n + 14) + 2025.0*bF14*x2*y2*z2*z2*z2*pw(-2.0*e, n + 14) + 10125.0*bF14*x2*y4*z4*pw(-2.0*e, n + 14) + 225.0*bF14*x4*y2*y2*y2*pw(-2.0*e, n + 14) + 10125.0*bF14*x4*y2*z4*pw(-2.0*e, n + 14) + 10125.0*bF14*x4*y4*z2*pw(-2.0*e, n + 14) + 225.0*bF14*x4*z2*z2*z2*pw(-2.0*e, n + 14) + 225.0*bF14*y2*y2*y2*z4*pw(-2.0*e, n + 14) + 225.0*bF14*y4*z2*z2*z2*pw(-2.0*e, n + 14) + 15.0*bF15*x2*x2*x2*y2*y2*y2*pw(-2.0*e, n + 15) + 675.0*bF15*x2*x2*x2*y2*z4*pw(-2.0*e, n + 15) + 675.0*bF15*x2*x2*x2*y4*z2*pw(-2.0*e, n + 15) + 15.0*bF15*x2*x2*x2*z2*z2*z2*pw(-2.0*e, n + 15) + 675.0*bF15*x2*y2*y2*y2*z4*pw(-2.0*e, n + 15) + 675.0*bF15*x2*y4*z2*z2*z2*pw(-2.0*e, n + 15) + 675.0*bF15*x4*y2*y2*y2*z2*pw(-2.0*e, n + 15) + 675.0*bF15*x4*y2*z2*z2*z2*pw(-2.0*e, n + 15) + 3375.0*bF15*x4*y4*z4*pw(-2.0*e, n + 15) + 15.0*bF15*y2*y2*y2*z2*z2*z2*pw(-2.0*e, n + 15) + 45.0*bF16*x2*x2*x2*y2*y2*y2*z2*pw(-2.0*e, n + 16) + 45.0*bF16*x2*x2*x2*y2*z2*z2*z2*pw(-2.0*e, n + 16) + 225.0*bF16*x2*x2*x2*y4*z4*pw(-2.0*e, n + 16) + 45.0*bF16*x2*y2*y2*y2*z2*z2*z2*pw(-2.0*e, n + 16) + 225.0*bF16*x4*y2*y2*y2*z4*pw(-2.0*e, n + 16) + 225.0*bF16*x4*y4*z2*z2*z2*pw(-2.0*e, n + 16) + 15.0*bF17*x2*x2*x2*y2*y2*y2*z4*pw(-2.0*e, n + 17) + 15.0*bF17*x2*x2*x2*y4*z2*z2*z2*pw(-2.0*e, n + 17) + 15.0*bF17*x4*y2*y2*y2*z2*z2*z2*pw(-2.0*e, n + 17) + 1.0*bF18*x2*x2*x2*y2*y2*y2*z2*z2*z2*pw(-2.0*e, n + 18) + 3375.0*bF9*pw(-2.0*e, n + 9);
        }
    }
    return val;

}

// float ccHelper766(int n, float orbExp, vec3 r12) {
//     return 6.0*coulombCoefficientHelper(ivec3(5, 6, 6), n, orbExp, r12)
//         + r12.x*coulombCoefficientHelper(ivec3(6, 6, 6), n, orbExp, r12);
// }

// float ccHelper776(int n, float orbExp, vec2 r12) {
//     return 6.0*ccHelper765(n, orbExp, r12)
//         + r12.y*ccHelper766(n, orbExp, r12);
// }

// float ccHelper777(int n, float orbExp, vec2 r12) {
//     return 6.0*ccHelper765(n, orbExp, r12)
//         + r12.y*ccHelper766(n, orbExp, r12);
// }


float coulombCoefficient(ivec3 indices, int n, float orbExp, vec3 r12) {
    vec3 s12 = r12;
    ivec3 indices2 = indices;
    // 1 2 3
    if (indices2[2] > indices2[1]) {
        SWAP_I(indices2[2], indices2[1]);
        SWAP_F(s12.z, s12.y);
    }
    // 1 3 2
    if (indices2[1] > indices2[0]) {
        SWAP_I(indices2[0], indices2[1]);
        SWAP_F(s12.y, s12.x);
    }
    // 3 1 2
    if (indices2[2] > indices2[1]) {
        SWAP_I(indices2[2], indices2[1]);
        SWAP_F(s12.z, s12.y);
    }
    // if (indices[0] < 7) {
    return coulombCoefficientHelper(indices2, n, orbExp, s12);
    // }
    // else {
    //     int i = indices[0], j = indices[1], k = indices[2];
    //     int hexIJK = i*16*16 + j*16 + k;
    //     if (hexIJK == 0x766) {
    //         return ccHelper766(n, orbExp, r12);
    //     } else if (hexIJK == 0x776) {
    //         return ccHelper776(n, orbExp, r12);
    //     } else if (hexIJK == 0x777) {
    //         return ccHelper766(n, orbExp, r12);
    //     }
    // }
}
