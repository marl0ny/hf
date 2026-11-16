// #version 120
// #pragma optimize(off)

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
#define SWAP_I2(a, b) int tmp_i2 = (a); (a) = (b); (b) = (tmp_i2)

uniform sampler2D basisFunctionSpec1Tex;
uniform sampler2D basisFunctionSpec2Tex;
uniform sampler2D primitivesTex;
uniform sampler2D boysFunctionTex;
uniform sampler2D indicesTex;
uniform sampler2D previousSumTex;
uniform int numberOfBasisFunctions;
uniform int maxNumberOfPrimitives;
uniform ivec4 primitiveIndices;
uniform int maxAngularIter;
uniform int shellCount;
uniform int shellQuartetsCount;
uniform int shellTexWidth;

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
void computeBFValuesRecurseUpwards(inout float bF[16], float x, int nMax) {
    complex cVal = computeBFRecurseUpwardsIntermediary(x);
    float val = (mul(complex(0.5, 0.0), cVal)).r;
    if (nMax > 12) {
        for (int nIter = nMax; nIter > 12; nIter--)
            bF[nIter - 1] = val;
    }
    for (int nIter = 12; nIter > 0; nIter--) {
        val = x/(float(nIter) - 0.5)*val 
            + exp(-x)/(2.0*(float(nIter) - 0.5));
        bF[nIter - 1] = val;
    }
}

/* See "A fast algorithm for computing the Boys function"
by Gregory Beylkin and Sandeep Sharma, particularly the section containing
equation (4) for the n = 0 case for the recursion relation
that goes to zero. */
void computeBFValuesRecurseToZero(inout float bF[16], float x, int nMax) {
    float val = sqrt(PI)*erf(sqrt(x))/(2.0*sqrt(x));
    float bx = 0.5*exp(-x)/x;
    bF[0] = val;
    for (int nIter = 1; nIter <= nMax; nIter++) {
        val = ((float(nIter) - 0.5)/x)*val - bx;
        bF[nIter] = val;
    }
}

void computeBoysFunctionValues(inout float bF[16], float x, int nMax) {
    if (abs(x) < 1e-60) {
        for (int n = 0; n <= nMax; n++) {
            bF[n] = 1.0/(2.0*float(n) + 1.0);
        }
    }
    // float y = 1.0;
    // int nMax = 12;
    // for (int j = 1; j <= nMax; j++) // See I.
    //     y *= (j - 0.5);
    // float z = pow(y, 1.0/float(nMax));
    if (abs(x) < RECURSE_DIR_TURN_POINT)
        computeBFValuesRecurseUpwards(bF, x, nMax);
    else
        computeBFValuesRecurseToZero(bF, x, nMax);

}

int getSymmetric4Index1D(int a, int b, int c, int d, inout ivec4 extra) {
    int n = shellCount;
    int symmetricSize2 = (n*(n + 1))/2;
    int symmetricSize4 = shellQuartetsCount;
    if (b < a) {
        SWAP_I(a, b);
        SWAP_I2(extra[0], extra[1]);
    }
    if (d < c) {
        SWAP_I(c, d);
        SWAP_I2(extra[2], extra[3]);
    }
    int indAB 
        = symmetricSize2 - ((n - a)*((n - a) + 1))/2 + (b - a);
    int indCD
        = symmetricSize2 - ((n - c)*((n - c) + 1))/2 + (d - c);
    if (indCD < indAB) {
        SWAP_I(indAB, indCD);
        ivec4 tmpExtra = ivec4(extra[0], extra[1], extra[2], extra[3]);
        extra[0] = tmpExtra[2];
        extra[1] = tmpExtra[3];
        extra[2] = tmpExtra[0];
        extra[3] = tmpExtra[1];
    }
    return (symmetricSize4 
        - ((symmetricSize2 - indAB)*((symmetricSize2 - indAB) + 1))/2
        + (indCD - indAB));
}

int getSymmetric4Index1D(int a, int b, int c, int d) {
    int n = shellCount;
    int symmetricSize2 = (n*(n + 1))/2;
    int symmetricSize4 = shellQuartetsCount;
    if (b < a) {
        SWAP_I(a, b);
    }
    if (d < c) {
        SWAP_I(c, d);
    }
    int indAB 
        = symmetricSize2 - ((n - a)*((n - a) + 1))/2 + (b - a);
    int indCD
        = symmetricSize2 - ((n - c)*((n - c) + 1))/2 + (d - c);
    if (indCD < indAB) {
        SWAP_I(indAB, indCD);
    }
    return (symmetricSize4 
        - ((symmetricSize2 - indAB)*((symmetricSize2 - indAB) + 1))/2
        + (indCD - indAB));
}

float coulombCoefficientHelper000To332(
    inout float bF[16],
    ivec3 indices, float orbExp, vec3 r12) {
    vec3 s12 = r12;
    int i = indices[0], j = indices[1], k = indices[2];
    int hexIJK = i*16*16 + j*16 + k;
    float e = orbExp;
    float r2 = dot(r12, r12);
    float val = 0.0;
    float x = s12.x, y = s12.y, z = s12.z;
    if (hexIJK == 0x000) {
        return bF[0];
    }
    else if (hexIJK == 0x100) {
        float ep1 = e;
        return -2.0*bF[1]*ep1*x;
    }
    else if (hexIJK == 0x110) {
        float ep2 = e*e;
        return 4.0*bF[2]*ep2*x*y;
    }
    else if (hexIJK == 0x111) {
        float ep3 = e*e*e;
        return -8*bF[3]*ep3*x*y*z;
    }
    else if (hexIJK == 0x200) {
        float ep1 = e;
        float ep2 = e*e;
        float x2 = x*x;
        return -2.0*bF[1]*ep1 + 4.0*bF[2]*ep2*x2;
    }
    else if (hexIJK == 0x210) {
        float ep3 = e*e*e;
        float ep2 = e*e;
        float x2 = x*x;
        return 4.0*bF[2]*ep2*y - 8.0*bF[3]*ep3*x2*y;
    }
    else if (hexIJK == 0x211) {
        float ep3 = e*e*e;
        float ep4 = e*e*e*e;
        float x2 = x*x;
        return -8.0*bF[3]*ep3*y*z + 16.0*bF[4]*ep4*x2*y*z;
    }
    else if (hexIJK == 0x220) {
        float ep4 = e*e*e*e;
        float ep3 = e*e*e;
        float ep2 = e*e;
        float x2 = x*x;
        float y2 = y*y;
        return 4.0*bF[2]*ep2 - 8.0*bF[3]*ep3*x2 - 8.0*bF[3]*ep3*y2 + 16.0*bF[4]*ep4*x2*y2;
    }
    else if (hexIJK == 0x221) {
        float ep3 = e*e*e;
        float ep4 = e*e*e*e;
        float ep5 = e*e*e*e*e;
        float x2 = x*x;
        float y2 = y*y;
        return -8.0*bF[3]*ep3*z + 16.0*bF[4]*ep4*x2*z + 16.0*bF[4]*ep4*y2*z - 32.0*bF[5]*ep5*x2*y2*z;
    }
    else if (hexIJK == 0x222) {
        float ep6 = e*e*e*e*e*e;
        float ep3 = e*e*e;
        float ep4 = e*e*e*e;
        float ep5 = e*e*e*e*e;
        float x2 = x*x;
        float y2 = y*y;
        float z2 = z*z;
        return -8.0*bF[3]*ep3 + 16.0*bF[4]*ep4*x2 + 16.0*bF[4]*ep4*y2 + 16.0*bF[4]*ep4*z2 - 32.0*bF[5]*ep5*x2*y2 - 32.0*bF[5]*ep5*x2*z2 - 32.0*bF[5]*ep5*y2*z2 + 64.0*bF[6]*ep6*x2*y2*z2;
    }
    else if (hexIJK == 0x300) {
        float ep3 = e*e*e;
        float ep2 = e*e;
        float x2 = x*x;
        return 12.0*bF[2]*ep2*x - 8.0*bF[3]*ep3*x*x2;
    }
    else if (hexIJK == 0x310) {
        float ep3 = e*e*e;
        float ep4 = e*e*e*e;
        float x2 = x*x;
        return -24.0*bF[3]*ep3*x*y + 16.0*bF[4]*ep4*x*x2*y;
    }
    else if (hexIJK == 0x311) {
        float ep4 = e*e*e*e;
        float ep5 = e*e*e*e*e;
        float x2 = x*x;
        return 48.0*bF[4]*ep4*x*y*z - 32.0*bF[5]*ep5*x*x2*y*z;
    }
    else if (hexIJK == 0x320) {
        float ep3 = e*e*e;
        float ep4 = e*e*e*e;
        float ep5 = e*e*e*e*e;
        float x2 = x*x;
        float y2 = y*y;
        return -24.0*bF[3]*ep3*x + 16.0*bF[4]*ep4*x*x2 + 48.0*bF[4]*ep4*x*y2 - 32.0*bF[5]*ep5*x*x2*y2;
    }
    else if (hexIJK == 0x321) {
        float ep6 = e*e*e*e*e*e;
        float ep4 = e*e*e*e;
        float ep5 = e*e*e*e*e;
        float x2 = x*x;
        float y2 = y*y;
        return 48.0*bF[4]*ep4*x*z - 32.0*bF[5]*ep5*x*x2*z - 96.0*bF[5]*ep5*x*y2*z + 64.0*bF[6]*ep6*x*x2*y2*z;
    }
    else if (hexIJK == 0x322) {
        float ep6 = e*e*e*e*e*e;
        float ep7 = e*e*e*e*e*e*e;
        float ep4 = e*e*e*e;
        float ep5 = e*e*e*e*e;
        float x2 = x*x;
        float y2 = y*y;
        float z2 = z*z;
        return 48.0*bF[4]*ep4*x - 32.0*bF[5]*ep5*x*x2 - 96.0*bF[5]*ep5*x*y2 - 96.0*bF[5]*ep5*x*z2 + 64.0*bF[6]*ep6*x*x2*y2 + 64.0*bF[6]*ep6*x*x2*z2 + 192.0*bF[6]*ep6*x*y2*z2 - 128.0*bF[7]*ep7*x*x2*y2*z2;
    }
    else if (hexIJK == 0x330) {
        float ep6 = e*e*e*e*e*e;
        float ep4 = e*e*e*e;
        float ep5 = e*e*e*e*e;
        float x2 = x*x;
        float y2 = y*y;
        return 144.0*bF[4]*ep4*x*y - 96.0*bF[5]*ep5*x*x2*y - 96.0*bF[5]*ep5*x*y*y2 + 64.0*bF[6]*ep6*x*x2*y*y2;
    }
    else if (hexIJK == 0x331) {
        float ep6 = e*e*e*e*e*e;
        float ep7 = e*e*e*e*e*e*e;
        float ep5 = e*e*e*e*e;
        float x2 = x*x;
        float y2 = y*y;
        return -288.0*bF[5]*ep5*x*y*z + 192.0*bF[6]*ep6*x*x2*y*z + 192.0*bF[6]*ep6*x*y*y2*z - 128.0*bF[7]*ep7*x*x2*y*y2*z;
    }
    else if (hexIJK == 0x332) {
        float ep6 = e*e*e*e*e*e;
        float ep8 = e*e*e*e*e*e*e*e;
        float ep7 = e*e*e*e*e*e*e;
        float ep5 = e*e*e*e*e;
        float x2 = x*x;
        float y2 = y*y;
        float z2 = z*z;
        return -288.0*bF[5]*ep5*x*y + 192.0*bF[6]*ep6*x*x2*y + 192.0*bF[6]*ep6*x*y*y2 + 576.0*bF[6]*ep6*x*y*z2 - 128.0*bF[7]*ep7*x*x2*y*y2 - 384.0*bF[7]*ep7*x*x2*y*z2 - 384.0*bF[7]*ep7*x*y*y2*z2 + 256.0*bF[8]*ep8*x*x2*y*y2*z2;
    }
}


float coulombCoefficientHelper400To800(
    inout float bF[16],
    ivec3 indices, float orbExp, vec3 r12) {
    vec3 s12 = r12;
    int i = indices[0], j = indices[1], k = indices[2];
    int hexIJK = i*16*16 + j*16 + k;
    float e = orbExp;
    float r2 = dot(r12, r12);
    float val = 0.0;
    float x = s12.x, y = s12.y, z = s12.z;
    if (hexIJK == 0x400) {
        float ep4 = e*e*e*e;
        float ep3 = e*e*e;
        float ep2 = e*e;
        float x4 = x*x*x*x;
        float x2 = x*x;
        return 12.0*bF[2]*ep2 - 48.0*bF[3]*ep3*x2 + 16.0*bF[4]*ep4*x4;
    }
    else if (hexIJK == 0x410) {
        float ep3 = e*e*e;
        float ep4 = e*e*e*e;
        float ep5 = e*e*e*e*e;
        float x4 = x*x*x*x;
        float x2 = x*x;
        return -24.0*bF[3]*ep3*y + 96.0*bF[4]*ep4*x2*y - 32.0*bF[5]*ep5*x4*y;
    }
    else if (hexIJK == 0x411) {
        float ep6 = e*e*e*e*e*e;
        float ep4 = e*e*e*e;
        float ep5 = e*e*e*e*e;
        float x4 = x*x*x*x;
        float x2 = x*x;
        return 48.0*bF[4]*ep4*y*z - 192.0*bF[5]*ep5*x2*y*z + 64.0*bF[6]*ep6*x4*y*z;
    }
    else if (hexIJK == 0x420) {
        float ep6 = e*e*e*e*e*e;
        float ep3 = e*e*e;
        float ep4 = e*e*e*e;
        float ep5 = e*e*e*e*e;
        float x4 = x*x*x*x;
        float x2 = x*x;
        float y2 = y*y;
        return -24.0*bF[3]*ep3 + 96.0*bF[4]*ep4*x2 + 48.0*bF[4]*ep4*y2 - 192.0*bF[5]*ep5*x2*y2 - 32.0*bF[5]*ep5*x4 + 64.0*bF[6]*ep6*x4*y2;
    }
    else if (hexIJK == 0x421) {
        float ep6 = e*e*e*e*e*e;
        float ep7 = e*e*e*e*e*e*e;
        float ep4 = e*e*e*e;
        float ep5 = e*e*e*e*e;
        float x4 = x*x*x*x;
        float x2 = x*x;
        float y2 = y*y;
        return 48.0*bF[4]*ep4*z - 192.0*bF[5]*ep5*x2*z - 96.0*bF[5]*ep5*y2*z + 384.0*bF[6]*ep6*x2*y2*z + 64.0*bF[6]*ep6*x4*z - 128.0*bF[7]*ep7*x4*y2*z;
    }
    else if (hexIJK == 0x422) {
        float ep6 = e*e*e*e*e*e;
        float ep8 = e*e*e*e*e*e*e*e;
        float ep4 = e*e*e*e;
        float ep5 = e*e*e*e*e;
        float ep7 = e*e*e*e*e*e*e;
        float x4 = x*x*x*x;
        float x2 = x*x;
        float y2 = y*y;
        float z2 = z*z;
        return 48.0*bF[4]*ep4 - 192.0*bF[5]*ep5*x2 - 96.0*bF[5]*ep5*y2 - 96.0*bF[5]*ep5*z2 + 384.0*bF[6]*ep6*x2*y2 + 384.0*bF[6]*ep6*x2*z2 + 64.0*bF[6]*ep6*x4 + 192.0*bF[6]*ep6*y2*z2 - 768.0*bF[7]*ep7*x2*y2*z2 - 128.0*bF[7]*ep7*x4*y2 - 128.0*bF[7]*ep7*x4*z2 + 256.0*bF[8]*ep8*x4*y2*z2;
    }
    else if (hexIJK == 0x430) {
        float ep6 = e*e*e*e*e*e;
        float ep7 = e*e*e*e*e*e*e;
        float ep4 = e*e*e*e;
        float ep5 = e*e*e*e*e;
        float x4 = x*x*x*x;
        float x2 = x*x;
        float y2 = y*y;
        return 144.0*bF[4]*ep4*y - 576.0*bF[5]*ep5*x2*y - 96.0*bF[5]*ep5*y*y2 + 384.0*bF[6]*ep6*x2*y*y2 + 192.0*bF[6]*ep6*x4*y - 128.0*bF[7]*ep7*x4*y*y2;
    }
    else if (hexIJK == 0x431) {
        float ep6 = e*e*e*e*e*e;
        float ep8 = e*e*e*e*e*e*e*e;
        float ep7 = e*e*e*e*e*e*e;
        float ep5 = e*e*e*e*e;
        float x4 = x*x*x*x;
        float x2 = x*x;
        float y2 = y*y;
        return -288.0*bF[5]*ep5*y*z + 1152.0*bF[6]*ep6*x2*y*z + 192.0*bF[6]*ep6*y*y2*z - 768.0*bF[7]*ep7*x2*y*y2*z - 384.0*bF[7]*ep7*x4*y*z + 256.0*bF[8]*ep8*x4*y*y2*z;
    }
    else if (hexIJK == 0x440) {
        float ep6 = e*e*e*e*e*e;
        float ep8 = e*e*e*e*e*e*e*e;
        float ep4 = e*e*e*e;
        float ep5 = e*e*e*e*e;
        float ep7 = e*e*e*e*e*e*e;
        float x4 = x*x*x*x;
        float x2 = x*x;
        float y4 = y*y*y*y;
        float y2 = y*y;
        return 144.0*bF[4]*ep4 - 576.0*bF[5]*ep5*x2 - 576.0*bF[5]*ep5*y2 + 2304.0*bF[6]*ep6*x2*y2 + 192.0*bF[6]*ep6*x4 + 192.0*bF[6]*ep6*y4 - 768.0*bF[7]*ep7*x2*y4 - 768.0*bF[7]*ep7*x4*y2 + 256.0*bF[8]*ep8*x4*y4;
    }
    else if (hexIJK == 0x500) {
        float ep3 = e*e*e;
        float ep4 = e*e*e*e;
        float ep5 = e*e*e*e*e;
        float x4 = x*x*x*x;
        float x2 = x*x;
        return -120.0*bF[3]*ep3*x + 160.0*bF[4]*ep4*x*x2 - 32.0*bF[5]*ep5*x*x4;
    }
    else if (hexIJK == 0x510) {
        float ep6 = e*e*e*e*e*e;
        float ep4 = e*e*e*e;
        float ep5 = e*e*e*e*e;
        float x4 = x*x*x*x;
        float x2 = x*x;
        return 240.0*bF[4]*ep4*x*y - 320.0*bF[5]*ep5*x*x2*y + 64.0*bF[6]*ep6*x*x4*y;
    }
    else if (hexIJK == 0x511) {
        float ep6 = e*e*e*e*e*e;
        float ep7 = e*e*e*e*e*e*e;
        float ep5 = e*e*e*e*e;
        float x4 = x*x*x*x;
        float x2 = x*x;
        return -480.0*bF[5]*ep5*x*y*z + 640.0*bF[6]*ep6*x*x2*y*z - 128.0*bF[7]*ep7*x*x4*y*z;
    }
    else if (hexIJK == 0x520) {
        float ep6 = e*e*e*e*e*e;
        float ep7 = e*e*e*e*e*e*e;
        float ep4 = e*e*e*e;
        float ep5 = e*e*e*e*e;
        float x4 = x*x*x*x;
        float x2 = x*x;
        float y2 = y*y;
        return 240.0*bF[4]*ep4*x - 320.0*bF[5]*ep5*x*x2 - 480.0*bF[5]*ep5*x*y2 + 640.0*bF[6]*ep6*x*x2*y2 + 64.0*bF[6]*ep6*x*x4 - 128.0*bF[7]*ep7*x*x4*y2;
    }
    else if (hexIJK == 0x521) {
        float ep6 = e*e*e*e*e*e;
        float ep8 = e*e*e*e*e*e*e*e;
        float ep7 = e*e*e*e*e*e*e;
        float ep5 = e*e*e*e*e;
        float x4 = x*x*x*x;
        float x2 = x*x;
        float y2 = y*y;
        return -480.0*bF[5]*ep5*x*z + 640.0*bF[6]*ep6*x*x2*z + 960.0*bF[6]*ep6*x*y2*z - 1280.0*bF[7]*ep7*x*x2*y2*z - 128.0*bF[7]*ep7*x*x4*z + 256.0*bF[8]*ep8*x*x4*y2*z;
    }
    else if (hexIJK == 0x530) {
        float ep6 = e*e*e*e*e*e;
        float ep8 = e*e*e*e*e*e*e*e;
        float ep7 = e*e*e*e*e*e*e;
        float ep5 = e*e*e*e*e;
        float x4 = x*x*x*x;
        float x2 = x*x;
        float y2 = y*y;
        return -1440.0*bF[5]*ep5*x*y + 1920.0*bF[6]*ep6*x*x2*y + 960.0*bF[6]*ep6*x*y*y2 - 1280.0*bF[7]*ep7*x*x2*y*y2 - 384.0*bF[7]*ep7*x*x4*y + 256.0*bF[8]*ep8*x*x4*y*y2;
    }
    else if (hexIJK == 0x600) {
        float ep6 = e*e*e*e*e*e;
        float ep3 = e*e*e;
        float ep4 = e*e*e*e;
        float ep5 = e*e*e*e*e;
        float x6 = x*x*x*x*x*x;
        float x4 = x*x*x*x;
        float x2 = x*x;
        return -120.0*bF[3]*ep3 + 720.0*bF[4]*ep4*x2 - 480.0*bF[5]*ep5*x4 + 64.0*bF[6]*ep6*x6;
    }
    else if (hexIJK == 0x610) {
        float ep6 = e*e*e*e*e*e;
        float ep7 = e*e*e*e*e*e*e;
        float ep4 = e*e*e*e;
        float ep5 = e*e*e*e*e;
        float x6 = x*x*x*x*x*x;
        float x4 = x*x*x*x;
        float x2 = x*x;
        return 240.0*bF[4]*ep4*y - 1440.0*bF[5]*ep5*x2*y + 960.0*bF[6]*ep6*x4*y - 128.0*bF[7]*ep7*x6*y;
    }
    else if (hexIJK == 0x611) {
        float ep6 = e*e*e*e*e*e;
        float ep8 = e*e*e*e*e*e*e*e;
        float ep7 = e*e*e*e*e*e*e;
        float ep5 = e*e*e*e*e;
        float x6 = x*x*x*x*x*x;
        float x4 = x*x*x*x;
        float x2 = x*x;
        return -480.0*bF[5]*ep5*y*z + 2880.0*bF[6]*ep6*x2*y*z - 1920.0*bF[7]*ep7*x4*y*z + 256.0*bF[8]*ep8*x6*y*z;
    }
    else if (hexIJK == 0x620) {
        float ep6 = e*e*e*e*e*e;
        float ep8 = e*e*e*e*e*e*e*e;
        float ep4 = e*e*e*e;
        float ep5 = e*e*e*e*e;
        float ep7 = e*e*e*e*e*e*e;
        float x6 = x*x*x*x*x*x;
        float x4 = x*x*x*x;
        float x2 = x*x;
        float y2 = y*y;
        return 240.0*bF[4]*ep4 - 1440.0*bF[5]*ep5*x2 - 480.0*bF[5]*ep5*y2 + 2880.0*bF[6]*ep6*x2*y2 + 960.0*bF[6]*ep6*x4 - 1920.0*bF[7]*ep7*x4*y2 - 128.0*bF[7]*ep7*x6 + 256.0*bF[8]*ep8*x6*y2;
    }
    else if (hexIJK == 0x700) {
        float ep6 = e*e*e*e*e*e;
        float ep7 = e*e*e*e*e*e*e;
        float ep4 = e*e*e*e;
        float ep5 = e*e*e*e*e;
        float x6 = x*x*x*x*x*x;
        float x4 = x*x*x*x;
        float x2 = x*x;
        return 1680.0*bF[4]*ep4*x - 3360.0*bF[5]*ep5*x*x2 + 1344.0*bF[6]*ep6*x*x4 - 128.0*bF[7]*ep7*x*x6;
    }
    else if (hexIJK == 0x710) {
        float ep6 = e*e*e*e*e*e;
        float ep8 = e*e*e*e*e*e*e*e;
        float ep7 = e*e*e*e*e*e*e;
        float ep5 = e*e*e*e*e;
        float x6 = x*x*x*x*x*x;
        float x4 = x*x*x*x;
        float x2 = x*x;
        return -3360.0*bF[5]*ep5*x*y + 6720.0*bF[6]*ep6*x*x2*y - 2688.0*bF[7]*ep7*x*x4*y + 256.0*bF[8]*ep8*x*x6*y;
    }
    else if (hexIJK == 0x800) {
        float ep6 = e*e*e*e*e*e;
        float ep8 = e*e*e*e*e*e*e*e;
        float ep4 = e*e*e*e;
        float ep5 = e*e*e*e*e;
        float ep7 = e*e*e*e*e*e*e;
        float x6 = x*x*x*x*x*x;
        float x4 = x*x*x*x;
        float x2 = x*x;
        float x8 = x4*x4;
        return 1680.0*bF[4]*ep4 - 13440.0*bF[5]*ep5*x2 + 13440.0*bF[6]*ep6*x4 - 3584.0*bF[7]*ep7*x6 + 256.0*bF[8]*ep8*x8;
    }
}

float coulombCoefficientHelper(
    inout float bF[16],
    ivec3 indices, float orbExp, vec3 r12) {
    if (indices[0] <= 3)
        return coulombCoefficientHelper000To332(bF, indices, orbExp, r12);
    return coulombCoefficientHelper400To800(bF, indices, orbExp, r12);
}

float coulombCoefficient(
    inout float bF[16],
    ivec3 indices, float orbExp, vec3 r12) {
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
    // if (indices2[0] > 6)
    //     return 0.0;
    return coulombCoefficientHelper(
        bF, indices2, orbExp, s12);
}


void fillOverlapCoefficients(
    inout vec4[8] overlaps, int index,
    int a1, int a2, float r, float e1, float e2, float expVal
) {
    if (a2 < a1) {
        SWAP_I(a2, a1);
        SWAP_F(e1, e2);
        r = -r;
    }
    int hexA1A2 = 16*a1 + a2;
    for (int i = 0; i < 10; i++)
        overlaps[i][index] = 0.0;
    if (hexA1A2 == 0x00) {
        overlaps[0][index] = expVal;
    }
    if (hexA1A2 == 0x01) {
        overlaps[0][index] = e1*expVal*r/(e1 + e2);
        overlaps[1][index] = 0.5*expVal/(e1 + e2);
    }
    if (hexA1A2 == 0x02) {
        float e1e2_2 = (e1 + e2)*(e1 + e2);
        float e1_2 = e1*e1;
        float r2 = r*r;
        overlaps[0][index] = expVal*(0.5*e1 + e1_2*r2 + 0.5*e2)/e1e2_2;
        overlaps[1][index] = 1.0*e1*expVal*r/e1e2_2;
        overlaps[2][index] = 0.25*expVal/e1e2_2;
    }
    if (hexA1A2 == 0x03) {
        float e1e2_2 = (e1 + e2)*(e1 + e2);
        float e1_2 = e1*e1;
        float r2 = r*r;
        float e1e2_3 = e1e2_2*(e1 + e2);
        overlaps[0][index] = e1*expVal*r*(1.5*e1 + e1_2*r2 + 1.5*e2)/e1e2_3;
        overlaps[1][index] = expVal*(0.75*e1 + 1.5*e1_2*r2 + 0.75*e2)/e1e2_3;
        overlaps[2][index] = 0.75*e1*expVal*r/e1e2_3;
        overlaps[3][index] = 0.125*expVal/e1e2_3;
    }
    if (hexA1A2 == 0x11) {
        float r2 = r*r;
        float e1e2_2 = (e1 + e2)*(e1 + e2);
        overlaps[0][index] = expVal*(-e1*e2*r2 + 0.5*e1 + 0.5*e2)/e1e2_2;
        overlaps[1][index] = 0.5*expVal*r*(e1 - e2)/e1e2_2;
        overlaps[2][index] = 0.25*expVal/e1e2_2;
    }
    if (hexA1A2 == 0x12) {
        float r2 = r*r;
        float e1_2 = e1*e1;
        float e1e2_2 = (e1 + e2)*(e1 + e2);
        float e1e2_3 = e1e2_2*(e1 + e2);
        overlaps[0][index] = expVal*r*(1.0*e1*(e1 + e2) - e2*(0.5*e1 + e1_2*r2 + 0.5*e2))/e1e2_3;
        overlaps[1][index] = expVal*(-1.0*e1*e2*r2 + 0.75*e1 + 0.5*e1_2*r2 + 0.75*e2)/e1e2_3;
        overlaps[2][index] = expVal*r*(0.5*e1 - 0.25*e2)/e1e2_3;
        overlaps[3][index] = 0.125*expVal/e1e2_3;
    }
    if (hexA1A2 == 0x13) {
        float r2 = r*r;
        float e1_2 = e1*e1;
        float e1e2_2 = (e1 + e2)*(e1 + e2);
        float e1e2_3 = e1e2_2*(e1 + e2);
        float e1e2_4 = e1e2_2*e1e2_2;
        float e2_2 = e2*e2;
        float e2_3 = e2_2*e2;
        float e2_4 = e2_2*e2_2;
        float e2_5 = e2_4*e2;
        float e2_6 = e2_3*e2_3;
        float e2_7 = e2_6*e2;
        float e2_8 = e2_4*e2_4;
        float e1e2_5 = e1e2_4*(e1 + e2);
        float e1e2_6 = e1e2_3*e1e2_3;
        float e1e2_7 = e1e2_6*(e1 + e2);
        float e1e2_8 = e1e2_4*e1e2_4;
        overlaps[0][index] = expVal*(-e1*e2*r2*(1.5*e1 + e1_2*r2 + 1.5*e2) + 1.0*e1_2*r2*(e1 + e2) + 0.5*e1e2_2 + (e1 + e2)*(0.25*e1 + 0.5*e1_2*r2 + 0.25*e2))/e1e2_4;
        overlaps[1][index] = expVal*r*(1.5*e1*(e1 + e2) + 0.5*e1*(1.5*e1 + e1_2*r2 + 1.5*e2) - e2*(0.75*e1 + 1.5*e1_2*r2 + 0.75*e2))/e1e2_4;
        overlaps[2][index] = 0.75*expVal*(-e1*e2*r2 + e1 + e1_2*r2 + e2)/e1e2_4;
        overlaps[3][index] = expVal*r*(0.375*e1 - 0.125*e2)/e1e2_4;
        overlaps[4][index] = 0.0625*expVal/e1e2_4;
    }
    if (hexA1A2 == 0x22) {
        float r2 = r*r;
        float r4 = r2*r2;
        float e1_2 = e1*e1;
        float e1e2_2 = (e1 + e2)*(e1 + e2);
        float e1e2_3 = e1e2_2*(e1 + e2);
        float e1e2_4 = e1e2_2*e1e2_2;
        float e2_2 = e2*e2;
        float e1_3 = e1_2*e1;
        float e1_4 = e1_2*e1_2;
        float e2_3 = e2_2*e2;
        float e2_4 = e2_2*e2_2;
        float e1_5 = e1_4*e1;
        float e2_5 = e2_4*e2;
        float e1e2_5 = e1e2_4*(e1 + e2);
        float e1e2_6 = e1e2_3*e1e2_3;
        float e1e2_7 = e1e2_6*(e1 + e2);
        float e1e2_8 = e1e2_4*e1e2_4;
        overlaps[0][index] = expVal*(-1.0*e1*e2*r2*(e1 + e2) + 0.5*e1e2_2 - e2*r2*(1.0*e1*(e1 + e2) - e2*(0.5*e1 + e1_2*r2 + 0.5*e2)) + (e1 + e2)*(0.25*e1 + 0.5*e1_2*r2 + 0.25*e2))/e1e2_4;
        overlaps[1][index] = expVal*r*(0.5*e1*(e1 + e2) - 0.5*e2*(0.5*e1 + e1_2*r2 + 0.5*e2) - e2*(-1.0*e1*e2*r2 + 0.75*e1 + 0.5*e1_2*r2 + 0.75*e2) + (e1 - 0.5*e2)*(e1 + e2))/e1e2_4;
        overlaps[2][index] = expVal*(-0.5*e1*e2*r2 + 0.75*e1 + 0.25*e1_2*r2 - e2*r2*(0.5*e1 - 0.25*e2) + 0.75*e2)/e1e2_4;
        overlaps[3][index] = 0.25*expVal*r*(e1 - e2)/e1e2_4;
        overlaps[4][index] = 0.0625*expVal/e1e2_4;
    }
    if (hexA1A2 == 0x23) {
        float r2 = r*r;
        float r4 = r2*r2;
        float e1_2 = e1*e1;
        float e1e2_2 = (e1 + e2)*(e1 + e2);
        float e1e2_3 = e1e2_2*(e1 + e2);
        float e1e2_4 = e1e2_2*e1e2_2;
        float e2_2 = e2*e2;
        float e1_3 = e1_2*e1;
        float e1_4 = e1_2*e1_2;
        float e2_3 = e2_2*e2;
        float e2_4 = e2_2*e2_2;
        float e1_5 = e1_4*e1;
        float e2_5 = e2_4*e2;
        float e1e2_5 = e1e2_4*(e1 + e2);
        float e1e2_6 = e1e2_3*e1e2_3;
        float e1e2_7 = e1e2_6*(e1 + e2);
        float e1e2_8 = e1e2_4*e1e2_4;
        overlaps[0][index] = expVal*r*(1.5*e1*e1e2_2 + 0.5*e1*(e1 + e2)*(1.5*e1 + e1_2*r2 + 1.5*e2) - e2*(e1 + e2)*(0.75*e1 + 1.5*e1_2*r2 + 0.75*e2) - e2*(-e1*e2*r2*(1.5*e1 + e1_2*r2 + 1.5*e2) + 1.0*e1_2*r2*(e1 + e2) + 0.5*e1e2_2 + (e1 + e2)*(0.25*e1 + 0.5*e1_2*r2 + 0.25*e2)))/e1e2_5;
        overlaps[1][index] = expVal*(3.75*e1*e2 - 3.75*e1*e2_2*r2 - 2.25*e1_2*e2*r2 + 1.5*e1_2*e2_2*r4 + 1.875*e1_2 - 1.0*e1_3*e2*r4 + 2.25*e1_3*r2 + 1.875*e2_2 + 0.75*e2_3*r2)/(5.0*e1*e2_4 + 10.0*e1_2*e2_3 + 10.0*e1_3*e2_2 + 5.0*e1_4*e2 + 1.0*e1_5 + 1.0*e2_5);
        overlaps[2][index] = expVal*r*(0.75*e1*(e1 + e2) + 0.25*e1*(1.5*e1 + e1_2*r2 + 1.5*e2) - 0.5*e2*(0.75*e1 + 1.5*e1_2*r2 + 0.75*e2) - 0.75*e2*(-e1*e2*r2 + e1 + e1_2*r2 + e2) + (e1 + e2)*(1.125*e1 - 0.375*e2))/e1e2_5;
        overlaps[3][index] = expVal*(-0.375*e1*e2*r2 + 0.625*e1 + 0.375*e1_2*r2 - e2*r2*(0.375*e1 - 0.125*e2) + 0.625*e2)/e1e2_5;
        overlaps[4][index] = expVal*r*(0.1875*e1 - 0.125*e2)/e1e2_5;
        overlaps[5][index] = 0.03125*expVal/e1e2_5;
    }
    if (hexA1A2 == 0x33) {
        float r2 = r*r;
        float r4 = r2*r2;
        float r6 = r2*r2*r2;
        float e1_2 = e1*e1;
        float e1e2_2 = (e1 + e2)*(e1 + e2);
        float e1e2_3 = e1e2_2*(e1 + e2);
        float e1e2_4 = e1e2_2*e1e2_2;
        float e2_2 = e2*e2;
        float e1_3 = e1_2*e1;
        float e1_4 = e1_2*e1_2;
        float e2_3 = e2_2*e2;
        float e2_4 = e2_2*e2_2;
        float e1_5 = e1_4*e1;
        float e1_6 = e1_5*e1;
        float e2_5 = e2_4*e2;
        float e2_6 = e2_5*e1;
        float e1e2_5 = e1e2_4*(e1 + e2);
        float e1e2_6 = e1e2_3*e1e2_3;
        float e1e2_7 = e1e2_6*(e1 + e2);
        float e1e2_8 = e1e2_4*e1e2_4;
        overlaps[0][index] = expVal*(5.625*e1*e2_2 - 2.25*e1*e2_3*r2 - 1.5*e1*e2_4*r4 + 5.625*e1_2*e2 - 9.0*e1_2*e2_2*r2 + 3.0*e1_2*e2_3*r4 - 2.25*e1_3*e2*r2 + 3.0*e1_3*e2_2*r4 - 1.0*e1_3*e2_3*r6 + 1.875*e1_3 - 1.5*e1_4*e2*r4 + 2.25*e1_4*r2 + 1.875*e2_3 + 2.25*e2_4*r2)/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
        overlaps[1][index] = expVal*r*(-5.625*e1*e2_2 + 6.0*e1*e2_3*r2 + 5.625*e1_2*e2 - 1.5*e1_2*e2_3*r4 - 6.0*e1_3*e2*r2 + 1.5*e1_3*e2_2*r4 + 5.625*e1_3 + 0.75*e1_4*r2 - 5.625*e2_3 - 0.75*e2_4*r2)/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
        overlaps[2][index] = expVal*(5.625*e1*e2 - 4.5*e1*e2_2*r2 - 0.75*e1*e2_3*r4 - 4.5*e1_2*e2*r2 + 2.25*e1_2*e2_2*r4 + 2.8125*e1_2 - 0.75*e1_3*e2*r4 + 2.25*e1_3*r2 + 2.8125*e2_2 + 2.25*e2_3*r2)/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
        overlaps[3][index] = expVal*r*(1.125*e1*e2_2*r2 - 1.125*e1_2*e2*r2 + 1.875*e1_2 + 0.125*e1_3*r2 - 1.875*e2_2 - 0.125*e2_3*r2)/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
        overlaps[4][index] = expVal*(-0.1875*e1*e2*r2 + 0.46875*e1 + 0.1875*e1_2*r2 + e2*r2*(-0.1875*e1 + 0.125*e2) - 0.5*e2*r2*(0.375*e1 - 0.125*e2) + 0.46875*e2)/e1e2_6;
        overlaps[5][index] = 0.09375*expVal*r*(e1 - e2)/e1e2_6;
        overlaps[6][index] = 0.015625*expVal/e1e2_6;
    }
}

vec3 productCenter(
    vec3 gPosition, float gExp, vec3 hPosition, float hExp) {
    return ((gExp*gPosition) + (hExp*hPosition)) / (gExp + hExp);
}

void fillOverlaps(
    inout vec4[8] overlaps,
    vec3 r1, ivec3 a1, float e1, vec3 r2, ivec3 a2, float e2) {
    vec3 r = r1 - r2;
    float expValX = exp(-e1*e2*r.x*r.x/(e1 + e2));
    float expValY = exp(-e1*e2*r.y*r.y/(e1 + e2));
    float expValZ = exp(-e1*e2*r.z*r.z/(e1 + e2));
    int xInd = 0, yInd = 1, zInd = 2;
    fillOverlapCoefficients(overlaps, xInd, a1.x, a2.x, r.x, e1, e2, expValX);
    fillOverlapCoefficients(overlaps, yInd, a1.y, a2.y, r.y, e1, e2, expValY);
    fillOverlapCoefficients(overlaps, zInd, a1.z, a2.z, r.z, e1, e2, expValZ);
}

float repulsion(
    vec2 shellCoord,
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
    float eps = 1e-40;
    float val = 0.0;
    vec3 grad1Val = vec3(0.0), grad2Val = vec3(0.0);
    vec3 grad3Val = vec3(0.0), grad4Val = vec3(0.0);
    int angXLMax = a1.x + a2.x;
    int angXRMax = a3.x + a4.x;
    int angYLMax = a1.y + a2.y;
    int angYRMax = a3.y + a4.y;
    int angZLMax = a1.z + a2.z;
    int angZRMax = a3.z + a4.z;
    int maxAx = a1.x + a2.x + a3.x + a4.x;
    int maxAy = a1.y + a2.y + a3.y + a4.y;
    int maxAz = a1.z + a2.z + a3.z + a4.z;

    vec4 overlapsL[8];
    fillOverlaps(overlapsL, r1, a1, e1, r2, a2, e2);
    vec4 overlapsR[8];
    fillOverlaps(overlapsR, r3, a3, e3, r4, a4, e4);
    float bF[16];
    // vec4 bFVec = texture2D(boysFunctionTex, shellCoord);
    // bF[0] = bFVec[0];
    // bF[1] = bFVec[1];
    // bF[2] = bFVec[2];
    // bF[3] = bFVec[3];
    computeBoysFunctionValues(
        bF, orbExp*dot(rLR, rLR), maxAx + maxAy + maxAz);
    for (int ix = 0; ix <= angXLMax; ix++) {
        float oXL = overlapsL[ix].x;
        float oXL1 = 1.0;
        float oXL2 = 1.0;
        for (int iy = 0; iy <= angYLMax; iy++) {
            float oYL = overlapsL[iy].y;
            float oYL1 = 1.0;
            float oYL2 = 1.0;
            for (int iz = 0; iz <= angZLMax; iz++) {
                float oZL = overlapsL[iz].z;
                float oZL1 = 1.0;
                float oZL2 = 1.0;
                for (int jx = 0; jx <= angXRMax; jx++) {
                    float oXR = overlapsR[jx].x;
                    float oXR3 = 1.0;
                    float oXR4 = 1.0;
                    for (int jy = 0; jy <= angYRMax; jy++) {
                        float oYR = overlapsR[jy].y;
                        float oYR3 = 1.0;
                        float oYR4 = 1.0;
                        for (int jz = 0; jz <= angZRMax; jz++) {
                            float oZR = overlapsR[jz].z;
                            float oZR3 = 1.0;
                            float oZR4 = 1.0;
                            float overlaps = oXL*oXR*oYL*oYR*oZL*oZR;
                            float cc = coulombCoefficient(
                                bF, 
                                ivec3(ix + jx, iy + jy, iz + jz),
                                orbExp, rLR);
                            float modVal = mod(float(jx + jy + jz), 2.0);
                            float factor = (modVal > eps)? -1.0: 1.0;
                            val += factor * overlaps * cc;
                            vec3 cc1 = vec3(1.0);
                            vec3 cc2 = vec3(1.0);
                            vec3 cc3 = vec3(1.0);
                            vec3 cc4 = vec3(1.0);
                            grad1Val += factor*vec3(
                                oXL1*oYL *oZL *oXR*oYR*oZR + overlaps*cc1.x,
                                oXL *oYL1*oZL *oXR*oYR*oZR + overlaps*cc1.y,
                                oXL *oYL *oZL1*oXR*oYR*oZR + overlaps*cc1.z
                            );
                            grad2Val += factor*vec3(
                                oXL2*oYL *oZL *oXR*oYR*oZR + overlaps*cc2.x,
                                oXL *oYL2*oZL *oXR*oYR*oZR + overlaps*cc2.y,
                                oXL *oYL *oZL2*oXR*oYR*oZR + overlaps*cc2.z
                            );
                            grad3Val += factor*vec3(
                                oXL*oYL*oZL*oXR3*oYR *oZR  + overlaps*cc3.x,
                                oXL*oYL*oZL*oXR *oYR3*oZR  + overlaps*cc3.y,
                                oXL*oYL*oZL*oXR *oYR *oZR3 + overlaps*cc3.z
                            );
                            grad4Val += factor*vec3(
                                oXL*oYL*oZL*oXR4*oYR *oZR  + overlaps*cc4.x,
                                oXL*oYL*oZL*oXR *oYR4*oZR  + overlaps*cc4.y,
                                oXL*oYL*oZL*oXR *oYR *oZR4 + overlaps*cc4.z
                            );
                        }
                    }
                }
            }
        }
    }
    // dFdx1 = factor * (ddx1(oXL)*oXR*oYL*oYR*oZL*oZR*cc + overlaps*ddx1(cc))
    // dFdy1 = factor * (oXL*oXR*ddy1(oYL)*oYR*oZL*oZR*cc + overlaps*ddy1(cc))
    // dFdz1 = factor * (oXL*oXR*oYL*oYR*ddz1(oZL)*oZR*cc + overlaps*ddz1(cc))
    // dFdx2 = factor * (ddx2(oXL)*oXR*oYL*oYR*oZL*oZR*cc + overlaps*ddx2(cc))
    // dFdy2 = factor * (oXL*oXR*ddy2(oYL)*oYR*oZL*oZR*cc + overlaps*ddy2(cc))
    // dFdz2 = factor * (oXL*oXR*oYL*oYR*ddz2(oZL)*oZR*cc + overlaps*ddz2(cc))

    // dFdx3 = factor * (oXL*ddx3(oXR)*oYL*oYR*oZL*oZR*cc + overlaps*ddx3(cc))
    // dFdy3 = factor * (oXL*oXR*oYL*ddy3(oYR)*oZL*oZR*cc + overlaps*ddy3(cc))
    // dFdz3 = factor * (oXL*oXR*oYL*oYR*oZL*ddz3(oZR)*cc + overlaps*ddz3(cc))
    // dFdx4 = factor * (oXL*ddx4(oXR)*oYL*oYR*oZL*oZR*cc + overlaps*ddx4(cc))
    // dFdy4 = factor * (oXL*oXR*oYL*ddy4(oYR)*oZL*oZR*cc + overlaps*ddy4(cc))
    // dFdz4 = factor * (oXL*oXR*oYL*oYR*oZL*ddz4(oZR)*cc + overlaps*ddz4(cc))
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
    int shellI = int(texture2D(basisFunctionSpec2Tex, indI).w);
    int shellJ = int(texture2D(basisFunctionSpec2Tex, indJ).w);
    int shellK = int(texture2D(basisFunctionSpec2Tex, indK).w);
    int shellL = int(texture2D(basisFunctionSpec2Tex, indL).w);
    ivec4 primitiveIndices2 = primitiveIndices;
    
    // int shellIndex1D = getSymmetric4Index1D(
    //     shellI, shellJ, shellK, shellL, primitiveIndices2);
    int shellIndex1D = getSymmetric4Index1D(
        shellI, shellJ, shellK, shellL);
    vec2 shellIndex2D = vec2(
        mod(shellIndex1D, shellTexWidth) + 0.5,
        floor(float(shellIndex1D/shellTexWidth)) + 0.5
    ) / float(shellTexWidth);

    if (primitiveIndices2[0] >= countI ||
        primitiveIndices2[1] >= countJ ||
        primitiveIndices2[2] >= countK ||
        primitiveIndices2[3] >= countL) {
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
            (float(primitiveIndices2[0]) + 0.5)/float(maxNumberOfPrimitives),
            indI[0]);
        vec2 primIndJ = vec2(
            (float(primitiveIndices2[1]) + 0.5)/float(maxNumberOfPrimitives),
            indJ[0]);
        vec2 primIndK = vec2(
            (float(primitiveIndices2[2]) + 0.5)/float(maxNumberOfPrimitives),
            indK[0]);
        vec2 primIndL = vec2(
            (float(primitiveIndices2[3]) + 0.5)/float(maxNumberOfPrimitives),
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
            shellIndex2D,
            ampI, rI, angularI, eI,
            ampJ, rJ, angularJ, eJ,
            ampK, rK, angularK, eK,
            ampL, rL, angularL, eL
        );
        fragColor = vec4(sumTerm) + texture2D(previousSumTex, UV);
    }
}