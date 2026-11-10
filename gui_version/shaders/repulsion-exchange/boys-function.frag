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
void computeBFValuesRecurseUpwards(inout float bF[6], float x, int nMax) {
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
void computeBFValuesRecurseToZero(inout float bF[6], float x, int nMax) {
    float val = sqrt(PI)*erf(sqrt(x))/(2.0*sqrt(x));
    float bx = 0.5*exp(-x)/x;
    bF[0] = val;
    for (int nIter = 1; nIter <= nMax; nIter++) {
        val = ((float(nIter) - 0.5)/x)*val - bx;
        bF[nIter] = val;
    }
}

void computeBoysFunctionValues(inout float bF[6], float x, int nMax) {
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

vec3 productCenter(
    vec3 gPosition, float gExp, vec3 hPosition, float hExp) {
    return ((gExp*gPosition) + (hExp*hPosition)) / (gExp + hExp);
}

void main() {
    vec4 indices = texture2D(indicesTex, UV);
    vec2 indI = vec2((indices[0] + 0.5)/float(numberOfBasisFunctions), 0.5);
    vec2 indJ = vec2((indices[1] + 0.5)/float(numberOfBasisFunctions), 0.5);
    vec2 indK = vec2((indices[2] + 0.5)/float(numberOfBasisFunctions), 0.5);
    vec2 indL = vec2((indices[3] + 0.5)/float(numberOfBasisFunctions), 0.5);
    int countI = int(texture2D(basisFunctionSpec1Tex, indI).w);
    int countJ = int(texture2D(basisFunctionSpec1Tex, indJ).w);
    int countK = int(texture2D(basisFunctionSpec1Tex, indK).w);
    int countL = int(texture2D(basisFunctionSpec1Tex, indL).w);
    if (primitiveIndices[0] >= countI ||
        primitiveIndices[1] >= countJ ||
        primitiveIndices[2] >= countK ||
        primitiveIndices[3] >= countL) {
        return;
    }
    vec3 rI = texture2D(basisFunctionSpec1Tex, indI).xyz;
    vec3 rJ = texture2D(basisFunctionSpec1Tex, indJ).xyz;
    vec3 rK = texture2D(basisFunctionSpec1Tex, indK).xyz;
    vec3 rL = texture2D(basisFunctionSpec1Tex, indL).xyz;
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
    float orbExpL = eI + eJ;
    float orbExpR = eK + eL;
    float orbExp = orbExpL*orbExpR/(orbExpL + orbExpR);
    vec3 rLR = productCenter(rI, eI, rJ, eJ) - productCenter(rK, eK, rL, eL);
    float bF[6];
    ivec3 angSum = angularI + angularJ + angularK + angularL;
    computeBoysFunctionValues(
        bF, orbExp*dot(rLR, rLR), angSum.x + angSum.y + angSum.z);
    fragColor = vec4(bF[0], bF[1], bF[2], bF[3]);

}