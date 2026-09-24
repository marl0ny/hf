/* Generate a new wavepacket. */
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
#define complex2 vec4

#define PI 3.141592653589793

// wave number of the wave packet (w.r.t. simulation domains)
uniform vec3 waveNumber;
// Position offset of the wave packet in texture coordinates
uniform vec3 offsetTexCoord;
// Amplitude of the wave packet
uniform float amplitude;
// Standard deviation of the wave packet, in texture coordinates
uniform vec3 sigmaTexCoord;
uniform complex2 spinor;

uniform ivec3 texelDimensions3D;
uniform ivec2 texelDimensions2D;
uniform vec3 dimensions3D;

#define TWO_PI_POW_3_OVER_4 3.9685778240728022

complex conj(complex z) {
    return complex(z.x, -z.y);
}

complex2 conj(complex2 z) {
    return complex2(conj(z.rg), conj(z.ba));
}

complex mul(complex a, complex b) {
    return complex(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

/* Multiply a complex scalar c1 with a two-component complex vector c2.*/
complex2 c1C2(complex c1, complex2 c2) {
    complex a = complex(c2[0], c2[1]);
    complex b = complex(c2[2], c2[3]);
    return complex2(mul(c1, a), mul(c1, b));
}

complex2 innerProd(complex2 a, complex2 b) {
    return complex2(mul(conj(a.xy), b.xy), mul(conj(a.zw), b.zw));
}

complex frac(complex z1, complex z2) {
    complex invZ2 = conj(z2)/(z2.x*z2.x + z2.y*z2.y);
    return mul(z1, invZ2);
}

vec3 to3DTextureCoordinates(vec2 uv) {
    int width3D = texelDimensions3D[0];
    int height3D = texelDimensions3D[1];
    int length3D = texelDimensions3D[2];
    int width2D = texelDimensions2D[0];
    int height2D = texelDimensions2D[1];
    float wStack = float(width2D)/float(width3D);
    float hStack = float(height2D)/float(height3D);
    float u = mod(uv[0]*wStack, 1.0);
    float v = mod(uv[1]*hStack, 1.0);
    float w = (floor(uv[1]*hStack)*wStack
               + floor(uv[0]*wStack) + 0.5)/float(length3D);
    return vec3(u, v, w);
}

float gaussian(vec3 r) {
    float sx = sigmaTexCoord.x;
    float sy = sigmaTexCoord.y;
    float sz = sigmaTexCoord.z;
    return exp(-0.25*dot(r/sigmaTexCoord, r/sigmaTexCoord))
        / (sqrt(sx*sy*sz)*TWO_PI_POW_3_OVER_4);
}

complex scalarWavepacketPeriodic(vec3 r) {
    complex phase = complex(cos(2.0*PI*dot(waveNumber, r)),
                            sin(2.0*PI*dot(waveNumber, r)));
    float g = 0.0;
    for (float i = -1.0; i < 2.0; i += 1.0)
        for (float j = -1.0; j < 2.0; j += 1.0)
            for (float k = -1.0; k < 2.0; k += 1.0)
                g += gaussian(r + vec3(i, j, k));
    return amplitude*g*phase;
}

void main() {
    vec3 r = to3DTextureCoordinates(UV) - offsetTexCoord;
    fragColor = vec4(c1C2(scalarWavepacketPeriodic(r), spinor));
}
