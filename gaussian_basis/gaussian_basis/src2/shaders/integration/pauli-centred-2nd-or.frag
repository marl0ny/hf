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

uniform float q;
uniform float m;
uniform float c;
uniform float hbar;
uniform float dt;

uniform bool useAbsorbingBoundaries;

uniform sampler2D psi0Tex;
uniform sampler2D psi1Tex;
uniform sampler2D potentialTex;

uniform bool enableVectorPotential;
uniform sampler2D vectorPotentialTex;
uniform sampler2D magneticFieldTex;

uniform vec3 dimensions3D;
uniform ivec2 texelDimensions2D;
uniform ivec3 texelDimensions3D;

#define complex vec2
#define complex2 vec4

const complex2 IMAG_UNIT = complex2(0.0, 1.0, 0.0, 1.0);

#define hermitian2x2 vec4

const hermitian2x2 SIGMA_X = hermitian2x2(0.0, 0.0, complex(1.0, 0.0));
const hermitian2x2 SIGMA_Y = hermitian2x2(0.0, 0.0, complex(0.0, -1.0));
const hermitian2x2 SIGMA_Z = hermitian2x2(1.0, -1.0, complex(0.0));


complex mul(complex a, complex b) {
    return complex(a[0]*b[0] - a[1]*b[1], a[0]*b[1] + a[1]*b[0]);
}

complex2 mul(complex2 a, complex2 b) {
    return complex2(mul(a.xy, b.xy), mul(a.zw, b.zw));
}

complex conj(complex z) {
    return complex(z[0], -z[1]);
}

complex2 matrixMul(hermitian2x2 m, complex2 v) {
    complex m00 = complex(m[0], 0.0);
    complex m11 = complex(m[1], 0.0);
    complex m01 = complex(m[2], m[3]);
    complex m10 = conj(m01);
    complex v0 = v.rg;
    complex v1 = v.ba;
    return complex2(mul(m00, v0) + mul(m01, v1),
                    mul(m10, v0) + mul(m11, v1));
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

vec2 to2DTextureCoordinates(vec3 uvw) {
    int width2D = texelDimensions2D[0];
    int height2D = texelDimensions2D[1];
    int width3D = texelDimensions3D[0];
    int height3D = texelDimensions3D[1];
    int length3D = texelDimensions3D[2];
    float wStack = float(width2D)/float(width3D);
    // float hStack = float(height2D)/float(height3D);
    float xIndex = float(width3D)*mod(uvw[0], 1.0);
    float yIndex = float(height3D)*mod(uvw[1], 1.0);
    float zIndex = mod(floor(float(length3D)*uvw[2]), float(length3D));
    float uIndex = mod(zIndex, wStack)*float(width3D) + xIndex; 
    float vIndex = floor(zIndex / wStack)*float(height3D) + yIndex; 
    return vec2(uIndex/float(width2D), vIndex/float(height2D));
}

vec4 sampleAt(sampler2D tex, vec3 uvw) {
    return texture2D(tex, to2DTextureCoordinates(uvw));
}

complex2 laplacian1D2ndOrder3pt(
    complex2 f, complex2 c, complex2 b, float d
) {
    return (f - 2.0*c + b)/(d*d);
}

complex2 laplacian2ndOrder7pt(sampler2D tex) {
    float du = 1.0/float(texelDimensions3D[0]);
    float dv = 1.0/float(texelDimensions3D[1]);
    float dw = 1.0/float(texelDimensions3D[2]);
    float dx = dimensions3D[0]/float(texelDimensions3D[0]);
    float dy = dimensions3D[1]/float(texelDimensions3D[1]);
    float dz = dimensions3D[2]/float(texelDimensions3D[2]);
    complex2 up, down, left, right, front, back, center;
    vec3 uvw = to3DTextureCoordinates(UV);
    front = sampleAt(tex, uvw + vec3(0.0, 0.0, dw));
    back = sampleAt(tex, uvw + vec3(0.0, 0.0, -dw));
    up = sampleAt(tex, uvw + vec3(0.0, dv, 0.0));
    down = sampleAt(tex, uvw + vec3(0.0, -dv, 0.0));
    left = sampleAt(tex, uvw + vec3(-du, 0.0, 0.0));
    right = sampleAt(tex, uvw + vec3(du, 0.0, 0.0));
    center = sampleAt(tex, uvw);
    return laplacian1D2ndOrder3pt(right, center, left, dx)
        + laplacian1D2ndOrder3pt(up, center, down, dy)
        + laplacian1D2ndOrder3pt(front, center, back, dz);
}

complex2 laplacian1D4thOrder5pt(
    complex2 f2, complex2 f1, complex2 c, complex2 b1, complex2 b2, float d
) {
    return (-f2 + 16.0*f1 - 30.0*c + 16.0*b1 - b2)/(12.0*d*d);
}

complex2 laplacian4thOrder13pt(sampler2D tex) {
    float du = 1.0/float(texelDimensions3D[0]);
    float dv = 1.0/float(texelDimensions3D[1]);
    float dw = 1.0/float(texelDimensions3D[2]);
    float dx = dimensions3D[0]/float(texelDimensions3D[0]);
    float dy = dimensions3D[1]/float(texelDimensions3D[1]);
    float dz = dimensions3D[2]/float(texelDimensions3D[2]);
    complex2 up2, up1, center, down1, down2;
    complex2 left2, left1, right1, right2;
    complex2 back2, back1, front1, front2;
    vec3 uvw = to3DTextureCoordinates(UV);
    front2 = sampleAt(tex, uvw + vec3(0.0, 0.0, 2.0*dw));
    front1 = sampleAt(tex, uvw + vec3(0.0, 0.0, dw));
    back1 = sampleAt(tex, uvw + vec3(0.0, 0.0, -dw));
    back2 = sampleAt(tex, uvw + vec3(0.0, 0.0, -2.0*dw));
    up2 = sampleAt(tex, uvw + vec3(0.0, 2.0*dv, 0.0));
    up1 = sampleAt(tex, uvw + vec3(0.0, dv, 0.0));
    down1 = sampleAt(tex, uvw + vec3(0.0, -dv, 0.0));
    down2 = sampleAt(tex, uvw + vec3(0.0, -2.0*dv, 0.0));
    left2 = sampleAt(tex, uvw + vec3(-2.0*du, 0.0, 0.0));
    left1 = sampleAt(tex, uvw + vec3(-du, 0.0, 0.0));
    right1 = sampleAt(tex, uvw + vec3(du, 0.0, 0.0));
    right2 = sampleAt(tex, uvw + vec3(2.0*du, 0.0, 0.0));
    center = sampleAt(tex, uvw);
    return laplacian1D4thOrder5pt(right2, right1, center, left1, left2, dx)
        + laplacian1D4thOrder5pt(up2, up1, center, down1, down2, dy)
        + laplacian1D4thOrder5pt(front2, front1, center, back1, back2, dz);
}

void gradientAndLaplacianZ4thOrder(
    inout vec4 gradientVal, inout vec4 laplacianVal,
    vec4 center, sampler2D tex) {
    float dw = 1.0/float(texelDimensions3D[2]);
    float dz = dimensions3D[2]/float(texelDimensions3D[2]);
    complex2 backward2, backward1, forward1, forward2;
    vec3 uvw = to3DTextureCoordinates(UV);
    forward2 = sampleAt(tex, uvw + vec3(0.0, 0.0, 2.0*dw));
    forward1 = sampleAt(tex, uvw + vec3(0.0, 0.0, dw));
    backward1 = sampleAt(tex, uvw + vec3(0.0, 0.0, -dw));
    backward2 = sampleAt(tex, uvw + vec3(0.0, 0.0, -2.0*dw));
    laplacianVal 
        = (-forward2/12.0 + 4.0*forward1/3.0 - 5.0*center/2.0
	       + 4.0*backward1/3.0 - backward2/12.0)/(dz*dz);
    gradientVal = (
        -forward2/12.0 + 2.0*forward1/3.0 
        - 2.0*backward1/3.0 + backward2/12.0)/dz;
}

void gradientAndLaplacianY4thOrder(
    inout vec4 gradientVal, inout vec4 laplacianVal,
    vec4 center, sampler2D tex) {
    float dv = 1.0/float(texelDimensions3D[1]);
    float dy = dimensions3D[1]/float(texelDimensions3D[1]);
    complex2 down2, down1, up1, up2;
    vec3 uvw = to3DTextureCoordinates(UV);
    up2 = sampleAt(tex, uvw + vec3(0.0, 2.0*dv, 0.0));
    up1 = sampleAt(tex, uvw + vec3(0.0, dv, 0.0));
    down1 = sampleAt(tex, uvw + vec3(0.0, -dv, 0.0));
    down2 = sampleAt(tex, uvw + vec3(0.0, -2.0*dv, 0.0));
    laplacianVal 
        = (-up2/12.0 + 4.0*up1/3.0 - 5.0*center/2.0
	        + 4.0*down1/3.0 - down2/12.0)/(dy*dy);
    gradientVal 
        = (-up2/12.0 + 2.0*up1/3.0 - 2.0*down1/3.0 + down2/12.0)/dy;
}

void gradientAndLaplacianX4thOrder(
    inout vec4 gradientVal, inout vec4 laplacianVal,
    vec4 center, sampler2D tex) {
    float du = 1.0/float(texelDimensions3D[0]);
    float dx = dimensions3D[0]/float(texelDimensions3D[0]);
    complex2 left2, left1, right1, right2;
    vec3 uvw = to3DTextureCoordinates(UV);
    right2 = sampleAt(tex, uvw + vec3(2.0*du, 0.0, 0.0));
    right1 = sampleAt(tex, uvw + vec3(du, 0.0, 0.0));
    left1 = sampleAt(tex, uvw + vec3(-du, 0.0, 0.0));
    left2 = sampleAt(tex, uvw + vec3(-2.0*du, 0.0, 0.0));
    laplacianVal 
        = (-right2/12.0 + 4.0*right1/3.0 - 5.0*center/2.0 
	        + 4.0*left1/3.0 - left2/12.0)/(dx*dx);
    gradientVal 
        = (-right2/12.0 + 2.0*right1/3.0 
            - 2.0*left1/3.0 + left2/12.0)/dx;
}

complex2 hamiltonian(sampler2D psiTex, sampler2D potentialTex) {
    complex2 psi = texture2D(psiTex, UV);
    float potential = texture2D(potentialTex, UV)[0];
    complex2 laplacianPsi = laplacian4thOrder13pt(psiTex);
    return (-hbar*hbar)/(2.0*m)*laplacianPsi + psi*potential;
    /* if (!enableVectorPotential) {
        complex2 laplacianPsi = laplacian4thOrder13pt(psiTex);
        return (-hbar*hbar)/(2.0*m)*laplacianPsi + psi*potential;
    }
    vec3 vectorPotential = texture2D(vectorPotentialTex, UV).xyz;
    float ax = vectorPotential.x, ay = vectorPotential.z;
    float az = vectorPotential.z;
    vec3 magneticField = texture(magneticFieldTex, UV).xyz;
    float bx = magneticField.x, by = magneticField.y;
    float bz = magneticField.z;
    complex2 gradXPsi, gradYPsi, gradZPsi;
    complex2 d2Psidx2, d2Psidy2, d2Psidz2;
    gradientAndLaplacianX4thOrder(gradXPsi, d2Psidx2, psi, psiTex);
    gradientAndLaplacianY4thOrder(gradYPsi, d2Psidy2, psi, psiTex);
    gradientAndLaplacianZ4thOrder(gradZPsi, d2Psidz2, psi, psiTex);
    complex2 laplacianPsi = d2Psidx2 + d2Psidy2 + d2Psidz2;
    return (-hbar*hbar)/(2.0*m)*laplacianPsi
        - q/(m*c)*mul(-IMAG_UNIT*hbar, ax*gradXPsi)
        - q/(m*c)*mul(-IMAG_UNIT*hbar, ay*gradYPsi)
        - q/(m*c)*mul(-IMAG_UNIT*hbar, az*gradZPsi)
        - (q*hbar)/(2.0*m*c)*bx*matrixMul(SIGMA_X, psi)
        - (q*hbar)/(2.0*m*c)*by*matrixMul(SIGMA_Y, psi)
        - (q*hbar)/(2.0*m*c)*bz*matrixMul(SIGMA_Z, psi)
        + psi*potential;*/
}

complex2 getAbsorbingPotential() {
    vec3 coord = to3DTextureCoordinates(UV);
    float x = coord[0], y = coord[1], z = coord[2];
    float dampPot = 0.0;
    float s = 0.02;
    float a = 1.0;
    dampPot += a*exp(-0.5*x*x/(s*s));
    dampPot += a*exp(-0.5*(x-1.0)*(x-1.0)/(s*s));
    dampPot += a*exp(-0.5*y*y/(s*s));
    dampPot += a*exp(-0.5*(y-1.0)*(y-1.0)/(s*s));
    dampPot += a*exp(-0.5*z*z/(s*s));
    dampPot += a*exp(-0.5*(z-1.0)*(z-1.0)/(s*s));
    return complex2(complex(0.0, -dampPot), complex(0.0, -dampPot));
}

void main() {
    complex2 psi0 = texture2D(psi0Tex, UV);
    complex2 iDt = complex2(0.0, dt, 0.0, dt);
    // complex2 imV = complex2(
    //     complex(0.0, texture2D(potentialTex, UV)[1]), 
    //     complex(0.0, texture2D(potentialTex, UV)[1]));
    complex2 psi2 = psi0 
        - mul(iDt/hbar, hamiltonian(psi1Tex, potentialTex) 
        + mul(psi0, getAbsorbingPotential())
        );
    fragColor = psi2;
}
