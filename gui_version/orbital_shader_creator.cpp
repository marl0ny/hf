#include <iostream>
#include <fstream>

#include "array_helpers.hpp"
#include "basis_function_array.hpp"


static const std::string START
= R"(#if (__VERSION__ >= 330) || (defined(GL_ES) && __VERSION__ >= 300)
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

#define PI 3.141592653589793

uniform float scale;
uniform bool showTotalDensity;
uniform bool isClosedShell;
uniform int orbitalCount;
uniform int orbitalIndex;

uniform ivec3 texelDimensions3D;
uniform ivec2 texelDimensions2D;

uniform vec3 dr; // Spartial step sizes
uniform vec3 center;
uniform vec3 dimensions3D; // Dimensions of simulation

#define fp_type float

vec3 argumentToColor(float argVal) {
    float maxCol = 1.0;
    float minCol = 50.0/255.0;
    float colRange = maxCol - minCol;
    if (argVal <= PI/3.0 && argVal >= 0.0) {
        return vec3(maxCol,
                    minCol + colRange*argVal/(PI/3.0), minCol);
    } else if (argVal > PI/3.0 && argVal <= 2.0*PI/3.0){
        return vec3(maxCol - colRange*(argVal - PI/3.0)/(PI/3.0),
                    maxCol, minCol);
    } else if (argVal > 2.0*PI/3.0 && argVal <= PI){
        return vec3(minCol, maxCol,
                    minCol + colRange*(argVal - 2.0*PI/3.0)/(PI/3.0));
    } else if (argVal < 0.0 && argVal > -PI/3.0){
        return vec3(maxCol, minCol,
                    minCol - colRange*argVal/(PI/3.0));
    } else if (argVal <= -PI/3.0 && argVal > -2.0*PI/3.0){
        return vec3(maxCol + (colRange*(argVal + PI/3.0)/(PI/3.0)),
                    minCol, maxCol);
    } else if (argVal <= -2.0*PI/3.0 && argVal >= -PI){
        return vec3(minCol,
                    minCol - (colRange*(argVal + 2.0*PI/3.0)/(PI/3.0)), 
                    maxCol);
    }
    else {
        return vec3(minCol, maxCol, maxCol);
    }
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

float square(vec3 r) {
    return dot(r, r);
}

float pow2(float a) {
    return a*a;
}

float pow3(float a) {
    return a*a*a;
}

float pow4(float a) {
    return a*a*a*a;
}    

)";

static const std::string MAIN_FUNC
= R"(void main() {
    vec3 uvw = to3DTextureCoordinates(UV);
    vec3 r = vec3(
        uvw.x*dimensions3D.x,
        uvw.y*dimensions3D.y,
        uvw.z*dimensions3D.z
    ) - center;
    float val = getOrbitalValue(r, orbitalIndex);
    if (showTotalDensity) {
        vec4 density = vec4(0.0);
        for (int i = 0; i < min(orbitalCount, orbitalIndex + 1); i++) {
            float argVal = 2.0*PI*float(i)/float(orbitalCount);
            vec3 col = argumentToColor(argVal);
            float orbitalAmplitude = getOrbitalValue(r, i);
            density += vec4(col, 1.0)
                *orbitalAmplitude*orbitalAmplitude;
        }
        fragColor = vec4(density);
    } else {
        float argVal = PI*float(orbitalIndex)/float(orbitalCount);
        vec3 col = argumentToColor((sign(val) < 0.0)?(-PI + argVal): argVal);
        fragColor = float(orbitalCount)*val*val*vec4(col, 1.0);
        // fragColor = vec4(val, 0.0, -val, abs(val));
        if (orbitalIndex >= orbitalCount)
            fragColor = vec4(0.0, 0.0, 0.0, 0.0);
    }
    fragColor *= scale;
    if (!isClosedShell)
        fragColor *= 0.5;
}
)";

static std::string double2str(double value) {
    if (value == 0.0)
        return "0.0";
    char buff[64] = {'\0'};
    snprintf(buff, 64, "%10e", value);
    return std::string(&buff[0]);
}

static std::string get_angular_part(
    int angular_val, std::string var, std::string offset_str) {
    if (angular_val == 0)
        return "";
    else if (angular_val == 1)
        return "*(r." + var + " - (" + offset_str + "))";
    else if (angular_val == 2)
        return "*pow2(r." + var + " - (" + offset_str + "))";
    else if (angular_val == 3)
        return "*pow3(r." + var + " - (" + offset_str + "))";
    else if (angular_val == 4)
        return "*pow4(r." + var + " - (" + offset_str + "))";
    else
        return "*pow(r." + var + " - (" + offset_str + "), " 
        + std::to_string(angular_val) + ")";
}

std::string express_orbitals_as_function(
    const array_helpers::Array2D &orbitals,
    const BasisFunctionArray &arr) {
    std::string st = "fp_type getOrbitalValue(vec3 r, int index) {\n    ";
    for (int n = 0; n < orbitals.row_count(); n++) {
        st += "if (index == " + std::to_string(n) + ") ";
        st += "return \n    ";
        // st += "" + std::to_string(n) + "  \n";
        for (int i = 0; i < orbitals.column_count(); i++) {
            if (abs(orbitals(n, i)) > 0.0) {
                double coeff = orbitals(n, i);
                for (int p = 0; p < arr.primitive_count_at(i); p++) {
                    Gaussian3D g = arr.get_primitive(i, p);
                    double amplitude = g.amplitude();
                    if (abs(coeff*amplitude) > 9.5e-7) {
                        double orb_exp = g.orbital_exponent();
                        spatial::Vector angular = g.angular();
                        spatial::Vector position = g.position();
                        std::string str_a = double2str(coeff*amplitude);
                        std::string str_e = double2str(orb_exp);
                        std::string str_x = double2str(position.x);
                        std::string str_y = double2str(position.y);
                        std::string str_z = double2str(position.z);
                        std::string angular_expr = "";
                        angular_expr += get_angular_part(
                            angular.x, "x", str_x);
                        angular_expr += get_angular_part(
                            angular.y, "y", str_y);
                        angular_expr += get_angular_part(
                            angular.z, "z", str_z);
                        std::string primitive_expr = " + (" + str_a + ")"
                            + angular_expr
                            + "*exp(-" + str_e + "*square(r - "
                            + "vec3(" + str_x + ", " + str_y + ", " + str_z + ")"
                            + "))";
                        st += "    " + primitive_expr + "\n    ";
                    }
                }
            }
        }
        st += ";\n    ";
    }
    st += "\n}\n";
    return st;
}

std::string get_shader_text(
    const array_helpers::Array2D &orbitals,
    const BasisFunctionArray &arr
) {
    std::string function = express_orbitals_as_function(orbitals, arr);
    std::string contents = START + function + MAIN_FUNC;
    return contents;    
}

void write_orbital_to_file(
    const array_helpers::Array2D &orbitals,
    const BasisFunctionArray &arr) {
    std::string function = express_orbitals_as_function(orbitals, arr);
    std::string contents = START + function + MAIN_FUNC;
    std::string filename = "orbitals.frag";
    std::fstream s(filename, s.out);
    s << contents;
}
