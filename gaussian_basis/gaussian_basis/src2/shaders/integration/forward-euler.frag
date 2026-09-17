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

uniform sampler2D qTex;
uniform sampler2D qDotTex;
uniform float dt;
uniform vec4 weights;

void main() {
    vec4 q = texture2D(qTex, UV);
    vec4 qDot = texture2D(qDotTex, UV); 
    fragColor = q + dt*weights*qDot;
}