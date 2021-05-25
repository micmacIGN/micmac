#ifndef SAISIEGLSL_GLSL
#define SAISIEGLSL_GLSL

#define GLSL(version, shader)  "#version " #version "\n" #shader

// VERTEX SHADER
const char * vertexShader =GLSL(120,

uniform highp mat4 matrix;
varying vec2 vTexCoord;

void main(void)
{
    vTexCoord   = gl_MultiTexCoord0.xy;
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;

}


);

// FRAGMENT SHADER
const char * fragmentGamma =GLSL(120,

uniform mediump vec4 color;
uniform sampler2D tex;
uniform float gamma;
varying vec2 vTexCoord;

void main(void)
{
    vec3 colorTex       = texture2D(tex, vTexCoord).rgb;
    gl_FragColor.rgb    = pow(colorTex, vec3(gamma));
    gl_FragColor.a      = 1.0f;
}

);

#endif //SAISIEGLSL_GLSL
