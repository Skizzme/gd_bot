#version 120

void main() {
    gl_TexCoord[0] = gl_MultiTexCoord0;
    vec4 pos = gl_Vertex;
//    pos.x = pos.x-pos.y/5; // Possible italics?
    gl_Position = gl_ModelViewProjectionMatrix * pos;
}