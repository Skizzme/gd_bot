#version 120

uniform sampler2D u_texture;

void main() {
    gl_FragColor = texture2D(u_texture, gl_TexCoord[0].xy);
//    gl_FragColor = vec4(1,1,1,1);
}
