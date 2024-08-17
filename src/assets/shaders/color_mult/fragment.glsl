#version 120

uniform sampler2D texture0;
uniform sampler2D texture1;

void main() {
    vec2 coord = gl_TexCoord[0].xy;
    vec4 tex0 = texture2D(texture0, coord);
    vec4 tex1 = texture2D(texture1, coord);
//    gl_FragColor = vec4(tex1.rgb, tex0.a*tex1.a);
    gl_FragColor = tex0*tex1;
//    gl_FragColor = vec4(0.0, 0.0, 0.0, coord.x);
//    gl_FragColor = vec4(tex0.r*tex1.r, 1, 1, tex0.a*tex1.a);
//    gl_FragColor = vec4(gl_TexCoord[0].xy, 1, texture2D(texture0, gl_TexCoord[0].xy).a);
}
