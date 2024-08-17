#version 120

uniform vec4 u_color;
uniform sampler2D u_texture;
uniform float u_smoothing;
uniform float atlas_width;
uniform float i_scale;

void main() {
    float distance = texture2D(u_texture, gl_TexCoord[0].xy).a;
//    gl_FragColor = vec4(u_color.rgb, texture2D(u_texture, gl_TexCoord[0].xy).a);
//    float alpha = smoothstep(0.5 - u_smoothing, 0.5 + u_smoothing, distance);
////    gl_FragColor = vec4(u_color.rgb, u_color.a * alpha);
    float r = smoothstep(0.5 - u_smoothing, 0.5 + u_smoothing, texture2D(u_texture, gl_TexCoord[0].xy).a);
    float g = smoothstep(0.5 - u_smoothing, 0.5 + u_smoothing, texture2D(u_texture, gl_TexCoord[0].xy + vec2(1 / atlas_width / 3 * 1 * i_scale, 0.0)).a);
    float b = smoothstep(0.5 - u_smoothing, 0.5 + u_smoothing, texture2D(u_texture, gl_TexCoord[0].xy + vec2(1 / atlas_width / 3 * 2 * i_scale, 0.0)).a);
    gl_FragColor = vec4(u_color.rgb, u_color.a * (r+g+b)/3);
////    gl_FragColor = vec4(u_color.r * r, u_color.g * g, u_color.b * b, u_color.a * alpha);
}
