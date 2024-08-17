#version 120

uniform vec4 u_color;
uniform vec2 u_size;
uniform float u_radius;
uniform float u_time;

const float PI = 3.1415926535897932384626433832795;

vec2 quadrant_vectors(float rotation) {
    if (rotation >= 3*PI/2) {
        return vec2(1, -1);
    } else if (rotation >= PI) {
        return vec2(-1, -1);
    } else if (rotation >= PI/2) {
        return vec2(-1, 1);
    } else if (rotation >= 0) {
        return vec2(1, 1);
    }
}

void main() {
    vec2 halfSize = u_size * .5;
    //smoothstep(0.98, 1.0,
    float a = 1-(length(vec2(0.5, 0.5) - gl_TexCoord[0].xy)*2.0); // Outer circlec
//    float a = 1; // Outer circlec
    float normal_x = gl_TexCoord[0].x*2-1;
    float normal_y = gl_TexCoord[0].y*-2+1;

    float hyp = length(gl_TexCoord[0].xy-vec2(0.5, 0.5)); // Inner circle
    float adj = normal_x;
    float opp = normal_y;
//    float rad = mod(u_time, 2*PI);
    float rad = 0;
    float rad2 = mod(u_time+0.3, 2*PI);
    vec2 dir = quadrant_vectors(rad2);
//    float rad = (mod(u_time, 2*PI))*2;
    // normal_x*dir.x > 0 && normal_y*dir.y > 0 &&
    // opp/adj > tan(rad)
    if (opp/adj < tan(rad2)*dir.x) {
        a = 0;
    }
    gl_FragColor = vec4(1, 1, 1, smoothstep(0.0, 0.02, a));
//    gl_FragColor = vec4(adj/hyp, opp/hyp, 1, 1);
}
