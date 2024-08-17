#version 330
uniform mat4 u_projection;

in vec2 position;
in vec2 vertexTexCoord;

out vec2 texCoord;

void main() {
    gl_Position = u_projection * vec4(position.xy, 0, 1);
    texCoord = vertexTexCoord;
}
