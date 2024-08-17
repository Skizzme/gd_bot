out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D texture0;

void main() {
    FragColor = texture(texture0, texCoord);
//    FragColor = vec4(texCoord.x, texCoord.y, 1, 1);
}
