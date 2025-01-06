#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texCoord;
layout (location = 3) in vec3 instancePosition;
layout (location = 4) in vec3 instanceColor;

uniform mat4 view;
uniform mat4 projection;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;
out vec3 Color;

void main()
{
    mat4 model = mat4(1.0);
    model[3] = vec4(instancePosition, 1.0);
    
    vec4 worldPos = model * vec4(position, 1.0);
    FragPos = worldPos.xyz;
    Normal = mat3(transpose(inverse(model))) * normal;
    TexCoord = texCoord;
    Color = instanceColor;
    
    gl_Position = projection * view * worldPos;
} 