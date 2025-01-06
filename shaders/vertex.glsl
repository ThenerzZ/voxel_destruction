#version 330 core

// Vertex attributes
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
layout (location = 3) in vec3 aInstancePos;
layout (location = 4) in vec3 aInstanceColor;

// Uniforms
uniform mat4 projection;
uniform mat4 view;

// Outputs to fragment shader
out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;
out vec3 Color;

void main() {
    // Calculate world position
    vec4 worldPos = vec4(aPos + aInstancePos, 1.0);
    FragPos = worldPos.xyz;
    
    // Transform normal to world space (assuming no non-uniform scaling)
    Normal = aNormal;
    
    // Pass through texture coordinates and color
    TexCoord = aTexCoord;
    Color = aInstanceColor;
    
    // Calculate final position
    gl_Position = projection * view * worldPos;
} 