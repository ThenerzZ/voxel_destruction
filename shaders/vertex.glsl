#version 330 core

// Vertex attributes
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texCoord;
layout (location = 3) in vec3 instancePosition;
layout (location = 4) in vec3 instanceColor;

// Uniforms
uniform mat4 view;
uniform mat4 projection;

// Outputs to fragment shader
out vec3 FragPos;
out vec3 Normal;
out vec3 Color;

void main() {
    // Calculate world position
    vec4 worldPos = vec4(position + instancePosition, 1.0);
    FragPos = worldPos.xyz;
    
    // Transform normal to world space (assuming no non-uniform scaling)
    Normal = normal;
    
    // Pass through color
    Color = instanceColor;
    
    // Calculate final position
    gl_Position = projection * view * worldPos;
} 