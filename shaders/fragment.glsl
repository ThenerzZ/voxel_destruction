#version 330 core

in vec3 FragPos;
in vec3 Normal;
in vec3 Color;

out vec4 FragColor;

uniform vec3 lightPos;
uniform vec3 viewPos;

void main()
{
    // Ambient lighting
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * Color;
    
    // Diffuse lighting
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * Color;
    
    // Specular lighting
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * vec3(1.0);
    
    // Edge highlighting
    float edgeFactor = 1.0 - max(0.0, dot(norm, viewDir));
    float edgeIntensity = pow(edgeFactor, 3.0) * 0.3;
    
    // Combine lighting
    vec3 result = ambient + diffuse + specular;
    
    // Add edge highlight
    result += vec3(edgeIntensity);
    
    // Output with full opacity
    FragColor = vec4(result, 1.0);
} 