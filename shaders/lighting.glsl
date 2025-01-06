#version 330 core

// Light structures
struct DirectionalLight {
    vec3 direction;
    vec3 color;
    float intensity;
    float shadows;
};

struct PointLight {
    vec3 position;
    vec3 color;
    float intensity;
    float radius;
    float shadows;
};

struct AmbientLight {
    vec3 color;
    float intensity;
};

// Maximum number of lights
const int MAX_DIRECTIONAL_LIGHTS = 4;
const int MAX_POINT_LIGHTS = 16;

// Light uniforms
uniform AmbientLight ambientLight;
uniform DirectionalLight directionalLights[MAX_DIRECTIONAL_LIGHTS];
uniform PointLight pointLights[MAX_POINT_LIGHTS];
uniform int numDirectionalLights;
uniform int numPointLights;
uniform float timeOfDay;

// Material properties
const float roughness = 0.7;
const float metallic = 0.0;
const float specularStrength = 0.5;

// Constants
const float PI = 3.14159265359;

// Lighting calculation functions
vec3 calculateDirectionalLight(DirectionalLight light, vec3 normal, vec3 viewDir, vec3 baseColor) {
    vec3 lightDir = normalize(-light.direction);
    
    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * light.color;
    
    // Specular
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);
    vec3 specular = specularStrength * spec * light.color;
    
    // Combine
    return (diffuse + specular) * light.intensity * baseColor;
}

vec3 calculatePointLight(PointLight light, vec3 fragPos, vec3 normal, vec3 viewDir, vec3 baseColor) {
    vec3 lightDir = normalize(light.position - fragPos);
    float distance = length(light.position - fragPos);
    
    // Attenuation
    float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
    
    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * light.color;
    
    // Specular
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);
    vec3 specular = specularStrength * spec * light.color;
    
    // Combine and apply attenuation
    return (diffuse + specular) * light.intensity * attenuation * baseColor;
}

// Main lighting function
vec3 calculateLighting(vec3 fragPos, vec3 normal, vec3 baseColor) {
    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 result = ambientLight.color * ambientLight.intensity * baseColor;
    
    // Add directional lights
    for(int i = 0; i < numDirectionalLights; i++) {
        if(i >= MAX_DIRECTIONAL_LIGHTS) break;
        result += calculateDirectionalLight(directionalLights[i], normal, viewDir, baseColor);
    }
    
    // Add point lights
    for(int i = 0; i < numPointLights; i++) {
        if(i >= MAX_POINT_LIGHTS) break;
        result += calculatePointLight(pointLights[i], fragPos, normal, viewDir, baseColor);
    }
    
    // Time of day adjustment
    float dayNightFactor = smoothstep(0.0, 1.0, sin(timeOfDay * PI / 12.0));
    result = mix(result * 0.3, result, dayNightFactor);  // Darker at night
    
    return result;
} 