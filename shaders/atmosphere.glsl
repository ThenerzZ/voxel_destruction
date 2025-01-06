#version 330 core

// Atmosphere parameters
struct Atmosphere {
    vec3 rayleighCoefficients;
    float rayleighHeight;
    vec3 mieCoefficients;
    float mieHeight;
    float mieDirection;
    float atmosphereHeight;
    float planetRadius;
    float cloudCoverage;
    float cloudDensity;
    float humidity;
    float fogStart;
    float fogEnd;
    vec3 fogColor;
    vec3 windDirection;
    float windSpeed;
    float time;
};

uniform Atmosphere atmosphere;
uniform vec3 viewPos;

// Constants
const float PI = 3.14159265359;
const int ATMOSPHERE_SAMPLES = 16;
const int LIGHT_SAMPLES = 8;
const float MAX_SCATTER_DISTANCE = 100.0;

// Noise functions for clouds
float hash(vec3 p) {
    p = fract(p * vec3(443.8975, 397.2973, 491.1871));
    p += dot(p.zxy, p.yxz + 19.19);
    return fract(p.x * p.y * p.z);
}

float noise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    
    return mix(
        mix(
            mix(hash(i), hash(i + vec3(1,0,0)), f.x),
            mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x),
            f.y
        ),
        mix(
            mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
            mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x),
            f.y
        ),
        f.z
    );
}

// Cloud density calculation
float calculateCloudDensity(vec3 position) {
    vec3 windOffset = atmosphere.windDirection * atmosphere.windSpeed * atmosphere.time;
    float baseNoise = noise((position + windOffset) * 0.001);
    float detailNoise = noise((position + windOffset) * 0.01);
    
    float density = mix(baseNoise, detailNoise, 0.5);
    density = smoothstep(1.0 - atmosphere.cloudCoverage, 1.0, density);
    return density * atmosphere.cloudDensity;
}

// Atmospheric scattering calculation
vec3 calculateScattering(vec3 start, vec3 dir, float maxDistance) {
    float stepSize = maxDistance / float(ATMOSPHERE_SAMPLES);
    vec3 rayleighScattering = vec3(0.0);
    vec3 mieScattering = vec3(0.0);
    
    for(int i = 0; i < ATMOSPHERE_SAMPLES; i++) {
        vec3 pos = start + dir * (float(i) * stepSize);
        float height = length(pos) - atmosphere.planetRadius;
        
        if(height < 0.0 || height > atmosphere.atmosphereHeight) continue;
        
        // Calculate densities
        float rayleighDensity = exp(-height / atmosphere.rayleighHeight);
        float mieDensity = exp(-height / atmosphere.mieHeight);
        
        // Accumulate scattering
        rayleighScattering += rayleighDensity * atmosphere.rayleighCoefficients;
        mieScattering += mieDensity * atmosphere.mieCoefficients;
    }
    
    // Phase functions
    float cosTheta = dot(dir, normalize(viewPos));
    float rayleighPhase = 3.0 / (16.0 * PI) * (1.0 + cosTheta * cosTheta);
    float miePhase = 3.0 / (8.0 * PI) * ((1.0 - atmosphere.mieDirection * atmosphere.mieDirection) * 
        (1.0 + cosTheta * cosTheta)) / ((2.0 + atmosphere.mieDirection * atmosphere.mieDirection) * 
        pow(1.0 + atmosphere.mieDirection * atmosphere.mieDirection - 
        2.0 * atmosphere.mieDirection * cosTheta, 1.5));
    
    return rayleighScattering * rayleighPhase + mieScattering * miePhase;
}

// Fog calculation
float calculateFog(float distance) {
    return smoothstep(atmosphere.fogStart, atmosphere.fogEnd, distance);
}

// Main atmosphere function
vec4 calculateAtmosphericEffects(vec3 worldPos, vec3 baseColor) {
    vec3 viewDir = normalize(viewPos - worldPos);
    float distance = length(viewPos - worldPos);
    
    // Calculate atmospheric scattering
    vec3 scattering = calculateScattering(worldPos, viewDir, min(distance, MAX_SCATTER_DISTANCE));
    
    // Calculate cloud shadows
    float cloudShadow = 1.0 - calculateCloudDensity(worldPos) * 0.5;
    
    // Apply fog
    float fogFactor = calculateFog(distance);
    vec3 color = mix(baseColor * cloudShadow, atmosphere.fogColor, fogFactor);
    
    // Add atmospheric scattering
    color += scattering;
    
    // Apply humidity effects (haziness)
    float hazeFactor = atmosphere.humidity * 0.3;
    color = mix(color, atmosphere.fogColor, hazeFactor);
    
    return vec4(color, 1.0);
} 