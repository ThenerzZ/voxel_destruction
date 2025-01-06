import numpy as np
from dataclasses import dataclass
import math

@dataclass
class DirectionalLight:
    direction: np.ndarray  # Normalized direction vector
    color: np.ndarray     # RGB color
    intensity: float      # Light intensity
    shadows: bool         # Whether this light casts shadows
    
@dataclass
class PointLight:
    position: np.ndarray  # Position in world space
    color: np.ndarray     # RGB color
    intensity: float      # Light intensity
    radius: float        # Light radius (for attenuation)
    shadows: bool        # Whether this light casts shadows

@dataclass
class AmbientLight:
    color: np.ndarray     # RGB color
    intensity: float      # Light intensity

class LightingSystem:
    def __init__(self):
        # Initialize default lights
        self.directional_lights = []
        self.point_lights = []
        self.ambient_light = AmbientLight(
            color=np.array([0.6, 0.7, 1.0], dtype=np.float32),  # Sky color
            intensity=0.3
        )
        
        # Add default sun
        self.add_directional_light(
            direction=np.array([0.5, -1.0, 0.3], dtype=np.float32),
            color=np.array([1.0, 0.95, 0.8], dtype=np.float32),  # Warm sunlight
            intensity=1.0,
            shadows=True
        )
        
        # Time of day system
        self.time_of_day = 0.0  # 0.0 to 24.0
        self.day_cycle_enabled = True
        
    def add_directional_light(self, direction, color, intensity=1.0, shadows=True):
        direction = direction / np.linalg.norm(direction)
        light = DirectionalLight(direction, color, intensity, shadows)
        self.directional_lights.append(light)
        return light
        
    def add_point_light(self, position, color, intensity=1.0, radius=10.0, shadows=False):
        light = PointLight(position, color, intensity, radius, shadows)
        self.point_lights.append(light)
        return light
        
    def update(self, delta_time):
        if self.day_cycle_enabled:
            # Update time of day (24-hour cycle)
            self.time_of_day = (self.time_of_day + delta_time * 0.1) % 24.0
            self._update_sun_position()
            
    def _update_sun_position(self):
        # Calculate sun position based on time of day
        sun_angle = (self.time_of_day / 24.0) * 2.0 * math.pi
        sun_height = math.sin(sun_angle)
        sun_x = math.cos(sun_angle)
        
        # Update main directional light (sun)
        if self.directional_lights:
            sun = self.directional_lights[0]
            sun.direction = np.array([sun_x, sun_height, 0.3], dtype=np.float32)
            sun.direction /= np.linalg.norm(sun.direction)
            
            # Adjust sun color and intensity based on time of day
            if 6.0 <= self.time_of_day <= 18.0:
                # Daytime
                sun.intensity = 1.0
                sun.color = np.array([1.0, 0.95, 0.8], dtype=np.float32)
            elif self.time_of_day < 6.0:
                # Dawn
                t = self.time_of_day / 6.0
                sun.intensity = t * 0.8
                sun.color = np.array([1.0, 0.7, 0.4], dtype=np.float32)
            else:
                # Dusk
                t = (24.0 - self.time_of_day) / 6.0
                sun.intensity = t * 0.8
                sun.color = np.array([1.0, 0.6, 0.3], dtype=np.float32)
                
            # Adjust ambient light based on time of day
            if 6.0 <= self.time_of_day <= 18.0:
                # Daytime sky
                self.ambient_light.color = np.array([0.6, 0.7, 1.0], dtype=np.float32)
                self.ambient_light.intensity = 0.3
            else:
                # Night sky
                self.ambient_light.color = np.array([0.1, 0.1, 0.2], dtype=np.float32)
                self.ambient_light.intensity = 0.1
                
    def get_lighting_uniforms(self):
        """Returns a dictionary of uniform values for the shader"""
        uniforms = {
            'ambientLight.color': self.ambient_light.color,
            'ambientLight.intensity': self.ambient_light.intensity,
            'numDirectionalLights': len(self.directional_lights),
            'numPointLights': len(self.point_lights),
            'timeOfDay': self.time_of_day
        }
        
        # Add directional lights
        for i, light in enumerate(self.directional_lights):
            prefix = f'directionalLights[{i}].'
            uniforms[prefix + 'direction'] = light.direction
            uniforms[prefix + 'color'] = light.color
            uniforms[prefix + 'intensity'] = light.intensity
            uniforms[prefix + 'shadows'] = 1.0 if light.shadows else 0.0
            
        # Add point lights
        for i, light in enumerate(self.point_lights):
            prefix = f'pointLights[{i}].'
            uniforms[prefix + 'position'] = light.position
            uniforms[prefix + 'color'] = light.color
            uniforms[prefix + 'intensity'] = light.intensity
            uniforms[prefix + 'radius'] = light.radius
            uniforms[prefix + 'shadows'] = 1.0 if light.shadows else 0.0
            
        return uniforms 