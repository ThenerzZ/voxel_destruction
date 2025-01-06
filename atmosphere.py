import numpy as np
from dataclasses import dataclass
import math

@dataclass
class AtmosphereParams:
    # Rayleigh scattering coefficients
    rayleigh_coefficients: np.ndarray  # RGB coefficients for air molecules
    rayleigh_height: float            # Scale height for Rayleigh scattering
    
    # Mie scattering coefficients
    mie_coefficients: np.ndarray      # RGB coefficients for larger particles
    mie_height: float                 # Scale height for Mie scattering
    mie_direction: float              # Mie preferred scattering direction
    
    # Atmosphere properties
    atmosphere_height: float          # Height of the atmosphere in world units
    planet_radius: float             # Radius of the planet in world units
    
    # Weather properties
    cloud_coverage: float            # 0.0 to 1.0
    cloud_density: float             # 0.0 to 1.0
    humidity: float                  # 0.0 to 1.0
    
    # Fog properties
    fog_start: float                 # Distance where fog starts
    fog_end: float                   # Distance where fog is fully opaque
    fog_color: np.ndarray           # RGB color of the fog

class AtmosphereSystem:
    def __init__(self):
        # Initialize with Earth-like parameters
        self.params = AtmosphereParams(
            # Rayleigh scattering (causes blue sky)
            rayleigh_coefficients=np.array([5.8e-6, 13.5e-6, 33.1e-6], dtype=np.float32),
            rayleigh_height=8000.0,
            
            # Mie scattering (causes whitish haze near horizon)
            mie_coefficients=np.array([21e-6, 21e-6, 21e-6], dtype=np.float32),
            mie_height=1200.0,
            mie_direction=0.758,
            
            # Atmosphere setup
            atmosphere_height=100.0,  # Scaled down for game world
            planet_radius=1000.0,    # Scaled down for game world
            
            # Weather
            cloud_coverage=0.5,
            cloud_density=0.3,
            humidity=0.5,
            
            # Fog
            fog_start=10.0,
            fog_end=100.0,
            fog_color=np.array([0.5, 0.6, 0.7], dtype=np.float32)
        )
        
        self.time_of_day = 0.0
        self.weather_time = 0.0
        self.wind_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.wind_speed = 1.0
        
    def update(self, delta_time, time_of_day):
        self.time_of_day = time_of_day
        self.weather_time += delta_time
        
        # Update weather parameters
        self._update_weather(delta_time)
        # Update atmospheric parameters based on time of day
        self._update_atmosphere_params()
        
    def _update_weather(self, delta_time):
        # Update cloud movement based on wind
        wind_movement = self.wind_direction * self.wind_speed * delta_time
        
        # Update cloud coverage with some natural variation
        noise = math.sin(self.weather_time * 0.1) * 0.1
        self.params.cloud_coverage = np.clip(self.params.cloud_coverage + noise, 0.0, 1.0)
        
        # Update humidity based on time of day (higher in morning, lower in afternoon)
        time_factor = math.sin(self.time_of_day * math.pi / 12.0)
        self.params.humidity = 0.5 + time_factor * 0.2
        
        # Update fog based on humidity and time of day
        self._update_fog()
        
    def _update_fog(self):
        # More fog in the morning and evening
        time_factor = abs(math.sin(self.time_of_day * math.pi / 12.0))
        humidity_factor = self.params.humidity
        
        # Adjust fog distance based on conditions
        base_fog_start = 10.0
        base_fog_end = 100.0
        
        self.params.fog_start = base_fog_start * (1.0 - humidity_factor)
        self.params.fog_end = base_fog_end * (1.0 - humidity_factor * 0.5)
        
        # Adjust fog color based on time of day
        if 5.0 <= self.time_of_day <= 8.0:  # Dawn
            self.params.fog_color = np.array([0.8, 0.7, 0.6], dtype=np.float32)
        elif 8.0 <= self.time_of_day <= 16.0:  # Day
            self.params.fog_color = np.array([0.5, 0.6, 0.7], dtype=np.float32)
        elif 16.0 <= self.time_of_day <= 19.0:  # Dusk
            self.params.fog_color = np.array([0.7, 0.6, 0.5], dtype=np.float32)
        else:  # Night
            self.params.fog_color = np.array([0.1, 0.1, 0.15], dtype=np.float32)
            
    def _update_atmosphere_params(self):
        # Adjust scattering based on time of day
        time_factor = math.sin(self.time_of_day * math.pi / 12.0)
        
        # Increase Mie scattering during sunrise/sunset
        sunset_factor = 1.0 - abs(time_factor)
        self.params.mie_coefficients *= (1.0 + sunset_factor)
        
        # Adjust Rayleigh scattering for time of day
        if time_factor > 0:  # Daytime
            self.params.rayleigh_coefficients = np.array([5.8e-6, 13.5e-6, 33.1e-6], dtype=np.float32)
        else:  # Nighttime
            self.params.rayleigh_coefficients = np.array([2.9e-6, 6.75e-6, 16.55e-6], dtype=np.float32)
            
    def get_atmosphere_uniforms(self):
        """Returns a dictionary of uniform values for the shader"""
        return {
            'atmosphere.rayleighCoefficients': self.params.rayleigh_coefficients,
            'atmosphere.rayleighHeight': self.params.rayleigh_height,
            'atmosphere.mieCoefficients': self.params.mie_coefficients,
            'atmosphere.mieHeight': self.params.mie_height,
            'atmosphere.mieDirection': self.params.mie_direction,
            'atmosphere.atmosphereHeight': self.params.atmosphere_height,
            'atmosphere.planetRadius': self.params.planet_radius,
            'atmosphere.cloudCoverage': self.params.cloud_coverage,
            'atmosphere.cloudDensity': self.params.cloud_density,
            'atmosphere.humidity': self.params.humidity,
            'atmosphere.fogStart': self.params.fog_start,
            'atmosphere.fogEnd': self.params.fog_end,
            'atmosphere.fogColor': self.params.fog_color,
            'atmosphere.windDirection': self.wind_direction,
            'atmosphere.windSpeed': self.wind_speed,
            'atmosphere.time': self.weather_time
        } 