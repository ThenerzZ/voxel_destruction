import numpy as np
import noise
from physics import MaterialType, VoxelData
from enum import Enum
import random
import time
import traceback

class BiomeType(Enum):
    PLAINS = 0
    FOREST = 1
    MOUNTAINS = 2
    DESERT = 3
    TUNDRA = 4

class BiomeProperties:
    def __init__(self, base_height, height_variation, temperature, humidity, 
                 surface_material, underground_material, vegetation_density):
        self.base_height = base_height
        self.height_variation = height_variation
        self.temperature = temperature
        self.humidity = humidity
        self.surface_material = surface_material
        self.underground_material = underground_material
        self.vegetation_density = vegetation_density

BIOME_PROPERTIES = {
    BiomeType.PLAINS: BiomeProperties(
        base_height=0.5,
        height_variation=0.1,
        temperature=0.6,
        humidity=0.5,
        surface_material=MaterialType.WOOD,  # Grass/dirt would be better with more materials
        underground_material=MaterialType.CONCRETE,
        vegetation_density=0.1
    ),
    BiomeType.FOREST: BiomeProperties(
        base_height=0.55,
        height_variation=0.2,
        temperature=0.7,
        humidity=0.7,
        surface_material=MaterialType.WOOD,
        underground_material=MaterialType.CONCRETE,
        vegetation_density=0.8
    ),
    BiomeType.MOUNTAINS: BiomeProperties(
        base_height=0.8,
        height_variation=0.5,
        temperature=0.3,
        humidity=0.4,
        surface_material=MaterialType.CONCRETE,
        underground_material=MaterialType.METAL,
        vegetation_density=0.2
    ),
    BiomeType.DESERT: BiomeProperties(
        base_height=0.4,
        height_variation=0.15,
        temperature=0.9,
        humidity=0.1,
        surface_material=MaterialType.CONCRETE,
        underground_material=MaterialType.CONCRETE,
        vegetation_density=0.05
    ),
    BiomeType.TUNDRA: BiomeProperties(
        base_height=0.45,
        height_variation=0.1,
        temperature=0.1,
        humidity=0.3,
        surface_material=MaterialType.GLASS,  # Ice
        underground_material=MaterialType.CONCRETE,
        vegetation_density=0.05
    )
}

class WorldGenerator:
    def __init__(self, seed=None):
        self.seed = seed if seed is not None else int(time.time())
        print(f"Initializing WorldGenerator with seed: {self.seed}")
        
    def generate_chunk(self, world, chunk_x, chunk_y, chunk_z, chunk_size):
        try:
            print(f"Starting chunk generation at {chunk_x}, {chunk_y}, {chunk_z}")
            # Calculate world space coordinates
            start_x = chunk_x * chunk_size
            start_y = chunk_y * chunk_size
            start_z = chunk_z * chunk_size
            
            print(f"Chunk world coordinates: ({start_x}, {start_y}, {start_z})")
            
            # Generate terrain for this chunk
            for x in range(start_x, min(start_x + chunk_size, world.width)):
                for z in range(start_z, min(start_z + chunk_size, world.depth)):
                    try:
                        # Generate height using multiple noise octaves
                        height = self._generate_height(x, z)
                        height = int(height * world.height / 2 + world.height / 4)
                        
                        # Fill in terrain from bottom to height
                        for y in range(min(height, world.height)):
                            material = self._determine_material(x, y, z, height)
                            world.voxels[x, y, z] = VoxelData(material)
                            
                    except Exception as e:
                        print(f"Error generating terrain at ({x}, {z}): {e}")
                        traceback.print_exc()
                        
            print(f"Finished generating chunk at {chunk_x}, {chunk_y}, {chunk_z}")
            
        except Exception as e:
            print(f"Error generating chunk at ({chunk_x}, {chunk_y}, {chunk_z}): {e}")
            traceback.print_exc()
            
    def _generate_height(self, x, z):
        try:
            # Use simpler noise generation to avoid memory issues
            scale = 25.0  # Larger scale for smoother terrain
            
            # Generate base noise
            try:
                base_noise = noise.pnoise2(
                    x / scale,
                    z / scale,
                    octaves=1,
                    persistence=0.5,
                    lacunarity=2.0,
                    repeatx=1024,
                    repeaty=1024,
                    base=self.seed % 1024  # Ensure seed is within reasonable range
                )
                
                if base_noise is None:
                    print(f"Warning: Base noise returned None for x={x}, z={z}")
                    base_noise = 0.0
                    
            except Exception as e:
                print(f"Error generating base noise: {e}")
                traceback.print_exc()
                base_noise = 0.0
                
            # Add some variation with a second noise layer
            try:
                detail_noise = noise.pnoise2(
                    x / (scale * 0.5),
                    z / (scale * 0.5),
                    octaves=1,
                    persistence=0.5,
                    lacunarity=2.0,
                    repeatx=1024,
                    repeaty=1024,
                    base=(self.seed + 1) % 1024  # Offset seed for variation
                ) * 0.5  # Reduce influence of detail noise
                
                if detail_noise is None:
                    print(f"Warning: Detail noise returned None for x={x}, z={z}")
                    detail_noise = 0.0
                    
            except Exception as e:
                print(f"Error generating detail noise: {e}")
                traceback.print_exc()
                detail_noise = 0.0
                
            # Combine noises and normalize to [0, 1]
            height = (base_noise + detail_noise + 1.0) * 0.5
            
            # Clamp height to valid range
            height = max(0.0, min(1.0, height))
            
            return height
            
        except Exception as e:
            print(f"Error in height generation at ({x}, {z}): {e}")
            traceback.print_exc()
            return 0.5  # Return default height on error
            
    def _determine_material(self, x, y, z, height):
        try:
            # Simple material distribution based on height
            if y < height * 0.3:
                return MaterialType.METAL
            elif y < height * 0.6:
                return MaterialType.CONCRETE
            elif y < height * 0.9:
                return MaterialType.WOOD
            else:
                return MaterialType.GLASS
                
        except Exception as e:
            print(f"Error determining material at ({x}, {y}, {z}): {e}")
            traceback.print_exc()
            return MaterialType.CONCRETE  # Return default material on error 