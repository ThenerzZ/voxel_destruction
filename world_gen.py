import numpy as np
import noise
from physics import MaterialType, VoxelData
from enum import Enum
import random
import time
import traceback

class TerrainType(Enum):
    OCEAN = 0
    BEACH = 1
    PLAINS = 2
    FOREST = 3
    MOUNTAINS = 4
    DESERT = 5
    SNOW = 6
    JUNGLE = 7

class TerrainProperties:
    def __init__(self, base_height, height_variation, roughness, material_layers):
        self.base_height = base_height  # Base terrain height [0-1]
        self.height_variation = height_variation  # How much height can vary
        self.roughness = roughness  # How rough/smooth the terrain is
        self.material_layers = material_layers  # List of (height_threshold, material) tuples

# Define terrain properties for each type
TERRAIN_PROPERTIES = {
    TerrainType.OCEAN: TerrainProperties(
        base_height=0.25,
        height_variation=0.1,
        roughness=0.3,
        material_layers=[
            (0.0, MaterialType.CONCRETE),  # Ocean floor
            (0.2, MaterialType.GLASS),     # Water
        ]
    ),
    TerrainType.BEACH: TerrainProperties(
        base_height=0.3,
        height_variation=0.1,
        roughness=0.2,
        material_layers=[
            (0.0, MaterialType.CONCRETE),  # Sand
        ]
    ),
    TerrainType.PLAINS: TerrainProperties(
        base_height=0.4,
        height_variation=0.2,
        roughness=0.4,
        material_layers=[
            (0.0, MaterialType.CONCRETE),  # Stone
            (0.3, MaterialType.WOOD),      # Dirt/Grass
        ]
    ),
    TerrainType.FOREST: TerrainProperties(
        base_height=0.5,
        height_variation=0.3,
        roughness=0.5,
        material_layers=[
            (0.0, MaterialType.CONCRETE),  # Stone
            (0.3, MaterialType.WOOD),      # Dirt/Grass
            (0.9, MaterialType.WOOD),      # Trees
        ]
    ),
    TerrainType.MOUNTAINS: TerrainProperties(
        base_height=0.7,
        height_variation=0.6,
        roughness=0.7,
        material_layers=[
            (0.0, MaterialType.METAL),     # Deep stone
            (0.4, MaterialType.CONCRETE),  # Stone
            (0.8, MaterialType.GLASS),     # Snow caps
        ]
    ),
    TerrainType.DESERT: TerrainProperties(
        base_height=0.35,
        height_variation=0.25,
        roughness=0.45,
        material_layers=[
            (0.0, MaterialType.CONCRETE),  # Sandstone
            (0.2, MaterialType.WOOD),      # Sand
        ]
    ),
    TerrainType.SNOW: TerrainProperties(
        base_height=0.45,
        height_variation=0.2,
        roughness=0.3,
        material_layers=[
            (0.0, MaterialType.CONCRETE),  # Stone
            (0.2, MaterialType.GLASS),     # Snow
        ]
    ),
    TerrainType.JUNGLE: TerrainProperties(
        base_height=0.5,
        height_variation=0.4,
        roughness=0.6,
        material_layers=[
            (0.0, MaterialType.CONCRETE),  # Stone
            (0.3, MaterialType.WOOD),      # Dirt
            (0.9, MaterialType.WOOD),      # Dense vegetation
        ]
    ),
}

class WorldGenerator:
    def __init__(self, seed=None):
        self.seed = seed if seed is not None else int(time.time())
        print(f"Initializing WorldGenerator with seed: {self.seed}")
        
        # Initialize different noise generators for various features
        self.temperature_seed = self.seed % 1000
        self.humidity_seed = (self.seed + 1) % 1000
        self.elevation_seed = (self.seed + 2) % 1000
        self.detail_seed = (self.seed + 3) % 1000
        
    def _get_terrain_type(self, x, z):
        """Determine terrain type based on temperature and humidity"""
        # Generate temperature and humidity noise
        temperature = noise.pnoise2(
            x / 100.0, z / 100.0,
            octaves=1,
            persistence=0.5,
            lacunarity=2.0,
            repeatx=1024,
            repeaty=1024,
            base=self.temperature_seed
        )
        
        humidity = noise.pnoise2(
            x / 100.0, z / 100.0,
            octaves=1,
            persistence=0.5,
            lacunarity=2.0,
            repeatx=1024,
            repeaty=1024,
            base=self.humidity_seed
        )
        
        # Normalize to [0,1] range
        temperature = (temperature + 1) * 0.5
        humidity = (humidity + 1) * 0.5
        
        # Determine terrain type based on temperature and humidity
        if temperature < 0.2:  # Cold
            return TerrainType.SNOW
        elif temperature > 0.8:  # Hot
            if humidity < 0.3:
                return TerrainType.DESERT
            else:
                return TerrainType.JUNGLE
        else:  # Moderate temperature
            if humidity < 0.3:
                return TerrainType.PLAINS
            elif humidity < 0.6:
                return TerrainType.FOREST
            else:
                return TerrainType.MOUNTAINS
                
    def _generate_height(self, x, z, terrain_type):
        """Generate height using multiple noise layers"""
        props = TERRAIN_PROPERTIES[terrain_type]
        
        # Base terrain noise
        base_scale = 50.0
        base_height = noise.pnoise2(
            x / base_scale,
            z / base_scale,
            octaves=4,
            persistence=0.5,
            lacunarity=2.0,
            repeatx=1024,
            repeaty=1024,
            base=self.elevation_seed
        )
        
        # Add detail noise
        detail_scale = 25.0
        detail = noise.pnoise2(
            x / detail_scale,
            z / detail_scale,
            octaves=2,
            persistence=0.5,
            lacunarity=2.0,
            repeatx=1024,
            repeaty=1024,
            base=self.detail_seed
        ) * props.roughness
        
        # Combine and normalize
        height = (base_height + detail) * props.height_variation
        height = height + props.base_height
        
        # Clamp to valid range
        return max(0.0, min(1.0, height))
        
    def _determine_material(self, x, y, z, terrain_type, relative_height):
        """Determine material based on terrain type and height"""
        props = TERRAIN_PROPERTIES[terrain_type]
        
        # Find appropriate material layer
        for threshold, material in reversed(props.material_layers):
            if relative_height >= threshold:
                return material
                
        return props.material_layers[0][1]  # Default to bottom layer
        
    def generate_chunk(self, world, chunk_x, chunk_y, chunk_z, chunk_size):
        try:
            print(f"Starting chunk generation at {chunk_x}, {chunk_y}, {chunk_z}")
            start_x = chunk_x * chunk_size
            start_y = chunk_y * chunk_size
            start_z = chunk_z * chunk_size
            
            # Generate terrain for this chunk
            for x in range(start_x, min(start_x + chunk_size, world.width)):
                for z in range(start_z, min(start_z + chunk_size, world.depth)):
                    # Determine terrain type for this column
                    terrain_type = self._get_terrain_type(x, z)
                    
                    # Generate height for this column
                    height_value = self._generate_height(x, z, terrain_type)
                    height = int(height_value * world.height)
                    
                    # Fill in terrain
                    for y in range(min(height, world.height)):
                        relative_height = y / height
                        material = self._determine_material(x, y, z, terrain_type, relative_height)
                        world.voxels[x, y, z] = VoxelData(material)
                        
            print(f"Finished generating chunk at {chunk_x}, {chunk_y}, {chunk_z}")
            
        except Exception as e:
            print(f"Error generating chunk: {e}")
            traceback.print_exc() 