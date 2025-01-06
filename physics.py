from enum import Enum
import numpy as np
from collections import defaultdict

class MaterialType(Enum):
    AIR = 0
    CONCRETE = 1
    WOOD = 2
    GLASS = 3
    METAL = 4

class MaterialProperties:
    def __init__(self, durability, fragment_size, fragment_count, deformation_resistance):
        self.durability = durability  # How much damage it can take
        self.fragment_size = fragment_size  # Size of debris
        self.fragment_count = fragment_count  # How many pieces it breaks into
        self.deformation_resistance = deformation_resistance  # How much it resists bending/warping

MATERIAL_PROPERTIES = {
    MaterialType.AIR: MaterialProperties(0.0, 0.0, 0, 0.0),  # Air has no physical properties
    MaterialType.CONCRETE: MaterialProperties(100.0, 0.4, 12, 0.9),
    MaterialType.WOOD: MaterialProperties(60.0, 0.3, 8, 0.5),
    MaterialType.GLASS: MaterialProperties(30.0, 0.2, 16, 0.1),
    MaterialType.METAL: MaterialProperties(120.0, 0.35, 10, 0.8),
}

class VoxelData:
    def __init__(self, material_type, health=None):
        self.material_type = material_type
        if material_type == MaterialType.AIR:
            self.health = 0.0
            self.damage_state = 0
            self.deformation = 0.0
        else:
            self.health = MATERIAL_PROPERTIES[material_type].durability if health is None else health
            self.damage_state = 0  # 0: pristine, 1: slightly damaged, 2: heavily damaged
            self.deformation = 0.0  # How much the block has been deformed

class VoxelFragment:
    def __init__(self, size, position, velocity=None, material_type=MaterialType.CONCRETE):
        """A smaller piece of a broken voxel"""
        self.size = size  # Size of the fragment (0.0 to 1.0)
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.zeros(3) if velocity is None else np.array(velocity)
        self.angular_velocity = np.random.uniform(-5, 5, 3)  # Random spin
        self.rotation = np.eye(3)
        self.lifetime = 5.0  # Fragments disappear after a few seconds
        self.material_type = material_type
        
    def update(self, dt):
        # Apply gravity
        self.velocity[1] -= 9.81 * dt
        
        # Apply air resistance (more for smaller pieces)
        air_resistance = 0.3 * (1.0 - self.size)
        self.velocity *= (1.0 - air_resistance * dt)
        
        # Update position
        self.position += self.velocity * dt
        
        # Update rotation
        angle = np.linalg.norm(self.angular_velocity) * dt
        if angle > 0:
            axis = self.angular_velocity / np.linalg.norm(self.angular_velocity)
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            rotation_matrix = np.array([
                [cos_angle + axis[0]**2 * (1 - cos_angle),
                 axis[0] * axis[1] * (1 - cos_angle) - axis[2] * sin_angle,
                 axis[0] * axis[2] * (1 - cos_angle) + axis[1] * sin_angle],
                [axis[1] * axis[0] * (1 - cos_angle) + axis[2] * sin_angle,
                 cos_angle + axis[1]**2 * (1 - cos_angle),
                 axis[1] * axis[2] * (1 - cos_angle) - axis[0] * sin_angle],
                [axis[2] * axis[0] * (1 - cos_angle) - axis[1] * sin_angle,
                 axis[2] * axis[1] * (1 - cos_angle) + axis[0] * sin_angle,
                 cos_angle + axis[2]**2 * (1 - cos_angle)]
            ])
            self.rotation = np.dot(rotation_matrix, self.rotation)
            
        # Update lifetime
        self.lifetime -= dt

class VoxelChunk:
    def __init__(self, voxels, position):
        self.voxels = voxels
        self.dimensions = np.array(voxels.shape)
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.rotation = np.eye(3)
        self.lifetime = 10.0  # Chunks disappear after this time if they don't settle
        
    def update(self, dt):
        # Apply gravity
        self.velocity[1] -= 9.81 * dt
        
        # Update position
        self.position += self.velocity * dt
        
        # Update rotation
        angle = np.linalg.norm(self.angular_velocity) * dt
        if angle > 0:
            axis = self.angular_velocity / np.linalg.norm(self.angular_velocity)
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            rotation_matrix = np.array([
                [cos_angle + axis[0]**2 * (1 - cos_angle),
                 axis[0] * axis[1] * (1 - cos_angle) - axis[2] * sin_angle,
                 axis[0] * axis[2] * (1 - cos_angle) + axis[1] * sin_angle],
                [axis[1] * axis[0] * (1 - cos_angle) + axis[2] * sin_angle,
                 cos_angle + axis[1]**2 * (1 - cos_angle),
                 axis[1] * axis[2] * (1 - cos_angle) - axis[0] * sin_angle],
                [axis[2] * axis[0] * (1 - cos_angle) - axis[1] * sin_angle,
                 axis[2] * axis[1] * (1 - cos_angle) + axis[0] * sin_angle,
                 cos_angle + axis[2]**2 * (1 - cos_angle)]
            ])
            self.rotation = np.dot(rotation_matrix, self.rotation)
            
        # Update lifetime
        self.lifetime -= dt

class PhysicsSystem:
    def __init__(self, world):
        self.world = world
        self.chunks = []  # List of active physics chunks
        self.fragments = []  # List of block fragments
        self.gravity = 9.81
        self.chunk_connections = defaultdict(set)
        self.stress_map = np.zeros((world.width, world.height, world.depth))
        self.support_threshold = 0.7  # Minimum support needed for stability
        
    def apply_damage(self, x, y, z, damage, radius=1.0):
        """Apply damage to blocks within a radius"""
        center = np.array([x, y, z])
        for dx in range(-int(radius), int(radius) + 1):
            for dy in range(-int(radius), int(radius) + 1):
                for dz in range(-int(radius), int(radius) + 1):
                    pos = np.array([x + dx, y + dy, z + dz])
                    if not self.world.is_valid_position(*pos):
                        continue
                        
                    # Calculate distance-based damage falloff
                    distance = np.linalg.norm(pos - center)
                    if distance > radius:
                        continue
                        
                    damage_at_point = damage * (1.0 - distance / radius)
                    voxel = self.world.get_voxel_data(*pos)
                    if voxel and voxel.material_type != MaterialType.AIR:
                        self._damage_block(*pos, damage_at_point)
                        
    def _damage_block(self, x, y, z, damage):
        """Apply damage to a single block"""
        voxel = self.world.get_voxel_data(x, y, z)
        if not voxel or voxel.material_type == MaterialType.AIR:
            return
            
        # Apply damage
        voxel.health -= damage
        
        # Update damage state
        props = MATERIAL_PROPERTIES[voxel.material_type]
        health_percentage = voxel.health / props.durability
        
        if health_percentage <= 0:
            self.destroy_voxel(x, y, z)
        elif health_percentage < 0.3:
            voxel.damage_state = 2
        elif health_percentage < 0.7:
            voxel.damage_state = 1
            
        # Apply deformation for certain materials
        if voxel.material_type in [MaterialType.METAL, MaterialType.WOOD]:
            deform_amount = damage / props.durability * (1.0 - props.deformation_resistance)
            voxel.deformation = min(1.0, voxel.deformation + deform_amount)
        
    def flood_fill(self, start_x, start_y, start_z, checked):
        """Find all connected voxels using flood fill"""
        to_check = [(start_x, start_y, start_z)]
        chunk_voxels = set()
        
        while to_check:
            x, y, z = to_check.pop()
            
            if (x, y, z) in checked:
                continue
                
            checked.add((x, y, z))
            
            voxel = self.world.get_voxel_data(x, y, z)
            if not voxel or voxel.material_type == MaterialType.AIR:
                continue
                
            chunk_voxels.add((x, y, z))
            
            # Add adjacent voxels to check
            for dx, dy, dz in [
                (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)
            ]:
                next_x, next_y, next_z = x + dx, y + dy, z + dz
                if self.world.is_valid_position(next_x, next_y, next_z):
                    to_check.append((next_x, next_y, next_z))
                    
        return chunk_voxels
        
    def is_chunk_supported(self, chunk_voxels):
        """Check if a chunk has support (connected to ground or stable structure)"""
        # Check if any voxel in the chunk is at ground level
        if any(y == 0 for _, y, _ in chunk_voxels):
            return True
            
        # Check if any voxel is connected to a supported structure
        for x, y, z in chunk_voxels:
            # Check the voxel below
            if y > 0 and self.world.is_valid_position(x, y-1, z):
                if self.world.voxels[x, y-1, z] == 1 and (x, y-1, z) not in chunk_voxels:
                    return True
                    
        return False
        
    def check_chunk_world_collision(self, chunk):
        """Check if a chunk collides with world voxels"""
        chunk_pos = chunk.position.astype(int)
        
        # Check each voxel in the chunk
        for x in range(chunk.dimensions[0]):
            for y in range(chunk.dimensions[1]):
                for z in range(chunk.dimensions[2]):
                    if chunk.voxels[x, y, z] == 0:
                        continue
                        
                    world_x = chunk_pos[0] + x
                    world_y = chunk_pos[1] + y
                    world_z = chunk_pos[2] + z
                    
                    # Check if position below is solid
                    check_y = world_y - 1
                    if check_y >= 0:
                        if self.world.is_valid_position(world_x, check_y, world_z):
                            if self.world.voxels[world_x, check_y, world_z] == 1:
                                return True
                                
        return False
        
    def merge_chunk_with_world(self, chunk):
        """Merge a physics chunk back into the world"""
        chunk_pos = chunk.position.astype(int)
        
        # Add chunk voxels back to world
        for x in range(chunk.dimensions[0]):
            for y in range(chunk.dimensions[1]):
                for z in range(chunk.dimensions[2]):
                    if chunk.voxels[x, y, z] == 1:
                        world_x = chunk_pos[0] + x
                        world_y = chunk_pos[1] + y
                        world_z = chunk_pos[2] + z
                        
                        if self.world.is_valid_position(world_x, world_y, world_z):
                            self.world.voxels[world_x, world_y, world_z] = 1
            
    def destroy_voxel(self, x, y, z):
        """Destroy a voxel and create fragments"""
        voxel = self.world.get_voxel_data(x, y, z)
        if not voxel or voxel.material_type == MaterialType.AIR:
            return
            
        # Get material properties
        props = MATERIAL_PROPERTIES[voxel.material_type]
        
        # Create fragments
        for _ in range(props.fragment_count):
            # Random offset within the block
            offset = np.random.uniform(-0.2, 0.2, 3)
            position = np.array([x, y, z]) + offset
            
            # Random initial velocity (explosion effect)
            velocity = np.random.uniform(-3, 3, 3)
            
            # Create fragment with material properties
            fragment = VoxelFragment(
                size=props.fragment_size * np.random.uniform(0.8, 1.2),
                position=position,
                velocity=velocity,
                material_type=voxel.material_type
            )
            self.fragments.append(fragment)
            
        # Set voxel to air
        self.world.voxels[x, y, z] = VoxelData(MaterialType.AIR)
        
        # Check structural integrity
        self.check_structural_integrity(x, y, z)
        
    def check_structural_integrity(self, x, y, z):
        """Check if removing a block causes other blocks to fall"""
        checked = set()
        to_check = set()
        
        # Add adjacent blocks to check
        for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            check_x, check_y, check_z = x + dx, y + dy, z + dz
            if self.world.is_valid_position(check_x, check_y, check_z):
                voxel = self.world.get_voxel_data(check_x, check_y, check_z)
                if voxel and voxel.material_type != MaterialType.AIR:
                    to_check.add((check_x, check_y, check_z))
        
        # Find disconnected blocks
        while to_check:
            pos = to_check.pop()
            if pos in checked:
                continue
                
            chunk_voxels = self.flood_fill(*pos, checked)
            if chunk_voxels and not self.is_chunk_supported(chunk_voxels):
                self.create_falling_chunk(chunk_voxels)
                
    def create_falling_chunk(self, chunk_voxels):
        """Create a new physics chunk with initial velocity based on position"""
        if not chunk_voxels:
            return
            
        # Find chunk bounds
        positions = np.array(list(chunk_voxels))
        min_pos = np.min(positions, axis=0)
        max_pos = np.max(positions, axis=0)
        dimensions = max_pos - min_pos + 1
        
        # Create voxel array for chunk
        chunk_array = np.full(dimensions, VoxelData(MaterialType.AIR), dtype=object)
        center_of_mass = np.zeros(3)
        total_mass = 0
        
        for x, y, z in chunk_voxels:
            local_x = x - min_pos[0]
            local_y = y - min_pos[1]
            local_z = z - min_pos[2]
            voxel = self.world.get_voxel_data(x, y, z)
            chunk_array[local_x, local_y, local_z] = voxel
            center_of_mass += np.array([x, y, z])
            total_mass += 1
            # Remove from world
            self.world.voxels[x, y, z] = VoxelData(MaterialType.AIR)
            
        center_of_mass /= total_mass
        
        # Create chunk with initial velocity and angular velocity
        chunk = VoxelChunk(chunk_array, min_pos)
        
        # Add random initial velocities for more interesting destruction
        chunk.velocity = np.random.uniform(-2, 2, 3)
        chunk.angular_velocity = np.random.uniform(-2, 2, 3)
        
        self.chunks.append(chunk)
        
    def update(self, dt):
        """Update physics simulation"""
        # Update fragments
        for fragment in self.fragments:
            fragment.update(dt)
            
            # Handle fragment collisions with ground
            if fragment.position[1] <= 0:
                fragment.position[1] = 0
                fragment.velocity[1] = -fragment.velocity[1] * 0.3  # Bounce with energy loss
                fragment.angular_velocity *= 0.7
                
        # Remove fragments that have expired
        self.fragments = [f for f in self.fragments if f.lifetime > 0]
        
        # Update chunks
        for chunk in self.chunks:
            chunk.update(dt)
            
            # Handle chunk collisions
            if chunk.position[1] <= 0:
                # Merge chunk with world when it hits ground
                self.merge_chunk_with_world(chunk)
            elif self.check_chunk_world_collision(chunk):
                # Merge chunk with world when it hits other blocks
                self.merge_chunk_with_world(chunk)
                
        # Remove chunks that have been merged or expired
        self.chunks = [chunk for chunk in self.chunks if chunk.lifetime > 0] 