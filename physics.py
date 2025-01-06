import numpy as np
from collections import defaultdict

class VoxelChunk:
    def __init__(self, voxels, position, velocity=None):
        self.voxels = voxels  # 3D numpy array of voxels
        self.position = np.array(position, dtype=np.float32)  # World position
        self.velocity = np.zeros(3) if velocity is None else np.array(velocity)
        self.dimensions = np.array(voxels.shape)
        self.mass = np.sum(voxels)  # Mass based on number of solid voxels
        
    def update(self, dt):
        # Apply gravity
        self.velocity[1] -= 9.81 * dt  # Gravity in Y direction
        self.position += self.velocity * dt

class PhysicsSystem:
    def __init__(self, world):
        self.world = world
        self.chunks = []  # List of active physics chunks
        self.gravity = 9.81
        self.chunk_connections = defaultdict(set)  # Track connected chunks
        
    def destroy_voxel(self, x, y, z):
        """Handle voxel destruction and chunk creation"""
        if not self.world.is_valid_position(x, y, z) or self.world.voxels[x, y, z] == 0:
            return
            
        # Remove the voxel
        self.world.voxels[x, y, z] = 0
        
        # Check surrounding voxels and create new chunks if needed
        self.check_chunk_separation(x, y, z)
        
    def check_chunk_separation(self, x, y, z):
        """Check if destroying a voxel separates the structure into chunks"""
        # Get all solid voxels connected to this position
        checked = set()
        chunks = []
        
        # Check all adjacent positions
        for dx, dy, dz in [
            (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)
        ]:
            check_x, check_y, check_z = x + dx, y + dy, z + dz
            
            if not self.world.is_valid_position(check_x, check_y, check_z):
                continue
                
            if self.world.voxels[check_x, check_y, check_z] == 1 and (check_x, check_y, check_z) not in checked:
                # Found a new potential chunk
                chunk_voxels = self.flood_fill(check_x, check_y, check_z, checked)
                if chunk_voxels:
                    chunks.append(chunk_voxels)
        
        # Create physics chunks for disconnected pieces
        for chunk_voxels in chunks:
            if not self.is_chunk_supported(chunk_voxels):
                self.create_falling_chunk(chunk_voxels)
                
    def flood_fill(self, start_x, start_y, start_z, checked):
        """Find all connected voxels using flood fill"""
        to_check = [(start_x, start_y, start_z)]
        chunk_voxels = set()
        
        while to_check:
            x, y, z = to_check.pop()
            
            if (x, y, z) in checked:
                continue
                
            checked.add((x, y, z))
            
            if not self.world.is_valid_position(x, y, z) or self.world.voxels[x, y, z] == 0:
                continue
                
            chunk_voxels.add((x, y, z))
            
            # Add adjacent voxels to check
            for dx, dy, dz in [
                (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)
            ]:
                to_check.append((x + dx, y + dy, z + dz))
                
        return chunk_voxels
        
    def is_chunk_supported(self, chunk_voxels):
        """Check if a chunk has support (connected to ground or stable structure)"""
        # For now, consider a chunk unsupported if it has no voxels at y=0
        return any(y == 0 for _, y, _ in chunk_voxels)
        
    def create_falling_chunk(self, chunk_voxels):
        """Create a new physics chunk from a set of voxel positions"""
        if not chunk_voxels:
            return
            
        # Find chunk bounds
        positions = np.array(list(chunk_voxels))
        min_pos = np.min(positions, axis=0)
        max_pos = np.max(positions, axis=0)
        dimensions = max_pos - min_pos + 1
        
        # Create voxel array for chunk
        chunk_array = np.zeros(dimensions, dtype=np.int32)
        for x, y, z in chunk_voxels:
            local_x = x - min_pos[0]
            local_y = y - min_pos[1]
            local_z = z - min_pos[2]
            chunk_array[local_x, local_y, local_z] = 1
            # Remove from world
            self.world.voxels[x, y, z] = 0
            
        # Create new chunk
        chunk = VoxelChunk(chunk_array, min_pos)
        self.chunks.append(chunk)
        
    def update(self, dt):
        """Update physics simulation"""
        # Update all active chunks
        for chunk in self.chunks:
            chunk.update(dt)
            
        # Handle collisions
        self.handle_collisions()
        
        # Remove chunks that have settled
        self.chunks = [chunk for chunk in self.chunks if not self.has_chunk_settled(chunk)]
        
    def handle_collisions(self):
        """Handle collisions between chunks and world"""
        for chunk in self.chunks:
            # Check collision with world bounds and other voxels
            chunk_bounds = np.array([
                chunk.position,
                chunk.position + chunk.dimensions
            ])
            
            # Floor collision
            if chunk_bounds[0][1] <= 0:
                chunk.position[1] = 0
                chunk.velocity[1] = 0
                self.merge_chunk_with_world(chunk)
                
            # World voxel collision
            elif self.check_chunk_world_collision(chunk):
                self.merge_chunk_with_world(chunk)
                
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
                            
    def has_chunk_settled(self, chunk):
        """Check if a physics chunk has come to rest"""
        return (
            abs(chunk.velocity[1]) < 0.01 and  # Very small vertical velocity
            chunk.position[1] <= 0.01  # Very close to ground
        ) or self.check_chunk_world_collision(chunk)  # Or colliding with world 