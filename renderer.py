import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
import pyrr
import os
import ctypes
from lighting import LightingSystem
from atmosphere import AtmosphereSystem
from physics import MaterialType

class VoxelRenderer:
    def __init__(self):
        # Initialize basic properties first
        self.max_instances = 16384  # Maximum number of instances to render at once
        self.shader = None
        self.vao = None
        self.vbo = None
        self.ebo = None
        self.instance_vbo = None
        
        # Face direction vectors for culling checks
        self.face_directions = [
            ( 0,  0,  1),  # Front  (+Z)
            ( 0,  0, -1),  # Back   (-Z)
            ( 0,  1,  0),  # Top    (+Y)
            ( 0, -1,  0),  # Bottom (-Y)
            ( 1,  0,  0),  # Right  (+X)
            (-1,  0,  0),  # Left   (-X)
        ]
        
        # Initialize lighting and atmosphere systems
        self.lighting_system = LightingSystem()
        self.atmosphere_system = AtmosphereSystem()
        
        # Setup OpenGL resources
        self.setup_shaders()
        self.setup_matrices()
        self.setup_cube()
        
        # Material colors
        self.material_colors = {
            MaterialType.AIR: np.array([0.0, 0.0, 0.0, 0.0]),  # Transparent
            MaterialType.CONCRETE: np.array([0.7, 0.7, 0.7]),  # Gray
            MaterialType.WOOD: np.array([0.6, 0.4, 0.2]),      # Brown
            MaterialType.GLASS: np.array([0.8, 0.9, 1.0]),     # Light blue
            MaterialType.METAL: np.array([0.8, 0.8, 0.9]),     # Metallic gray
        }
        
        # Damage state modifiers
        self.damage_modifiers = [
            1.0,    # Pristine
            0.8,    # Slightly damaged
            0.6,    # Heavily damaged
        ]
        
    def should_render_face(self, world, x, y, z, dx, dy, dz):
        """Check if a face should be rendered based on adjacent voxels"""
        # Check if the adjacent block position is valid
        adj_x = x + dx
        adj_y = y + dy
        adj_z = z + dz
        
        # If adjacent position is outside world bounds, always render the face
        if not (0 <= adj_x < world.width and 
                0 <= adj_y < world.height and 
                0 <= adj_z < world.depth):
            return True
        
        # Get adjacent voxel data
        adj_voxel = world.get_voxel_data(adj_x, adj_y, adj_z)
        
        # Render face if adjacent block is air or doesn't exist
        return adj_voxel is None or adj_voxel.material_type == MaterialType.AIR
        
    def setup_shaders(self):
        with open('shaders/vertex.glsl', 'r') as file:
            vertex_shader = shaders.compileShader(file.read(), GL_VERTEX_SHADER)
        with open('shaders/fragment.glsl', 'r') as file:
            fragment_shader = shaders.compileShader(file.read(), GL_FRAGMENT_SHADER)
            
        self.shader = shaders.compileProgram(vertex_shader, fragment_shader)
        glUseProgram(self.shader)
        
    def setup_matrices(self):
        self.projection = pyrr.matrix44.create_perspective_projection(
            70.0, 800.0/600.0, 0.1, 1000.0, dtype=np.float32
        )
        self.view = pyrr.matrix44.create_look_at(
            np.array([5.0, 5.0, 5.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            dtype=np.float32
        )
        
    def setup_cube(self):
        # Define vertices for a unit cube (using full 1.0 size)
        vertices = np.array([
            # Front face (+Z)
            -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  0.0, 0.0,  # Full size cube
             0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  1.0, 0.0,  # for no gaps
             0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  1.0, 1.0,
            -0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  0.0, 1.0,

            # Back face (-Z)
            -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  1.0, 0.0,
             0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  0.0, 0.0,
             0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  0.0, 1.0,
            -0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  1.0, 1.0,

            # Top face (+Y)
            -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  0.0, 1.0,
             0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  1.0, 1.0,
             0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  1.0, 0.0,
            -0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  0.0, 0.0,

            # Bottom face (-Y)
            -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  0.0, 0.0,
             0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  1.0, 0.0,
             0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  1.0, 1.0,
            -0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  0.0, 1.0,

            # Right face (+X)
             0.5, -0.5, -0.5,  1.0,  0.0,  0.0,  1.0, 0.0,
             0.5,  0.5, -0.5,  1.0,  0.0,  0.0,  1.0, 1.0,
             0.5,  0.5,  0.5,  1.0,  0.0,  0.0,  0.0, 1.0,
             0.5, -0.5,  0.5,  1.0,  0.0,  0.0,  0.0, 0.0,

            # Left face (-X)
            -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,  0.0, 0.0,
            -0.5,  0.5, -0.5, -1.0,  0.0,  0.0,  0.0, 1.0,
            -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,  1.0, 1.0,
            -0.5, -0.5,  0.5, -1.0,  0.0,  0.0,  1.0, 0.0,
        ], dtype=np.float32)

        # Define indices for all faces
        indices = np.array([
            0,  1,  2,    2,  3,  0,   # Front
            4,  5,  6,    6,  7,  4,   # Back
            8,  9,  10,   10, 11, 8,   # Top
            12, 13, 14,   14, 15, 12,  # Bottom
            16, 17, 18,   18, 19, 16,  # Right
            20, 21, 22,   22, 23, 20   # Left
        ], dtype=np.uint32)

        # Create and bind VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # Create and bind VBO for cube vertices
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        # Create and bind EBO
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # Set up vertex attributes
        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # Normal attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

        # Texture coordinate attribute
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(24))
        glEnableVertexAttribArray(2)

        # Create and set up instance buffer
        self.instance_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
        
        # Each instance needs: position (3), color (3)
        instance_stride = 6 * 4  # 6 floats * 4 bytes
        buffer_size = self.max_instances * instance_stride
        glBufferData(GL_ARRAY_BUFFER, buffer_size, None, GL_DYNAMIC_DRAW)
        
        # Instance position attribute (location = 3)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, instance_stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)
        
        # Instance color attribute (location = 4)
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, instance_stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(4)
        glVertexAttribDivisor(4, 1)
        
        # Cleanup
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    def update(self, delta_time):
        """Update lighting and atmosphere systems"""
        self.lighting_system.update(delta_time)
        self.atmosphere_system.update(delta_time, self.lighting_system.time_of_day)
        
    def set_uniforms(self):
        """Set all shader uniforms"""
        glUseProgram(self.shader)
        
        # Set camera uniforms
        view_loc = glGetUniformLocation(self.shader, "view")
        proj_loc = glGetUniformLocation(self.shader, "projection")
        
        if view_loc != -1:
            glUniformMatrix4fv(view_loc, 1, GL_FALSE, self.view)
        if proj_loc != -1:
            glUniformMatrix4fv(proj_loc, 1, GL_FALSE, self.projection)
        
        # Set view position
        view_pos = pyrr.Vector3([self.view[3][0], self.view[3][1], self.view[3][2]])
        view_pos_loc = glGetUniformLocation(self.shader, "viewPos")
        if view_pos_loc != -1:
            glUniform3fv(view_pos_loc, 1, view_pos)
        
    def render_world(self, world):
        glUseProgram(self.shader)
        
        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        
        # Disable face culling for now to see all faces
        glDisable(GL_CULL_FACE)
        
        # Enable blending for transparent materials
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Update and set uniforms
        self.set_uniforms()
        
        # Render static world voxels
        self._render_static_voxels(world)
        
        # Render falling chunks
        self._render_chunks(world)
        
        # Render fragments
        self._render_fragments(world)
        
    def _render_static_voxels(self, world):
        """Render the static (non-moving) voxels in the world"""
        # Pre-allocate arrays for better performance
        max_instances = world.width * world.height * world.depth * 6  # 6 faces per voxel maximum
        instance_data = np.zeros(max_instances * 6, dtype=np.float32)
        instance_count = 0
        
        # Render each voxel
        for x in range(world.width):
            for y in range(world.height):
                for z in range(world.depth):
                    voxel = world.get_voxel_data(x, y, z)
                    if voxel and voxel.material_type != MaterialType.AIR:
                        # Check each face
                        for face_idx, (dx, dy, dz) in enumerate(self.face_directions):
                            if self.should_render_face(world, x, y, z, dx, dy, dz):
                                # Calculate offset into instance_data array
                                offset = instance_count * 6
                                
                                # Position
                                instance_data[offset:offset+3] = [x, y, z]
                                
                                # Get base color for material
                                base_color = self.material_colors[voxel.material_type][:3]  # Only take RGB components
                                
                                # Apply damage state modifier
                                damage_modifier = self.damage_modifiers[voxel.damage_state]
                                
                                # Apply deformation effect (darken and add redness for deformed metal/wood)
                                if voxel.deformation > 0:
                                    deform_color = np.array([0.8, 0.2, 0.2])  # Reddish tint
                                    base_color = base_color * (1.0 - voxel.deformation) + deform_color * voxel.deformation
                                
                                # Final color
                                color = base_color * damage_modifier
                                instance_data[offset+3:offset+6] = color
                                
                                instance_count += 1
                                
                                if instance_count >= self.max_instances:
                                    self._render_batch(instance_data[:instance_count * 6], instance_count)
                                    instance_count = 0
        
        if instance_count > 0:
            self._render_batch(instance_data[:instance_count * 6], instance_count)
            
    def _render_chunks(self, world):
        """Render falling chunks"""
        for chunk in world.physics.chunks:
            # Pre-allocate arrays for this chunk
            max_instances = chunk.dimensions[0] * chunk.dimensions[1] * chunk.dimensions[2] * 6
            instance_data = np.zeros(max_instances * 6, dtype=np.float32)
            instance_count = 0
            
            # Render each voxel in the chunk
            for x in range(chunk.dimensions[0]):
                for y in range(chunk.dimensions[1]):
                    for z in range(chunk.dimensions[2]):
                        voxel = chunk.voxels[x, y, z]
                        if voxel and voxel.material_type != MaterialType.AIR:
                            # Calculate world position
                            world_pos = chunk.position + np.array([x, y, z])
                            
                            # Add instance data for each face
                            for face_idx, (dx, dy, dz) in enumerate(self.face_directions):
                                # Always render all faces for falling chunks
                                offset = instance_count * 6
                                
                                # Position
                                instance_data[offset:offset+3] = world_pos
                                
                                # Get base color for material
                                base_color = self.material_colors[voxel.material_type][:3]
                                
                                # Apply damage state modifier
                                damage_modifier = self.damage_modifiers[voxel.damage_state]
                                
                                # Final color
                                color = base_color * damage_modifier
                                instance_data[offset+3:offset+6] = color
                                
                                instance_count += 1
                                
            if instance_count > 0:
                self._render_batch(instance_data[:instance_count * 6], instance_count)
                
    def _render_fragments(self, world):
        """Render debris fragments"""
        if world.physics.fragments:
            fragment_data = np.zeros(len(world.physics.fragments) * 6, dtype=np.float32)
            for i, fragment in enumerate(world.physics.fragments):
                offset = i * 6
                fragment_data[offset:offset+3] = fragment.position
                
                # Get base color for fragment material
                base_color = self.material_colors[fragment.material_type][:3]
                
                # Add some variation based on fragment lifetime
                intensity = 0.6 + 0.2 * np.sin(fragment.lifetime * 5.0)
                fragment_data[offset+3:offset+6] = base_color * intensity
            
            self._render_batch(fragment_data, len(world.physics.fragments))
            
    def _render_batch(self, instance_data, instance_count):
        """Render a batch of instances efficiently"""
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, instance_data.nbytes, instance_data)
        
        glBindVertexArray(self.vao)
        glDrawElementsInstanced(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None, instance_count)
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        
    def render_cube(self):
        """Render a single cube"""
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        
    def render_chunk(self, chunk):
        """Render a chunk of voxels"""
        glBindVertexArray(self.vao)
        for x in range(chunk.dimensions[0]):
            for y in range(chunk.dimensions[1]):
                for z in range(chunk.dimensions[2]):
                    if chunk.voxels[x, y, z] == 1:
                        model = pyrr.matrix44.create_from_translation([x, y, z])
                        glUniformMatrix4fv(
                            glGetUniformLocation(self.shader, "model"),
                            1, GL_FALSE, model
                        )
                        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)
        glBindVertexArray(0) 