import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
import pyrr
import os
import ctypes
from lighting import LightingSystem
from atmosphere import AtmosphereSystem

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
        
        # If adjacent block is air (0), render the face
        return world.voxels[adj_x, adj_y, adj_z] == 0
        
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
        # Define vertices for a unit cube centered at origin
        vertices = np.array([
            # Front face (+Z)
            -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  0.0, 0.0,  # 0
             0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  1.0, 0.0,  # 1
             0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  1.0, 1.0,  # 2
            -0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  0.0, 1.0,  # 3

            # Back face (-Z)
            -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  1.0, 0.0,  # 4
             0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  0.0, 0.0,  # 5
             0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  0.0, 1.0,  # 6
            -0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  1.0, 1.0,  # 7

            # Top face (+Y)
            -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  0.0, 1.0,  # 8
             0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  1.0, 1.0,  # 9
             0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  1.0, 0.0,  # 10
            -0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  0.0, 0.0,  # 11

            # Bottom face (-Y)
            -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  0.0, 0.0,  # 12
             0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  1.0, 0.0,  # 13
             0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  1.0, 1.0,  # 14
            -0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  0.0, 1.0,  # 15

            # Right face (+X)
             0.5, -0.5, -0.5,  1.0,  0.0,  0.0,  1.0, 0.0,  # 16
             0.5,  0.5, -0.5,  1.0,  0.0,  0.0,  1.0, 1.0,  # 17
             0.5,  0.5,  0.5,  1.0,  0.0,  0.0,  0.0, 1.0,  # 18
             0.5, -0.5,  0.5,  1.0,  0.0,  0.0,  0.0, 0.0,  # 19

            # Left face (-X)
            -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,  0.0, 0.0,  # 20
            -0.5,  0.5, -0.5, -1.0,  0.0,  0.0,  0.0, 1.0,  # 21
            -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,  1.0, 1.0,  # 22
            -0.5, -0.5,  0.5, -1.0,  0.0,  0.0,  1.0, 0.0,  # 23
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
        
        # Disable face culling for now to debug visibility issues
        glDisable(GL_CULL_FACE)
        
        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Update and set uniforms
        self.set_uniforms()
        
        # Pre-allocate arrays for better performance
        max_faces = world.width * world.height * world.depth * 6  # Maximum possible faces
        instance_data = np.zeros(max_faces * 6, dtype=np.float32)  # 6 floats per instance
        instance_count = 0
        
        # Face colors for different orientations
        face_colors = [
            [1.00, 0.80, 0.80],  # Front  (+Z) - Salmon pink
            [0.80, 0.80, 1.00],  # Back   (-Z) - Light blue
            [1.00, 1.00, 1.00],  # Top    (+Y) - White
            [0.70, 0.70, 0.70],  # Bottom (-Y) - Gray
            [0.80, 1.00, 0.80],  # Right  (+X) - Light green
            [1.00, 0.90, 0.80],  # Left   (-X) - Peach
        ]
        
        # Render voxels
        for x in range(world.width):
            for y in range(world.height):
                for z in range(world.depth):
                    if world.voxels[x, y, z] == 1:
                        # Check each face
                        for face_idx, (dx, dy, dz) in enumerate(self.face_directions):
                            adj_x = x + dx
                            adj_y = y + dy
                            adj_z = z + dz
                            
                            # Render face if it's at the edge or next to an air block
                            if (not (0 <= adj_x < world.width and 
                                   0 <= adj_y < world.height and 
                                   0 <= adj_z < world.depth) or
                                world.voxels[adj_x, adj_y, adj_z] == 0):
                                
                                # Calculate offset into instance_data array
                                offset = instance_count * 6
                                # Position
                                instance_data[offset:offset+3] = [x, y, z]
                                # Color
                                instance_data[offset+3:offset+6] = face_colors[face_idx]
                                instance_count += 1
                                
                                if instance_count >= self.max_instances:
                                    self._render_batch(instance_data[:instance_count * 6], instance_count)
                                    instance_count = 0
        
        if instance_count > 0:
            self._render_batch(instance_data[:instance_count * 6], instance_count)
        
        # Cleanup
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _render_batch(self, instance_data, instance_count):
        instance_data = np.array(instance_data, dtype=np.float32)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, instance_data.nbytes, instance_data)
        
        glBindVertexArray(self.vao)
        glDrawElementsInstanced(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None, instance_count)
        
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0) 