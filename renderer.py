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
        
        # Initialize lighting and atmosphere systems
        self.lighting_system = LightingSystem()
        self.atmosphere_system = AtmosphereSystem()
        
        # Setup OpenGL resources
        self.setup_shaders()
        self.setup_matrices()
        self.setup_cube()
        
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
            -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  0.0, 0.0,
             0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  1.0, 0.0,
             0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  1.0, 1.0,
            -0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  0.0, 1.0,

            # Back face (-Z)
             0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  0.0, 0.0,
            -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  1.0, 0.0,
            -0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  1.0, 1.0,
             0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  0.0, 1.0,

            # Top face (+Y)
            -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  0.0, 0.0,
            -0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  0.0, 1.0,
             0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  1.0, 1.0,
             0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  1.0, 0.0,

            # Bottom face (-Y)
            -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  0.0, 0.0,
             0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  1.0, 0.0,
             0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  1.0, 1.0,
            -0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  0.0, 1.0,

            # Right face (+X)
             0.5, -0.5, -0.5,  1.0,  0.0,  0.0,  0.0, 0.0,
             0.5, -0.5,  0.5,  1.0,  0.0,  0.0,  1.0, 0.0,
             0.5,  0.5,  0.5,  1.0,  0.0,  0.0,  1.0, 1.0,
             0.5,  0.5, -0.5,  1.0,  0.0,  0.0,  0.0, 1.0,

            # Left face (-X)
            -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,  0.0, 0.0,
            -0.5, -0.5,  0.5, -1.0,  0.0,  0.0,  1.0, 0.0,
            -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,  1.0, 1.0,
            -0.5,  0.5, -0.5, -1.0,  0.0,  0.0,  0.0, 1.0,
        ], dtype=np.float32)

        indices = np.array([
            0,  1,  2,    0,  2,  3,   # Front
            4,  5,  6,    4,  6,  7,   # Back
            8,  9,  10,   8,  10, 11,  # Top
            12, 13, 14,   12, 14, 15,  # Bottom
            16, 17, 18,   16, 18, 19,  # Right
            20, 21, 22,   20, 22, 23   # Left
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
        
        # Each instance needs: position (3), color (3), face index (1)
        instance_stride = (3 + 3 + 1) * 4  # 7 floats * 4 bytes
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
        
        # Instance face index attribute (location = 5)
        glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, instance_stride, ctypes.c_void_p(24))
        glEnableVertexAttribArray(5)
        glVertexAttribDivisor(5, 1)
        
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
        
        # Set lighting uniforms
        lighting_uniforms = self.lighting_system.get_lighting_uniforms()
        for name, value in lighting_uniforms.items():
            loc = glGetUniformLocation(self.shader, name)
            if loc != -1:  # Only set if uniform exists in shader
                try:
                    if isinstance(value, np.ndarray):
                        if value.size == 3:
                            glUniform3fv(loc, 1, value)
                        elif value.size == 16:
                            glUniformMatrix4fv(loc, 1, GL_FALSE, value)
                    elif isinstance(value, (int, float)):
                        glUniform1f(loc, float(value))
                except Exception as e:
                    print(f"Error setting uniform {name}: {e}")
                
        # Set atmosphere uniforms
        atmosphere_uniforms = self.atmosphere_system.get_atmosphere_uniforms()
        for name, value in atmosphere_uniforms.items():
            loc = glGetUniformLocation(self.shader, name)
            if loc != -1:  # Only set if uniform exists in shader
                try:
                    if isinstance(value, np.ndarray):
                        if value.size == 3:
                            glUniform3fv(loc, 1, value)
                    elif isinstance(value, (int, float)):
                        glUniform1f(loc, float(value))
                except Exception as e:
                    print(f"Error setting uniform {name}: {e}")
        
    def render_world(self, world):
        glUseProgram(self.shader)
        
        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        
        # Update and set uniforms
        self.set_uniforms()
        
        # Collect visible voxels
        instance_data = []
        instance_count = 0
        
        # Face colors for different orientations
        face_colors = [
            [0.8, 0.8, 0.8],  # Front
            [0.7, 0.7, 0.7],  # Back
            [0.9, 0.9, 0.9],  # Top
            [0.6, 0.6, 0.6],  # Bottom
            [0.75, 0.75, 0.75],  # Right
            [0.7, 0.7, 0.7],  # Left
        ]
        
        for x in range(world.width):
            for y in range(world.height):
                for z in range(world.depth):
                    if world.voxels[x, y, z] == 1:
                        # Add each face with its position, color, and face index
                        for face_idx, color in enumerate(face_colors):
                            instance_data.extend([x, y, z] + color + [float(face_idx)])
                            instance_count += 1
                            
                            if instance_count >= self.max_instances:
                                self._render_batch(instance_data, instance_count)
                                instance_data = []
                                instance_count = 0
        
        if instance_count > 0:
            self._render_batch(instance_data, instance_count)
            
    def _render_batch(self, instance_data, instance_count):
        instance_data = np.array(instance_data, dtype=np.float32)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, instance_data.nbytes, instance_data)
        
        glBindVertexArray(self.vao)
        glDrawElementsInstanced(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None, instance_count)
        
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0) 