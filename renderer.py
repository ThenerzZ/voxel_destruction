import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
import pyrr
import os
import ctypes

class VoxelRenderer:
    def __init__(self):
        self.setup_shaders()
        self.setup_cube()
        self.setup_matrices()
        self.max_instances = 16384  # Maximum number of instances to render at once
        self.setup_instance_buffers()
        
    def setup_shaders(self):
        with open('shaders/vertex.glsl', 'r') as file:
            vertex_shader = shaders.compileShader(file.read(), GL_VERTEX_SHADER)
        with open('shaders/fragment.glsl', 'r') as file:
            fragment_shader = shaders.compileShader(file.read(), GL_FRAGMENT_SHADER)
            
        self.shader = shaders.compileProgram(vertex_shader, fragment_shader)
        glUseProgram(self.shader)
        
    def setup_cube(self):
        # Define cube vertices with normals and texture coordinates
        vertices = np.array([
            # Front face
            -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  0.0, 0.0,
             0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  1.0, 0.0,
             0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  1.0, 1.0,
            -0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  0.0, 1.0,
            # Back face
            -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  1.0, 0.0,
             0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  0.0, 0.0,
             0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  0.0, 1.0,
            -0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  1.0, 1.0,
            # Right face
             0.5, -0.5, -0.5,  1.0,  0.0,  0.0,  0.0, 0.0,
             0.5,  0.5, -0.5,  1.0,  0.0,  0.0,  1.0, 0.0,
             0.5,  0.5,  0.5,  1.0,  0.0,  0.0,  1.0, 1.0,
             0.5, -0.5,  0.5,  1.0,  0.0,  0.0,  0.0, 1.0,
            # Left face
            -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,  1.0, 0.0,
            -0.5,  0.5, -0.5, -1.0,  0.0,  0.0,  0.0, 0.0,
            -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,  0.0, 1.0,
            -0.5, -0.5,  0.5, -1.0,  0.0,  0.0,  1.0, 1.0,
            # Top face
            -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  0.0, 0.0,
             0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  1.0, 0.0,
             0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  1.0, 1.0,
            -0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  0.0, 1.0,
            # Bottom face
            -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  0.0, 0.0,
             0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  1.0, 0.0,
             0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  1.0, 1.0,
            -0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  0.0, 1.0,
        ], dtype=np.float32)
        
        indices = np.array([
            0,  1,  2,  2,  3,  0,  # Front
            4,  5,  6,  6,  7,  4,  # Back
            8,  9,  10, 10, 11, 8,  # Right
            12, 13, 14, 14, 15, 12, # Left
            16, 17, 18, 18, 19, 16, # Top
            20, 21, 22, 22, 23, 20  # Bottom
        ], dtype=np.uint32)
        
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        
        # Vertex buffer
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        # Element buffer
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Normal attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        
        # Texture coordinate attribute
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(24))
        glEnableVertexAttribArray(2)
        
    def setup_instance_buffers(self):
        # Create and bind the instance VBO
        self.instance_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
        
        # Allocate buffer for instance data (position + color for each instance)
        buffer_size = self.max_instances * 6 * 4  # 6 floats per instance (3 for position, 3 for color) * 4 bytes per float
        glBufferData(GL_ARRAY_BUFFER, buffer_size, None, GL_DYNAMIC_DRAW)
        
        # Instance position attribute (location = 3)
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glVertexAttribDivisor(3, 1)
        
        # Instance color attribute (location = 4)
        glEnableVertexAttribArray(4)
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glVertexAttribDivisor(4, 1)
        
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
        
    def render_world(self, world):
        glUseProgram(self.shader)
        
        # Set uniforms
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader, "view"),
            1, GL_FALSE, self.view
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader, "projection"),
            1, GL_FALSE, self.projection
        )
        glUniform3fv(
            glGetUniformLocation(self.shader, "lightPos"),
            1, np.array([50.0, 50.0, 50.0], dtype=np.float32)
        )
        glUniform3fv(
            glGetUniformLocation(self.shader, "viewPos"),
            1, np.array([0.0, 0.0, 0.0], dtype=np.float32)
        )
        
        # Collect visible voxels
        instance_data = []
        instance_count = 0
        
        for x in range(world.width):
            for y in range(world.height):
                for z in range(world.depth):
                    if world.voxels[x, y, z] == 1:
                        # Add position (3 floats) and color (3 floats)
                        instance_data.extend([x, y, z, 0.8, 0.8, 0.8])
                        instance_count += 1
                        
                        if instance_count >= self.max_instances:
                            self._render_batch(instance_data, instance_count)
                            instance_data = []
                            instance_count = 0
        
        if instance_count > 0:
            self._render_batch(instance_data, instance_count)
            
    def _render_batch(self, instance_data, instance_count):
        # Convert instance data to numpy array
        instance_data = np.array(instance_data, dtype=np.float32)
        
        # Update instance buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, instance_data.nbytes, instance_data)
        
        # Draw instances
        glBindVertexArray(self.vao)
        glDrawElementsInstanced(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None, instance_count) 