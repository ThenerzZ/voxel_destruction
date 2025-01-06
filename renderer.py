import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
import pyrr
import os
import ctypes
from lighting import LightingSystem
from atmosphere import AtmosphereSystem
from physics import MaterialType
import traceback

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
        print("\nCompiling shaders...")
        vertex_shader = None
        fragment_shader = None
        
        try:
            # Compile vertex shader
            print("Reading vertex shader source...")
            try:
                with open('shaders/vertex.glsl', 'r') as file:
                    vertex_source = file.read()
                    print("Vertex shader source loaded successfully")
            except Exception as e:
                print(f"Failed to read vertex shader: {e}")
                raise
            
            print("Compiling vertex shader...")
            try:
                vertex_shader = shaders.compileShader(vertex_source, GL_VERTEX_SHADER)
                print("Vertex shader compiled successfully")
            except Exception as e:
                print(f"Vertex shader compilation failed:")
                print(f"Error: {str(e)}")
                raise
            
            # Compile fragment shader
            print("Reading fragment shader source...")
            try:
                with open('shaders/fragment.glsl', 'r') as file:
                    fragment_source = file.read()
                    print("Fragment shader source loaded successfully")
            except Exception as e:
                print(f"Failed to read fragment shader: {e}")
                if vertex_shader:
                    glDeleteShader(vertex_shader)
                raise
            
            print("Compiling fragment shader...")
            try:
                fragment_shader = shaders.compileShader(fragment_source, GL_FRAGMENT_SHADER)
                print("Fragment shader compiled successfully")
            except Exception as e:
                print(f"Fragment shader compilation failed:")
                print(f"Error: {str(e)}")
                if vertex_shader:
                    glDeleteShader(vertex_shader)
                raise
            
            # Link shader program
            print("Linking shader program...")
            try:
                self.shader = shaders.compileProgram(vertex_shader, fragment_shader)
                print("Shader program linked successfully")
            except Exception as e:
                print(f"Shader program linking failed:")
                print(f"Error: {str(e)}")
                if vertex_shader:
                    glDeleteShader(vertex_shader)
                if fragment_shader:
                    glDeleteShader(fragment_shader)
                raise
            
            # Clean up individual shaders
            glDeleteShader(vertex_shader)
            glDeleteShader(fragment_shader)
            
            # Activate shader program
            glUseProgram(self.shader)
            print("Shader program activated successfully\n")
            
        except Exception as e:
            print(f"\nFatal error in shader setup: {e}")
            traceback.print_exc()
            raise
            
    def setup_matrices(self):
        self.projection = pyrr.matrix44.create_perspective_projection(
            70.0, 800.0/600.0, 0.1, 2000.0, dtype=np.float32
        )
        self.view = pyrr.matrix44.create_look_at(
            np.array([64.0, 48.0, 64.0]),
            np.array([64.0 + np.cos(-np.pi/4), 48.0 - 0.5, 64.0 + np.sin(-np.pi/4)]),
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
        try:
            glUseProgram(self.shader)
            
            # Set camera uniforms
            view_loc = glGetUniformLocation(self.shader, "view")
            proj_loc = glGetUniformLocation(self.shader, "projection")
            
            if view_loc != -1:
                glUniformMatrix4fv(view_loc, 1, GL_FALSE, self.view)
            else:
                print("Warning: Could not find view uniform")
                
            if proj_loc != -1:
                glUniformMatrix4fv(proj_loc, 1, GL_FALSE, self.projection)
            else:
                print("Warning: Could not find projection uniform")
            
            # Set view position for specular lighting
            view_pos = pyrr.Vector3([self.view[3][0], self.view[3][1], self.view[3][2]])
            view_pos_loc = glGetUniformLocation(self.shader, "viewPos")
            if view_pos_loc != -1:
                glUniform3fv(view_pos_loc, 1, view_pos)
            else:
                print("Warning: Could not find viewPos uniform")
                
            # Set light position (temporary static light)
            light_pos_loc = glGetUniformLocation(self.shader, "lightPos")
            if light_pos_loc != -1:
                light_pos = pyrr.Vector3([50.0, 100.0, 50.0])  # Light high above the world
                glUniform3fv(light_pos_loc, 1, light_pos)
            else:
                print("Warning: Could not find lightPos uniform")
                
        except Exception as e:
            print(f"Error setting uniforms: {e}")
            traceback.print_exc()
        
    def render_world(self, world):
        try:
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
            
        except Exception as e:
            print(f"Error in render_world: {e}")
            traceback.print_exc()
        
    def _calculate_frustum_planes(self, view_proj):
        """Calculate view frustum planes from view-projection matrix"""
        # Extract planes from view-projection matrix
        m = view_proj
        planes = []
        
        # Left plane
        planes.append(np.array([
            m[0,3] + m[0,0],
            m[1,3] + m[1,0],
            m[2,3] + m[2,0],
            m[3,3] + m[3,0]
        ]))
        
        # Right plane
        planes.append(np.array([
            m[0,3] - m[0,0],
            m[1,3] - m[1,0],
            m[2,3] - m[2,0],
            m[3,3] - m[3,0]
        ]))
        
        # Bottom plane
        planes.append(np.array([
            m[0,3] + m[0,1],
            m[1,3] + m[1,1],
            m[2,3] + m[2,1],
            m[3,3] + m[3,1]
        ]))
        
        # Top plane
        planes.append(np.array([
            m[0,3] - m[0,1],
            m[1,3] - m[1,1],
            m[2,3] - m[2,1],
            m[3,3] - m[3,1]
        ]))
        
        # Near plane
        planes.append(np.array([
            m[0,3] + m[0,2],
            m[1,3] + m[1,2],
            m[2,3] + m[2,2],
            m[3,3] + m[3,2]
        ]))
        
        # Far plane
        planes.append(np.array([
            m[0,3] - m[0,2],
            m[1,3] - m[1,2],
            m[2,3] - m[2,2],
            m[3,3] - m[3,2]
        ]))
        
        # Normalize planes
        normalized_planes = []
        for plane in planes:
            length = np.sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2])
            if length > 0:
                normalized_planes.append(plane / length)
            else:
                normalized_planes.append(plane)
        
        return normalized_planes
        
    def _is_sphere_in_frustum(self, frustum, center, radius):
        """Check if a sphere is inside or intersects the view frustum"""
        for plane in frustum:
            # Calculate signed distance from sphere center to plane
            distance = (
                plane[0] * center[0] +
                plane[1] * center[1] +
                plane[2] * center[2] +
                plane[3]
            )
            
            # If sphere is completely behind any plane, it's outside
            if distance < -radius:
                return False
            
        return True
        
    def _render_static_voxels(self, world):
        """Render the static (non-moving) voxels in the world"""
        try:
            # Pre-allocate arrays for better performance
            max_batch_size = 16384  # Limit batch size for better performance
            instance_data = np.zeros(max_batch_size * 6, dtype=np.float32)
            instance_count = 0
            
            # Calculate view frustum for culling
            view_proj = np.array(self.projection @ self.view)
            frustum = self._calculate_frustum_planes(view_proj)
            
            # Get camera position for distance culling
            camera_pos = np.array([self.view[3][0], self.view[3][1], self.view[3][2]])
            max_render_distance = 300.0  # Increased render distance
            
            # Chunk-based rendering
            chunk_size = 16
            chunks_x = (world.width + chunk_size - 1) // chunk_size
            chunks_z = (world.depth + chunk_size - 1) // chunk_size
            
            for cx in range(chunks_x):
                for cz in range(chunks_z):
                    # Calculate chunk bounds
                    chunk_min_x = cx * chunk_size
                    chunk_min_z = cz * chunk_size
                    chunk_max_x = min((cx + 1) * chunk_size, world.width)
                    chunk_max_z = min((cz + 1) * chunk_size, world.depth)
                    
                    # Check if chunk is in view frustum
                    chunk_center = np.array([
                        (chunk_min_x + chunk_max_x) * 0.5,
                        world.height * 0.5,
                        (chunk_min_z + chunk_max_z) * 0.5
                    ])
                    chunk_radius = np.sqrt(chunk_size * chunk_size * 2 + world.height * world.height) * 0.5
                    
                    # Skip frustum culling for nearby chunks
                    chunk_dist = np.linalg.norm(chunk_center - camera_pos)
                    if chunk_dist > 32.0 and not self._is_sphere_in_frustum(frustum, chunk_center, chunk_radius):
                        continue
                    
                    # Check chunk distance from camera
                    if chunk_dist > max_render_distance + chunk_radius:
                        continue
                    
                    # Render voxels in this chunk
                    for x in range(chunk_min_x, chunk_max_x):
                        for z in range(chunk_min_z, chunk_max_z):
                            for y in range(world.height):
                                voxel = world.get_voxel_data(x, y, z)
                                if voxel and voxel.material_type != MaterialType.AIR:
                                    # Quick distance check for individual voxels
                                    voxel_pos = np.array([x, y, z])
                                    if np.linalg.norm(voxel_pos - camera_pos) > max_render_distance:
                                        continue
                                        
                                    # Check each face
                                    for face_idx, (dx, dy, dz) in enumerate(self.face_directions):
                                        if self.should_render_face(world, x, y, z, dx, dy, dz):
                                            # Calculate offset into instance_data array
                                            offset = instance_count * 6
                                            
                                            # Position
                                            instance_data[offset:offset+3] = [x, y, z]
                                            
                                            # Get base color for material
                                            base_color = self.material_colors[voxel.material_type][:3]
                                            
                                            # Apply damage state modifier
                                            damage_modifier = self.damage_modifiers[voxel.damage_state]
                                            
                                            # Final color
                                            color = base_color * damage_modifier
                                            instance_data[offset+3:offset+6] = color
                                            
                                            instance_count += 1
                                            
                                            # Render batch if full
                                            if instance_count >= max_batch_size:
                                                self._render_batch(instance_data, instance_count)
                                                instance_count = 0
                
            # Render remaining instances
            if instance_count > 0:
                self._render_batch(instance_data[:instance_count * 6], instance_count)
                
        except Exception as e:
            print(f"Error in static voxel rendering: {e}")
            traceback.print_exc()
        
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
        try:
            # Only print batch info occasionally for debugging
            if instance_count > 100:  # Only print for larger batches
                print(f"Rendering batch: {instance_count} instances")
                
            glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
            glBufferSubData(GL_ARRAY_BUFFER, 0, instance_data.nbytes, instance_data)
            
            glBindVertexArray(self.vao)
            glDrawElementsInstanced(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None, instance_count)
            glBindVertexArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            
            if instance_count > 100:  # Only print for larger batches
                print("Batch rendering complete")
            
        except Exception as e:
            print(f"Error in batch rendering: {e}")
            traceback.print_exc()
        
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