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
        self.max_instances = 32768
        self.instance_buffer_size = self.max_instances * 6 * 4  # 6 floats per instance, 4 bytes per float
        
        # Initialize other properties
        self.face_directions = [
            ( 0,  0,  1),  # Front  (+Z)
            ( 0,  0, -1),  # Back   (-Z)
            ( 0,  1,  0),  # Top    (+Y)
            ( 0, -1,  0),  # Bottom (-Y)
            ( 1,  0,  0),  # Right  (+X)
            (-1,  0,  0),  # Left   (-X)
        ]
        
        # Initialize systems
        self.lighting_system = LightingSystem()
        self.atmosphere_system = AtmosphereSystem()
        
        # Setup shaders first
        self.setup_shaders()
        
        # Create and bind VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        
        # Create vertex buffer for cube geometry
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        
        # Define cube vertices with position, normal, and UV
        vertices = np.array([
            # Front face
            -0.5, -0.5,  0.5,   0.0,  0.0,  1.0,   0.0, 0.0,  # Bottom-left
             0.5, -0.5,  0.5,   0.0,  0.0,  1.0,   1.0, 0.0,  # Bottom-right
             0.5,  0.5,  0.5,   0.0,  0.0,  1.0,   1.0, 1.0,  # Top-right
            -0.5,  0.5,  0.5,   0.0,  0.0,  1.0,   0.0, 1.0,  # Top-left
            
            # Back face
            -0.5, -0.5, -0.5,   0.0,  0.0, -1.0,   1.0, 0.0,
             0.5, -0.5, -0.5,   0.0,  0.0, -1.0,   0.0, 0.0,
             0.5,  0.5, -0.5,   0.0,  0.0, -1.0,   0.0, 1.0,
            -0.5,  0.5, -0.5,   0.0,  0.0, -1.0,   1.0, 1.0,
            
            # Top face
            -0.5,  0.5, -0.5,   0.0,  1.0,  0.0,   0.0, 1.0,
             0.5,  0.5, -0.5,   0.0,  1.0,  0.0,   1.0, 1.0,
             0.5,  0.5,  0.5,   0.0,  1.0,  0.0,   1.0, 0.0,
            -0.5,  0.5,  0.5,   0.0,  1.0,  0.0,   0.0, 0.0,
            
            # Bottom face
            -0.5, -0.5, -0.5,   0.0, -1.0,  0.0,   0.0, 0.0,
             0.5, -0.5, -0.5,   0.0, -1.0,  0.0,   1.0, 0.0,
             0.5, -0.5,  0.5,   0.0, -1.0,  0.0,   1.0, 1.0,
            -0.5, -0.5,  0.5,   0.0, -1.0,  0.0,   0.0, 1.0,
            
            # Right face
             0.5, -0.5, -0.5,   1.0,  0.0,  0.0,   1.0, 0.0,
             0.5,  0.5, -0.5,   1.0,  0.0,  0.0,   1.0, 1.0,
             0.5,  0.5,  0.5,   1.0,  0.0,  0.0,   0.0, 1.0,
             0.5, -0.5,  0.5,   1.0,  0.0,  0.0,   0.0, 0.0,
            
            # Left face
            -0.5, -0.5, -0.5,  -1.0,  0.0,  0.0,   0.0, 0.0,
            -0.5,  0.5, -0.5,  -1.0,  0.0,  0.0,   0.0, 1.0,
            -0.5,  0.5,  0.5,  -1.0,  0.0,  0.0,   1.0, 1.0,
            -0.5, -0.5,  0.5,  -1.0,  0.0,  0.0,   1.0, 0.0,
        ], dtype=np.float32)
        
        # Upload vertex data
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        # Set up vertex attributes for cube geometry
        glEnableVertexAttribArray(0)  # position
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        
        glEnableVertexAttribArray(1)  # normal
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        
        glEnableVertexAttribArray(2)  # uv
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(24))
        
        # Create element buffer for cube indices
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        
        # Define cube indices
        indices = np.array([
            0,  1,  2,  2,  3,  0,  # Front
            4,  5,  6,  6,  7,  4,  # Back
            8,  9,  10, 10, 11, 8,  # Top
            12, 13, 14, 14, 15, 12, # Bottom
            16, 17, 18, 18, 19, 16, # Right
            20, 21, 22, 22, 23, 20  # Left
        ], dtype=np.uint32)
        
        # Upload index data
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Create and initialize instance buffer
        self.instance_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.instance_buffer_size, None, GL_DYNAMIC_DRAW)
        
        # Set up instance attributes
        glEnableVertexAttribArray(3)  # instance position
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glVertexAttribDivisor(3, 1)
        
        glEnableVertexAttribArray(4)  # instance color
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glVertexAttribDivisor(4, 1)
        
        # Unbind buffers
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        
        # Setup matrices
        self.setup_matrices()
        
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
        
        # Face visibility lookup tables
        self.face_masks = {
            'top': np.array([0, 1, 0]),     # +Y
            'bottom': np.array([0, -1, 0]),  # -Y
            'front': np.array([0, 0, 1]),    # +Z
            'back': np.array([0, 0, -1]),    # -Z
            'right': np.array([1, 0, 0]),    # +X
            'left': np.array([-1, 0, 0]),    # -X
        }
        
        # Lookup table for face indices
        self.face_indices = {
            'top': 0,
            'bottom': 1,
            'front': 2,
            'back': 3,
            'right': 4,
            'left': 5
        }
        
        # Occlusion rules for different materials
        self.occlusion_rules = {
            MaterialType.AIR: lambda x: True,  # Air never occludes
            MaterialType.GLASS: lambda x: False,  # Glass doesn't occlude same material
            MaterialType.CONCRETE: lambda x: True,  # Concrete always occludes
            MaterialType.WOOD: lambda x: True,  # Wood always occludes
            MaterialType.METAL: lambda x: True,  # Metal always occludes
        }
        
    def should_render_face(self, world, x, y, z, dx, dy, dz):
        """Determine if a face should be rendered based on neighbor voxels"""
        # Check if neighbor position is within world bounds
        nx, ny, nz = x + dx, y + dy, z + dz
        if not (0 <= nx < world.width and 0 <= ny < world.height and 0 <= nz < world.depth):
            return True  # Always render faces at world boundaries
            
        # Get neighbor voxel
        neighbor = world.get_voxel_data(nx, ny, nz)
        
        # Render face if neighbor is air or transparent
        return (neighbor is None or 
                neighbor.material_type == MaterialType.AIR or 
                neighbor.material_type == MaterialType.GLASS)
        
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
            
            # Enable depth testing and face culling
            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LEQUAL)  # Changed from GL_LESS to handle coplanar faces better
            
            # Only cull back faces when they're not visible
            glEnable(GL_CULL_FACE)
            glCullFace(GL_BACK)
            glFrontFace(GL_CCW)  # Counter-clockwise winding
            
            # Enable blending for transparent materials
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            # Update and set uniforms
            self.set_uniforms()
            
            # Render static world voxels
            self._render_static_voxels(world)
            
            # Only render physics objects if they exist
            if world.physics.chunks:
                self._render_chunks(world)
            if world.physics.fragments:
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
        
    def _check_face_visibility(self, world, chunk_cache, x, y, z, face_dir, current_material):
        """
        Sophisticated face visibility check using material-specific rules and chunk caching
        Returns: (is_visible, should_merge)
        """
        nx, ny, nz = x + face_dir[0], y + face_dir[1], z + face_dir[2]
        
        # Check world bounds
        if not (0 <= nx < world.width and 0 <= ny < world.height and 0 <= nz < world.depth):
            return True, False  # Visible but can't merge at world edges
        
        # Get neighbor from cache if possible
        chunk_x, chunk_z = nx // 16, nz // 16
        cache_key = (chunk_x, chunk_z)
        
        if cache_key in chunk_cache:
            neighbor = chunk_cache[cache_key].get((nx % 16, ny, nz % 16))
        else:
            # Cache miss - load chunk data
            chunk_data = {}
            cx_start = chunk_x * 16
            cz_start = chunk_z * 16
            for local_x in range(16):
                for local_z in range(16):
                    for local_y in range(world.height):
                        world_x = cx_start + local_x
                        world_z = cz_start + local_z
                        if 0 <= world_x < world.width and 0 <= world_z < world.depth:
                            voxel = world.get_voxel_data(world_x, local_y, world_z)
                            if voxel and voxel.material_type != MaterialType.AIR:
                                chunk_data[(local_x, local_y, local_z)] = voxel
            chunk_cache[cache_key] = chunk_data
            neighbor = chunk_data.get((nx % 16, ny, nz % 16))
        
        if not neighbor or neighbor.material_type == MaterialType.AIR:
            return True, False  # Visible but can't merge with air
            
        # Check material-specific occlusion rules
        if current_material == MaterialType.GLASS:
            if neighbor.material_type == MaterialType.GLASS:
                # Special case: only show glass-glass interfaces if damage states differ
                return neighbor.damage_state != current_material.damage_state, True
            return True, False  # Glass is visible next to non-glass
            
        # Apply material-specific occlusion rules
        occludes = self.occlusion_rules[neighbor.material_type](current_material)
        can_merge = (neighbor.material_type == current_material.material_type and 
                    neighbor.damage_state == current_material.damage_state)
        
        return not occludes, can_merge

    def _render_static_voxels(self, world):
        """Render the static (non-moving) voxels in the world with optimized face culling"""
        try:
            instance_data = np.zeros(self.max_instances * 6, dtype=np.float32)
            instance_count = 0
            
            # Setup view frustum culling
            view_proj = np.array(self.projection @ self.view)
            frustum = self._calculate_frustum_planes(view_proj)
            camera_pos = np.array([self.view[3][0], self.view[3][1], self.view[3][2]])
            
            # Chunk processing setup
            chunk_size = 16
            camera_chunk_x = int(camera_pos[0] / chunk_size)
            camera_chunk_z = int(camera_pos[2] / chunk_size)
            max_render_distance = 300.0
            min_render_distance = 32.0
            
            # Cache for chunk data to reduce world queries
            chunk_cache = {}
            
            # Process chunks in spiral order
            chunks_to_render = self._get_chunks_to_render(world, camera_chunk_x, camera_chunk_z, 
                                                        max_render_distance, chunk_size)
            
            for cx, cz, dist in chunks_to_render:
                chunk_min_x = cx * chunk_size
                chunk_max_x = min((cx + 1) * chunk_size, world.width)
                chunk_min_z = cz * chunk_size
                chunk_max_z = min((cz + 1) * chunk_size, world.depth)
                
                # Frustum culling for distant chunks
                if dist > min_render_distance:
                    chunk_center = np.array([
                        (chunk_min_x + chunk_max_x) * 0.5,
                        world.height * 0.5,
                        (chunk_min_z + chunk_max_z) * 0.5
                    ])
                    if not self._is_sphere_in_frustum(frustum, chunk_center, 
                            np.sqrt(chunk_size * chunk_size * 2 + world.height * world.height) * 0.5):
                        continue
                
                # Process voxels in optimal order
                for y in range(world.height - 1, -1, -1):  # Top to bottom
                    for x in range(chunk_min_x, chunk_max_x):
                        for z in range(chunk_min_z, chunk_max_z):
                            voxel = world.get_voxel_data(x, y, z)
                            if not voxel or voxel.material_type == MaterialType.AIR:
                                continue
                                
                            # Check each face with material-specific rules
                            for face_name, face_dir in self.face_masks.items():
                                is_visible, can_merge = self._check_face_visibility(
                                    world, chunk_cache, x, y, z, face_dir, voxel)
                                    
                                if is_visible and not can_merge:
                                    if instance_count >= self.max_instances:
                                        self._render_batch(instance_data, instance_count)
                                        instance_count = 0
                                    
                                    offset = instance_count * 6
                                    instance_data[offset:offset+3] = [x, y, z]
                                    
                                    base_color = self.material_colors[voxel.material_type][:3]
                                    damage_modifier = self.damage_modifiers[voxel.damage_state]
                                    instance_data[offset+3:offset+6] = base_color * damage_modifier
                                    
                                    instance_count += 1
                
                # Flush batch if needed
                if instance_count >= self.max_instances * 0.75:
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
            if instance_count > 0:
                # Ensure we don't exceed buffer size
                if instance_count > self.max_instances:
                    instance_count = self.max_instances
                    instance_data = instance_data[:self.max_instances * 6]
                
                data_size = instance_count * 6 * 4  # 6 floats per instance, 4 bytes per float
                
                glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
                glBufferSubData(GL_ARRAY_BUFFER, 0, data_size, instance_data)
                
                glBindVertexArray(self.vao)
                glDrawElementsInstanced(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None, instance_count)
                glBindVertexArray(0)
                glBindBuffer(GL_ARRAY_BUFFER, 0)
                
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

    def _get_chunks_to_render(self, world, camera_chunk_x, camera_chunk_z, max_render_distance, chunk_size):
        """Get chunks to render in a spiral pattern starting from the camera position"""
        chunks = []
        max_chunk_radius = int(max_render_distance / chunk_size)
        
        # Start at camera position
        x, z = camera_chunk_x, camera_chunk_z
        dx, dz = 0, -1  # Initial direction: up
        radius = 0  # Current radius of spiral
        steps = 0   # Steps taken in current direction
        
        # Generate spiral pattern
        while radius <= max_chunk_radius:
            # Calculate chunk position and distance
            chunk_center_x = (x + 0.5) * chunk_size
            chunk_center_z = (z + 0.5) * chunk_size
            dist = ((chunk_center_x - camera_chunk_x * chunk_size) ** 2 + 
                   (chunk_center_z - camera_chunk_z * chunk_size) ** 2) ** 0.5
            
            # Add chunk if within world bounds and render distance
            if (0 <= x * chunk_size < world.width and 
                0 <= z * chunk_size < world.depth and 
                dist <= max_render_distance):
                chunks.append((x, z, dist))
            
            # Take a step in current direction
            x += dx
            z += dz
            steps += 1
            
            # Check if we need to change direction
            if steps == radius:
                steps = 0
                # Rotate 90 degrees clockwise
                dx, dz = -dz, dx
                # Increase radius when completing half a circle
                if dx == 0:
                    radius += 1
        
        # Sort chunks by distance for better rendering
        chunks.sort(key=lambda c: c[2])
        return chunks 