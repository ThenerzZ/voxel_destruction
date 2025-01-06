import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
import pyrr
import noise
from renderer import VoxelRenderer
import time
from physics import PhysicsSystem
from camera import Camera

class VoxelWorld:
    def __init__(self, width=32, height=32, depth=32):
        self.width = width
        self.height = height
        self.depth = depth
        self.voxels = np.zeros((width, height, depth), dtype=np.int32)
        self.physics = PhysicsSystem(self)
        self.generate_terrain()
        
    def generate_terrain(self):
        scale = 20.0
        for x in range(self.width):
            for z in range(self.depth):
                height = int(noise.pnoise2(x/scale, 
                                       z/scale, 
                                       octaves=3, 
                                       persistence=0.5) * 10 + self.height/2)
                for y in range(height):
                    self.voxels[x, y, z] = 1  # 1 represents solid block
                    
    def is_valid_position(self, x, y, z):
        """Check if a position is within world bounds"""
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                0 <= z < self.depth)
                    
    def destroy_voxel(self, x, y, z):
        """Destroy a voxel and handle physics"""
        if self.is_valid_position(x, y, z):
            self.physics.destroy_voxel(x, y, z)
            
    def update(self, dt):
        """Update world physics"""
        self.physics.update(dt)

class Game:
    def __init__(self, width=800, height=600):
        if not glfw.init():
            raise Exception("GLFW initialization failed")
            
        # Configure GLFW window
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            
        self.window = glfw.create_window(width, height, "Voxel Destruction Sandbox", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Window creation failed")
            
        glfw.make_context_current(self.window)
        
        # Set up input handling
        glfw.set_key_callback(self.window, self.key_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_callback)
        
        # Capture mouse cursor
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        
        self.world = VoxelWorld()
        self.renderer = VoxelRenderer()
        self.camera = Camera(position=np.array([16.0, 16.0, 32.0]))  # Start with a better view
        
        self.last_frame = glfw.get_time()
        self.delta_time = 0.0
        
        self.setup_gl()
        
    def setup_gl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glFrontFace(GL_CCW)
        glClearColor(0.5, 0.7, 1.0, 1.0)
        
    def cursor_callback(self, window, xpos, ypos):
        """Handle mouse movement for camera control"""
        self.camera.handle_mouse_movement(xpos, ypos)
        
    def key_callback(self, window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        else:
            self.camera.handle_keyboard(key, action)
            
    def mouse_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            # Ray cast from camera and destroy voxel
            hit, pos = self.ray_cast_from_camera()
            if hit:
                x, y, z = map(int, pos)
                self.world.destroy_voxel(x, y, z)
                
    def ray_cast_from_camera(self):
        """Cast a ray from camera position in view direction"""
        ray_start = self.camera.position
        ray_dir = self.camera.front
        
        # Simple ray marching
        step = 0.1
        max_dist = 50.0
        current_dist = 0.0
        
        while current_dist < max_dist:
            pos = ray_start + ray_dir * current_dist
            x, y, z = map(int, pos)
            
            if self.world.is_valid_position(x, y, z):
                if self.world.voxels[x, y, z] == 1:
                    return True, pos
                    
            current_dist += step
            
        return False, None
            
    def run(self):
        while not glfw.window_should_close(self.window):
            current_frame = glfw.get_time()
            self.delta_time = current_frame - self.last_frame
            self.last_frame = current_frame
            
            # Process input and update camera
            self.camera.update(self.delta_time)
            
            # Update physics
            self.world.update(self.delta_time)
            
            # Update view matrix
            self.renderer.view = self.camera.get_view_matrix()
            
            # Clear buffers
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Render
            self.renderer.render_world(self.world)
            
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            
        glfw.terminate()

if __name__ == "__main__":
    game = Game()
    game.run() 