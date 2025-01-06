import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
import pyrr
import noise
from renderer import VoxelRenderer
import time

class Camera:
    def __init__(self, position=None):
        self.position = position if position is not None else np.array([8.0, 12.0, 8.0], dtype=np.float32)
        self.front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        
        self.yaw = -90.0
        self.pitch = 0.0
        
        self.movement_speed = 5.0
        self.mouse_sensitivity = 0.1
        
        self.last_x = 400
        self.last_y = 300
        self.first_mouse = True
        
    def get_view_matrix(self):
        return pyrr.matrix44.create_look_at(
            self.position,
            self.position + self.front,
            self.up,
            dtype=np.float32
        )
        
    def process_keyboard(self, direction, delta_time):
        velocity = self.movement_speed * delta_time
        if direction == "FORWARD":
            self.position += self.front * velocity
        if direction == "BACKWARD":
            self.position -= self.front * velocity
        if direction == "LEFT":
            self.position -= self.right * velocity
        if direction == "RIGHT":
            self.position += self.right * velocity
        if direction == "UP":
            self.position += self.up * velocity
        if direction == "DOWN":
            self.position -= self.up * velocity
            
    def process_mouse_movement(self, xoffset, yoffset, constrain_pitch=True):
        xoffset *= self.mouse_sensitivity
        yoffset *= self.mouse_sensitivity
        
        self.yaw += xoffset
        self.pitch += yoffset
        
        if constrain_pitch:
            if self.pitch > 89.0:
                self.pitch = 89.0
            if self.pitch < -89.0:
                self.pitch = -89.0
                
        self.update_camera_vectors()
        
    def update_camera_vectors(self):
        front = np.array([
            np.cos(np.radians(self.yaw)) * np.cos(np.radians(self.pitch)),
            np.sin(np.radians(self.pitch)),
            np.sin(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
        ])
        self.front = front / np.linalg.norm(front)
        self.right = np.cross(self.front, np.array([0.0, 1.0, 0.0])) / np.linalg.norm(np.cross(self.front, np.array([0.0, 1.0, 0.0])))
        self.up = np.cross(self.right, self.front) / np.linalg.norm(np.cross(self.right, self.front))

class VoxelWorld:
    def __init__(self, width=32, height=32, depth=32):
        self.width = width
        self.height = height
        self.depth = depth
        self.voxels = np.zeros((width, height, depth), dtype=np.int32)
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
                    
    def destroy_voxel(self, x, y, z):
        if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth:
            self.voxels[x, y, z] = 0

class Game:
    def __init__(self, width=800, height=600):
        if not glfw.init():
            raise Exception("GLFW initialization failed")
            
        # Configure GLFW
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, 4)  # Enable MSAA
            
        self.window = glfw.create_window(width, height, "Voxel Destruction Sandbox", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Window creation failed")
            
        glfw.make_context_current(self.window)
        
        # Set callbacks
        glfw.set_key_callback(self.window, self.key_callback)
        glfw.set_cursor_pos_callback(self.window, self.mouse_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback)
        
        # Capture mouse
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        
        self.camera = Camera()
        self.world = VoxelWorld(32, 32, 32)  # Larger world
        self.setup_gl()
        self.renderer = VoxelRenderer()
        
        # Timing
        self.delta_time = 0.0
        self.last_frame = 0.0
        
        # Input state
        self.keys = {}
        
    def setup_gl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glEnable(GL_MULTISAMPLE)
        glClearColor(0.5, 0.7, 1.0, 1.0)
        
    def framebuffer_size_callback(self, window, width, height):
        glViewport(0, 0, width, height)
        
    def key_callback(self, window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
            
        if action == glfw.PRESS:
            self.keys[key] = True
        elif action == glfw.RELEASE:
            self.keys[key] = False
            
    def process_input(self):
        if glfw.KEY_W in self.keys and self.keys[glfw.KEY_W]:
            self.camera.process_keyboard("FORWARD", self.delta_time)
        if glfw.KEY_S in self.keys and self.keys[glfw.KEY_S]:
            self.camera.process_keyboard("BACKWARD", self.delta_time)
        if glfw.KEY_A in self.keys and self.keys[glfw.KEY_A]:
            self.camera.process_keyboard("LEFT", self.delta_time)
        if glfw.KEY_D in self.keys and self.keys[glfw.KEY_D]:
            self.camera.process_keyboard("RIGHT", self.delta_time)
        if glfw.KEY_SPACE in self.keys and self.keys[glfw.KEY_SPACE]:
            self.camera.process_keyboard("UP", self.delta_time)
        if glfw.KEY_LEFT_SHIFT in self.keys and self.keys[glfw.KEY_LEFT_SHIFT]:
            self.camera.process_keyboard("DOWN", self.delta_time)
            
    def mouse_callback(self, window, xpos, ypos):
        if self.camera.first_mouse:
            self.camera.last_x = xpos
            self.camera.last_y = ypos
            self.camera.first_mouse = False
            
        xoffset = xpos - self.camera.last_x
        yoffset = self.camera.last_y - ypos
        
        self.camera.last_x = xpos
        self.camera.last_y = ypos
        
        self.camera.process_mouse_movement(xoffset, yoffset)
            
    def mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            # TODO: Implement ray casting for voxel destruction
            # For now, just destroy a random voxel for testing
            x = np.random.randint(0, self.world.width)
            y = np.random.randint(0, self.world.height)
            z = np.random.randint(0, self.world.depth)
            self.world.destroy_voxel(x, y, z)
            
    def run(self):
        while not glfw.window_should_close(self.window):
            current_frame = glfw.get_time()
            self.delta_time = current_frame - self.last_frame
            self.last_frame = current_frame
            
            self.process_input()
            
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Update view matrix based on camera
            self.renderer.view = self.camera.get_view_matrix()
            
            # Render the world
            self.renderer.render_world(self.world)
            
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            
        glfw.terminate()

if __name__ == "__main__":
    game = Game()
    game.run() 