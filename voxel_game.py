import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
import pyrr
import noise
from renderer import VoxelRenderer
import time
from physics import PhysicsSystem, MaterialType, VoxelData
from camera import Camera
from world_gen import WorldGenerator
import sys
import traceback
import signal
import faulthandler

# Enable faulthandler to help debug crashes
faulthandler.enable()

# Set higher recursion limit for world generation
sys.setrecursionlimit(10000)

def signal_handler(signum, frame):
    print("\n" + "="*60)
    print(f"Received signal {signum}")
    print("Current stack trace:")
    print("-"*60)
    traceback.print_stack(frame)
    print("="*60 + "\n")
    sys.exit(1)

# Register signal handlers
signal.signal(signal.SIGABRT, signal_handler)
signal.signal(signal.SIGSEGV, signal_handler)
signal.signal(signal.SIGILL, signal_handler)
signal.signal(signal.SIGFPE, signal_handler)

class VoxelWorld:
    def __init__(self, width=32, height=16, depth=32, seed=None):
        try:
            print(f"Creating VoxelWorld with dimensions: {width}x{height}x{depth}")
            self.width = width
            self.height = height
            self.depth = depth
            
            # Initialize all voxels as air
            print("Initializing voxel array...")
            try:
                self.voxels = np.full((width, height, depth), VoxelData(MaterialType.AIR), dtype=object)
                print("Voxel array initialized successfully")
            except Exception as e:
                print(f"Error initializing voxel array: {e}")
                traceback.print_exc()
                raise
            
            print("Creating physics system...")
            try:
                self.physics = PhysicsSystem(self)
                print("Physics system created successfully")
            except Exception as e:
                print(f"Error creating physics system: {e}")
                traceback.print_exc()
                raise
            
            print("Creating world generator...")
            try:
                self.generator = WorldGenerator(seed)
                print("World generator created successfully")
            except Exception as e:
                print(f"Error creating world generator: {e}")
                traceback.print_exc()
                raise
            
            print("Starting world generation...")
            try:
                self.generate_world()
                print("World generation completed successfully")
            except Exception as e:
                print(f"Error during world generation: {e}")
                traceback.print_exc()
                raise
                
        except Exception as e:
            print(f"Fatal error in VoxelWorld initialization: {e}")
            traceback.print_exc()
            raise
            
    def generate_world(self):
        """Generate the initial world"""
        try:
            # Calculate number of chunks
            chunk_size = 16
            chunks_x = (self.width + chunk_size - 1) // chunk_size
            chunks_z = (self.depth + chunk_size - 1) // chunk_size
            
            print(f"Generating world with {chunks_x}x{chunks_z} chunks...")
            total_chunks = chunks_x * chunks_z
            chunks_done = 0
            
            # Generate each chunk
            for cx in range(chunks_x):
                for cz in range(chunks_z):
                    try:
                        print(f"Generating chunk {cx},{cz} ({chunks_done}/{total_chunks})")
                        self.generator.generate_chunk(self, cx, 0, cz, chunk_size)
                        chunks_done += 1
                        sys.stdout.flush()  # Force print to show immediately
                    except Exception as e:
                        print(f"Error generating chunk at {cx},{cz}: {e}")
                        traceback.print_exc()
                        # Continue with next chunk instead of failing completely
                        continue
                        
        except Exception as e:
            print(f"Error in world generation: {e}")
            traceback.print_exc()
            raise
                    
    def is_valid_position(self, x, y, z):
        """Check if a position is within world bounds"""
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                0 <= z < self.depth)
                    
    def get_voxel_data(self, x, y, z):
        """Get VoxelData at position"""
        if self.is_valid_position(x, y, z):
            return self.voxels[x, y, z]
        return None
                    
    def destroy_voxel(self, x, y, z):
        """Destroy a voxel and handle physics"""
        if self.is_valid_position(x, y, z):
            self.physics.destroy_voxel(x, y, z)
            
    def apply_damage(self, x, y, z, damage, radius=1.0):
        """Apply damage to blocks within a radius"""
        self.physics.apply_damage(x, y, z, damage, radius)
            
    def update(self, dt):
        """Update world physics"""
        self.physics.update(dt)

class Game:
    def __init__(self, width=800, height=600):
        print("Initializing GLFW...")
        if not glfw.init():
            raise Exception("GLFW initialization failed")
            
        # Configure GLFW window
        print("Creating window...")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.VISIBLE, True)
        glfw.window_hint(glfw.RESIZABLE, True)
        glfw.window_hint(glfw.DOUBLEBUFFER, True)
            
        # Create window before doing anything else
        self.window = glfw.create_window(width, height, "Voxel Destruction Sandbox", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Window creation failed")
            
        # Make context current
        glfw.make_context_current(self.window)
        print(f"OpenGL Version: {glGetString(GL_VERSION).decode()}")
        
        # Set up input handling
        print("Setting up input handlers...")
        glfw.set_key_callback(self.window, self.key_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_callback)
        
        # Process events to show window
        glfw.poll_events()
        
        # Basic OpenGL configuration
        self.setup_gl()
        
        # Capture mouse cursor
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        
        print("Creating world...")
        try:
            # Create world with random seed
            self.world = VoxelWorld(seed=int(time.time()))
            print("World created!")
            
            print("Initializing renderer...")
            self.renderer = VoxelRenderer()
            print("Renderer initialized!")
            
            # Start at a better position to see more of the world
            self.camera = Camera(position=np.array([32.0, 24.0, 32.0]))
            
            self.last_frame = glfw.get_time()
            self.delta_time = 0.0
            
            print("Game initialization complete!")
        except Exception as e:
            print(f"Error during initialization: {e}")
            traceback.print_exc()
            glfw.terminate()
            raise
        
    def setup_gl(self):
        print("Setting up OpenGL...")
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glFrontFace(GL_CCW)
        glClearColor(0.5, 0.7, 1.0, 1.0)
        print("OpenGL setup complete")
        
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
            # Ray cast from camera and apply damage
            hit, pos = self.ray_cast_from_camera()
            if hit:
                x, y, z = map(int, pos)
                # Apply damage with radius for more realistic destruction
                self.world.apply_damage(x, y, z, damage=30.0, radius=2.0)
                
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
                voxel = self.world.get_voxel_data(x, y, z)
                if voxel and voxel.material_type != MaterialType.AIR:
                    return True, pos
                    
            current_dist += step
            
        return False, None
            
    def run(self):
        print("Starting game loop...")
        try:
            while not glfw.window_should_close(self.window):
                try:
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
                    
                    # Swap buffers and poll events
                    glfw.swap_buffers(self.window)
                    glfw.poll_events()
                    
                except Exception as e:
                    print(f"Error in game loop: {e}")
                    traceback.print_exc()
                    break
                    
        except Exception as e:
            print(f"Fatal error in game loop: {e}")
            traceback.print_exc()
        finally:
            print("Shutting down...")
            glfw.terminate()

if __name__ == "__main__":
    try:
        print("\nStarting game initialization...")
        sys.stdout.flush()  # Force print
        
        game = Game()
        print("\nStarting game loop...")
        sys.stdout.flush()  # Force print
        
        game.run()
    except Exception as e:
        print("\n" + "="*60)
        print("FATAL ERROR:")
        print("-"*60)
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print("\nFull traceback:")
        print("-"*60)
        traceback.print_exc()
        print("="*60 + "\n")
        sys.stdout.flush()  # Force print
        
        print("\nPress Enter to exit...")
        sys.stdout.flush()
        input()  # Wait for user input before closing
    finally:
        print("\nAttempting clean shutdown...")
        try:
            glfw.terminate()
            print("GLFW terminated successfully")
        except Exception as e:
            print(f"Error during GLFW cleanup: {e}")
        print("Game shutdown complete")
        print("\nPress Enter to exit...")
        sys.stdout.flush()
        input()  # Wait for user input before final exit 