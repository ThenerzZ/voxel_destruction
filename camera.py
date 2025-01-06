import numpy as np
import glfw
import pyrr

class Camera:
    def __init__(self, position=None):
        # Camera position and orientation
        self.position = np.array([5.0, 5.0, 5.0]) if position is None else np.array(position)
        self.front = np.array([0.0, 0.0, -1.0])
        self.up = np.array([0.0, 1.0, 0.0])
        self.right = np.array([1.0, 0.0, 0.0])
        
        # Camera angles
        self.yaw = -90.0
        self.pitch = 0.0
        
        # Camera settings
        self.movement_speed = 50.0  # Faster base movement
        self.mouse_sensitivity = 0.05  # Lower for more precise control
        
        # Mouse tracking
        self.last_x = 400
        self.last_y = 300
        self.first_mouse = True
        
        # Movement state
        self.moving_forward = False
        self.moving_backward = False
        self.moving_left = False
        self.moving_right = False
        self.moving_up = False
        self.moving_down = False
        
        # Sprint state
        self.sprinting = False
        
        # Update vectors
        self.update_camera_vectors()
        
    def get_view_matrix(self):
        return pyrr.matrix44.create_look_at(
            self.position,
            self.position + self.front,
            self.up
        )
        
    def handle_keyboard(self, key, action):
        """Handle keyboard input for camera movement"""
        if action == glfw.PRESS:
            if key == glfw.KEY_W:
                self.moving_forward = True
            elif key == glfw.KEY_S:
                self.moving_backward = True
            elif key == glfw.KEY_A:
                self.moving_left = True
            elif key == glfw.KEY_D:
                self.moving_right = True
            elif key == glfw.KEY_SPACE:
                self.moving_up = True
            elif key == glfw.KEY_LEFT_SHIFT:
                self.moving_down = True
            elif key == glfw.KEY_LEFT_CONTROL:
                self.sprinting = True
        elif action == glfw.RELEASE:
            if key == glfw.KEY_W:
                self.moving_forward = False
            elif key == glfw.KEY_S:
                self.moving_backward = False
            elif key == glfw.KEY_A:
                self.moving_left = False
            elif key == glfw.KEY_D:
                self.moving_right = False
            elif key == glfw.KEY_SPACE:
                self.moving_up = False
            elif key == glfw.KEY_LEFT_SHIFT:
                self.moving_down = False
            elif key == glfw.KEY_LEFT_CONTROL:
                self.sprinting = False
                
    def handle_mouse_movement(self, xpos, ypos):
        """Handle mouse movement for camera rotation"""
        if self.first_mouse:
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = False
            
        xoffset = xpos - self.last_x
        yoffset = self.last_y - ypos  # Reversed since y-coordinates go from bottom to top
        self.last_x = xpos
        self.last_y = ypos
        
        xoffset *= self.mouse_sensitivity
        yoffset *= self.mouse_sensitivity
        
        self.yaw += xoffset
        self.pitch += yoffset
        
        # Constrain pitch
        if self.pitch > 89.0:
            self.pitch = 89.0
        if self.pitch < -89.0:
            self.pitch = -89.0
            
        # Update front vector
        self.update_camera_vectors()
        
    def update_camera_vectors(self):
        """Calculate the new front vector"""
        front = np.array([
            np.cos(np.radians(self.yaw)) * np.cos(np.radians(self.pitch)),
            np.sin(np.radians(self.pitch)),
            np.sin(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
        ])
        self.front = front / np.linalg.norm(front)
        self.right = np.cross(self.front, np.array([0.0, 1.0, 0.0]))
        self.right = self.right / np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.front)
        self.up = self.up / np.linalg.norm(self.up)
        
    def update(self, delta_time):
        """Update camera position based on movement state"""
        # Calculate actual movement speed (sprint is 2x normal speed)
        current_speed = self.movement_speed * (2.0 if self.sprinting else 1.0)
        velocity = current_speed * delta_time
        
        if self.moving_forward:
            self.position += self.front * velocity
        if self.moving_backward:
            self.position -= self.front * velocity
        if self.moving_right:
            self.position += self.right * velocity
        if self.moving_left:
            self.position -= self.right * velocity
        if self.moving_up:
            self.position += self.up * velocity
        if self.moving_down:
            self.position -= self.up * velocity 