# Voxel Destruction Sandbox

A simple voxel-based destruction simulator built with Python and OpenGL. This project allows you to explore a procedurally generated voxel world and destroy blocks.

## Features

- Procedurally generated terrain using Perlin noise
- First-person camera controls
- Voxel destruction mechanics
- Basic lighting system

## Requirements

- Python 3.7+
- OpenGL support
- Required Python packages (install via pip):
  - numpy
  - PyOpenGL
  - noise
  - glfw
  - pyrr

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the game:
```bash
python voxel_game.py
```

### Controls

- WASD: Move camera
- Mouse: Look around
- Left Click: Destroy blocks (currently random blocks for testing)
- ESC: Exit game

## Project Structure

- `voxel_game.py`: Main game loop and initialization
- `renderer.py`: OpenGL rendering code
- `shaders/`: GLSL shader files
  - `vertex.glsl`: Vertex shader
  - `fragment.glsl`: Fragment shader

## Future Improvements

- Ray casting for precise block destruction
- Different block types and materials
- Physics simulation for falling blocks
- Block placement mechanics
- Texture support
- Chunk-based world generation for larger worlds 