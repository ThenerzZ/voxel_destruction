# Personal Voxel Game Project

A small experiment with voxel-based destructible environments using Python and OpenGL. Built this to learn more about graphics programming and physics simulations.

## What it's supposed to do
- Generate a voxel world with different materials (concrete, wood, glass, metal)
- Allow for destruction and physics simulation of falling blocks
- Have basic lighting and atmosphere effects
- Support chunk-based rendering for performance

## Current State
The project is currently broken and probably won't be fixed since I've learned what I wanted from it. Main issues:
- Rendering is extremely slow
- Chunks don't display properly
- Face culling is buggy
- OpenGL buffer management needs work

## Tech Stack
- Python 3.x
- PyOpenGL for graphics
- NumPy for math/arrays
- Pyrr for matrix operations

## Dependencies
```
numpy
pyopengl
pyrr
```

This is just a personal learning project - not meant for actual use or distribution. Keeping it around for reference and maybe future experiments with graphics programming. 