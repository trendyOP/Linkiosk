# Game Engine Integration Sample

This sample demonstrates how to integrate the Beam Eye Tracker SDK into a game engine architecture. It provides a practical example of implementing eye and head tracking features in a game environment, using patterns compatible with major game engines like Unity and Unreal Engine 5.

## Overview

The sample implements a simplified game engine architecture to demonstrate:

- Integration of the Beam Eye Tracker API as a custom game engine input device
- Head and eye tracking for immersive camera control
- Dynamic visibility of HUD elements based on whether the user is looking at them, for increased immersion
- Game settings mapped to input device parameters
- Viewport geometry management
- Camera recentering via SPACE key
- Auto-start functionality for seamless initialization

## Files and Structure

- `my_game_engine.h`: Implements dummy classes emulating a game engine OOP pattern, similar to Unity or Unreal Engine, with base interfaces having functions like `BeginPlay`, `Tick`, `Update`, etc. This is the least interesting part of the sample, as it is meant to be replaced by functionality existing in your own engine.
- `bet_game_engine_device.h`: Demonstrates how to implement a class that wraps the Beam Eye Tracker API and exposes it into a game engine, in this case: the `MyGameEngineBeamEyeTrackerDevice` class.
- `main.cpp`: Shows how to use your custom device class to interact with other objects in the engine (such as controlling a Camera object pose or HUD opacity) and effectively implement the head and eye tracking driven features.

## Building and Running

### Building the Sample

1. Ensure CMake is installed
2. Go to the parent folder (cpp/samples)
3. Run `build_samples.bat` in the sample directory
4. Find the compiled executable in the `bin` directory

### Running the Sample

The program runs for 30 seconds, showcasing:

- Real-time head tracking responses
- Real-time HUD opacity changes based on gaze
- Camera recentering (activated by SPACE key)

The console displays:

- Real-time camera position and rotation (only z translation and camera yaw are shown)
- Current HUD opacity values for the top-left corner
- Recentering status updates (when pressing SPACE)

Moreover, you can observe that the camera's z translation continuously increases to simulate the base camera world pose changes through the 3D scene, while the dynamic camera pose changes are additive to this base pose.

## Sample Implementation Details

### MyGameEngineBeamEyeTrackerDevice Class

This class emulates a custom input device in the engine. It:

- Manages the API instance and communication with the Beam Eye Tracker
- Processes incoming tracking data
- Handles auto-start requests for the Beam Eye Tracker
- Provides ready-to-use HUD opacity values based on gaze
- Outputs camera transforms in the game engine's coordinate system
- Processes game settings configuration and action controls (e.g., camera recentering)

### Supporting Classes

- `MyGameEngineImmersiveHUD`: Demonstrates HUD element management using opacity controls from `MyGameEngineBeamEyeTrackerDevice`
- `MyGameEngineCharacterHead`: Simulates a reference pose for the in-game camera
- `MyGameEngineImmersiveCamera`: Implements an in-game camera as a child of `MyGameEngineCharacterHead`, showing how `MyGameEngineBeamEyeTrackerDevice` controls its relative pose
- `MyGameEngineHotkeysMapper`: Maps input controls to `MyGameEngineBeamEyeTrackerDevice` actions

### Viewport Configuration

- In this sample, we hardcode a viewport emulating the middle of three 1920x1080 monitors
- Customize `get_rendering_area_viewport_geometry_from_engine()` for your display configuration when testing the sample

### Coordinate Systems

- The simplified Game Engine assumes Unity-style coordinate system
- Includes complete conversion from Beam's right-handed to Unity's left-handed coordinates
- Detailed comments explain coordinate transformations

### In-Game Settings

The `MyGameEngineBeamEyeTrackerDevice` holds a few public properties emulating potential in-game settings, exposed in the UI to the user, specifically:

- Head tracking and eye tracking sensitivity
- Whether the auto-start behavior is toggled ON

## Integration Guidelines

When implementing in your engine:

1. Implement accurate viewport geometry retrieval
2. Carefully handle coordinate system conversions
3. Map tracking sensitivities to appropriate game settings
4. Implement robust tracking interruption handling
5. Design a smooth user experience for the auto-start feature
6. Consider frame timing and update frequency

## Requirements

- Windows operating system
- CMake build system
- C++ compiler
- Beam Eye Tracker application (compatible with SDK v2.0 or higher)

## License

Copyright © 2025 Eyeware Tech SA. All rights reserved.
