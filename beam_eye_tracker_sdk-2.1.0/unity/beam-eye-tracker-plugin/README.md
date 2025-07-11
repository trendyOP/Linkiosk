# Beam Eye Tracker Unity SDK

Add support for the Beam Eye Tracker - Enable immersive and natural camera controls as well as a dynamic HUD that also boosts immersion. The Beam Eye Tracker uses only a webcam or smartphone making accessible to many users. Free demo is available on Steam. The plugin also delivers real-time data for foveated rendering, if your rendering pipeline supports it.

## Features
- Real-time head and eye tracking data
- Camera control based on eye/head tracking
- HUD fades out when not looked at for increased immersion
- Real-time data to implement foveated rendering
- Unity Input System integration

## Requirements
- Unity 2020.3 or newer
- Input System Package 1.3.0+
- Windows 10 or newer
- Webcam

## Quick Start

1. Install the Beam Eye Tracker application from [beam.eyeware.tech](https://beam.eyeware.tech)
2. Import this package via Unity Package Manager
3. Drag the `CameraControlBehaviour.cs` script into the camera you want to control based on head and eye tracking
4. Drag the `ImmersiveHUDPanelBehaviour.cs` script into Canvas sub-elements that you want to auto-hide when not looked at.
5. Or implement your own behaviours by inheriting from `BeamEyeTrackerMonoBehaviour.cs` (or replicate the code therein) and access the `BeamEyeTrackerInputDevice`

## Demo Sample

The `CamAndHUDControl.unity` sample demonstrates these features in a simple scene.

## Documentation
Full documentation available at [docs.beam.eyeware.tech](https://docs.beam.eyeware.tech)

## Support
For support, visit our [Discord](https://discord.gg/dqm2KbFWJd) or email us at support@eyeware.tech

© 2025 Eyeware Tech SA. All rights reserved.
