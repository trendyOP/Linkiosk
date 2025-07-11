# Changelog

All notable changes to the Beam Eye Tracker SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2025-04-10

### Added

- Foveated rendering support
  - FoveatedRenderingState: gaze coordinates and radii that define high to low resolution areas
- Unity support
  - BeamEyeTrackerInputDevice class that provides a custom input device based on Unity InputSystem
  - BeamEyeTrackerMonoBehaviour class that allows easy access to the input device
  - CameraControlBehaviour and ImmersiveHUDPanelBehaviour classes that implement the in-game camera control and immersive HUD features in Unity
  - BeamEyeTrackerControls.inputactions that define actions like camera recentering
- Documentation
  - Added comprehensive documentation for SDK concepts
  - C/C++/C#/Python API references
  - Unity guide

### Fixed

- Potential crash on shutdown

## [2.0.1] - 2025-02-15

### Fixed

- Fixed handling of paths containing non-Western characters

## [2.0.0] - 2025-02-01

### Added

- Releasing brand new API with these features:
  - UserState: head pose and gaze coordinates parameters (now supporting multiple displays)
  - SimGameCameraState: ready to use 6 DOF camera controls
  - GameImmersiveHUDState: signal indicating if the user is looking at either 4 corners or edges
  - Synchronous, asynchronous and polling-based tracking data access
  - Privacy control - Beam app can manage data access per-client
  - Clients can launch the Beam and start the tracking/webcam
- C/C++/C#/Python APIs
- Future-proof design:
  - Features logic fully delegated to the Beam application (not hardcoded in the client)
    - Gives room to improve user experience without increasing client work load
  - C/C++ APIs designed to be ABI stable and easy to extend with new features
  - Forward and backward client compatibility with Beam app versions (from v2.4.0)
    - Handshaking mechanism
    - Per-client tracking data messaging filtering
    - Future-proof serialization

### Changed

- Complete API redesign focusing on stability and extensibility
- Enhanced documentation and samples

### Removed

- API v1.0
