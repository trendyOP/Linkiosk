# Beam Eye Tracker SDK

This SDK provides integration capabilities for the Beam Eye Tracker into games and applications.

## Contents

The SDK includes:

- C++ and C APIs located in:
  - `include/eyeware/beam_eye_tracker.h` (C++ API)
  - `include/eyeware/beam_eye_tracker_c.h` (C API)
- The `lib/beam_eye_tracker_client.lib` and `bin/beam_eye_tracker_client.dll` main library. It is built using `/MD` and thus requiring the Visual C++ Redistributable.
- C# bindings located in:
  - `csharp/src/Eyeware/BeamEyeTracker.cs` (Currently experimental)
- Python bindings for Python 3.7 through 3.12 located in (note that the DLL is not needed for these bindings)
  - `python/package`
- Samples

We also provide the library built with `/MT` (`lib/beam_eye_tracker_client_MT.lib` and `bin/beam_eye_tracker_client_MT.dll`) and thus not requiring the Visual C++ Redistributable at the host machine . However, in most cases, it is acceptable and even preferred to use the default library (`/MD` build).

## C++ Samples

The samples, available in the `cpp` folder, include a `build_samples.bat` script that will build the executables for you. The script assumes you have CMake installed and that CMake can find a suitable C++ compiler/generator on your system. Simply run `build_samples.bat` to compile the samples into a newly created `bin` folder. There are different samples available:

### Data Access Methods

The SDK provides three primary data access methods to interact with the Beam Eye Tracker:

1. **Polling Data Access**: This method allows you to retrieve tracking data without blocking the thread. It is suitable for scenarios where you can tolerate some latency in receiving updates. The polling method is demonstrated in the `bet_polling_data_access.cpp` sample.

2. **Synchronous Data Access**: This method is designed for low-latency data retrieval, blocking the thread until new tracking data is available. It is ideal for applications that require immediate updates to trigger further processing from the Beam Eye Tracker data. The synchronous method is illustrated in the `bet_sync_data_access.cpp` sample.

3. **Asynchronous Data Access**: This method enables you to receive tracking updates without blocking the main thread, providing low latency similar to synchronous access. The data is received through virtual methods of a listener class instance. The asynchronous method is showcased in the `bet_async_data_access.cpp` sample.

### Interpreting Data Structures

The `bet_sample_utils.h` file provides utility functions to interpret and display the received tracking data. It includes functions to print the tracking data reception status, interpret user state, and display the latest tracking state set. These utilities help in understanding the data flow and the state of the tracking system.

### Game Engine Integration

A detailed sample showing how to integrate the SDK into game engines is provided under `samples/game_engine_integration/`. The sample demonstrates integration patterns compatible with engines like Unity and Unreal Engine 5.

This sample is the recommended starting point for understanding how to use and integrate the Beam Eye Tracker API into your project. It contains detailed comments explaining each step of the integration process.

## Python Samples

The Python samples accurately replicate the C++ samples, except for the `Game Engine Integration` sample. Therefore, please refer to the C++ samples documentation for explanation on each of the samples.

These can be found in `python/samples`. To run them you need python with version 3.7...3.12, and numpy.

## Copyright

Copyright © 2025 Beam Eye Tracker. All rights reserved.

All rights reserved.
