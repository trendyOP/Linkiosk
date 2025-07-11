/**
 * Copyright (C) 2025 Eyeware Tech SA
 *
 * All rights reserved
 */
using UnityEngine;
using UnityEngine.InputSystem;

namespace Eyeware.BeamEyeTracker.Unity
{
    /// <summary>
    /// Base class for components that use the BeamEyeTrackerInputDevice.
    /// </summary>
    public abstract class BeamEyeTrackerMonoBehaviour : MonoBehaviour
    {
        // Beam eye tracker input device and its input actions / controls
        private static BeamEyeTrackerInputDevice _betInputDevice;
        private static BeamEyeTrackerControls _betControls;

        private static float beamInitializationRetryTime = 0f;
        private const float GET_BEAM_DEVICE_RETRY_INTERVAL_IN_SEC = 0.1f;

        // Properties to access controls and input device
        protected static BeamEyeTrackerControls betControls
        {
            get
            {
                if (_betControls == null)
                {
                    _betControls = new BeamEyeTrackerControls();
                    _betControls.Enable();
                }
                return _betControls;
            }
        }
        protected static BeamEyeTrackerInputDevice betInputDevice
        {
            get
            {
                if (_betInputDevice == null)
                {
                    if (Time.time >= beamInitializationRetryTime)
                    {
                        _betInputDevice = InputSystem.GetDevice<BeamEyeTrackerInputDevice>();
                        beamInitializationRetryTime =
                            Time.time + GET_BEAM_DEVICE_RETRY_INTERVAL_IN_SEC;
                    }
                }
                return _betInputDevice;
            }
        }

        protected virtual void OnDestroy()
        {
            // Only disable and dispose controls if this is the last consumer being destroyed
            if (
                _betControls != null
                && gameObject.scene.isLoaded
                && FindObjectsOfType<BeamEyeTrackerMonoBehaviour>().Length <= 1
            )
            {
                _betControls.Disable();
                _betControls.Dispose();
                _betControls = null;
            }
        }
    }
}
