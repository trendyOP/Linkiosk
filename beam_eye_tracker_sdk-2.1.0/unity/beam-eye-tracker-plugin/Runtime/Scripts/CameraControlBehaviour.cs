/**
 * Copyright (C) 2025 Eyeware Tech SA
 *
 * All rights reserved
 */
using Eyeware.BeamEyeTracker.Unity;
using UnityEngine;
using UnityEngine.InputSystem;

namespace Eyeware.BeamEyeTracker.Unity
{
    [RequireComponent(typeof(Camera))]
    public class CameraControlBehaviour : BeamEyeTrackerMonoBehaviour
    {
        public bool cameraControlIsPaused = false;
        private Transform cachedTransform;

        void Awake()
        {
            cachedTransform = transform;

            // Map controls to functionality of ingame camera controls
            betControls.Camera.Recenter.performed += OnRecenterPressed;
            betControls.Camera.Recenter.canceled += OnRecenterReleased;
            betControls.Camera.TogglePause.performed += OnCamControlTogglePause;
        }

        private void OnCamControlTogglePause(InputAction.CallbackContext context)
        {
            cameraControlIsPaused = !cameraControlIsPaused;
        }

        private void OnRecenterPressed(InputAction.CallbackContext context)
        {
            if (betInputDevice == null)
            {
                return;
            }
            if (cameraControlIsPaused || betInputDevice.trackingStatus.ReadValue() != 1)
            {
                cachedTransform.localPosition = Vector3.zero;
                cachedTransform.localEulerAngles = Vector3.zero;
            }
            else
            {
                betInputDevice.StartRecenterSimGameCamera();
            }
        }

        private void OnRecenterReleased(InputAction.CallbackContext context)
        {
            if (betInputDevice == null)
            {
                return;
            }
            betInputDevice.StopRecenterSimGameCamera();
        }

        private void Update()
        {
            if (betInputDevice == null)
            {
                return;
            }

            if (!cameraControlIsPaused && betInputDevice.trackingStatus.ReadValue() == 1)
            {
                cachedTransform.localPosition = betInputDevice.cameraPosition.ReadValue();
                cachedTransform.localEulerAngles = betInputDevice.cameraRotation.ReadValue();
            }
        }
    }
}
