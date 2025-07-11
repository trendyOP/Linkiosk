/**
 * Copyright (C) 2025 Eyeware Tech SA
 *
 * All rights reserved
 */
using System.Linq;
using Eyeware.BeamEyeTracker.Unity;
using UnityEngine;
using UnityEngine.InputSystem;

namespace Eyeware.BeamEyeTracker.Unity
{
    /// <summary>
    /// Moves the object to the position of the gaze on the screen.
    /// </summary>
    public class FollowGazeBehaviour : BeamEyeTrackerMonoBehaviour
    {
        void Update()
        {
            if (betInputDevice == null)
            {
                return;
            }

            Vector2 gazeOnViewport = betInputDevice.viewportGazePosition.ReadValue();

            Camera camera = GetComponentInParent<Camera>();
            if (camera == null)
            {
                return;
            }

            // Clamp gaze position to viewport bounds (0-1)
            gazeOnViewport.x = Mathf.Clamp01(gazeOnViewport.x);
            gazeOnViewport.y = Mathf.Clamp01(gazeOnViewport.y);

            // Convert screen (pixel) coordinates to world space
            Vector3 screenPosition = new Vector3(
                gazeOnViewport.x * Screen.width,
                gazeOnViewport.y * Screen.height,
                camera.nearClipPlane
            );
            Vector3 worldPosition = camera.ScreenToWorldPoint(screenPosition);

            // Use the converted worldPosition, but keep the original z position of the targetObject
            Vector3 newPosition = new Vector3(worldPosition.x, worldPosition.y, worldPosition.z);
            gameObject.transform.position = newPosition;
        }
    }
}
