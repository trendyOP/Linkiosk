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
    public class FoveatedRenderingLoggerBehaviour : BeamEyeTrackerMonoBehaviour
    {
        private void LateUpdate()
        {
            if (betInputDevice == null)
            {
                return;
            }

            // Demonstrates how to get the latest foveated rendering state from the eye tracker.
            FoveatedRenderingState? foveatedRenderingState =
                betInputDevice.GetFoveatedRenderingState();

            if (foveatedRenderingState.HasValue)
            {
                Debug.Log(
                    $"Foveated rendering center: {foveatedRenderingState.Value.NormalizedFoveationCenter.X}, {foveatedRenderingState.Value.NormalizedFoveationCenter.Y}, radii: {foveatedRenderingState.Value.NormalizedFoveationRadii.RadiusLevel1}, {foveatedRenderingState.Value.NormalizedFoveationRadii.RadiusLevel2}, {foveatedRenderingState.Value.NormalizedFoveationRadii.RadiusLevel3}, {foveatedRenderingState.Value.NormalizedFoveationRadii.RadiusLevel4}"
                );
            }
        }
    }
}
