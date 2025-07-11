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
    /// Automatically starts the Beam Eye Tracker at the first Update.
    /// </summary>
    public class AutoStartBeamEyeTrackerBehaviour : BeamEyeTrackerMonoBehaviour
    {
        public bool startBeamEyeTrackerOnStart = true;

        private static bool hasAttemptedStartingTracker = false;

        private void Update()
        {
            if (betInputDevice == null)
            {
                return;
            }

            if (!hasAttemptedStartingTracker && startBeamEyeTrackerOnStart){
                hasAttemptedStartingTracker = betInputDevice.AttemptStartingTheBeamEyeTracker();
            }
        }
    }
}
