/**
 * Copyright (C) 2025 Eyeware Tech SA
 *
 * All rights reserved
 */
using System;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using Eyeware.BeamEyeTracker;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Controls;
using UnityEngine.InputSystem.Layouts;
using UnityEngine.InputSystem.LowLevel;
using UnityEngine.InputSystem.Utilities;
#nullable enable
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace Eyeware.BeamEyeTracker.Unity
{
    /// <summary>
    /// State structure for the Beam Eye Tracker input device
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct BeamEyeTrackerState : IInputStateTypeInfo
    {
        public FourCC format => new FourCC('B', 'E', 'A', 'M');

        [InputControl(displayName = "Unified Screen Gaze Position", layout = "Vector2")]
        public Vector2 unifiedScreenGazePosition;

        [InputControl(displayName = "Viewport Gaze Position", layout = "Vector2")]
        public Vector2 viewportGazePosition;

        [InputControl(displayName = "Camera Rotation", layout = "Vector3")]
        public Vector3 cameraRotation;

        [InputControl(displayName = "Camera Position", layout = "Vector3")]
        public Vector3 cameraPosition;

        [InputControl(displayName = "Looking At Top Left Corner", layout = "Button")]
        public float isLookingAtTopLeftCorner;

        [InputControl(displayName = "Looking At Top Middle", layout = "Button")]
        public float isLookingAtTopMiddle;

        [InputControl(displayName = "Looking At Top Right Corner", layout = "Button")]
        public float isLookingAtTopRightCorner;

        [InputControl(displayName = "Looking At Center Left", layout = "Button")]
        public float isLookingAtCenterLeft;

        [InputControl(displayName = "Looking At Center Right", layout = "Button")]
        public float isLookingAtCenterRight;

        [InputControl(displayName = "Looking At Bottom Left Corner", layout = "Button")]
        public float isLookingAtBottomLeftCorner;

        [InputControl(displayName = "Looking At Bottom Middle", layout = "Button")]
        public float isLookingAtBottomMiddle;

        [InputControl(displayName = "Looking At Bottom Right Corner", layout = "Button")]
        public float isLookingAtBottomRightCorner;

        [InputControl(displayName = "Tracking Status", layout = "Integer")]
        public int trackingStatus;
    }

    /// <summary>
    /// Input device for the Beam Eye Tracker
    /// </summary>
    [InputControlLayout(stateType = typeof(BeamEyeTrackerState))]
#if UNITY_EDITOR
    [InitializeOnLoad]
#endif
    public class BeamEyeTrackerInputDevice : InputDevice, IInputUpdateCallbackReceiver, IDisposable
    {
        private API? betApi;
        private bool hasNewData;

        public Vector2Control unifiedScreenGazePosition { get; private set; } = null!;
        public Vector2Control viewportGazePosition { get; private set; } = null!;
        public Vector3Control cameraRotation { get; private set; } = null!;
        public Vector3Control cameraPosition { get; private set; } = null!;
        public IntegerControl trackingStatus { get; private set; } = null!;
        public ButtonControl isLookingAtTopLeftCorner { get; private set; } = null!;
        public ButtonControl isLookingAtTopMiddle { get; private set; } = null!;
        public ButtonControl isLookingAtTopRightCorner { get; private set; } = null!;
        public ButtonControl isLookingAtCenterLeft { get; private set; } = null!;
        public ButtonControl isLookingAtCenterRight { get; private set; } = null!;
        public ButtonControl isLookingAtBottomLeftCorner { get; private set; } = null!;
        public ButtonControl isLookingAtBottomMiddle { get; private set; } = null!;
        public ButtonControl isLookingAtBottomRightCorner { get; private set; } = null!;

        public bool IsDisposed { get; private set; }
        private float timeToMakeNextViewportGeometryCheck;
        private bool viewportGeometrySetOnce;

        private double lastReceivedTimestamp = Constants.NullDataTimestamp;

        private double lastReceivedFoveatedRenderingStateTimestamp = Constants.NullDataTimestamp;
        private FoveatedRenderingState latestFoveatedRenderingState =
            FoveatedRenderingState.Create();

        static BeamEyeTrackerInputDevice()
        {
            InputSystem.RegisterLayout<BeamEyeTrackerInputDevice>(
                matches: new InputDeviceMatcher().WithInterface("BeamEyeTracker")
            );
            if (!InputSystem.devices.Any(device => device is BeamEyeTrackerInputDevice))
                InputSystem.AddDevice<BeamEyeTrackerInputDevice>();
        }

        [RuntimeInitializeOnLoadMethod]
        private static void InitializeInPlayer() { }

        ~BeamEyeTrackerInputDevice()
        {
            Dispose();
        }

        protected override void FinishSetup()
        {
            base.FinishSetup();

            unifiedScreenGazePosition = GetChildControl<Vector2Control>(
                "unifiedScreenGazePosition"
            );
            viewportGazePosition = GetChildControl<Vector2Control>("viewportGazePosition");
            cameraRotation = GetChildControl<Vector3Control>("cameraRotation");
            cameraPosition = GetChildControl<Vector3Control>("cameraPosition");
            trackingStatus = GetChildControl<IntegerControl>("trackingStatus");
            isLookingAtTopLeftCorner = GetChildControl<ButtonControl>("isLookingAtTopLeftCorner");
            isLookingAtTopMiddle = GetChildControl<ButtonControl>("isLookingAtTopMiddle");
            isLookingAtTopRightCorner = GetChildControl<ButtonControl>("isLookingAtTopRightCorner");
            isLookingAtCenterLeft = GetChildControl<ButtonControl>("isLookingAtCenterLeft");
            isLookingAtCenterRight = GetChildControl<ButtonControl>("isLookingAtCenterRight");
            isLookingAtBottomLeftCorner = GetChildControl<ButtonControl>(
                "isLookingAtBottomLeftCorner"
            );
            isLookingAtBottomMiddle = GetChildControl<ButtonControl>("isLookingAtBottomMiddle");
            isLookingAtBottomRightCorner = GetChildControl<ButtonControl>(
                "isLookingAtBottomRightCorner"
            );

            Initialize();
        }

        private void Initialize()
        {
            Eyeware.BeamEyeTracker.ViewportGeometry? potentialGameViewportArea =
                GetGameViewportAreaInUnifiedScreen();
            Eyeware.BeamEyeTracker.ViewportGeometry gameViewportArea;
            if (potentialGameViewportArea.HasValue)
            {
                gameViewportArea = potentialGameViewportArea.Value;
                viewportGeometrySetOnce = true;
            }
            else
            {
                gameViewportArea = new Eyeware.BeamEyeTracker.ViewportGeometry(
                    new Eyeware.BeamEyeTracker.Point(0, 0),
                    new Eyeware.BeamEyeTracker.Point(0, 0)
                );
            }

            betApi = new API(Application.productName, gameViewportArea);
            timeToMakeNextViewportGeometryCheck = Time.realtimeSinceStartup;
        }

        public void Dispose()
        {
            if (IsDisposed)
            {
                return;
            }

            if (betApi != null)
            {
                betApi.Dispose();
                betApi = null;
            }

            IsDisposed = true;
        }

        public bool AttemptStartingTheBeamEyeTracker()
        {
            if (betApi == null)
            {
                return false;
            }
            betApi.AttemptStartingTheBeamEyeTracker();
            return true;
        }

        public void OnUpdate()
        {
            if (betApi == null)
            {
                return;
            }

            if (Time.realtimeSinceStartup >= timeToMakeNextViewportGeometryCheck)
            {
                // We update the viewport geometry in case the rendering area has changed, but we do it only
                // once in a while to avoid unnecessary overhead as this should not occur often in practice.

                // WARNING: When using the the editor, there is a strange behavior from Unity in which the
                // Screen object does not give the proper rendering resolution while the mouse moves.
                // Thus, for the GameView geometry to be captured at least once, we need to meet these conditions
                // at least once.
                // 1. The keyboard focus is on the GameView window (click on it)
                // 2. The mouse is hovering over the GameView window
                // 3. The mouse does not move.
                Eyeware.BeamEyeTracker.ViewportGeometry? newViewportGeometry =
                    GetGameViewportAreaInUnifiedScreen();
                if (newViewportGeometry.HasValue)
                {
                    betApi.UpdateViewportGeometry(newViewportGeometry.Value);
                    viewportGeometrySetOnce = true;
                }
                if (!viewportGeometrySetOnce)
                {
                    // We try frequently
                    timeToMakeNextViewportGeometryCheck = Time.realtimeSinceStartup + 0.1f;
                }
                else
                {
                    timeToMakeNextViewportGeometryCheck = Time.realtimeSinceStartup + 1.0f;
                }
            }

            // We check if there is a change in data reception status
            TrackingDataReceptionStatus currentDataReceptionStatus =
                betApi.GetTrackingDataReceptionStatus();

            if ((int)currentDataReceptionStatus != trackingStatus.ReadValue())
            {
                InputSystem.QueueDeltaStateEvent(trackingStatus, (int)currentDataReceptionStatus);
            }

            // We poll to check if there is new data, without blocking.
            if (betApi.WaitForNewTrackingData(ref lastReceivedTimestamp, 0))
            {
                TrackingStateSet latestTrackingStateSet = betApi.GetLatestTrackingStateSet();
                if (
                    latestTrackingStateSet.SimGameCameraState.TimestampInSeconds
                    != Constants.NullDataTimestamp
                )
                {
                    var cameraTransform = API.ComputeSimGameCameraTransformParameters(
                        latestTrackingStateSet.SimGameCameraState
                    );
                    InputSystem.QueueDeltaStateEvent(
                        cameraRotation,
                        new Vector3(
                            cameraTransform.PitchInRadians * Mathf.Rad2Deg,
                            -cameraTransform.YawInRadians * Mathf.Rad2Deg,
                            -cameraTransform.RollInRadians * Mathf.Rad2Deg
                        )
                    );
                    InputSystem.QueueDeltaStateEvent(
                        cameraPosition,
                        new Vector3(
                            -cameraTransform.XInMeters,
                            cameraTransform.YInMeters,
                            cameraTransform.ZInMeters
                        )
                    );
                }

                if (
                    latestTrackingStateSet.UserState.UnifiedScreenGaze.Confidence
                    != TrackingConfidence.LostTracking
                )
                {
                    InputSystem.QueueDeltaStateEvent(
                        unifiedScreenGazePosition,
                        new Vector2(
                            latestTrackingStateSet.UserState.UnifiedScreenGaze.PointOfRegard.X,
                            latestTrackingStateSet.UserState.UnifiedScreenGaze.PointOfRegard.Y
                        )
                    );
                }

                if (
                    latestTrackingStateSet.GameImmersiveHUDState.TimestampInSeconds
                    != Constants.NullDataTimestamp
                )
                {
                    InputSystem.QueueDeltaStateEvent(
                        isLookingAtTopLeftCorner,
                        latestTrackingStateSet.GameImmersiveHUDState.LookingAtViewportTopLeft > 0.5f
                            ? 1.0f
                            : 0.0f
                    );
                    InputSystem.QueueDeltaStateEvent(
                        isLookingAtTopMiddle,
                        latestTrackingStateSet.GameImmersiveHUDState.LookingAtViewportTopMiddle
                        > 0.5f
                            ? 1.0f
                            : 0.0f
                    );
                    InputSystem.QueueDeltaStateEvent(
                        isLookingAtTopRightCorner,
                        latestTrackingStateSet.GameImmersiveHUDState.LookingAtViewportTopRight
                        > 0.5f
                            ? 1.0f
                            : 0.0f
                    );
                    InputSystem.QueueDeltaStateEvent(
                        isLookingAtCenterLeft,
                        latestTrackingStateSet.GameImmersiveHUDState.LookingAtViewportCenterLeft
                        > 0.5f
                            ? 1.0f
                            : 0.0f
                    );
                    InputSystem.QueueDeltaStateEvent(
                        isLookingAtCenterRight,
                        latestTrackingStateSet.GameImmersiveHUDState.LookingAtViewportCenterRight
                        > 0.5f
                            ? 1.0f
                            : 0.0f
                    );
                    InputSystem.QueueDeltaStateEvent(
                        isLookingAtBottomLeftCorner,
                        latestTrackingStateSet.GameImmersiveHUDState.LookingAtViewportBottomLeft
                        > 0.5f
                            ? 1.0f
                            : 0.0f
                    );
                    InputSystem.QueueDeltaStateEvent(
                        isLookingAtBottomMiddle,
                        latestTrackingStateSet.GameImmersiveHUDState.LookingAtViewportBottomMiddle
                        > 0.5f
                            ? 1.0f
                            : 0.0f
                    );
                    InputSystem.QueueDeltaStateEvent(
                        isLookingAtBottomRightCorner,
                        latestTrackingStateSet.GameImmersiveHUDState.LookingAtViewportBottomRight
                        > 0.5f
                            ? 1.0f
                            : 0.0f
                    );
                }

                if (
                    latestTrackingStateSet.UserState.ViewportGaze.Confidence
                    != TrackingConfidence.LostTracking
                )
                {
                    InputSystem.QueueDeltaStateEvent(
                        viewportGazePosition,
                        new Vector2(
                            latestTrackingStateSet.UserState.ViewportGaze.NormalizedPointOfRegard.X,
                            latestTrackingStateSet.UserState.ViewportGaze.NormalizedPointOfRegard.Y
                        )
                    );
                }
            }
        }

        public bool StartRecenterSimGameCamera()
        {
            if (IsDisposed || betApi == null)
                return false;
            return betApi.StartRecenterSimGameCamera();
        }

        public void StopRecenterSimGameCamera()
        {
            if (IsDisposed || betApi == null)
                return;
            betApi.StopRecenterSimGameCamera();
        }

        /// <summary>
        /// Gets the latest foveated rendering state from the eye tracker.
        /// </summary>
        /// <remarks>
        /// Foveated rendering follows a different paradigm than other input device outputs: it is
        /// available with this explicit polling mechanism. This is behaviour is important and deliverate for
        /// two main reasons:
        /// - Foveated rendering does not affect other Game Objects as, for example, the cameraPosition and
        ///   cameraRotation do.
        /// - The standard input system queries for input at the start of the frame, but for foveated rendering
        ///   it is critical to query just before the rendering pass as to minimize latency and improve user
        ///   experience.
        /// </remarks>
        /// <returns>The latest foveated rendering state from the eye tracker, or null if the eye tracker is not connected or not tracking.</returns>
        public FoveatedRenderingState? GetFoveatedRenderingState()
        {
            if (IsDisposed || betApi == null)
                return null;

            if (
                betApi.GetTrackingDataReceptionStatus()
                != TrackingDataReceptionStatus.ReceivingTrackingData
            )
            {
                return null;
            }

            if (betApi.WaitForNewTrackingData(ref lastReceivedFoveatedRenderingStateTimestamp, 0))
            {
                TrackingStateSet trackingStateSet = betApi.GetLatestTrackingStateSet();
                if (
                    trackingStateSet.FoveatedRenderingState.TimestampInSeconds
                    != Constants.NullDataTimestamp
                )
                {
                    latestFoveatedRenderingState = trackingStateSet.FoveatedRenderingState;
                }
            }

            if (
                lastReceivedFoveatedRenderingStateTimestamp != Constants.NullDataTimestamp
                && latestFoveatedRenderingState.TimestampInSeconds != Constants.NullDataTimestamp
            )
            {
                return latestFoveatedRenderingState;
            }
            return null;
        }

        protected override void OnRemoved()
        {
            Dispose();
            base.OnRemoved();
        }

#if UNITY_EDITOR
        /// <summary>
        /// Gets the "viewport area" of the game rendering in OS Unified Screen coordinates
        /// </summary>
        /// <remarks>
        /// The viewport area is the entire rectangle where the "fullscreen" rendering is happening but
        /// it is referred to the OS Unified Screen coordinate system. In practice, it is possible that only a small
        /// region of this area is visible to the user, as in the UnityEditor the user may be using scaling or translation
        /// and much of the content may be outside the visible area, or even outside the screen itself. This is
        /// however ok, because our intention is to map gaze coordinates to the game viewport.
        /// </remarks>
        /// <returns>
        /// A <see cref="ViewportGeometry"/> representing the viewport area in OS Unified Screen coordinates
        /// The x and y properties of the ViewportGeometry represent the top-left corner of the game rendering area in unified screen coordinates
        /// (top-left), and the width and height properties represent the size of the game rendering area.
        /// </returns>
        private Eyeware.BeamEyeTracker.ViewportGeometry? GetGameViewportAreaInUnifiedScreen()
        {
            // We first retrieve the GameView window, which is the window where the rendering results are shown.
            try
            {
                System.Reflection.Assembly assembly = typeof(EditorWindow).Assembly;
                Type gameViewType = assembly.GetType("UnityEditor.GameView");

                EditorWindow focusedWindow = EditorWindow.focusedWindow;
                if (focusedWindow == null || focusedWindow.GetType() != gameViewType)
                {
                    return null;
                }
                EditorWindow gameViewWindowInFocus = focusedWindow;

                // The Screen.width and Screen.height are sometimes the resolution of the Window under the mouse
                // pointer, rather than the rendering resolution, for a reason I can't explain. Thus we need to
                // compare their geometry. However, for non-GameView windows, we can't properly query this resolution
                // based on the "viewInParent" property. Thus, for now, we filter out the cases in which the window
                // under the mouse pointer is not the GameView window.
                EditorWindow windowUnderMouse = EditorWindow.mouseOverWindow;
                if (windowUnderMouse == null || windowUnderMouse != gameViewWindowInFocus)
                {
                    return null;
                }

                // We will then retrieve the geometry for the area which ignores the toolbars and other decorations.
                PropertyInfo viewInParentProperty = gameViewType.GetProperty(
                    "viewInParent",
                    BindingFlags.Instance | BindingFlags.NonPublic
                );
                UnityEngine.Rect viewInParentRect = (UnityEngine.Rect)
                    viewInParentProperty.GetValue(gameViewWindowInFocus);

                // With this information, we can now calculate the top-left corner of the game view window in relation to the screen
                // for the region where the rendering occurs, but including the margins due to potential aspect ratio differences.
                UnityEngine.Vector2 game_view_without_toolbars_top_left_wrt_screen =
                    new UnityEngine.Vector2(
                        gameViewWindowInFocus.position.x + viewInParentRect.x,
                        gameViewWindowInFocus.position.y + viewInParentRect.y
                    );

                // We retrieve the rendering resolution.
                UnityEngine.Vector2 rendering_resolution = new UnityEngine.Vector2(
                    Screen.width,
                    Screen.height
                );

                // However, rendering resolution is in relation to the physical resolution, ignoring dpi whereas windows geometry
                // is handled logically by comprising the dpi scale. We need to account for this.
                float dpi_scale = Screen.dpi / 96.0f;
                UnityEngine.Vector2 logical_rendering_resolution = new UnityEngine.Vector2(
                    rendering_resolution.x / dpi_scale,
                    rendering_resolution.y / dpi_scale
                );

                if (
                    Mathf.Abs(
                        logical_rendering_resolution.x
                            - (viewInParentRect.x + viewInParentRect.width)
                    ) <= 1
                    && Mathf.Abs(
                        logical_rendering_resolution.y
                            - (viewInParentRect.y + viewInParentRect.height)
                    ) <= 1
                )
                {
                    // If the rendering resolution matches the window resolution (within 1 pixel)
                    // then we can't trust the values from Screen.width and Screen.height, and we assume
                    // it's Unity Editor going nuts.
                    return null;
                }

                // We then retrieve the scale and translation (panning) applied to the rendering window.
                var areaField = gameViewType.GetField(
                    "m_ZoomArea",
                    System.Reflection.BindingFlags.Instance
                        | System.Reflection.BindingFlags.NonPublic
                );
                var areaObj = areaField.GetValue(gameViewWindowInFocus);
                var translationField = areaObj
                    .GetType()
                    .GetField(
                        "m_Translation",
                        System.Reflection.BindingFlags.Instance
                            | System.Reflection.BindingFlags.NonPublic
                    );
                var translation = (UnityEngine.Vector2)translationField.GetValue(areaObj);
                var scaleField = areaObj
                    .GetType()
                    .GetField(
                        "m_Scale",
                        System.Reflection.BindingFlags.Instance
                            | System.Reflection.BindingFlags.NonPublic
                    );
                var scale = (UnityEngine.Vector2)scaleField.GetValue(areaObj);

                // We then combine the logical rendering resolution with scale and translation to figure out how
                // the entire viewport area is positioned in relation to the game view window.
                UnityEngine.Rect rendering_area_wrt_gameview = new UnityEngine.Rect(
                    translation.x - scale.x * logical_rendering_resolution.x / 2,
                    translation.y - scale.y * logical_rendering_resolution.y / 2,
                    scale.x * logical_rendering_resolution.x,
                    scale.y * logical_rendering_resolution.y
                );

                // We now refer that information to the virtual screen (logical, i.e., dpi scaled) coordinates.
                UnityEngine.Rect rendering_area_wrt_screen = new UnityEngine.Rect(
                    game_view_without_toolbars_top_left_wrt_screen.x
                        + rendering_area_wrt_gameview.x,
                    game_view_without_toolbars_top_left_wrt_screen.y
                        + rendering_area_wrt_gameview.y,
                    rendering_area_wrt_gameview.width,
                    rendering_area_wrt_gameview.height
                );

                // We now finally transform to virtual screen physical coordinates, by removing the dpi scaling.
                UnityEngine.Rect rendering_area_wrt_virtual_screen = new UnityEngine.Rect(
                    rendering_area_wrt_screen.x * dpi_scale,
                    rendering_area_wrt_screen.y * dpi_scale,
                    rendering_area_wrt_screen.width * dpi_scale,
                    rendering_area_wrt_screen.height * dpi_scale
                );

                return new Eyeware.BeamEyeTracker.ViewportGeometry(
                    // Point00 is the bottom-left corner of the rendering area
                    new Eyeware.BeamEyeTracker.Point(
                        (int)Mathf.Round(rendering_area_wrt_virtual_screen.x),
                        (int)
                            Mathf.Round(
                                rendering_area_wrt_virtual_screen.y
                                    + rendering_area_wrt_virtual_screen.height
                                    - 1
                            )
                    ),
                    // Point11 is the top-right corner of the rendering area
                    new Eyeware.BeamEyeTracker.Point(
                        (int)
                            Mathf.Round(
                                rendering_area_wrt_virtual_screen.x
                                    + rendering_area_wrt_virtual_screen.width
                                    - 1
                            ),
                        (int)Mathf.Round(rendering_area_wrt_virtual_screen.y)
                    )
                );
            }
            catch (System.Exception)
            {
                return null;
            }
        }
#elif UNITY_STANDALONE_WIN

        [DllImport("user32.dll", SetLastError = true)]
        private static extern IntPtr FindWindow(string lpClassName, string lpWindowName);

        [DllImport("user32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool GetWindowRect(IntPtr hWnd, out WindowRect lpRect);

        [StructLayout(LayoutKind.Sequential)]
        public struct WindowRect
        {
            public int Left;
            public int Top;
            public int Right;
            public int Bottom;
        }

        /// <summary>
        /// Get the coordinates of the top-left and bottom-right corners of the Game view window.
        /// </summary>
        /// <returns>A <see cref="ViewportGeometry"/> representing the window</returns>
        private Eyeware.BeamEyeTracker.ViewportGeometry? GetGameViewportAreaInUnifiedScreen()
        {
            string windowName = Application.productName;
            IntPtr pWindow = FindWindow(null, windowName);

            if (pWindow == null)
            {
                return null;
            }

            WindowRect lpRect;
            if (!GetWindowRect(pWindow, out lpRect))
            {
                return null;
            }

            return new Eyeware.BeamEyeTracker.ViewportGeometry(
                // Point00 is the bottom-left corner of the rendering area
                new Eyeware.BeamEyeTracker.Point(lpRect.Left, lpRect.Bottom),
                // Point11 is the top-right corner of the rendering area
                new Eyeware.BeamEyeTracker.Point(lpRect.Right, lpRect.Top)
            );
        }
#endif
    }
}
