/**
 * Copyright (C) 2025 Eyeware Tech SA
 *
 * All rights reserved
 */
using System;
using System.Linq;
using System.Collections.Generic;
using Eyeware.BeamEyeTracker.Unity;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UI;

namespace Eyeware.BeamEyeTracker.Unity
{
    /// <summary>
    /// Controls the opacity of a UI panel and its children using a CanvasGroup based on whether
    /// the user is looking at the element to increase immersion.
    ///
    /// This script should only be attached to a parent GameObject (e.g., a HUD panel)
    /// that contains child UI elements like Images, Text, or Sliders. Attaching it to
    /// individual UI elements may lead to unexpected behavior.
    /// </summary>
    public class ImmersiveHUDPanelBehaviour : BeamEyeTrackerMonoBehaviour
    {
        public enum OnScreenPosition
        {
            Undefined,
            TopLeft,
            TopMiddle,
            TopRight,
            CenterLeft,
            CenterRight,
            BottomLeft,
            BottomMiddle,
            BottomRight
        }

        private static readonly IReadOnlyDictionary<
            OnScreenPosition,
            Vector2
        > VIEWPORT_COORDINATES = new Dictionary<OnScreenPosition, Vector2>
        {
            { OnScreenPosition.Undefined, new Vector2(0.5f, 0.5f) },
            { OnScreenPosition.TopLeft, new Vector2(0.0f, 1.0f) },
            { OnScreenPosition.TopMiddle, new Vector2(0.5f, 1.0f) },
            { OnScreenPosition.TopRight, new Vector2(1.0f, 1.0f) },
            { OnScreenPosition.CenterLeft, new Vector2(0.0f, 0.5f) },
            { OnScreenPosition.CenterRight, new Vector2(1.0f, 0.5f) },
            { OnScreenPosition.BottomLeft, new Vector2(0.0f, 0.0f) },
            { OnScreenPosition.BottomMiddle, new Vector2(0.5f, 0.0f) },
            { OnScreenPosition.BottomRight, new Vector2(1.0f, 0.0f) }
        };

        // Config
        public float fadeOutSpeed = 2f;
        public float fadeInSpeed = 10f;
        public float minOpacity = 0.2f;

        // Unity components cached for performance
        private CanvasGroup canvasGroup;
        private Canvas canvas;
        private RectTransform rectTransform;

        // State
        private bool isFadingOut = false;
        private bool isFadingIn = false;

        void Start()
        {
            // Get the CanvasGroup component or add one if not present
            canvasGroup = GetComponent<CanvasGroup>();
            if (canvasGroup == null)
            {
                canvasGroup = gameObject.AddComponent<CanvasGroup>();
            }
            // Set initial opacity to fully visible
            canvasGroup.alpha = 1f;
            canvas = GetComponentInParent<Canvas>();
            rectTransform = GetComponent<RectTransform>();
        }

        void Update()
        {
            if (betInputDevice == null)
            {
                return;
            }

            OnScreenPosition onScreenPosition = getScreenPositionForThisElement();
            if (onScreenPosition == OnScreenPosition.Undefined)
            {
                return;
            }

            float likelihoodOfUserLookingThis = 1.0f;

            switch (onScreenPosition)
            {
                case OnScreenPosition.TopLeft:
                    likelihoodOfUserLookingThis =
                        betInputDevice.isLookingAtTopLeftCorner.ReadValue();
                    break;
                case OnScreenPosition.TopMiddle:
                    likelihoodOfUserLookingThis = betInputDevice.isLookingAtTopMiddle.ReadValue();
                    break;
                case OnScreenPosition.TopRight:
                    likelihoodOfUserLookingThis =
                        betInputDevice.isLookingAtTopRightCorner.ReadValue();
                    break;
                case OnScreenPosition.CenterLeft:
                    likelihoodOfUserLookingThis = betInputDevice.isLookingAtCenterLeft.ReadValue();
                    break;
                case OnScreenPosition.CenterRight:
                    likelihoodOfUserLookingThis = betInputDevice.isLookingAtCenterRight.ReadValue();
                    break;
                case OnScreenPosition.BottomLeft:
                    likelihoodOfUserLookingThis =
                        betInputDevice.isLookingAtBottomLeftCorner.ReadValue();
                    break;
                case OnScreenPosition.BottomMiddle:
                    likelihoodOfUserLookingThis =
                        betInputDevice.isLookingAtBottomMiddle.ReadValue();
                    break;
                case OnScreenPosition.BottomRight:
                    likelihoodOfUserLookingThis =
                        betInputDevice.isLookingAtBottomRightCorner.ReadValue();
                    break;
            }

            bool isLookingAtThis = likelihoodOfUserLookingThis > 0.5f;

            if (betInputDevice.trackingStatus.ReadValue() != 1)
            {
                // When there is no data coming from the eye tracker, we assume the user is
                // looking at the element as that is the safe choice from a UX perspective.
                isLookingAtThis = true;
            }

            if (!isLookingAtThis)
            {
                // Start fading out if not already fading out
                if (!isFadingOut)
                {
                    isFadingOut = true;
                    isFadingIn = false;
                    StopAllCoroutines();
                    StartCoroutine(FadeTo(minOpacity, fadeOutSpeed));
                }
            }
            else
            {
                // Start fading in if not already fading in
                if (!isFadingIn)
                {
                    isFadingIn = true;
                    isFadingOut = false;
                    StopAllCoroutines();
                    StartCoroutine(FadeTo(1f, fadeInSpeed));
                }
            }
        }

        private OnScreenPosition getScreenPositionForThisElement()
        {
            // We calculate the center of the element referred to the screen space, whose approach
            // depends on the canvas' render mode
            Vector2 screenSpaceCenter = new Vector2(0, 0);
            if (canvas.renderMode == RenderMode.ScreenSpaceOverlay)
            {
                Vector3[] worldCorners = new Vector3[4];
                rectTransform.GetWorldCorners(worldCorners);
                // add up opposite corners to later get the mean/center of the element
                screenSpaceCenter = new Vector2(
                    worldCorners[0].x + worldCorners[2].x,
                    worldCorners[0].y + worldCorners[2].y
                );
            }
            else
            {
                Vector3[] localCorners = new Vector3[4];
                rectTransform.GetLocalCorners(localCorners);
                // Get the camera to use according to the canvas' render mode
                Camera canvasCamera;
                if (canvas.renderMode == RenderMode.ScreenSpaceCamera)
                {
                    canvasCamera = canvas.worldCamera;
                }
                else
                {
                    canvasCamera = Camera.main;
                }
                if (canvasCamera == null)
                {
                    return OnScreenPosition.Undefined;
                }

                // Add the top left and bottom right corners
                foreach (var i in new[] { 0, 2 })
                {
                    Vector3 worldPos = rectTransform.TransformPoint(localCorners[i]);
                    Vector3 screenPoint = canvasCamera.WorldToScreenPoint(worldPos);
                    screenSpaceCenter += new Vector2(screenPoint.x, screenPoint.y);
                }
            }
            screenSpaceCenter /= 2f; // calculate middle point

            // We calculate the center of the element within the -normalized- screen space
            Vector2 viewportSpaceCenter = new Vector2(
                screenSpaceCenter[0] / Screen.width,
                screenSpaceCenter[1] / Screen.height
            );

            // We calculate the distance of the center of the element to each of the corners of the screen
            var distances = Enum.GetValues(typeof(OnScreenPosition))
                .Cast<OnScreenPosition>()
                .ToDictionary(
                    position => position,
                    position =>
                        Vector2.Distance(viewportSpaceCenter, VIEWPORT_COORDINATES[position])
                );

            return distances.OrderBy(d => d.Value).First().Key;
        }

        private System.Collections.IEnumerator FadeTo(float targetAlpha, float speed)
        {
            // Smoothly transition to the target alpha
            while (!Mathf.Approximately(canvasGroup.alpha, targetAlpha))
            {
                canvasGroup.alpha = Mathf.MoveTowards(
                    canvasGroup.alpha,
                    targetAlpha,
                    speed * Time.deltaTime
                );
                yield return null;
            }
        }
    }
}
