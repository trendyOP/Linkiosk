/**
 * Copyright (C) 2025 Eyeware Tech SA
 *
 * All rights reserved
 *
 * @file beam_eye_tracker_c.h
 * @brief C language API for the Beam Eye Tracker SDK
 *
 * This header provides a C interface for integrating eye tracking capabilities.
 * It wraps the C++ implementation, providing equivalent functionality through
 * a C-compatible interface. The API supports three data access methods:
 * - Polling: Non-blocking data retrieval with potential latency
 * - Synchronous: Blocking calls for immediate data updates
 * - Asynchronous: Callback-based updates via registered functions
 */
#ifndef _EYEWARE_BEAM_EYE_TRACKER_C_H_
#define _EYEWARE_BEAM_EYE_TRACKER_C_H_

#include "beam_eye_tracker/config.h"
#include "beam_eye_tracker/types.h"

#ifdef __cplusplus
using namespace eyeware::beam_eye_tracker;
extern "C" {
#endif

/**
 * @brief Opaque handle to the API instance
 *
 * This handle represents a connection to the Beam Eye Tracker.
 * All API functions require a valid handle obtained through EW_BET_API_Create().
 *
 * @see EW_BET_API_Create
 * @see EW_BET_NULL_HANDLE
 */
typedef void *EW_BET_API_HANDLE;

/**
 * @brief Opaque handle to tracking state data
 *
 * Represents a snapshot of tracking data including gaze, head pose, and other states.
 * Must be properly managed with Create/Destroy functions to avoid memory leaks except
 * when received through callbacks.
 */
typedef void *EW_BET_TRACKING_STATE_SET_HANDLE;

/**
 * @brief Opaque handle to callback registration
 *
 * Used to manage asynchronous callback registrations. Each registration creates
 * a unique handle that must be used to unregister the callbacks later.
 *
 * @see EW_BET_API_RegisterTrackingCallbacks
 * @see EW_BET_API_UnregisterTrackingCallbacks
 */
typedef void *EW_BET_CALLBACKS_HANDLE;

/**
 * @brief Invalid handle value
 *
 * Used to:
 * - Initialize handle variables
 * - Check for failed handle creation
 * - Mark handles as invalid after destruction
 */
#define EW_BET_NULL_HANDLE nullptr

/**
 * @brief Initialize the Beam Eye Tracker API
 *
 * @param friendly_name Application identifier (UTF-8, max 200 bytes) displayed in the Beam Eye
 * Tracker UI
 * @param initial_viewport_geometry Initial viewport configuration for coordinate mapping
 * @param[out] api_handle Pointer to store the created API handle
 * @return 0 on success, error code otherwise
 *
 * @rst
 * See :ref:`about_api_singleton` for a high-level explanation of the ``API`` object.
 * @endrst
 */
EW_BET_API_DECL int32_t EW_BET_API_Create(const char *friendly_name,
                                          EW_BET_ViewportGeometry initial_viewport_geometry,
                                          EW_BET_API_HANDLE *api_handle);

/**
 * @brief Clean up and release API resources
 *
 * Must be called when the application is shutting down or no longer needs
 * eye tracking functionality. After this call, the handle becomes invalid
 * and should be set to @ref EW_BET_NULL_HANDLE.
 *
 * @param api_handle Handle to destroy (becomes invalid after call)
 */
EW_BET_API_DECL void EW_BET_API_Destroy(EW_BET_API_HANDLE api_handle);

/**
 * @brief Get SDK version information
 *
 * @param api_handle Valid API handle
 * @param[out] version Structure to receive version info (already allocated)
 *
 * Example:
 * @code{.c}
 * EW_BET_Version version;
 * EW_BET_API_GetVersion(api, &version);
 * printf("Beam Eye Tracker SDK v%d.%d.%d (build %d)\n",
 *        version.major, version.minor,
 *        version.patch, version.build);
 * @endcode
 */
EW_BET_API_DECL void EW_BET_API_GetVersion(EW_BET_API_HANDLE api_handle, EW_BET_Version *version);

/**
 * @brief Update the viewport geometry for coordinate mapping
 *
 * @param api_handle Valid API handle.
 * @param new_viewport_geometry The new viewport geometry to set.
 *
 * @rst
 * .. note:: For a detailed explanation, see :ref:`viewport`.
 * @endrst
 */
EW_BET_API_DECL void
EW_BET_API_UpdateViewportGeometry(EW_BET_API_HANDLE api_handle,
                                  EW_BET_ViewportGeometry new_viewport_geometry);

/**
 * @brief Attempts to start the Beam Eye Tracker application and tracking output.
 *
 * @rst
 * .. note:: See :ref:`about_auto_start_tracking` for more information.
 * @endrst
 */
EW_BET_API_DECL void EW_BET_API_AttemptStartingTheBeamEyeTracker(EW_BET_API_HANDLE api_handle);

/********************************************************************************
 *                 Asynchronous tracking data access
 ********************************************************************************/

/**
 * @brief Callback function type for tracking data reception status changes
 */
typedef void (*EW_BET_TrackingDataReceptionStatusCallback)(
    EW_BET_TrackingDataReceptionStatus status, void *user_data);

/**
 * @brief Callback function to receive the latest tracking data as soon as it arrives.
 */
typedef void (*EW_BET_TrackingDataCallback)(
    const EW_BET_TRACKING_STATE_SET_HANDLE tracking_state_set_handle, EW_BET_Timestamp timestamp,
    void *user_data);

/**
 * Register callbacks for asynchronous tracking data reception
 *
 * @param api_handle Handle to the API instance
 * @param on_tracking_data_reception_status_changed Callback for tracking data reception status
 * @param on_tracking_state_set_update Callback for new tracking data
 * @param user_data User data pointer passed to callbacks
 * @param[out] callbacks_handle Handle to the callbacks to unregister
 * @return 0 on success, non-zero error code otherwise
 *
 * @warning The callbacks must remain valid until unregistered
 *
 * @rst
 * .. note:: For a detailed explanation, see :ref:`about_asynchronous_data_access`.
 * @endrst
 */
EW_BET_API_DECL int32_t EW_BET_API_RegisterTrackingCallbacks(
    EW_BET_API_HANDLE api_handle,
    EW_BET_TrackingDataReceptionStatusCallback on_tracking_data_reception_status_changed,
    EW_BET_TrackingDataCallback on_tracking_state_set_update, void *user_data,
    EW_BET_CALLBACKS_HANDLE *callbacks_handle);

/**
 * Deregisters the callbacks from receiving data from the Beam Eye Tracker application.
 *
 * @param api_handle Handle to the API instance
 * @param[in,out] callbacks_handle Handle to the callbacks to unregister, which will also be set to
 *                                 nullptr on success.
 */
EW_BET_API_DECL void
EW_BET_API_UnregisterTrackingCallbacks(EW_BET_API_HANDLE api_handle,
                                       EW_BET_CALLBACKS_HANDLE *callbacks_handle);

/********************************************************************************
 *                 Synchronous tracking data access
 ********************************************************************************/

/**
 * @brief Waits until new tracking data becomes available. This is a blocking call
 * lasting until @ref timeout_ms milliseconds.
 *
 * @param api_handle Valid API handle
 * @param[in,out] last_update_timestamp  On input: the timestamp of the last received frame,
 *        used to determine if new data is available. On output: if new data is available,
 *        this will be updated to the timestamp of the newly received frame.
 *        Prior to the first iteration, you can initialize to @ref EW_BET_NULL_DATA_TIMESTAMP.
 * @param timeout_ms The maximum time to wait for new tracking data. Set to 0 to return
 * immediately.
 *
 * @return true if new tracking data is available, false if the timeout was reached without
 * new data.
 *
 * @rst
 * .. note:: See :ref:`about_synchronous_data_access` and :ref:`about_polling_data_access`.
 * @endrst
 */
EW_BET_API_DECL bool EW_BET_API_WaitForNewTrackingStateSet(EW_BET_API_HANDLE api_handle,
                                                           EW_BET_Timestamp *last_update_timestamp,
                                                           uint32_t timeout_ms);

/**
 * @brief Returns the current status of the tracking data reception.
 *
 * @param api_handle Valid API handle
 * @return The current status of the tracking data reception.
 *
 * @rst
 * .. note:: See :ref:`about_tracking_data_reception_status` for more information.
 * @endrst
 */
EW_BET_API_DECL EW_BET_TrackingDataReceptionStatus
EW_BET_API_GetTrackingDataReceptionStatus(EW_BET_API_HANDLE api_handle);

/**
 * @brief Get the latest tracking state
 *
 * Creates and fills a new ``TrackingStateSet`` with the most recent data.
 * The caller is responsible for destroying the handle when done.
 *
 * @param api_handle Valid API handle
 * @param[out] tracking_state_set Handle to receive the new state
 * @return 0 on success, error code otherwise
 *
 * @warning Must call @ref EW_BET_API_DestroyTrackingStateSet when done
 *
 * Example:
 * @code{.c}
 * EW_BET_TRACKING_STATE_SET_HANDLE tss_handle;
 * if(EW_BET_API_CreateAndFillLatestTrackingStateSet(api, &tss_handle) == 0) {
 *   const EW_BET_UserState* user = EW_BET_API_GetUserState(tss_handle);
 *   // Do something with the user state
 *   EW_BET_API_DestroyTrackingStateSet(tss_handle);
 * }
 * @endcode
 *
 * @rst
 * .. note:: See :ref:`about_tracking_state_set` for more information.
 * @endrst
 */
EW_BET_API_DECL int32_t EW_BET_API_CreateAndFillLatestTrackingStateSet(
    EW_BET_API_HANDLE api_handle, EW_BET_TRACKING_STATE_SET_HANDLE *tracking_state_set);

/**
 * @brief Release tracking state resources
 *
 * Must be called to free memory associated with a tracking state handle.
 * After this call, the handle becomes invalid and should not be used.
 *
 * @param tracking_state_set_handle Handle to destroy (becomes invalid)
 */
EW_BET_API_DECL void
EW_BET_API_DestroyTrackingStateSet(EW_BET_TRACKING_STATE_SET_HANDLE tracking_state_set_handle);

/**
 * @brief Returns the user state including the user's gaze on screen and head pose.
 *
 * @param tracking_state_set_handle Handle to the tracking state set.
 * @return The user state.
 *
 * @rst
 * .. note:: See :ref:`about_real_time_tracking` for more information.
 * @endrst
 */
EW_BET_API_DECL const EW_BET_UserState *
EW_BET_API_GetUserState(EW_BET_TRACKING_STATE_SET_HANDLE tracking_state_set_handle);

/**
 * Access the latest game camera state to implement the immersive in-game camera controls.
 *
 * @param tracking_state_set_handle Handle to the tracking state set
 * @return The latest camera controls.
 *
 * @rst
 * .. note:: See :ref:`about_sim_game_camera_state` for more information.
 * @endrst
 */
EW_BET_API_DECL const EW_BET_SimGameCameraState *
EW_BET_API_GetSimGameCameraState(EW_BET_TRACKING_STATE_SET_HANDLE tracking_state_set_handle);

/**
 * @brief Returns the parameters to implement an immersive HUD in your game.
 *
 * @param tracking_state_set_handle Handle to the tracking state set
 * @return The latest game immersive HUD state.
 *
 * @rst
 * .. note:: See :ref:`about_game_immersive_hud_state` for more information.
 * @endrst
 */
EW_BET_API_DECL const EW_BET_GameImmersiveHUDState *
EW_BET_API_GetGameImmersiveHUDState(EW_BET_TRACKING_STATE_SET_HANDLE tracking_state_set_handle);

/**
 * @brief Returns the parameters to implement foveated rendering.
 *
 * @param tracking_state_set_handle Handle to the tracking state set
 * @return The latest foveated rendering state.
 *
 * @rst
 * .. note:: See :ref:`about_foveated_rendering` for more information.
 * @endrst
 */
EW_BET_API_DECL const EW_BET_FoveatedRenderingState *
EW_BET_API_GetFoveatedRenderingState(EW_BET_TRACKING_STATE_SET_HANDLE tracking_state_set_handle);


/**
 * @brief Compute the transform parameters to apply to the in-game camera.
 *
 * @param camera_state The current state of the in-game camera.
 * @param eye_tracking_weight The weight of the eye tracking component.
 * @param head_tracking_weight The weight of the head tracking component.
 * @return The transform parameters to apply to the in-game camera.
 *
 * @rst
 * See further explanation in the ``C++`` API function
 * :cpp:func:`eyeware::beam_eye_tracker::API::compute_sim_game_camera_transform_parameters`.
 * @endrst
 */
EW_BET_API_DECL EW_BET_SimCameraTransform3D EW_BET_API_ComputeSimGameCameraTransformParameters(
    EW_BET_SimGameCameraState *camera_state, float eye_tracking_weight, float head_tracking_weight);

/**
 * @brief Start recentering the sim game camera, adjusting it to current user state.
 *
 * @param api_handle Handle to the API instance
 * @return true if the start recentering process could be queued, false otherwise.
 *
 * @rst
 * .. note:: See :ref:`about_camera_recentering` for more information.
 * @endrst
 */
EW_BET_API_DECL bool EW_BET_API_RecenterSimGameCameraStart(EW_BET_API_HANDLE api_handle);

/**
 * End recentering the sim game camera.
 *
 * @param api_handle Handle to the API instance
 *
 * @rst
 * .. note:: See :ref:`about_camera_recentering` for more information.
 * @endrst
 */
EW_BET_API_DECL void EW_BET_API_RecenterSimGameCameraEnd(EW_BET_API_HANDLE api_handle);

#ifdef __cplusplus
}
#endif

#endif // _EYEWARE_BEAM_EYE_TRACKER_C_H_
