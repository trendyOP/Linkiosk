/**
 * Copyright (C) 2025 Eyeware Tech SA
 *
 * All rights reserved
 *
 * @file beam_eye_tracker.h
 * @brief Main C++ API for the Beam Eye Tracker SDK
 */
#ifndef _EYEWARE_BEAM_EYE_TRACKER_H_
#define _EYEWARE_BEAM_EYE_TRACKER_H_

#include "beam_eye_tracker/config.h"
#include "beam_eye_tracker/types.h"

namespace eyeware {
namespace beam_eye_tracker {

/**
 * @brief Default timeout in milliseconds for tracking data retrieval operations
 *
 * @see API::wait_for_new_tracking_state_set()
 */
constexpr unsigned int DEFAULT_TRACKING_DATA_TIMEOUT_MS = 1000;

// Forward declaration of implementation details
class TrackingStateSetData;

// Forward declarations of public classes, declared below.
class TrackingListener;
class TrackingStateSet;

typedef uint64_t TRACKING_LISTENER_HANDLE;
constexpr TRACKING_LISTENER_HANDLE INVALID_TRACKING_LISTENER_HANDLE = 0;

/**
 * @brief Main entry point for the Beam Eye Tracker SDK
 *
 * @rst
 * .. note:: See :ref:`about_api_singleton` for a high-level explanation of the ``API`` object.
 * @endrst
 *
 * @see TrackingListener For asynchronous data reception
 */
class EW_BET_API_DECL API {
  public:
    /**
     * @brief Create a new API instance to communicate with the Beam Eye Tracker application.
     *
     * @param friendly_name Application identifier displayed in the Beam Eye Tracker UI (UTF-8, max
     * 200 bytes)
     * @param initial_viewport_geometry Initial screen viewport configuration for coordinate mapping
     *
     * @throw std::invalid_argument If friendly_name is invalid or exceeds size limit
     * @throw std::runtime_error If API initialization fails
     */
    API(const char *friendly_name, const ViewportGeometry &initial_viewport_geometry);

    // Not copyable
    API(const API &) = delete;
    API &operator=(const API &) = delete;

    // Movable
    API(API &&);
    API &operator=(API &&);
    ~API();

    /**
     * @brief Get the current SDK version information
     *
     * @return Version structure containing major, minor, patch and build numbers
     *
     * Example:
     * @code{.cpp}
     * Version ver = api.get_version();
     * printf("SDK Version: %d.%d.%d.%d\n",
     *        ver.major, ver.minor, ver.patch, ver.build);
     * @endcode
     */
    Version get_version() const;

    /**
     * @brief Update the viewport geometry for coordinate mapping
     *
     * @param new_viewport_geometry The new viewport geometry to set.
     *
     * @rst
     * .. note:: For a detailed explanation, see :ref:`viewport`.
     * @endrst
     */
    void update_viewport_geometry(const ViewportGeometry &new_viewport_geometry);

    /**
     * @brief Attempts to start the Beam Eye Tracker application and tracking output.
     *
     * @rst
     * .. note:: See :ref:`about_auto_start_tracking` for more information.
     * @endrst
     */
    void attempt_starting_the_beam_eye_tracker();

    /********************************************************************************
     *                 Asynchronous tracking data access
     ********************************************************************************/

    /**
     * @brief Register a listener for asynchronous tracking data updates
     *
     * @param listener Pointer to listener implementation
     * @return Handle for the registered listener
     *
     * @warning The listener must remain valid until unregistered
     *
     * @rst
     * .. note:: For a detailed explanation, see :ref:`about_asynchronous_data_access`.
     * @endrst
     */
    TRACKING_LISTENER_HANDLE start_receiving_tracking_data_on_listener(TrackingListener *listener);

    /**
     * Deregisters the listener from receiving data from the Beam Eye Tracker application.
     *
     * @param listener_handle The handle to the listener to stop receiving tracking data.
     */
    void stop_receiving_tracking_data_on_listener(TRACKING_LISTENER_HANDLE listener_handle);

    /********************************************************************************
     *                 Synchronous tracking data access
     ********************************************************************************/

    /**
     * Waits until new tracking data becomes available. This is a blocking call
     * lasting until @ref timeout_ms milliseconds.
     *
     * @param last_update_timestamp [in,out] On input: the timestamp of the last received frame,
     *        used to determine if new data is available. On output: if new data is available,
     *        this will be updated to the timestamp of the newly received frame.
     *        Prior to the first iteration, you can initialize to @ref NULL_DATA_TIMESTAMP.
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
    bool
    wait_for_new_tracking_state_set(Timestamp &last_update_timestamp,
                                    unsigned int timeout_ms = DEFAULT_TRACKING_DATA_TIMEOUT_MS);

    /**
     * Returns the latest tracking state set. See @ref TrackingStateSet for more information.
     *
     * @rst
     * .. note:: See :ref:`about_tracking_state_set` for more information.
     * @endrst
     *
     * @return The latest tracking state set.
     */
    TrackingStateSet get_latest_tracking_state_set() const;

    /**
     * Returns the current status of the tracking data reception.
     *
     * @return The current status of the tracking data reception.
     *
     * @rst
     * .. note:: See :ref:`about_tracking_data_reception_status` for more information.
     * @endrst
     */
    TrackingDataReceptionStatus get_tracking_data_reception_status() const;

    /********************************************************************************
     *                 Sim game camera controls utils
     ********************************************************************************/

    /**
     * @brief Computes the transform you should apply to the in-game camera.
     *
     * @param state The current state of the in-game camera.
     * @param eye_tracking_weight The weight of the eye tracking component.
     * @param head_tracking_weight The weight of the head tracking component.
     *
     * Use the weight parameters to control the contributions of the eye and head tracking data. A
     * weight of 1.0 (for both) is the default, which would apply the camera movement as configured
     * by the user within the Beam Eye Tracker (which may have already applied curve mappings,
     * amplification/suppression or choosing only head or eye tracking).
     *
     * The weight values values affect how much the eye/head tracking influences camera
     * movement. A weight of 2.0 will make the camera move twice as much for the same head/eye
     * movement, while 0.5 will make it move half as much.
     *
     * @rst
     * .. note:: See :ref:`about_sim_game_camera_sensitivity_range_sliders` for more information.
     * @endrst
     * @return The transform parameters to apply to the in-game camera.
     */
    static SimCameraTransform3D
    compute_sim_game_camera_transform_parameters(const SimGameCameraState &state,
                                                 float eye_tracking_weight = 1.0f,
                                                 float head_tracking_weight = 1.0f);

    /**
     * @brief Start recentering the sim game camera, adjusting it to current user state.
     *
     * @return true if the start recentering process could be queued.
     *
     * @rst
     * .. note:: See :ref:`about_camera_recentering` for more information.
     * @endrst
     */
    bool recenter_sim_game_camera_start();

    /**
     * @brief End recentering the sim game camera.
     *
     * @rst
     * .. note:: See :ref:`about_camera_recentering` for more information.
     * @endrst
     */
    void recenter_sim_game_camera_end();

  private:
    class Impl;
    Impl *m_pimpl;
};

/**
 * @brief The TrackingStateSet is the key object holding tracking data for a time instanct.
 *
 * @rst
 * .. note:: See :ref:`about_tracking_state_set` for more information.
 * @endrst
 */
class EW_BET_API_DECL TrackingStateSet {
  public:
    /**
     * The TrackingStateSet is intended as an inmmutable data holder, which is only
     * generated by the API class and thus it is only movable by client code.
     */
    TrackingStateSet(TrackingStateSet &&);
    TrackingStateSet &operator=(TrackingStateSet &&);
    ~TrackingStateSet();

    /**
     * Returns the user state including the user's gaze on screen and head pose.
     *
     * @return The user state.
     *
     * @rst
     * .. note:: See :ref:`about_real_time_tracking` for more information.
     * @endrst
     */
    const UserState &user_state() const;

    /**
     * Access the latest game camera state to implement the immersive in-game camera controls.
     *
     * @return The latest camera controls.
     *
     * @rst
     * .. note:: See :ref:`about_sim_game_camera_state` for more information.
     * @endrst
     */
    const SimGameCameraState &sim_game_camera_state() const;

    /**
     * Returns the parameters to implement an immersive HUD in your game.
     *
     * @rst
     * .. note:: See :ref:`about_game_immersive_hud_state` for more information.
     * @endrst
     *
     * @return The latest game immersive HUD state.
     */
    const GameImmersiveHUDState &game_immersive_hud_state() const;

    /**
     * Returns the parameters to implement foveated rendering.
     *
     * @return The latest foveated rendering state.
     */
    const FoveatedRenderingState &foveated_rendering_state() const;

  private:
    friend class APIImpl;
    TrackingStateSet();
    TrackingStateSet(const TrackingStateSet &) = delete;
    TrackingStateSet &operator=(const TrackingStateSet &) = delete;
    TrackingStateSetData *m_pimpl;
};

/**
 * @brief Interface class which you need to inherit and override with your own callback logic.
 *
 * @rst
 * .. note:: See :ref:`about_asynchronous_data_access` for more information.
 * @endrst
 */
class EW_BET_API_DECL TrackingListener {
  public:
    virtual ~TrackingListener() = default;
    /**
     * Reimplement this method to receive the status of the tracking data reception.
     *
     * @param status The new status for the tracking data reception.
     */
    virtual void on_tracking_data_reception_status_changed(TrackingDataReceptionStatus status) = 0;

    /**
     * Reimplement this method to access the latest tracking data as soon as it arrives.
     *
     * @param state_set The latest tracking data.
     */
    virtual void on_tracking_state_set_update(const TrackingStateSet &tracking_state_set,
                                              const Timestamp timestamp) = 0;

  protected:
    TrackingListener() = default;
};

} // namespace beam_eye_tracker
} // namespace eyeware

#endif