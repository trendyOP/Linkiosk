/**
 * Copyright (C) 2025 Eyeware Tech SA.
 *
 * All rights reserved.
 */

#ifndef BET_GAME_ENGINE_DEVICE_H
#define BET_GAME_ENGINE_DEVICE_H

#include "eyeware/beam_eye_tracker.h"
#include "my_game_engine.h"
#include <algorithm>
#include <memory>

namespace {

constexpr float M_PI = 3.14159265358979323846;
constexpr float M_RADIANS_TO_DEGREES = 180.0f / M_PI;
constexpr float M_METERS_TO_INCHES = 39.3700787;

float update_hud_opacity(float prev_opacity, bool looking_at_hud, float delta_time) {
    // Implements asymmetric linear opacity update. You can use a nicer animation curve.
    const float opacity_rate_on_looking_at_hud = 10.0f;     // Fully visible in max 0.1 seconds
    const float opacity_rate_on_not_looking_at_hud = -1.0f; // Fully invisible in max 1 seconds
    const float min_opacity = 0.2f; // In case the HUD should not fully disappear

    const float opacity_update_rate =
        looking_at_hud ? opacity_rate_on_looking_at_hud : opacity_rate_on_not_looking_at_hud;
    return std::min(std::max(prev_opacity + opacity_update_rate * delta_time, min_opacity), 1.0f);
}

} // namespace

// Note, for this sample, we keep explicit mentions to the namespace eyeware::beam_eye_tracker
// to make it easier to separate API related code.
class MyGameEngineBeamEyeTrackerDevice : public eyeware::beam_eye_tracker::TrackingListener,
                                         public MyGameEngineObjectInterface {
  public:
    MyGameEngineBeamEyeTrackerDevice(MyGameEngineObjectInterface *parent)
        : MyGameEngineObjectInterface(parent) {
        // We only need one instance of the API. You can also create it on "Begin Play" if you
        // want to, but here it is created in the constructor for simplicity and not to check
        // on pointer validity.
        m_bet_api = std::make_unique<eyeware::beam_eye_tracker::API>(
            "Game Engine Integration Sample", get_rendering_area_viewport_geometry_from_engine());
    }

    ~MyGameEngineBeamEyeTrackerDevice() { stop_bet_api_tracking_data_reception(); }

    void BeginPlay() override {
        if (m_auto_start_tracking) {
            // If auto start is toggled on, this will request the Beam app to launch and
            // or to start the webcam and initialize the tracking.
            // HEADS UP! Be wise when you call this. Ideally you want to call it when the game
            // rendering starts and accepts device input, as otherwise may start the webcam
            // at a random time and confuse the user.
            m_bet_api->attempt_starting_the_beam_eye_tracker();
        }

        // Register itself as the listener to receive tracking data from the Beam Eye Tracker
        // application on the on_tracking_state_set_update method asynchronously.
        if (m_listener_handle == eyeware::beam_eye_tracker::INVALID_TRACKING_LISTENER_HANDLE) {
            m_listener_handle = m_bet_api->start_receiving_tracking_data_on_listener(this);
        }
    }

    void EndPlay() override { stop_bet_api_tracking_data_reception(); }

    void Tick(float delta_time) override {
        // For the purpose of this sample, we assume a custom device output is the HUD opacity,
        // which is updated here.

        // Animate the opacity change depending on whether the user is looking at HUD elements.
        device_output_top_left_hud_opacity = update_hud_opacity(
            device_output_top_left_hud_opacity, m_is_user_looking_at_top_left_corner, delta_time);
        device_output_top_right_hud_opacity = update_hud_opacity(
            device_output_top_right_hud_opacity, m_is_user_looking_at_top_right_corner, delta_time);
        device_output_bottom_left_hud_opacity =
            update_hud_opacity(device_output_bottom_left_hud_opacity,
                               m_is_user_looking_at_bottom_left_corner, delta_time);
        device_output_bottom_right_hud_opacity =
            update_hud_opacity(device_output_bottom_right_hud_opacity,
                               m_is_user_looking_at_bottom_right_corner, delta_time);

        // Update viewport every 3 seconds in case the rendering area geometry changed.
        // In the Beam Eye Tracker application API, this operation is light-weight
        // so you could call it more frequently, but 3 seconds balances the trade-off between
        // slight-overhead (inc. game engine retrieval of the geometry) and responsiveness in case
        // the game was resized or moved. If you have an specific event for game window geometry
        // changes, may be better suited for this purpose.
        m_time_since_last_viewport_geometry_update += delta_time;
        if (m_time_since_last_viewport_geometry_update >= 3.0f) {
            m_bet_api->update_viewport_geometry(get_rendering_area_viewport_geometry_from_engine());
            m_time_since_last_viewport_geometry_update = 0.0f;
        }
    }

    // Functions for recentering the camera. Likely mapped to a hotkey press/release event.
    void recenter_camera_start() { m_bet_api->recenter_sim_game_camera_start(); }
    void recenter_camera_end() { m_bet_api->recenter_sim_game_camera_end(); }

    eyeware::beam_eye_tracker::ViewportGeometry get_rendering_area_viewport_geometry_from_engine() {
        // Implement here your Game Engine specific logic where you retrieve the rendering area
        // geometry, i.e., the viewport. We need to keep the eyeware::beam_eye_tracker::API up to
        // date with changes in the viewport geometry.

        // For this demo, assume this configuration: three physical monitors from left to right of
        // resolutions 1920x1080, 1920x1080, 1920x1080. The left-most monitor is configured in
        // Windows settings as the "Main display" (thus, it defines the (0, 0) coordinates in the
        // Windows Virtual Screen), and the game is rendering full screen in the center monitor.
        // Moreover, lets assume this game engine follows Unity's viewport standard, where the the
        // viewport (0, 0) coordinates are at the bottom-left corner of the rendering area.
        // Thus this coordinates would represent that configuration:
        eyeware::beam_eye_tracker::Point point_00 = {1920, 1079};
        eyeware::beam_eye_tracker::Point point_11 = {1920 + 1920 - 1, 0};

        return {point_00, point_11};
    }

    void on_tracking_data_reception_status_changed(
        eyeware::beam_eye_tracker::TrackingDataReceptionStatus status) override {

        // See TrackingDataReceptionStatus for explanation on all possible statuses.
        if (status ==
            eyeware::beam_eye_tracker::TrackingDataReceptionStatus::NOT_RECEIVING_TRACKING_DATA) {
            // If the tracking data reception status is NOT_RECEIVING_TRACKING_DATA is because the
            // Beam app is not open, the webcam is not active, the client was rejected from using
            // the API, the user is not signed in, etc. etc.. To "fix" this, manual intervention
            // from the user is required. Note that this state could be reached shortly after a call
            // to attempt_starting_the_beam_eye_tracker, which failed to achieve the auto-start.
            // You can try calling attempt_starting_the_beam_eye_tracker again, but it is a question
            // of user experience, as the user may be manually toggling off, but the game insists on
            // toggling on.

            // In this situation, makes sense to reset the device output to default values.
            // Animation curves could be used for a smoother transition.
            device_output_camera_local_transform = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

            device_output_top_left_hud_opacity = 1.0f;
            device_output_top_right_hud_opacity = 1.0f;
            device_output_bottom_left_hud_opacity = 1.0f;
            device_output_bottom_right_hud_opacity = 1.0f;

            m_is_user_looking_at_top_left_corner = true;
            m_is_user_looking_at_top_right_corner = true;
            m_is_user_looking_at_bottom_left_corner = true;
            m_is_user_looking_at_bottom_right_corner = true;
        }
    }

    void on_tracking_state_set_update(
        const eyeware::beam_eye_tracker::TrackingStateSet &tracking_state_set,
        const eyeware::beam_eye_tracker::Timestamp timestamp) override {
        // Async callback to retrieve the tracking data.

        update_device_viewport_gaze_state_from_bet_api_input(tracking_state_set.user_state());

        update_device_sim_game_camera_state_from_bet_api_input(
            tracking_state_set.sim_game_camera_state());

        update_device_game_immersive_hud_state_from_bet_api_input(
            tracking_state_set.game_immersive_hud_state());
    }

    void update_device_viewport_gaze_state_from_bet_api_input(
        const eyeware::beam_eye_tracker::UserState &user_state) {

        // Eye tracking coordinates referred to the viewport area.
        if (user_state.timestamp_in_seconds != eyeware::beam_eye_tracker::NULL_DATA_TIMESTAMP &&
            eyeware::beam_eye_tracker::cast_confidence(user_state.viewport_gaze.confidence) !=
                eyeware::beam_eye_tracker::TrackingConfidence::LOST_TRACKING) {

            // Normalized gaze coordinates in the viewport. Normalized as it is in the range [0, 1],
            // however, values outside this range are possible.
            device_output_viewport_normalized_gaze_x =
                user_state.viewport_gaze.normalized_point_of_regard.x;
            device_output_viewport_normalized_gaze_y =
                user_state.viewport_gaze.normalized_point_of_regard.y;
        }
    }

    void update_device_sim_game_camera_state_from_bet_api_input(
        const eyeware::beam_eye_tracker::SimGameCameraState &sim_game_camera_state) {

        if (sim_game_camera_state.timestamp_in_seconds !=
            eyeware::beam_eye_tracker::NULL_DATA_TIMESTAMP) {

            // Mapping sensitivity (default 0.5) to weight (default 1.0). Note this mapping
            // could be more complex, but the assumption is that a weight of 1.0 would make the
            // signal as configured by the user within the Beam Eye Tracker application.
            const float sim_game_camera_eye_tracking_weight =
                2.0 * m_sim_game_camera_eye_tracking_sensitivity;
            const float sim_game_camera_head_tracking_weight =
                2.0 * m_sim_game_camera_head_tracking_sensitivity;

            // This combines the signals into one transform.
            eyeware::beam_eye_tracker::SimCameraTransform3D bet_camera_local_transform =
                eyeware::beam_eye_tracker::API::compute_sim_game_camera_transform_parameters(
                    sim_game_camera_state, sim_game_camera_eye_tracking_weight,
                    sim_game_camera_head_tracking_weight);

            // Now, we need to map the beam eye tracker coordinates to the game engine
            // coordinates. See the documentation of the API for SimCameraTransform3D explaining
            // the API in detail. Assuming the game engine is using Unity's coordinate system,
            // which is the same as Beam, except that x is inverted, and the rotations are
            // left-handed, not right-handed. The rotation order for roll, pitch, yaw is
            // consistent with Beam's.

            // Rotations going from right-handed to left-handed coordinate system.
            device_output_camera_local_transform.rotation_x_degrees =
                bet_camera_local_transform.pitch_in_radians * M_RADIANS_TO_DEGREES;
            device_output_camera_local_transform.rotation_y_degrees =
                -bet_camera_local_transform.yaw_in_radians * M_RADIANS_TO_DEGREES;
            device_output_camera_local_transform.rotation_z_degrees =
                -bet_camera_local_transform.roll_in_radians * M_RADIANS_TO_DEGREES;

            // Translations going from right-handed to left-handed coordinate system.
            device_output_camera_local_transform.translation_x_inches =
                -bet_camera_local_transform.x_in_meters * M_METERS_TO_INCHES;
            device_output_camera_local_transform.translation_y_inches =
                bet_camera_local_transform.y_in_meters * M_METERS_TO_INCHES;
            device_output_camera_local_transform.translation_z_inches =
                bet_camera_local_transform.z_in_meters * M_METERS_TO_INCHES;

        } else {
            // There could be multiple reasons to receive a NULL_DATA_TIMESTAMP in the callback.
            // But in general it means an interruption of the normal tracking, the feature
            // itself, or other.

            // For user experience, the camera should NOT be reset to the default position
            // immediately (m_latest_sim_game_camera_transform being 0.0f), as that would be
            // confusing: imagine the user going briefly off-frame to connect a cable, but
            // suddenly the camera snaps to zero. Instead, we suggest to keep the
            // m_latest_sim_game_camera_transform as is with the latest valid data.
            //
            // However, you may also choose to set it to zeros after a reasonable time, and
            // perhaps even slowly. But that's your choice.
        }
    }

    void update_device_game_immersive_hud_state_from_bet_api_input(
        const eyeware::beam_eye_tracker::GameImmersiveHUDState &game_immersive_hud_state) {
        if (game_immersive_hud_state.timestamp_in_seconds !=
            eyeware::beam_eye_tracker::NULL_DATA_TIMESTAMP) {

            // Note: the input values are interpreted a "likelihood" or as a "probability", so
            // you can simply threshold it.
            m_is_user_looking_at_top_left_corner =
                game_immersive_hud_state.looking_at_viewport_top_left > 0.5f;
            m_is_user_looking_at_top_right_corner =
                game_immersive_hud_state.looking_at_viewport_top_right > 0.5f;
            m_is_user_looking_at_bottom_left_corner =
                game_immersive_hud_state.looking_at_viewport_bottom_left > 0.5f;
            m_is_user_looking_at_bottom_right_corner =
                game_immersive_hud_state.looking_at_viewport_bottom_right > 0.5f;

        } else {
            // There could be multiple reasons to receive a NULL_DATA_TIMESTAMP in the callback.
            // But in general it means an interruption of the normal tracking, the feature
            // itself, or other.

            // In this case, it makes sense to "reset" at set all the HUD as visible. For
            // example, assume the user is off-camera.
            m_is_user_looking_at_top_left_corner = true;
            m_is_user_looking_at_top_right_corner = true;
            m_is_user_looking_at_bottom_left_corner = true;
            m_is_user_looking_at_bottom_right_corner = true;
        }
    }

    //*********  VARIABLES ASSUMED TO BE LINKED TO IN-GAME SETTINGS ****************** */
    // Note: we don't define the setters to avoid clutter.
    /**
     * @brief Implement a user interface that allows to change this value.
     */
    bool m_auto_start_tracking = true;

    // For the in-game camera controls, assumed to be in the range [0, 1].
    float m_sim_game_camera_eye_tracking_sensitivity = 0.5f;
    float m_sim_game_camera_head_tracking_sensitivity = 0.5f;

    //************ VARIABLES REPRESENTING THE DEVICE STATE/OUTPUT ************ */
    // Note: we don't define the getters to avoid clutter.

    // Sim game camera local/additive transform
    MyGameEngineTransform device_output_camera_local_transform{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Immersive HUD state. Here it is default opaque,  i.e., all the HUD is visible until
    // tracking data says otherwise. It's different than the boolean counterparts, as the
    // opacity changes are smooth across frames.
    float device_output_top_left_hud_opacity = 1.0f;
    float device_output_top_right_hud_opacity = 1.0f;
    float device_output_bottom_left_hud_opacity = 1.0f;
    float device_output_bottom_right_hud_opacity = 1.0f;

    // Normalized gaze coordinates in the viewport.
    float device_output_viewport_normalized_gaze_x = 0.0f;
    float device_output_viewport_normalized_gaze_y = 0.0f;

    //************************************************************************************ */

  private:
    bool m_is_user_looking_at_top_left_corner = true;
    bool m_is_user_looking_at_top_right_corner = true;
    bool m_is_user_looking_at_bottom_left_corner = true;
    bool m_is_user_looking_at_bottom_right_corner = true;

    void stop_bet_api_tracking_data_reception() {
        if (m_listener_handle != eyeware::beam_eye_tracker::INVALID_TRACKING_LISTENER_HANDLE) {
            m_bet_api->stop_receiving_tracking_data_on_listener(m_listener_handle);
            m_listener_handle = eyeware::beam_eye_tracker::INVALID_TRACKING_LISTENER_HANDLE;
        }
    }

    std::unique_ptr<eyeware::beam_eye_tracker::API> m_bet_api;
    eyeware::beam_eye_tracker::TRACKING_LISTENER_HANDLE m_listener_handle{
        eyeware::beam_eye_tracker::INVALID_TRACKING_LISTENER_HANDLE};
    float m_time_since_last_viewport_geometry_update{0.0f};
};

#endif // BET_GAME_ENGINE_DEVICE_H