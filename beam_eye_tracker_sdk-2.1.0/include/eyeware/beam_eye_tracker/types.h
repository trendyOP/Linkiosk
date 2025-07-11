/**
 * Copyright (C) 2025 Eyeware Tech SA.
 *
 * All rights reserved.
 *
 * @file types.h
 * @brief Core type definitions for the Beam Eye Tracker SDK
 *
 * This header defines all data structures and types used across the SDK.
 * It includes coordinate systems, tracking states, and configuration types.
 */

#ifndef _EYEWARE_BEAM_EYE_TRACKER_TYPES_H_
#define _EYEWARE_BEAM_EYE_TRACKER_TYPES_H_

#include <stdint.h>

static_assert(sizeof(float) == 4, "Float must be 32-bit");
static_assert(sizeof(double) == 8, "Double must be 64-bit");

#if defined(_MSC_VER)
#define EW_BET_ALIGN(x) __declspec(align(x))
#else
#define EW_BET_ALIGN(x) __attribute__((aligned(x)))
#endif

/**
 * @brief Timestamp type for tracking data
 *
 * Represents time in seconds since tracking started. The counter may reset
 * when tracking is stopped and restarted.
 *
 * @warning Not guaranteed to be strictly monotonic due to potential tracking restarts
 */
typedef double EW_BET_Timestamp;

/**
 * @brief Special value indicating an invalid timestamp
 *
 * @rst
 * .. note:: See :ref:`about_timestamps` for more information.
 * @endrst
 */
#define EW_BET_NULL_DATA_TIMESTAMP (EW_BET_Timestamp)(-1.0)

// ----------------------------------------------------------------------------
// ------------------- EW_BET_TrackingDataReceptionStatus ---------------------
// ----------------------------------------------------------------------------
/**
 * @brief  Represents the status of the tracking data reception.
 *
 * This state indicates whether the client is receiving frame-by-frame tracking data or not,
 * regardless of whether the user is being tracked or not.
 *
 * @rst
 * Alternatively, see :ref:`about_tracking_data_reception_status` for more information.
 * @endrst
 */
typedef int32_t EW_BET_TrackingDataReceptionStatus;

/**
 * @brief The client is not currently receiving data from the Beam Eye Tracker.
 *
 * There could be multiple reasons why this is the case but in general it means that the user
 * should manually start the Beam Eye Tracker application (if not yet launched), sign in, and/or
 * successfully activate "Gaming Extensions" (as of Beam Eye Tracker v2.0).
 *
 *
 *
 * @rst
 * In general, this is when manual user intervention is needed to configure the tracker.
 * Alternatively, see :ref:`about_auto_start_tracking`.
 * @endrst
 */
#define EW_BET_NOT_RECEIVING_TRACKING_DATA (EW_BET_TrackingDataReceptionStatus)0

/**
 * @brief It is actively connected to the Beam Eye Tracker and regularly receiving tracking data.
 *
 * Please note this does not imply that the user is being successfully tracked.
 * It merely indicates that the Beam Eye Tracker is active and sending updates,
 * even if the user is not being tracked.
 */
#define EW_BET_RECEIVING_TRACKING_DATA (EW_BET_TrackingDataReceptionStatus)1

/**
 * @rst
 * It is trying to to launch the Beam Eye Tracker and/or start its tracking after a explicit
 * request to :ref:`about_auto_start_tracking`.
 * @endrst
 *
 * While in this state, there are multiple things that could be happening behind the scenes:
 *
 * - Checking if the Beam Eye Tracker is installed.
 * - Checking if the Beam Eye Tracker is running.
 * - Launching the Beam Eye Tracker.
 * - Requesting the Beam Eye Tracker to activate Gaming Extensions (or other API enabling mode)
 * - ...
 *
 * Thus, depending on the state of the Beam Eye Tracker, this could fail, succeed quickly (~100ms)
 * or succeed taking a while (~10 seconds).
 *
 */
#define EW_BET_ATTEMPTING_TRACKING_AUTO_START (EW_BET_TrackingDataReceptionStatus)2

// ----------------------------------------------------------------------------
// ------------------- EW_BET_TrackingConfidence -------------------------------
// ----------------------------------------------------------------------------
/**
 * @brief Realibility measure for obtained tracking results.
 *
 * @rst
 * .. note:: See :ref:`tracking_confidence` for a detailed explanation.
 * @endrst
 */
typedef int32_t EW_BET_TrackingConfidence;
/**
 * The signal/data in question is unavailable and should be discarded.
 */
#define EW_BET_LOST_TRACKING (EW_BET_TrackingConfidence)0
/**
 * Tracking is present but highly uncertain.
 */
#define EW_BET_LOW (EW_BET_TrackingConfidence)1
/**
 * Tracking reliability is fair.
 */
#define EW_BET_MEDIUM (EW_BET_TrackingConfidence)2
/**
 * Tracking is as reliable as it gets.
 */
#define EW_BET_HIGH (EW_BET_TrackingConfidence)3

#ifdef __cplusplus
namespace eyeware {
namespace beam_eye_tracker {
#endif

// ----------------------------------------------------------------------------
// ---------------------------  Structs ---------------------------------------
// ----------------------------------------------------------------------------

/**
 * @brief Test for Doxygen
 */
struct DoxyTest {
    int a;
    int b;
};

/**
 * @brief SDK version information
 */
typedef struct EW_BET_ALIGN(8) EW_BET_Version {
    uint32_t major; /**< Major version number */
    uint32_t minor; /**< Minor version number */
    uint32_t patch; /**< Patch level */
    uint32_t build; /**< Build number */
} EW_BET_Version;

/**
 * @brief 2D integer point coordinates
 *
 * Used primarily for screen coordinates in the unified coordinate system
 */
typedef struct EW_BET_ALIGN(8) EW_BET_Point {
    int32_t x; /**< X coordinate */
    int32_t y; /**< Y coordinate */
} EW_BET_Point;

/**
 * @brief 2D floating point coordinates
 *
 * Used for normalized viewport coordinates and precise positioning
 */
typedef struct EW_BET_ALIGN(8) EW_BET_PointF {
    float x; /**< X coordinate */
    float y; /**< Y coordinate */
} EW_BET_PointF;

/**
 * @brief Viewport geometry definition.
 *
 * It is used to map from unified screen coordinates to the viewport normalized coordinates, ranging
 * [0.0, 1.0] for a point inside the viewport rectangle.
 *
 * @rst
 * .. note:: See :ref:`viewport` for more information.
 * @endrst
 */
typedef struct EW_BET_ALIGN(8) EW_BET_ViewportGeometry {
    /**
     * Point, in unified screen coordinates, where the (0.0, 0.0) point of the viewport is.
     *
     * The point is inclusive, i.e., considered part of the border of the viewport rectangle.
     */
    EW_BET_Point point_00;
    /**
     * Point, in unified screen coordinates, where the (1.0, 1.0) point of the viewport is.
     *
     * The point is inclusive, i.e., considered part of the border of the viewport rectangle. Thus:
     *
     * width = point_11.x + 1 - point_00.x
     * height = point_11.y + 1 - point_00.y
     */
    EW_BET_Point point_11;
} EW_BET_ViewportGeometry;

/**
 * Matrix of 3x3, implemented as an array of arrays (row-major).
 *
 * \code{.cpp}
 * Matrix3x3 my_matrix; // Assume a Matrix3x3 instance is available
 * int row = 1;
 * int col = 2;
 * float coefficient = my_matrix[row][col];
 * \endcode
 *
 */
typedef float EW_BET_Matrix3x3[3][3];

/**
 * Representation of a 3D vector or 3D point.
 */
typedef struct EW_BET_ALIGN(8) EW_BET_Vector3D {
    /**
     * x coordinate.
     */
    float x;
    /**
     * y coordinate.
     */
    float y;
    /**
     * z coordinate.
     */
    float z;
    uint32_t _padding;
} EW_BET_Vector3D;

/**
 * Represents information on how the user attention relates to the plugged-in displays.
 *
 * Point coordinates are referred to the unified screen coordinate system. Accuracy is expected to
 * be lower for the screens for which the eye tracking was not calibrated, and that lead to the
 * users head to have large angles with respect to the camera, when looking at them.
 */
typedef struct EW_BET_ALIGN(8) EW_BET_UnifiedScreenGaze {
    /**
     * The confidence of the tracking result.
     */
    EW_BET_TrackingConfidence confidence;

    /**
     * Point where the user is looking at, kept within bounds of the screen(s) resolution(s).
     */
    EW_BET_Point point_of_regard;

    /**
     * Point where the user is looking at, which may be outside the physical screen space.
     *
     * This alternative to @ref point_of_regard is important because:
     * - having a continuous signal crossing the screen boundaries is useful for smoother
     *   animations, or controlling elements that are not confined to the screen (e.g. the
     *   eye tracking overlay implemented in the Beam Eye Tracker software);
     * - it allows yo to implement heuristics to account for eye tracker inaccuracies
     *   nearby the screen bounds.
     */
    EW_BET_Point unbounded_point_of_regard;

} EW_BET_UnifiedScreenGaze;

typedef struct EW_BET_ALIGN(8) EW_BET_ViewportGaze {
    /**
     * The confidence of the tracking result.
     */
    EW_BET_TrackingConfidence confidence;

    /**
     * Point where the user is looking at, normalized such that, if the gaze is inside the viewport,
     * then the values are in the range [0, 1]. However, it can be negative or exceed 1, if the
     * gaze is outside the viewport.
     */
    EW_BET_PointF normalized_point_of_regard;

} EW_BET_ViewportGaze;

/**
 * Represents information of the head pose for the given time instant.
 *
 * @rst
 * .. note:: See :ref:`head_pose` for a detailed explanation.
 * @endrst
 */
typedef struct EW_BET_ALIGN(8) EW_BET_HeadPose {
    /**
     * The confidence of the tracking result.
     */
    EW_BET_TrackingConfidence confidence;

    /**
     * Rotation matrix, with respect to the World Coordinate System (WCS).
     */
    EW_BET_Matrix3x3 rotation_from_hcs_to_wcs;

    /**
     * Translation vector, with respect to the World Coordinate System (WCS).
     */
    EW_BET_Vector3D translation_from_hcs_to_wcs;

    /**
     * Indicates the ID of the session of uninterrupted consecutive tracking.
     *
     * When a user is being tracked over consecutive frames, the track_session_uid
     * is kept unchanged. However, if the user goes out of frame or turns around such
     * that they can no longer be tracker, then this number is incremented once the
     * user is detected again.
     */
    uint64_t track_session_uid;
} EW_BET_HeadPose;

/**
 * @brief Complete user tracking state
 *
 * @rst
 * .. note:: See :ref:`about_real_time_tracking` for a detailed explanation.
 * @endrst
 */
typedef struct EW_BET_ALIGN(8) EW_BET_UserState {
    uint64_t struct_version;                      /**< Structure version for compatibility */
    EW_BET_Timestamp timestamp_in_seconds;        /**< Data timestamp */
    EW_BET_HeadPose head_pose;                    /**< 3D head position and orientation */
    EW_BET_UnifiedScreenGaze unified_screen_gaze; /**< Gaze in screen coordinates */
    EW_BET_ViewportGaze viewport_gaze;            /**< Normalized viewport gaze */
    uint8_t reserved[128];                        /**< Reserved for future use */
} EW_BET_UserState;

/**
 * @brief Represents the 3D transform parameters to be applied to the in-game camera.
 *
 * @rst
 * .. warning:: Please see :ref:`sim_mapping` for explanation on how to interpret the parameters.
 * @endrst
 */
typedef struct EW_BET_ALIGN(8) EW_BET_SimCameraTransform3D {
    /**
     * Roll, in radians.
     */
    float roll_in_radians;
    /**
     * Pitch, in radians.
     */
    float pitch_in_radians;
    /**
     * Yaw, in radians.
     */
    float yaw_in_radians;
    /**
     * X translation, in meters.
     */
    float x_in_meters;
    /**
     * Y translation, in meters.
     */
    float y_in_meters;
    /**
     * Z translation, in meters.
     */
    float z_in_meters;
} EW_BET_SimCameraTransform3D;

/**
 * @brief Holds the required data to achieve real-time immersive controls of the in-game camera.
 *
 * To consume the parameters, we do not recommend accessing the @ref eye_tracking_pose_component
 * and @ref head_tracking_pose_component directly, but instead, use the provided method that
 * applies a weighted combination of the two components.
 *
 * @rst
 * .. note:: See :ref:`about_sim_game_camera_state` for a detailed explanation.
 * @endrst
 */
typedef struct EW_BET_ALIGN(8) EW_BET_SimGameCameraState {
    uint64_t struct_version; // Struct verion
    /**
     * @brief The timestamp of this update, in seconds. If it is equal to
     * EW_BET_NULL_DATA_TIMESTAMP, then the rest of the data is invalid and should be ignored.
     *
     * This is effectively a counter since the tracking started. Note that given that the user can
     * turn off/on the tracking at will, this counter can't be assumed to be strictly monotonic.
     */
    EW_BET_Timestamp timestamp_in_seconds;

    /**
     * The camera transform if based solely on the eye tracking data.
     *
     * See @ref EW_BET_SimCameraTransform3D for interpretation of the parameters.
     *
     * We do not recommend using this signal and, instead, use the @ref get method to get the final
     * camera transform.
     */
    EW_BET_SimCameraTransform3D eye_tracking_pose_component;

    /**
     * The camera transform if based solely on the head tracking data.
     *
     * See @ref EW_BET_SimCameraTransform3D for interpretation of the parameters.
     *
     * We do not recommend using this signal and, instead, use the @ref get method to get the final
     * camera transform.
     */
    EW_BET_SimCameraTransform3D head_tracking_pose_component;

    // For future use
    uint64_t reserved[128];
} EW_BET_SimGameCameraState;

/**
 * @brief Represents the information you need to implement an immersive HUD in your game.
 *
 * In many games, the HUD is implemented by UI elements on the 4 corners of the screen, but this
 * struct provides values for all non-center 8 regions of the screen (corners and mid-edges).
 *
 * The values are in the range [0, 1], where 0 means the user is not looking at the element, and 1
 * means the user is looking at the element. In most cases, you can simply map to a boolean value
 * using 0.5 as threshold.
 *
 * @rst
 * .. note:: See :ref:`about_game_immersive_hud_ready_to_use_signals` for more information.
 * @endrst
 */
typedef struct EW_BET_ALIGN(8) EW_BET_GameImmersiveHUDState {
    uint64_t struct_version; // Struct version
    /**
     * @brief The timestamp of this update, in seconds. If it is equal to
     * EW_BET_NULL_DATA_TIMESTAMP, then the rest of the data is invalid and should be ignored.
     *
     * This is effectively a counter since the tracking started. Note that given that the user can
     * turn off/on the tracking at will, this counter can't be assumed to be strictly monotonic.
     */
    EW_BET_Timestamp timestamp_in_seconds;

    /**
     * @brief Signals of whether the user is looking at the top-left region of the screen.
     */
    float looking_at_viewport_top_left;

    /**
     * @brief Signal of whether the user is looking at the top-middle region of the screen.
     */
    float looking_at_viewport_top_middle;

    /**
     * @brief Signal of whether the user is looking at the top-right region of the screen.
     */
    float looking_at_viewport_top_right;

    /**
     * @brief Signal of whether the user is looking at the center-left region of the screen.
     */
    float looking_at_viewport_center_left;

    /**
     * @brief Signal of whether the user is looking at the center-right region of the screen.
     */
    float looking_at_viewport_center_right;

    /**
     * @brief Signal of whether the user is looking at the bottom-left region of the screen.
     */
    float looking_at_viewport_bottom_left;

    /**
     * @brief Signal of whether the user is looking at the bottom-middle region of the screen.
     */
    float looking_at_viewport_bottom_middle;

    /**
     * @brief Signal of whether the user is looking at the bottom-right region of the screen.
     */
    float looking_at_viewport_bottom_right;

    // For future use
    uint8_t reserved[128];
} EW_BET_GameImmersiveHUDState;

/**
 * Representation of the radii of the foveated rendering regions.
 */
typedef struct EW_BET_ALIGN(8) EW_BET_FoveationRadii {
    /**
     * Inner area should be rendered at highest definition.
     */
    float radius_level_1;
    /**
     * Second level of definition.
     */
    float radius_level_2;
    /**
     * Third level of definition.
     */
    float radius_level_3;
    /**
     * Outer area should be rendered at lowest definition.
     */
    float radius_level_4;
} EW_BET_FoveationRadii;

/**
 * @brief Holds the required data to achieve foveated rendering.
 *
 * @rst
 * .. note:: See :ref:`about_foveated_rendering` for a detailed explanation.
 * @endrst
 */
typedef struct EW_BET_ALIGN(8) EW_BET_FoveatedRenderingState {
    uint64_t struct_version; // Struct verion
    /**
     * @brief The timestamp of this update, in seconds. If it is equal to
     * EW_BET_NULL_DATA_TIMESTAMP, then the rest of the data is invalid and should be ignored.
     *
     * This is effectively a counter since the tracking started. Note that given that the user can
     * turn off/on the tracking at will, this counter can't be assumed to be strictly monotonic.
     */
    EW_BET_Timestamp timestamp_in_seconds;

    /**
     * Point where to place the foveated rendering regions, it is normalized
     * by the viewport width and height like @ref EW_BET_ViewportGaze::normalized_point_of_regard.
     */
    EW_BET_PointF normalized_foveation_center;

    /**
     * The radii of the foveated rendering regions normalized by the viewport width.
     *
     * See @ref EW_BET_FoveationRadii for interpretation of the parameters.
     */
    EW_BET_FoveationRadii normalized_foveation_radii;

    // For future use
    uint64_t reserved[128];
} EW_BET_FoveatedRenderingState;

#ifdef __cplusplus

// C++ style aliases (accessible through the eyeware::beam_eye_tracker namespace)
using Timestamp = EW_BET_Timestamp;

/**
 * @brief See @ref EW_BET_NULL_DATA_TIMESTAMP
 */
constexpr Timestamp NULL_DATA_TIMESTAMP = EW_BET_NULL_DATA_TIMESTAMP;

enum class TrackingDataReceptionStatus : EW_BET_TrackingDataReceptionStatus {
    /**
     * @brief Same meaning as @ref EW_BET_NOT_RECEIVING_TRACKING_DATA
     */
    NOT_RECEIVING_TRACKING_DATA = EW_BET_NOT_RECEIVING_TRACKING_DATA,
    /**
     * @brief Same meaning as @ref EW_BET_RECEIVING_TRACKING_DATA
     */
    RECEIVING_TRACKING_DATA = EW_BET_RECEIVING_TRACKING_DATA,
    /**
     * @brief Same meaning as @ref EW_BET_ATTEMPTING_TRACKING_AUTO_START
     */
    ATTEMPTING_TRACKING_AUTO_START = EW_BET_ATTEMPTING_TRACKING_AUTO_START,
};

/**
 * @brief See @ref EW_BET_TrackingConfidence
 */
enum class TrackingConfidence : EW_BET_TrackingConfidence {
    LOST_TRACKING = EW_BET_LOST_TRACKING, /**< Same meaning as @ref EW_BET_LOST_TRACKING */
    LOW = EW_BET_LOW,                     /**< Same meaning as @ref EW_BET_LOW */
    MEDIUM = EW_BET_MEDIUM,               /**< Same meaning as @ref EW_BET_MEDIUM */
    HIGH = EW_BET_HIGH,                   /**< Same meaning as @ref EW_BET_HIGH */
};

using Version = EW_BET_Version;
using Point = EW_BET_Point;
using PointF = EW_BET_PointF;
using ViewportGeometry = EW_BET_ViewportGeometry;
using Matrix3x3 = EW_BET_Matrix3x3;
using Vector3D = EW_BET_Vector3D;
using UnifiedScreenGaze = EW_BET_UnifiedScreenGaze;
using ViewportGaze = EW_BET_ViewportGaze;
using HeadPose = EW_BET_HeadPose;
using UserState = EW_BET_UserState;
using SimGameCameraState = EW_BET_SimGameCameraState;
using SimCameraTransform3D = EW_BET_SimCameraTransform3D;
using GameImmersiveHUDState = EW_BET_GameImmersiveHUDState;
using FoveationRadii = EW_BET_FoveationRadii;
using FoveatedRenderingState = EW_BET_FoveatedRenderingState;

/**
 * Convenience function to cast from the C-style enum confidence, which is a member
 * of most tracking state structs, to the C++ enum.
 */
inline TrackingConfidence cast_confidence(EW_BET_TrackingConfidence confidence) {
    return static_cast<TrackingConfidence>(confidence);
}

} // namespace beam_eye_tracker
} // namespace eyeware

#endif
#endif //_EYEWARE_EW_BET_TYPES_H_
