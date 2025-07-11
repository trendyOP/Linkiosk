/**
 * Copyright (c) 2025, Eyeware Tech SA.
 *
 * All rights reserved.
 *
 * @brief This file demonstrates how to interpret the received data.
 *
 * @note The timestamp fields are very informative within the API. They convey whether a data
 * structure is valid at all (when holding NULL_DATA_TIMESTAMP) and whether "State" components  (eg.
 * UserState, SimGameCameraState, etc.) are running at different framerates. These functions and
 * comments are meant to provide guidance on how to interpret them. Accessing other data fields of a
 * data structure with NULL_DATA_TIMESTAMP is undefined behavior.
 */
#ifndef BET_SAMPLES_UTILS_H
#define BET_SAMPLES_UTILS_H

#include "eyeware/beam_eye_tracker.h"
#include <iostream>

using eyeware::beam_eye_tracker::NULL_DATA_TIMESTAMP;
using eyeware::beam_eye_tracker::TrackingDataReceptionStatus;

/**
 * @brief Prints a message indicating that no tracking data is being received.
 *
 * We just want this function to make the flow a bit more explicit in console.
 */
void print_tracking_data_reception_status(TrackingDataReceptionStatus status) {
    std::cout << "****************************************************" << std::endl;
    switch (status) {
    case TrackingDataReceptionStatus::RECEIVING_TRACKING_DATA:
        std::cout << "Receiving tracking data." << std::endl;
        break;
    case TrackingDataReceptionStatus::NOT_RECEIVING_TRACKING_DATA:
        std::cout << "Not receiving tracking data." << std::endl;
        break;
    case TrackingDataReceptionStatus::ATTEMPTING_TRACKING_AUTO_START:
        std::cout << "Attempting to auto start tracking." << std::endl;
        break;
    }
    std::cout << "****************************************************" << std::endl;
}

/**
 * @brief Shows how to interpret the tracking data reception status.
 */
void print_tracking_data_reception_status_if_changed(TrackingDataReceptionStatus previous_status,
                                                     TrackingDataReceptionStatus status) {
    if (previous_status == status) {
        // No change in the status.
        return;
    }
    std::cout << "****************************************************" << std::endl;
    switch (status) {
    case TrackingDataReceptionStatus::RECEIVING_TRACKING_DATA:
        if (previous_status == TrackingDataReceptionStatus::ATTEMPTING_TRACKING_AUTO_START) {
            std::cout << "Successfully auto started the Beam Eye Tracker" << std::endl;
        } else {
            std::cout << "Receiving tracking data." << std::endl;
        }
        break;
    case TrackingDataReceptionStatus::NOT_RECEIVING_TRACKING_DATA:
        if (previous_status == TrackingDataReceptionStatus::ATTEMPTING_TRACKING_AUTO_START) {
            std::cout << "Failed to auto start the Beam Eye Tracker" << std::endl;
        } else {
            std::cout << "Not receiving tracking data." << std::endl;
        }
        break;
    case TrackingDataReceptionStatus::ATTEMPTING_TRACKING_AUTO_START:
        std::cout << "Attempting to auto start tracking." << std::endl;
        break;
    }
    std::cout << "****************************************************" << std::endl;
}

/**
 * @brief Shows how to interpret the user state.
 */
void print_user_state(const eyeware::beam_eye_tracker::UserState &user_state) {
    if (user_state.timestamp_in_seconds == NULL_DATA_TIMESTAMP) {
        std::cout << "UserState data is not valid and should be ignored. " << std::endl;
        return;
    }

    if (eyeware::beam_eye_tracker::cast_confidence(user_state.head_pose.confidence) ==
        eyeware::beam_eye_tracker::TrackingConfidence::LOST_TRACKING) {
        // Scenario where the user face is not detected, the user goes away from frame, etc.
        // @warning: Using data from other fields of the UserState is undefined behavior.
        std::cout << "Tracking is active but the user is not being tracked. " << std::endl;
        return;
    } else {
        std::cout << "Head pose: X = " << user_state.head_pose.translation_from_hcs_to_wcs.x
                  << ", Y= " << user_state.head_pose.translation_from_hcs_to_wcs.y
                  << ", Z = " << user_state.head_pose.translation_from_hcs_to_wcs.z << " ";
    }

    if (eyeware::beam_eye_tracker::cast_confidence(user_state.unified_screen_gaze.confidence) ==
        eyeware::beam_eye_tracker::TrackingConfidence::LOST_TRACKING) {
        // Screen gaze data is not being updated.
        std::cout << "User is not looking at the screen. ";
        return;
    } else {
        std::cout << "Point of regard: (" << user_state.unified_screen_gaze.point_of_regard.x << ","
                  << user_state.unified_screen_gaze.point_of_regard.y << ") ";
    }
    std::cout << std::endl;
}

void print_latest_tracking_state_set(
    const eyeware::beam_eye_tracker::TrackingStateSet &tracking_state_set,
    const eyeware::beam_eye_tracker::Timestamp timestamp_of_tracking_state_set) {

    if (timestamp_of_tracking_state_set == NULL_DATA_TIMESTAMP) {
        // Typically, if following the sync, async or polling data access methods, it is unlikely to
        // end up reading a TrackingStateSet with an associated NULL_DATA_TIMESTAMP, but we still
        // include this case for completeness.
        std::cout << "Not receiving data from the Beam Eye Tracker. ";
        return;
    }

    // In general, the async or

    // As of Beam 2.4.0, all "State" components (eg. UserState, SimGameCameraState, etc.) are
    // expected to be updated at the same time and thus holding the same timestamp as the
    // "timestamp_of_tracking_state_set" whenever the user is being tracked. However, a
    // future-proof implementation is to assume that only a subset the "State" components is
    // updated. One way of doing that, is to compare timestamps, as done in each case below.

    if (tracking_state_set.user_state().timestamp_in_seconds != NULL_DATA_TIMESTAMP &&
        tracking_state_set.user_state().timestamp_in_seconds != timestamp_of_tracking_state_set) {
        // This is the case when we keep receiving TrackingStateSet updates.
        std::cout << "UserState data not updated." << std::endl;
    } else {
        print_user_state(tracking_state_set.user_state());
    }

    // if (tracking_state_set.user_state().timestamp_in_seconds == NULL_DATA_TIMESTAMP) {
    //     std::cout << "Tracking is active but the user is not being tracked. ";
    // } else if (timestamp_of_tracking_state_set ==
    //            tracking_state_set.user_state().timestamp_in_seconds) {
    //     print_user_state(tracking_state_set.user_state());
    // } else {
    //     // This is the case when a TrackingStateSet is updated, but not the UserState it holds.
    //     std::cout << "The UserState data was  " << std::endl;
    // }

    // if (timestamp_of_tracking_state_set !=
    //     tracking_state_set.sim_game_camera_state().timestamp_in_seconds) {
    //     // print_sim_game_camera_state(tracking_state_set.sim_game_camera_state());
    // }

    // For example, just an update of the SimGameCameraState, while the latest UserState remained
    // the same.
}

#endif // BET_SAMPLES_UTILS_H
