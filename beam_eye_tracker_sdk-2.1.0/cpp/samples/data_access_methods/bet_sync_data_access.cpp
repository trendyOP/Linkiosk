/**
 * Copyright (c) 2025, Eyeware Tech SA.
 *
 * All rights reserved.
 *
 * @brief This sample demonstrates how to use the synchronous data access method.
 */
#include "../bet_sample_utils.h"
#include "eyeware/beam_eye_tracker.h"

void main() {
    eyeware::beam_eye_tracker::API bet_api("Sync Sample",
                                           eyeware::beam_eye_tracker::ViewportGeometry());

    // We hold a timestamp instance useful to sync the data reception.
    eyeware::beam_eye_tracker::Timestamp last_update_timestamp_sec{
        eyeware::beam_eye_tracker::NULL_DATA_TIMESTAMP};

    std::unique_ptr<eyeware::beam_eye_tracker::TrackingStateSet> last_received_tracking_state_set;

    eyeware::beam_eye_tracker::TrackingDataReceptionStatus previous_status =
        bet_api.get_tracking_data_reception_status();

    print_tracking_data_reception_status(previous_status);

    int count = 0;
    // Access 1 minute of data (assuming a 30fps webcam)
    while (count < 30 * 60) {
        const unsigned int wait_timeout_ms = 1000;

        // Follow up with the status of the tracking data reception.
        eyeware::beam_eye_tracker::TrackingDataReceptionStatus status =
            bet_api.get_tracking_data_reception_status();
        print_tracking_data_reception_status_if_changed(previous_status, status);
        previous_status = status;

        // Wait for a new frame. If this function returns true, then there is a new frame and
        // last_update_timestamp_sec is updated.
        if (bet_api.wait_for_new_tracking_state_set(last_update_timestamp_sec, wait_timeout_ms)) {
            // This is how we access the latest TrackingStateSet.
            // It is only movable, thus this is a way to retain it.
            last_received_tracking_state_set =
                std::make_unique<eyeware::beam_eye_tracker::TrackingStateSet>(
                    std::move(bet_api.get_latest_tracking_state_set()));

            print_latest_tracking_state_set(*last_received_tracking_state_set,
                                            last_update_timestamp_sec);
            count++;
        }
    }
}
