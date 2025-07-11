/**
 * Copyright (c) 2025, Eyeware Tech SA.
 *
 * All rights reserved.
 *
 * @brief This sample demonstrates how to use the polling data access method.
 */
#include "../bet_sample_utils.h"
#include "eyeware/beam_eye_tracker.h"
#include <chrono>
#include <memory>
#include <thread>

void main() {
    eyeware::beam_eye_tracker::API bet_api("Polling Sample",
                                           eyeware::beam_eye_tracker::ViewportGeometry());

    // We hold a timestamp instance useful to sync the data reception.
    eyeware::beam_eye_tracker::Timestamp last_update_timestamp_sec{
        eyeware::beam_eye_tracker::NULL_DATA_TIMESTAMP};

    std::unique_ptr<eyeware::beam_eye_tracker::TrackingStateSet> last_received_tracking_state_set;

    eyeware::beam_eye_tracker::TrackingDataReceptionStatus previous_status =
        bet_api.get_tracking_data_reception_status();

    print_tracking_data_reception_status(previous_status);

    int count = 0;
    // Access 1 minute of data (assuming a 10fps events)
    while (count < 10 * 60) {

        // Follow up with the status of the tracking data reception.
        eyeware::beam_eye_tracker::TrackingDataReceptionStatus status =
            bet_api.get_tracking_data_reception_status();
        print_tracking_data_reception_status_if_changed(previous_status, status);
        previous_status = status;

        // Polling for new data follows the synchronous data access method, but with a timeout of
        // 0ms. This means that the function will return immediately.
        // Similarly, last_update_timestamp_sec is updated in case of new data.
        if (bet_api.wait_for_new_tracking_state_set(last_update_timestamp_sec, 0)) {
            // This is how we access the latest TrackingStateSet.
            // It is only movable, thus this is a way to retain it.
            last_received_tracking_state_set =
                std::make_unique<eyeware::beam_eye_tracker::TrackingStateSet>(
                    std::move(bet_api.get_latest_tracking_state_set()));

            print_latest_tracking_state_set(*last_received_tracking_state_set,
                                            last_update_timestamp_sec);

            count++;
        }

        // To simulate an "external event" driving this thread at 10fps, we sleep 100ms to simulate
        // a thread loop driven by other events.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}