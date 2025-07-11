/**
 * Copyright (c) 2025, Eyeware Tech SA.
 *
 * All rights reserved.
 *
 * @brief This sample demonstrates how to use the asynchronous data access method.
 *
 * @warning: the TrackingListener instance must be alive in between the
 * start_receiving_tracking_data_on_listener and the stop_receiving_tracking_data_on_listener calls.
 */
#include "../bet_sample_utils.h"
#include "eyeware/beam_eye_tracker.h"
#include <chrono>
#include <thread>

class MyTrackingListener : public eyeware::beam_eye_tracker::TrackingListener {

  public:
    MyTrackingListener(eyeware::beam_eye_tracker::TrackingDataReceptionStatus status)
        : m_previous_status(status) {
        print_tracking_data_reception_status(status);
    }

    void on_tracking_state_set_update(
        const eyeware::beam_eye_tracker::TrackingStateSet &tracking_state_set,
        const eyeware::beam_eye_tracker::Timestamp timestamp) override {

        print_latest_tracking_state_set(tracking_state_set, timestamp);
    }

    void on_tracking_data_reception_status_changed(
        eyeware::beam_eye_tracker::TrackingDataReceptionStatus status) override {

        print_tracking_data_reception_status_if_changed(m_previous_status, status);
        m_previous_status = status;
    }

  private:
    eyeware::beam_eye_tracker::TrackingDataReceptionStatus m_previous_status =
        eyeware::beam_eye_tracker::TrackingDataReceptionStatus::NOT_RECEIVING_TRACKING_DATA;
};

void main() {
    eyeware::beam_eye_tracker::API bet_api("Async Sample",
                                           eyeware::beam_eye_tracker::ViewportGeometry{0, 0, 0, 0});

    // Create an instance of your custom listener, which is the class receiving updates.
    MyTrackingListener async_listener(bet_api.get_tracking_data_reception_status());

    // Start receiving tracking data.
    eyeware::beam_eye_tracker::TRACKING_LISTENER_HANDLE listener_handle =
        bet_api.start_receiving_tracking_data_on_listener(&async_listener);

    // ... Do whatever you need to do ;-)
    std::this_thread::sleep_for(std::chrono::seconds(30));

    bet_api.stop_receiving_tracking_data_on_listener(listener_handle);
}