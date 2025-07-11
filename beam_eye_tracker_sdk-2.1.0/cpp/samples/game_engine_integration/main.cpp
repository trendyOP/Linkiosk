#include "bet_game_engine_device.h"
#include "my_game_engine.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
#include <windows.h>
namespace {
std::vector<MyGameEngineObjectInterface *> devices;
}

// See bet_game_engine_device.h for showing the "integration" of the Beam API as a device in the
// engine See this file to see how it interacts with the other objects in the engine.
class MyGameEngineImmersiveHUD : public MyGameEngineObjectInterface {
  public:
    MyGameEngineImmersiveHUD(MyGameEngineObjectInterface *parent)
        : MyGameEngineObjectInterface(parent) {}

    void BeginPlay() {
        // Get a reference to the Beam Eye Tracker device.
        beam_eye_tracker_device = dynamic_cast<MyGameEngineBeamEyeTrackerDevice *>(devices.at(0));

        // Dummy hud ui_elements added to all corners
        ui_elements.push_back(MyGameEngineHUDElement{MyGameEngineHUDElement::Type::TOP_LEFT, this});
        ui_elements.push_back(
            MyGameEngineHUDElement{MyGameEngineHUDElement::Type::TOP_RIGHT, this});
        ui_elements.push_back(
            MyGameEngineHUDElement{MyGameEngineHUDElement::Type::BOTTOM_LEFT, this});
        ui_elements.push_back(
            MyGameEngineHUDElement{MyGameEngineHUDElement::Type::BOTTOM_RIGHT, this});
    }
    // Implemented in bet_game_engine_device.cpp to identify
    void Tick(float delta_time) override {
        for (auto &element : ui_elements) {
            switch (element.type) {
            case MyGameEngineHUDElement::Type::TOP_LEFT:
                element.opacity = beam_eye_tracker_device->device_output_top_left_hud_opacity;
                break;
            case MyGameEngineHUDElement::Type::TOP_RIGHT:
                element.opacity = beam_eye_tracker_device->device_output_top_right_hud_opacity;
                break;
            case MyGameEngineHUDElement::Type::BOTTOM_LEFT:
                element.opacity = beam_eye_tracker_device->device_output_bottom_left_hud_opacity;
                break;
            case MyGameEngineHUDElement::Type::BOTTOM_RIGHT:
                element.opacity = beam_eye_tracker_device->device_output_bottom_right_hud_opacity;
                break;
            }
        }
    }

    std::vector<MyGameEngineHUDElement> ui_elements;

    MyGameEngineBeamEyeTrackerDevice *beam_eye_tracker_device{nullptr};
};

class MyGameEngineImmersiveCamera : public MyGameEngineObjectInterface {
  public:
    MyGameEngineImmersiveCamera(MyGameEngineObjectInterface *parent)
        : MyGameEngineObjectInterface(parent) {}

    void BeginPlay() override {
        // Get a reference to the Beam Eye Tracker device.
        beam_eye_tracker_device = dynamic_cast<MyGameEngineBeamEyeTrackerDevice *>(devices.at(0));
    }

    void Tick(float delta_time) override {
        // Updates the local pose.
        // What is critical to notice is that this updates the world_transform by adding up the
        // parent's world_transform with the now given local_transform.
        this->set_local_transform(beam_eye_tracker_device->device_output_camera_local_transform);
    }

    MyGameEngineBeamEyeTrackerDevice *beam_eye_tracker_device{nullptr};
};

class MyGameEngineHotkeysMapper : public MyGameEngineObjectInterface {
  public:
    MyGameEngineHotkeysMapper(MyGameEngineObjectInterface *parent)
        : MyGameEngineObjectInterface(parent) {}

    void BeginPlay() override {
        // Get a reference to the Beam Eye Tracker device.
        beam_eye_tracker_device = dynamic_cast<MyGameEngineBeamEyeTrackerDevice *>(devices.at(0));
    }
    void Tick(float delta_time) override {
        // Check if R key is pressed
        bool recenter_key_pressed = GetAsyncKeyState(VK_SPACE) & 0x8000;
        if (recenter_key_pressed != was_recenter_key_pressed) {
            if (recenter_key_pressed) {
                beam_eye_tracker_device->recenter_camera_start();
            } else {
                beam_eye_tracker_device->recenter_camera_end();
            }
            was_recenter_key_pressed = recenter_key_pressed;
        }
    }

    bool was_recenter_key_pressed = false;
    MyGameEngineBeamEyeTrackerDevice *beam_eye_tracker_device{nullptr};
};

class MyGameEngineCharacterHead : public MyGameEngineObjectInterface {
  public:
    MyGameEngineCharacterHead(MyGameEngineObjectInterface *parent)
        : MyGameEngineObjectInterface(parent) {}

    void Tick(float delta_time) override {
        // Just pretend the character is moving forward very slowly.
        this->world_transform.translation_z_inches += 0.01f * M_METERS_TO_INCHES * delta_time;
    }
};

int main() {
    std::unique_ptr<MyGameEngineBeamEyeTrackerDevice> beam_eye_tracker_device =
        std::make_unique<MyGameEngineBeamEyeTrackerDevice>(nullptr);
    devices.push_back(beam_eye_tracker_device.get());

    // Basic components, whose parent will be ignored as that's irrelevant in this sample.
    MyGameEngineCharacterHead character_head(nullptr);
    MyGameEngineImmersiveHUD immersive_hud(nullptr);
    MyGameEngineHotkeysMapper hotkeys_mapper(nullptr);
    MyGameEngineImmersiveCamera immersive_camera(&character_head);

    // We could put all in a list, but will be made explicit for clarity.

    //============= INITIALIZING THE GAME RENDERING =============
    for (auto &device : devices) {
        // Initializes the Beam device and API
        device->BeginPlay();
    }
    hotkeys_mapper.BeginPlay();
    immersive_hud.BeginPlay();
    immersive_camera.BeginPlay();
    character_head.BeginPlay();

    //============= FRAME LOOP at 60 FPS =============
    auto prev_frame_time = std::chrono::system_clock::now();
    const auto end_time = prev_frame_time + std::chrono::seconds(30); // Run for 30 seconds.
    while (std::chrono::system_clock::now() < end_time) {
        auto frame_start = std::chrono::system_clock::now();
        const float delta_time =
            std::chrono::duration<float>(frame_start - prev_frame_time).count();
        prev_frame_time = frame_start;

        hotkeys_mapper.Tick(delta_time);
        // We assume that devices are updated before the HUD and camera.
        for (auto &device : devices) {
            device->Tick(delta_time);
        }
        // In theory, here the parent-child relationship would drive the ordering, but we'll just
        // fake it by updating the character head first, then the camera, then the HUD.
        character_head.Tick(delta_time);
        immersive_hud.Tick(delta_time);

        immersive_camera.Tick(delta_time);

        // Note: if you want to see "real" responses for the top-left HUD element opacity when you
        // look to the top-left corner of your display, please edit
        // MyGameEngineBeamEyeTrackerDevice::get_rendering_area_viewport_geometry_from_engine and
        // hard-code the correct geometry of your display.

        // Note: you should see the printed z values to grow slowly as the character head is moving
        // forward slowly, but also increase or decrease in values as you move towards or away from
        // the webcam. Press SPACE to recenter the camera.
        std::cout << "[Game cam: z_pos_inches=" << std::fixed << std::setprecision(2)
                  << immersive_camera.world_transform.translation_z_inches
                  << " ; yaw_degrees=" << std::fixed << std::setprecision(2)
                  << immersive_camera.world_transform.rotation_y_degrees << "] and ["
                  << "HUD top left opacity=" << std::fixed << std::setprecision(2)
                  << immersive_hud.ui_elements.at(0).opacity << "]";
        if (hotkeys_mapper.was_recenter_key_pressed) {
            std::cout << " Recentering!" << std::endl;
        } else {
            std::cout << std::endl;
        }

        // Sleep for 16ms to simulate 60-ish FPS
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
    //============= SHUTTING DOWN THE GAME RENDERING =============
    for (auto &device : devices) {
        device->EndPlay(); // Shuts down the Beam device and API
    }
    immersive_hud.EndPlay();
    immersive_camera.EndPlay();
    character_head.EndPlay();
}