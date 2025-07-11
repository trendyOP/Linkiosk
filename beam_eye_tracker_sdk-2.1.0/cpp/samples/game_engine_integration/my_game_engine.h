#ifndef MY_GAME_ENGINE_H
#define MY_GAME_ENGINE_H

#include <vector>
/**
 * Copyright (C) 2025 Eyeware Tech SA.
 *
 * All rights reserved.
 *
 * This file defines an extremely simplified game engine using OOP, similar to Unity and UE5
 * OOP paradigm. In real life we assume all of this is already defined in your engine.
 *
 * The good stuff is in files main.cpp and bet_game_engine_device.cpp.
 */

struct MyGameEngineTransform {
    // For this sample's purpose, we assume Unity's camera coordinate system which is the same as
    // Beam, except that x is inverted, and the rotations are left-handed, not right-handed.
    float rotation_x_degrees;
    float rotation_y_degrees;
    float rotation_z_degrees;
    float translation_x_inches;
    float translation_y_inches;
    float translation_z_inches;

    MyGameEngineTransform operator+(MyGameEngineTransform other) {
        return MyGameEngineTransform{this->rotation_x_degrees + other.rotation_x_degrees,
                                     this->rotation_y_degrees + other.rotation_y_degrees,
                                     this->rotation_z_degrees + other.rotation_z_degrees,
                                     this->translation_x_inches + other.translation_x_inches,
                                     this->translation_y_inches + other.translation_y_inches,
                                     this->translation_z_inches + other.translation_z_inches};
    }
};

class MyGameEngineObjectInterface {
  public:
    MyGameEngineObjectInterface(MyGameEngineObjectInterface *parent) : parent(parent) {}
    // Called frequently and periodically. delta_time in seconds.
    virtual void Tick(float delta_time) {};

    // Called when the rendering loop starts.
    virtual void BeginPlay() {};

    // Called when the rendering loop stops.
    virtual void EndPlay() {};

    void set_local_transform(MyGameEngineTransform local_transform) {
        this->local_transform = local_transform;
        if (this->parent) {
            this->world_transform = this->parent->world_transform + this->local_transform;
        } else {
            this->world_transform = this->local_transform;
        }
    }

    MyGameEngineTransform local_transform{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    MyGameEngineTransform world_transform{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    MyGameEngineObjectInterface *parent;
};

class MyGameEngineHUDElement : public MyGameEngineObjectInterface {
  public:
    enum class Type {
        TOP_LEFT,
        TOP_RIGHT,
        BOTTOM_LEFT,
        BOTTOM_RIGHT,
    };

    MyGameEngineHUDElement(MyGameEngineHUDElement::Type type, MyGameEngineObjectInterface *parent)
        : MyGameEngineObjectInterface(parent), type(type) {}

    Type type;
    float opacity = 1.0f;
};

#endif // MY_GAME_ENGINE_H