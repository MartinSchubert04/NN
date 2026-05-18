#pragma once

#include "../include/raylib.h"
#include "pch.h"

class Box {

public:
  Vector2 origin{0, 0};
  f32 width = 0;
  f32 height = 0;
  Vector2 center{0, 0};

  Box() = default;
  Box(Vector2 origin, f32 width, f32 height) : origin(origin), width(width), height(height) {}

  void init() { center = {origin.x + width / 2, origin.y + height / 2}; }
};
