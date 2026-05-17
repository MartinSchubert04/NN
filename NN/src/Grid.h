#pragma once

#include "pch.h"
#include "Math.h"
#include <array>
#include "../include/raylib.h"

namespace NN {

class Grid {

public:
  std::array<f32, 784> cells{};
  Vector2 pos;
  u32 cellSize;

  Grid() = default;
  Grid(Vector2 pos, u32 cellSize);

  void draw();
  void paint(std::array<f32, 784> &grid, int col, int row, f32 brushSize);
  void clear();
};

}  // namespace NN
