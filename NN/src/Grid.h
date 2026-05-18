#pragma once

#include "pch.h"
#include "Math.h"
#include <array>
#include "../include/raylib.h"
#include "Box.h"

namespace NN {

class Grid {

public:
  std::array<f32, 784> cells{};
  Box area;
  u32 cellSize;

  Grid() = default;
  Grid(Box area, u32 cellSize);

  void draw();
  void paint(std::array<f32, 784> &grid, int col, int row, f32 brushSize);
  void clear();
};

}  // namespace NN
