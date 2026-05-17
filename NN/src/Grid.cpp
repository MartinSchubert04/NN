#include "Grid.h"

namespace NN {

Grid::Grid(Vector2 pos, u32 cellSize) : pos(pos), cellSize(cellSize) {}

void Grid::draw() {
  for (size_t i{0}; i < 28; i++) {
    for (size_t j{0}; j < 28; j++) {
      f32 x = pos.x + j * cellSize;
      f32 y = pos.y + i * cellSize;
      u8 val = (u8)(cells[i * 28 + j] * 255);
      DrawRectangle(x, y, cellSize, cellSize, {val, val, val, 255});
      // DrawRectangleLines(x, y, cellSize, cellSize, {50, 50, 50, 255});
    }
  }

  // draw
  if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
    Vector2 mouse = GetMousePosition();
    int col = (mouse.x - pos.x) / cellSize;
    int row = (mouse.y - pos.y) / cellSize;

    if (col >= 0 && col < 28 && row >= 0 && row < 28)
      paint(cells, col, row, 0.15f);
  }

  // clear grid
  if (IsMouseButtonDown(MOUSE_RIGHT_BUTTON)) {
    for (int i{0}; i < cells.size(); i++)
      cells[i] = 0.0f;
  }
}

void Grid::paint(std::array<f32, 784> &grid, int col, int row, f32 brushSize) {
  for (int dy = -2; dy <= 2; dy++) {
    for (int dx = -2; dx <= 2; dx++) {
      int r = row + dy;
      int c = col + dx;
      if (r < 0 || r >= 28 || c < 0 || c >= 28)
        continue;

      // distancia al centro → intensidad
      f32 dist = sqrtf(dx * dx + dy * dy);
      f32 intensity = expf(-dist * dist / (2.0f * brushSize));  // gaussiana

      f32 &pixel = grid[r * 28 + c];
      pixel = MIN(1.0f, pixel + intensity);  // acumula, no reemplaza
    }
  }
}

}  // namespace NN
