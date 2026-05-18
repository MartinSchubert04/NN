#include "Grid.h"

namespace NN {

Grid::Grid(Box area, u32 cellSize) : area(area), cellSize(cellSize) {
  this->area.init();
}

void Grid::draw() {
  for (size_t i{0}; i < 28; i++) {
    for (size_t j{0}; j < 28; j++) {
      f32 x = area.origin.x + j * cellSize;
      f32 y = area.origin.y + i * cellSize;
      u8 val = (u8)(cells[i * 28 + j] * 255);
      DrawRectangle(x, y, cellSize, cellSize, {val, val, val, 255});
      // DrawRectangleLines(x, y, cellSize, cellSize, {50, 50, 50, 255});
    }
  }

  const char *text = "Right click: draw | Left click: clear";
  u8 fontSize = 20;
  auto textWidth = MeasureText(text, fontSize);

  DrawText(TextFormat("%s", text), area.center.x - textWidth / 2, area.origin.y + cellSize * 28 + 10, fontSize,
           RAYWHITE);

  const char *text2 = "<- -> keys | select an image from dataset";

  auto textWidth2 = MeasureText(text2, fontSize);
  DrawText(text2, area.center.x - textWidth2 / 2, area.origin.y + cellSize * 28 + 30, fontSize, RAYWHITE);

  // draw
  if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
    Vector2 mouse = GetMousePosition();
    int col = (mouse.x - area.origin.x) / cellSize;
    int row = (mouse.y - area.origin.y) / cellSize;

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
