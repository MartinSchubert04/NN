#include "UI.h"
#include "Box.h"
#include "NeuralNetwork.h"
#include "pch.h"
#include <algorithm>
#include <filesystem>

namespace NN {

void UI::init(NeuralNetwork::ModelContext ctx, Vector2 windowSize) {
  _windowSize = windowSize;
  grid = Grid(
      Box({100, (_windowSize.y / 2) - (_pixelSize * _imgWidth) / 2}, _imgWidth * _pixelSize, _imgWidth * _pixelSize),
      _pixelSize);

  _netArea = Box({_windowSize.x / 3, 0}, windowSize.x / 3 * 2, windowSize.y);
  _netArea.init();

  setupNetPositions(ctx);
}

UI::~UI() {
  for (auto &[index, img] : images) {
    UnloadTexture(*img.texture);
  }
}

void UI::setCurrentImg(u32 index) {
  _currentImg = index;
}

void UI::moveImg(int val) {
  int next = (int)_currentImg + val;
  if (next < 0)
    return;
  _currentImg = (u32)next;
}

void UI::loadTexture(std::vector<float> &data, u8 label, u32 index) {

  std::vector<unsigned char> pixels(28 * 28 * 4);

  for (size_t i{0}; i < data.size(); i++) {
    unsigned char val = (unsigned char)(data[i] * 255.0f);
    pixels[i * 4 + 0] = val;
    pixels[i * 4 + 1] = val;
    pixels[i * 4 + 2] = val;
    pixels[i * 4 + 3] = 255;
  }
  Image img;

  img.data = pixels.data();
  img.width = 28;
  img.height = 28;
  img.mipmaps = 1;
  img.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;

  Ref<Texture2D> texture = createRef<Texture2D>(LoadTextureFromImage(img));

  images[index] = {texture, data, label};
  _loadedIndexes.insert(index);
}

void UI::draw(u16 val, NeuralNetwork::ModelContext ctx) {

  // debbug
  // DrawRectangleLines(_netArea.origin.x, _netArea.origin.y, _netArea.width, _netArea.height, RED);
  // DrawRectangleLines(grid.area.origin.x - 1, grid.area.origin.y - 1, grid.area.width + 2, grid.area.height + 1, RED);

  grid.draw(val);
  drawNet(ctx);

  if (_loadedIndexes.size() == 0)
    return;

  const char *text2 = TextFormat("Train time: %.2f s | Train accuracy: %.2f  | Test accuracy: %.2f | Loss %.10f",
                                 _trainingTime, ctx.trainAcc * 100, ctx.testAcc * 100, ctx.loss);
  int netDataTextSize = MeasureText(text2, _fontSize);

  DrawText(text2, _windowSize.x - netDataTextSize - 10, _windowSize.y - _fontSize - 5, _fontSize, RAYWHITE);
}

void UI::drawNet(NeuralNetwork::ModelContext &ctx) {

  u32 radius = 10;

  for (auto &c : connections) {
    // DrawLine(c.start.x, c.start.y, c.end.x, c.end.y, {203, 203, 203, 255});
    DrawLine(c.start.x, c.start.y, c.end.x, c.end.y, RAYWHITE);
  }

  u32 neuronOffset = 0;

  for (int i{1}; i < ctx.layers.size(); i++) {
    Matrix &activations = ctx.forwardOut["A" + toString(i)];

    f32 maxVal = *std::max_element(activations.data.begin(), activations.data.end());
    f32 minVal = *std::min_element(activations.data.begin(), activations.data.end());

    for (size_t j{0}; j < activations.data.size(); j++) {
      f32 normalized = (maxVal == minVal) ? 0.0f : (activations.data[j] - minVal) / (maxVal - minVal);

      u8 c = (u8)(normalized * 255);
      // Color color = {255, (u8)(255 - c), (u8)(255 - c), 255};
      Color color = {c, c, c, 255};
      auto n = neurons[neuronOffset + j];
      DrawCircle(n.pos.x, n.pos.y, n.radius, color);
      DrawCircleLines(n.pos.x, n.pos.y, n.radius, RAYWHITE);
    }

    if (i == ctx.layers.size() - 1) {
      for (size_t j{0}; j < activations.data.size(); j++) {

        DrawText(TextFormat("%.1f", activations.data[j] * 100), neurons[neuronOffset + j].pos.x + 20,
                 neurons[neuronOffset + j].pos.y - _fontSize / 2, _fontSize, RAYWHITE);
      }
    }

    neuronOffset += ctx.layers[i];
  }
}

void UI::setupNetPositions(NeuralNetwork::ModelContext ctx) {

  const f32 H = _netArea.height;
  const f32 maxRadius = 15.f;
  const f32 minGap = 10.f;
  const f32 margin = maxRadius * 2.f + 80.f;

  u32 offsetX = _netArea.width / (ctx.layers.size() - 1);

  for (size_t i{1}; i < ctx.layers.size(); i++) {
    u32 n = ctx.layers[i];

    f32 radius, spacing;
    if (n == 1) {
      radius = maxRadius;
      spacing = 0.f;
    } else {
      // max radius so n neurons with minGap between edges fit exactly in H
      radius = std::min(maxRadius, (H - (n - 1) * minGap) / (2.f * n));
      radius = std::max(radius, 1.f);
      // spacing between centers so outermost neuron edges touch area bounds
      spacing = (H - margin) / (f32)(n - 1);
      if (i == ctx.layers.size() - 1) {
        spacing = maxRadius + 30.f;
      }
    }

    f32 totalHeight = (n > 1) ? (n - 1) * spacing : 0.f;

    for (size_t j{0}; j < n; j++) {
      float posX = _netArea.origin.x + offsetX / 2.f + offsetX * (i - 1);
      float posY = _netArea.center.y - totalHeight / 2.f + spacing * j;

      neurons.push_back({{posX, posY}, radius});
    }
  }

  std::vector<u32> layerOffset(ctx.layers.size(), 0);
  for (size_t i{2}; i < ctx.layers.size(); i++)
    layerOffset[i] = layerOffset[i - 1] + ctx.layers[i - 1];

  for (size_t i{1}; i < ctx.layers.size() - 1; i++) {
    for (size_t j{0}; j < ctx.layers[i]; j++) {
      for (size_t k{0}; k < ctx.layers[i + 1]; k++) {
        Line l;
        l.start = neurons[layerOffset[i] + j].pos;
        l.end = neurons[layerOffset[i + 1] + k].pos;
        connections.push_back(l);
      }
    }
  }
}

}  // namespace NN
