#include "UI.h"
#include "NeuralNetwork.h"

namespace NN {

void UI::init(NeuralNetwork::ModelContext ctx, Vector2 windowSize) {
  _windowSize = windowSize;

  Vector2 startPos{_windowSize.x / 3, _windowSize.y / 2};
  u32 offsetY = 30;
  u32 offsetX = 150;

  for (size_t i{1}; i < ctx.layers.size(); i++) {
    u32 n = ctx.layers[i];
    u32 totalHeight = (n - 1) * offsetY;

    for (size_t j{0}; j < n; j++) {
      float posX = startPos.x + offsetX * (i - 1);
      float posY = startPos.y - totalHeight / 2.0f + offsetY * j;

      neurons.push_back({posX, posY});
    }
  }

  std::vector<u32> layerOffset(ctx.layers.size(), 0);
  for (size_t i{2}; i < ctx.layers.size(); i++)
    layerOffset[i] = layerOffset[i - 1] + ctx.layers[i - 1];

  for (size_t i{1}; i < ctx.layers.size() - 1; i++) {
    for (size_t j{0}; j < ctx.layers[i]; j++) {
      for (size_t k{0}; k < ctx.layers[i + 1]; k++) {
        Line l;
        l.start = neurons[layerOffset[i] + j];
        l.end = neurons[layerOffset[i + 1] + k];
        connections.push_back(l);
      }
    }
  }
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

void UI::draw(u16 val) {

  drawNet();

  if (_loadedIndexes.size() == 0)
    return;

  f32 imgPosX = 100;
  u32 fontSize = 20;
  f32 scale = 10.0f;
  f32 imgPosY = _windowSize.y / 2 - (28 * scale / 2);

  ImageData img = images[_currentImg];

  const char *status = val == img.label ? "SUCCES" : "FAIL";

  const char *text = TextFormat("Expected: %d | Predicted: %d | %s", img.label, val, status);
  u16 textSize = MeasureText(text, fontSize);
  u32 centeredX = imgPosX + (28 * scale / 2 - textSize / 2);
  u32 centeredY = imgPosY - 10 - fontSize;

  DrawText(TextFormat("Train time: %.2f", _trainingTime), centeredX, centeredY - 30, fontSize, RAYWHITE);
  DrawText(text, centeredX, centeredY, fontSize, RAYWHITE);
  DrawTextureEx(*img.texture, {imgPosX, imgPosY}, 0.0f, scale, WHITE);
}

void UI::drawNet() {

  u32 radius = 10;

  for (auto &c : connections) {
    DrawLine(c.start.x, c.start.y, c.end.x, c.end.y, RAYWHITE);
  }

  for (auto &n : neurons) {
    DrawCircle(n.x, n.y, radius, RAYWHITE);
  }
}

}  // namespace NN
