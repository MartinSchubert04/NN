#include "UI.h"
#include "../include/raylib.h"
#include "pch.h"

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

void UI::draw() {

  if (_loadedIndexes.size() == 0)
    return;

  f32 imgPosY = 100;
  f32 imgPosX = 100;
  u32 fontSize = 20;
  f32 scale = 10.0f;

  ImageData img = images[_currentImg];

  const char *text = TextFormat("Expected: %d", img.label);
  u16 textSize = MeasureText(text, fontSize);
  u32 centeredX = imgPosX + (28 * scale / 2 - textSize / 2);
  u32 centeredY = imgPosY - 10 - fontSize;

  DrawText(text, centeredX, centeredY, fontSize, RAYWHITE);
  DrawTextureEx(*img.texture, {imgPosX, imgPosY}, 0.0f, scale, WHITE);
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

UI::~UI() {
  for (auto &[index, img] : images) {
    UnloadTexture(*img.texture);
  }
}
