#pragma once

#include "pch.h"
#include "../include/raylib.h"
#include <vector>

class UI {

  struct ImageData {
    Ref<Texture2D> texture;
    std::vector<f32> data;
    u8 label;
  };

public:
  ~UI();

  std::vector<ImageData> images;

  void loadImage(std::vector<float> &data, u8 label);
  void draw();
};
