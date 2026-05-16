#pragma once

#include "pch.h"
#include "../include/raylib.h"
#include <unordered_set>
#include <unordered_map>

class UI {

  struct ImageData {
    Ref<Texture2D> texture;
    std::vector<f32> data;
    u8 label;
  };

public:
  ~UI();

  std::unordered_map<u32, ImageData> images;

  void loadTexture(std::vector<float> &data, u8 label, u32 index);
  void draw(u16 val);

  void setCurrentImg(u32 index);
  u32 getCurrentImg() { return _currentImg; };
  void moveImg(int val);

  b32 isImgLoaded(u32 index) { return _loadedIndexes.count(index) > 0; };

private:
  std::unordered_set<u32> _loadedIndexes;
  u32 _currentImg = 0;
};
