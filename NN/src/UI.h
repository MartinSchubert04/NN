#pragma once

#include "NeuralNetwork.h"
#include "pch.h"
#include "../include/raylib.h"
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace NN {

struct Line {
  Vector2 start;
  Vector2 end;
};

class UI {

  struct ImageData {
    Ref<Texture2D> texture;
    std::vector<f32> data;
    u8 label;
  };

public:
  ~UI();

  std::unordered_map<u32, ImageData> images;
  std::vector<Vector2> neurons;
  std::vector<Line> connections;

  void loadTexture(std::vector<float> &data, u8 label, u32 index);
  void draw(u16 val);
  void init(NeuralNetwork::ModelContext ctx, Vector2 windowSize);
  void drawNet();

  void setCurrentImg(u32 index);
  u32 getCurrentImg() { return _currentImg; };
  void moveImg(int val);
  void setTrainingDuration(f32 time) { _trainingTime = time; }

  b32 isImgLoaded(u32 index) { return _loadedIndexes.count(index) > 0; };

private:
  f32 _trainingTime;
  Vector2 _windowSize;
  std::unordered_set<u32> _loadedIndexes;
  u32 _currentImg = 0;
};

}  // namespace NN
