#pragma once

#include "Grid.h"
#include "NeuralNetwork.h"
#include "pch.h"
#include "raylib.h"
#include <unordered_set>
#include <unordered_map>
#include <array>
#include "Box.h"
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
  UI() = default;
  ~UI();

  std::unordered_map<u32, ImageData> images;
  std::vector<Vector2> neurons;
  std::vector<Line> connections;

  Grid grid;

  void init(NeuralNetwork::ModelContext ctx, Vector2 windowSize);
  void loadTexture(std::vector<float> &data, u8 label, u32 index);
  void draw(u16 val, NeuralNetwork::ModelContext ctx);
  void drawNet(NeuralNetwork::ModelContext &ctx);

  void moveImg(int val);

  void setCurrentImg(u32 index);
  void setTrainingDuration(f32 time) { _trainingTime = time; }
  u32 getCurrentImg() { return _currentImg; };
  std::array<f32, 784> &getGridData() { return grid.cells; }

  b32 isImgLoaded(u32 index) { return _loadedIndexes.count(index) > 0; };

private:
  f32 _pixelSize = 10.f;
  u32 _imgWidth = 28;
  Box _netArea;
  Box _predictionArea;
  f32 _trainingTime;
  f32 _fontSize = 20.f;
  Vector2 _windowSize;
  std::unordered_set<u32> _loadedIndexes;
  u32 _currentImg = 0;
};

}  // namespace NN
