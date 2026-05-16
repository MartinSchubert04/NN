#pragma once

#include "Math.h"
#include <string>
#include <unordered_map>
#include <vector>
#include "Random.h"

namespace NN {

class NeuralNetwork {

public:
  struct ModelData {
    Matrix trainImages;
    Matrix trainLabels;
    Matrix testImages;
    Matrix testLabels;
  };

public:
  ModelData _modelData;

  f32 learningStep;
  u32 epochs;
  std::vector<u32> layers;
  f32 loss;

  std::unordered_map<std::string, Matrix> weights;
  std::unordered_map<std::string, Matrix> bias;
  std::unordered_map<std::string, Matrix> forwardOut;
  std::unordered_map<std::string, Matrix> backwardOut;

  NeuralNetwork() = default;
  NeuralNetwork(std::vector<u32> layers, f32 step, u32 epochs);

  Matrix loadFile(u32 rows, u32 cols, std::string filepath);
  std::vector<f32> oneHotEncode(std::vector<f32> &out, std::vector<f32> &data);
  Matrix crossEntropy(Matrix p, Matrix q);
  Matrix relu(Matrix mat);
  Matrix d_relu(Matrix mat);
  Matrix softmax(Matrix mat);
  void updateParameters();

  ModelData &getModelData() { return _modelData; }

  void forwardProp(Matrix input);
  void backwardProp(Matrix y);
  f32 accuracy(u32 numSamples);
  void train();
  u16 predict(std::vector<f32> img);
};

}  // namespace NN
