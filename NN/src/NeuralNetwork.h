#pragma once

#include "Math.h"
#include <vector>

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

  std::vector<f32> weights;
  std::vector<f32> bias;

  NeuralNetwork();

  Matrix loadFile(u32 rows, u32 cols, std::string filepath);
  std::vector<f32> oneHotEncode(std::vector<f32> &out, std::vector<f32> &data);
  Matrix crossEntropy(Matrix p, Matrix q);
  Matrix relu(Matrix mat);  // TODO move to nn class
  Matrix softmax(Matrix mat);  // TODO move to nn class

  ModelData &getModelData() { return _modelData; }
};

}  // namespace NN
