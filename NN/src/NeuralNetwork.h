#pragma once

#include "Math.h"
#include <vector>

namespace NN {

class NeuralNetwork {

public:
  std::vector<f32> weights;
  std::vector<f32> bias;

  NeuralNetwork() = default;

  Matrix loadFile(u32 rows, u32 cols, std::string filepath);

  Matrix crossEntropy(Matrix p, Matrix q);
  Matrix relu(Matrix mat);  // TODO move to nn class
  Matrix softmax(Matrix mat);  // TODO move to nn class
};

}  // namespace NN
