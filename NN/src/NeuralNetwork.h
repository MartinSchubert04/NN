#include "Math.h"
#include <vector>

namespace NN {

class NueralNetwork {

public:
  std::vector<f32> weights;
  std::vector<f32> bias;

  NueralNetwork() = default;

  Matrix loadFile(std::string filepath);

  Matrix crossEntropy(Matrix p, Matrix q);
  Matrix relu(Matrix mat);  // TODO move to nn class
  Matrix softmax(Matrix mat);  // TODO move to nn class
};

}  // namespace NN
