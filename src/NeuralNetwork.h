#include "Math.h"
#include <vector>

namespace NN {

class NueralNetwork {

public:
  std::vector<f32> weights;
  std::vector<f32> bias;

  NueralNetwork() = default;

  Matrix crossEntropy(Matrix p, Matrix q);
};

}  // namespace NN
