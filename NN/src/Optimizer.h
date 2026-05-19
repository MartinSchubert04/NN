#pragma once

#include "pch.h"
#include "Math.h"
#include <unordered_map>
#include <vector>

namespace NN {

class Optimizer {

public:
  Optimizer() = default;
  Optimizer(std::vector<Ref<Matrix>> _weights, std::vector<Ref<Matrix>> _bias, u32 layers);

  void step(std::unordered_map<std::string, Matrix> backwardOut);

private:
  f32 _step = 0.001;
  u32 _layers;
  std::vector<Ref<Matrix>> _weights;
  std::vector<Ref<Matrix>> _bias;

  std::vector<Matrix> _mediaW, _varianceW;
  std::vector<Matrix> _mediaB, _varianceB;
  u32 _t = 0;
  const f32 _beta1 = 0.9f;
  const f32 _beta2 = 0.999f;
};

}  // namespace NN
