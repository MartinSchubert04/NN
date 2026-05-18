#pragma once

#include "pch.h"
#include "Math.h"
#include "NeuralNetwork.h"

namespace NN {

class Serializer {

public:
  Serializer() = default;

  void save(const std::string &path, NeuralNetwork::ModelContext &ctx);
  void load(const std::string &path, NeuralNetwork::ModelContext &ctx);
};

}  // namespace NN
