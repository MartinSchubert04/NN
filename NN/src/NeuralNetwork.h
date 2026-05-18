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

  struct ModelContext {
    f32 loss;
    f32 trainAcc;
    f32 testAcc;
    std::vector<u32> layers;
    std::vector<Matrix> weights;
    std::vector<Matrix> bias;
    std::unordered_map<std::string, Matrix> forwardOut;
    std::unordered_map<std::string, Matrix> backwardOut;

    ModelContext(std::vector<u32> layers, std::vector<Matrix> weights, std::vector<Matrix> bias,
                 std::unordered_map<std::string, Matrix> forwardOut,
                 std::unordered_map<std::string, Matrix> backwardOut, f32 loss, f32 trainAcc, f32 testAcc) :
        layers(layers),
        weights(weights),
        bias(bias),
        forwardOut(forwardOut),
        backwardOut(backwardOut),
        loss(loss),
        trainAcc(trainAcc),
        testAcc(testAcc) {}
  };

public:
  ModelData _modelData;

  f32 learningStep;
  u32 epochs;
  std::vector<u32> layers;
  f32 loss;
  f32 trainAcc;
  f32 testAcc;
  u32 accuracyBatchSize = 1000;

  std::vector<Matrix> weights;
  std::vector<Matrix> bias;
  std::unordered_map<std::string, Matrix> forwardOut;
  std::unordered_map<std::string, Matrix> backwardOut;

  NeuralNetwork() = default;
  NeuralNetwork(std::vector<u32> layers, f32 step, u32 epochs);

  Matrix loadFile(u32 rows, u32 cols, std::string filepath);
  std::vector<f32> oneHotEncode(std::vector<f32> &out, std::vector<f32> &data);
  Matrix crossEntropy(const Matrix &p, const Matrix &q);
  Matrix relu(const Matrix &mat);
  Matrix d_relu(const Matrix &mat);
  Matrix softmax(const Matrix &mat);
  void updateParameters();

  ModelData &getModelData() { return _modelData; }

  ModelContext getModelContext() {
    return ModelContext(layers, weights, bias, forwardOut, backwardOut, loss, trainAcc, testAcc);
  }

  void forwardProp(Matrix input);
  void backwardProp(Matrix y);
  f32 accuracy(u32 numSamples, Matrix samples, Matrix labels);
  void train();
  u16 predict(std::vector<f32> img);
};

}  // namespace NN
