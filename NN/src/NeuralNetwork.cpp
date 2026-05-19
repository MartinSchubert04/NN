#include "NeuralNetwork.h"
#include "Math.h"
#include "Optimizer.h"
#include "pch.h"
#include <cmath>
#include <fstream>
#include <random>
#include <algorithm>

namespace NN {

NeuralNetwork::NeuralNetwork(std::vector<u32> layers, u32 epochs, u32 batchSize) :
    layers(layers), epochs(epochs), _batchSize(batchSize) {

  _modelData.trainImages = loadFile(60000, 784, DATA_PATH "train_images.mat");
  _modelData.testImages = loadFile(10000, 784, DATA_PATH "test_images.mat");

  Matrix trainLabelsFile = loadFile(60000, 1, DATA_PATH "train_labels.mat");
  Matrix testLabelsFile = loadFile(10000, 1, DATA_PATH "test_labels.mat");

  _modelData.trainLabels = Matrix(60000, 10);
  _modelData.testLabels = Matrix(10000, 10);

  oneHotEncode(_modelData.trainLabels.data, trainLabelsFile.data);
  oneHotEncode(_modelData.testLabels.data, testLabelsFile.data);

  // init w & b
  std::normal_distribution<f32> dist(0.0f, 1.0f);

  weights.resize(layers.size());
  bias.resize(layers.size());

  for (size_t i{1}; i < layers.size(); i++) {
    Ref<Matrix> w = createRef<Matrix>(layers[i], layers[i - 1]);

    f32 std_dev = std::sqrt(2.0f / layers[i - 1]);

    for (size_t j{0}; j < w->data.size(); j++) {
      w->data[j] = dist(rng()) * std_dev;
    }

    weights[i] = w;

    Ref<Matrix> b = createRef<Matrix>(layers[i], 1);
    b->fill(0);

    bias[i] = b;
  }

  _optimizer = createRef<Optimizer>(weights, bias, layers.size());
}

std::vector<f32> NeuralNetwork::oneHotEncode(std::vector<f32> &out, std::vector<f32> &data) {
  for (size_t i{0}; i < data.size(); i++) {
    u8 num = data[i];
    out[i * 10 + num] = 1.0f;
  }
  return out;
}

Matrix NeuralNetwork::loadFile(u32 rows, u32 cols, std::string filepath) {
  Matrix mat(rows, cols);
  std::ifstream file(filepath, std::ios::binary | std::ios::ate);

  if (!file.is_open()) {
    throw std::runtime_error("No se pudo abrir: " + filepath);
  }

  u64 fileSize = static_cast<u64>(file.tellg());
  file.seekg(0, std::ios::beg);

  u64 size = MIN(fileSize, sizeof(f32) * rows * cols);

  file.read(reinterpret_cast<char *>(mat.data.data()), size);

  file.close();

  return mat;
}

Matrix NeuralNetwork::relu(const Matrix &mat) {
  Matrix result(mat.rows, mat.cols);

  size_t size = mat.cols * mat.rows;

  for (size_t i{0}; i < size; i++) {
    result.data[i] = MAX(mat.data[i], 0);
  }

  return result;
}

Matrix NeuralNetwork::d_relu(const Matrix &mat) {
  Matrix result(mat.rows, mat.cols);

  size_t size = mat.cols * mat.rows;

  for (size_t i{0}; i < size; i++) {
    result.data[i] = mat.data[i] > 0.0f ? 1.0f : 0.0f;
  }

  return result;
}

Matrix NeuralNetwork::softmax(const Matrix &mat) {
  // a_i = e^a_i/ sum(e^a_i)

  Matrix result(mat.rows, mat.cols);

  size_t size = mat.cols * mat.rows;

  f32 maxVal = *std::max_element(mat.data.begin(), mat.data.end());
  f32 sum = 0.0f;
  for (size_t i{0}; i < size; i++) {
    result.data[i] = expf(mat.data[i] - maxVal);
    sum += result.data[i];
  }

  result = result * (1.0f / sum);

  return result;
}

Matrix NeuralNetwork::crossEntropy(const Matrix &p, const Matrix &q) {
  if (p.rows != q.rows || p.cols != q.cols) {
    throw std::invalid_argument("Dimesions do not align for crossentropy function");
  }

  Matrix result(p.rows, p.cols);

  size_t size = result.cols * result.rows;

  for (size_t i{0}; i < size; i++) {
    result.data[i] = p.data[i] == 0.0f ? 0.0f : p.data[i] * -logf(q.data[i]);
  }

  return result;
}

void NeuralNetwork::forwardProp(Matrix input) {
  u32 n = layers.size() - 1;  // avoid last layer

  forwardOut["A0"] = input;

  for (size_t i{1}; i < n; i++) {
    forwardOut["Z" + toString(i)] = *weights[i] * forwardOut["A" + toString(i - 1)] + *bias[i];

    forwardOut["A" + toString(i)] = relu(forwardOut["Z" + toString(i)]);
  }

  // last layers
  forwardOut["Z" + toString(n)] = *weights[n] * forwardOut["A" + toString(n - 1)] + *bias[n];
  forwardOut["A" + toString(n)] = softmax(forwardOut["Z" + toString(n)]);
}

void NeuralNetwork::backwardProp(Matrix y) {
  int n = layers.size() - 1;

  Matrix AL = forwardOut["A" + toString(n)];

  backwardOut["dZ" + toString(n)] = AL - y;

  backwardOut["dW" + toString(n)] = backwardOut["dZ" + toString(n)] * forwardOut["A" + toString(n - 1)].T();

  backwardOut["db" + toString(n)] = sumXaxis(backwardOut["dZ" + toString(n)]);

  for (int i{n - 1}; i > 0; i--) {
    backwardOut["dZ" + toString(i)] =
        ((*weights[i + 1]).T() * backwardOut["dZ" + toString(i + 1)]).elemMult(d_relu(forwardOut["Z" + toString(i)]));

    backwardOut["dW" + toString(i)] = backwardOut["dZ" + toString(i)] * forwardOut["A" + toString(i - 1)].T();

    backwardOut["db" + toString(i)] = sumXaxis(backwardOut["dZ" + toString(i)]);
  }
}

f32 NeuralNetwork::accuracy(u32 numSamples, Matrix samples, Matrix labels) {
  u32 n = layers.size() - 1;
  u32 correct = 0;

  for (u32 i = 0; i < numSamples; i++) {
    std::vector<f32> img(samples.data.begin() + i * 784, samples.data.begin() + i * 784 + 784);
    forwardProp(Matrix(784, 1, img));

    Matrix &output = forwardOut["A" + toString(n)];
    u32 predicted = std::max_element(output.data.begin(), output.data.end()) - output.data.begin();

    u32 actual = 0;
    for (u32 j = 0; j < 10; j++) {
      if (labels.data[i * 10 + j] == 1.0f) {
        actual = j;
        break;
      }
    }

    if (predicted == actual)
      correct++;
  }

  return (f32)correct / numSamples;
}

void NeuralNetwork::train() {

  for (size_t e{0}; e < epochs; e++) {

    std::unordered_map<std::string, Matrix> gradients;

    for (size_t i{0}; i < _batchSize; i++) {

      u32 index = std::uniform_int_distribution<u32>(0, _modelData.trainImages.rows - 1)(rng());
      std::vector<f32> sampleImage(_modelData.trainImages.data.begin() + index * 784,
                                   _modelData.trainImages.data.begin() + index * 784 + 784);

      Matrix x(784, 1, sampleImage);

      std::vector<f32> sampleLabel(_modelData.trainLabels.data.begin() + index * 10,
                                   _modelData.trainLabels.data.begin() + index * 10 + 10);
      Matrix y(10, 1, sampleLabel);

      forwardProp(x);

      backwardProp(y);

      for (size_t l{1}; l < layers.size(); l++) {
        std::string dW = "dW" + toString(l);
        std::string db = "db" + toString(l);

        if (gradients.find(dW) == gradients.end()) {
          gradients[dW] = backwardOut[dW];
          gradients[db] = backwardOut[db];
        } else {
          gradients[dW] = gradients[dW] + backwardOut[dW];
          gradients[db] = gradients[db] + backwardOut[db];
        }
      }
    }

    for (size_t l{1}; l < layers.size(); l++) {
      gradients["dW" + toString(l)] = gradients["dW" + toString(l)] * (1.0f / _batchSize);
      gradients["db" + toString(l)] = gradients["db" + toString(l)] * (1.0f / _batchSize);
    }

    _optimizer->step(gradients);

    if (e % (epochs / 5) == 0) {
      u32 index = std::uniform_int_distribution<u32>(0, _modelData.trainImages.rows - 1)(rng());
      std::vector<f32> sampleLabel(_modelData.trainLabels.data.begin() + index * 10,
                                   _modelData.trainLabels.data.begin() + index * 10 + 10);
      Matrix y(10, 1, sampleLabel);

      loss = crossEntropy(y, forwardOut["A" + toString(layers.size() - 1)]).sum();
      trainAcc = accuracy(accuracyBatchSize, _modelData.trainImages, _modelData.trainLabels);
      testAcc = accuracy(accuracyBatchSize, _modelData.testImages, _modelData.testLabels);
      std::cout << "loss: " << loss << " | train acc: " << trainAcc << " | test acc: " << testAcc << std::endl;
    }
  }
}

u16 NeuralNetwork::predict(std::vector<f32> img) {
  u32 n = layers.size() - 1;
  u32 correct = 0;

  forwardProp(Matrix(784, 1, img));

  Matrix &output = forwardOut["A" + toString(n)];
  u32 predicted = std::max_element(output.data.begin(), output.data.end()) - output.data.begin();

  return predicted;
}

void NeuralNetwork::onLoadContext(ModelContext context) {
  setWeights(context.weights);
  setBias(context.bias);
  setMetrics(context.loss, context.trainAcc, context.testAcc);
  _optimizer = createRef<Optimizer>(context.weights, context.bias, context.layers.size());
}

}  // namespace NN
