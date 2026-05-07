#include "NeuralNetwork.h"
#include <cstdio>

namespace NN {

Matrix loadFile(u32 rows, u32 cols, std::string filepath) {

  FILE *file = fopen(filepath.c_str(), "rb");

  fseek(file, 0, SEEK_END);
  u64 size = ftell(file);

  return mat;
}

Matrix NueralNetwork::relu(Matrix mat) {
  Matrix result(mat.rows, mat.cols);

  size_t size = mat.cols * mat.rows;

  for (size_t i{0}; i < size; i++) {
    result.data[i] = MAX(mat.data[i], 0);
  }

  return result;
}

Matrix NueralNetwork::softmax(Matrix mat) {
  // a_i = e^a_i/ sum(e^a_i)

  Matrix result(mat.rows, mat.cols);

  size_t size = mat.cols * mat.rows;

  f32 sum = 0.0f;
  for (size_t i{0}; i < size; i++) {
    result.data[i] = expf(mat.data[i]);
    sum += result.data[i];
  }

  result = result * (1.0f / sum);

  return result;
}

Matrix NueralNetwork::crossEntropy(Matrix p, Matrix q) {
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

}  // namespace NN
