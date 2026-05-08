#include "NeuralNetwork.h"
#include "Math.h"
#include <cstdio>
#include <fstream>
namespace NN {

Matrix NeuralNetwork::loadFile(u32 rows, u32 cols, std::string filepath) {

  // FILE *file = fopen(filepath.c_str(), "rb");
  // fseek(file, 0, SEEK_END);
  // u64 size = ftell(file);
  // size = MIN(size, sizeof(f32) * rows * cols);
  // fread(&mat.data, sizeof(f32), size, file);
  // fclose(file);

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

Matrix NeuralNetwork::relu(Matrix mat) {
  Matrix result(mat.rows, mat.cols);

  size_t size = mat.cols * mat.rows;

  for (size_t i{0}; i < size; i++) {
    result.data[i] = MAX(mat.data[i], 0);
  }

  return result;
}

Matrix NeuralNetwork::softmax(Matrix mat) {
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

Matrix NeuralNetwork::crossEntropy(Matrix p, Matrix q) {
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
