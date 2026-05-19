#include "Serializer.h"
#include "NeuralNetwork.h"

namespace NN {

void Serializer::save(const std::string &path, NeuralNetwork::ModelContext &ctx) {
  std::ofstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("Could not open file: " + path);

  u32 nBias = ctx.bias.size();
  u32 nWeights = ctx.weights.size();

  f.write(reinterpret_cast<const char *>(&nBias), sizeof(nBias));
  f.write(reinterpret_cast<const char *>(&nWeights), sizeof(nWeights));

  for (auto &mat : ctx.bias) {
    mat = createRef<Matrix>();

    u32 rows = mat->rows;
    u32 cols = mat->cols;
    f.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    f.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
    f.write(reinterpret_cast<const char *>(mat->data.data()), rows * cols * sizeof(f32));
  }

  for (auto &mat : ctx.weights) {
    mat = createRef<Matrix>();

    u32 rows = mat->rows;
    u32 cols = mat->cols;
    f.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    f.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
    f.write(reinterpret_cast<const char *>(mat->data.data()), rows * cols * sizeof(f32));
  }

  f.write(reinterpret_cast<const char *>(&ctx.loss), sizeof(f32));
  f.write(reinterpret_cast<const char *>(&ctx.trainAcc), sizeof(f32));
  f.write(reinterpret_cast<const char *>(&ctx.testAcc), sizeof(f32));
}

void Serializer::load(const std::string &path, NeuralNetwork::ModelContext &ctx) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("Could not open file: " + path);

  u32 nBias, nWeights;
  f.read(reinterpret_cast<char *>(&nBias), sizeof(nBias));
  f.read(reinterpret_cast<char *>(&nWeights), sizeof(nWeights));
  ctx.bias.resize(nBias);
  ctx.weights.resize(nWeights);

  for (auto &mat : ctx.bias) {
    mat = createRef<Matrix>();

    u32 fileRows, fileCols;
    f.read(reinterpret_cast<char *>(&fileRows), sizeof(fileRows));
    f.read(reinterpret_cast<char *>(&fileCols), sizeof(fileCols));

    // validate dimensions
    if (mat->rows != 0 && mat->cols != 0) {
      if (mat->rows != fileRows || mat->cols != fileCols)
        throw std::runtime_error("Dimension mismatch: expected: " + std::to_string(mat->rows) + "x" +
                                 std::to_string(mat->cols) + " actual: " + std::to_string(fileRows) + "x" +
                                 std::to_string(fileCols));
    }

    mat->rows = fileRows;
    mat->cols = fileCols;
    mat->data.resize(mat->rows * mat->cols);
    f.read(reinterpret_cast<char *>(mat->data.data()), mat->rows * mat->cols * sizeof(f32));
  }

  for (auto &mat : ctx.weights) {
    mat = createRef<Matrix>();

    u32 fileRows, fileCols;
    f.read(reinterpret_cast<char *>(&fileRows), sizeof(fileRows));
    f.read(reinterpret_cast<char *>(&fileCols), sizeof(fileCols));

    // validate dimensions
    if (mat->rows != 0 && mat->cols != 0) {
      if (mat->rows != fileRows || mat->cols != fileCols)
        throw std::runtime_error("Dimension mismatch: expected: " + std::to_string(mat->rows) + "x" +
                                 std::to_string(mat->cols) + " actual: " + std::to_string(fileRows) + "x" +
                                 std::to_string(fileCols));
    }

    mat->rows = fileRows;
    mat->cols = fileCols;
    mat->data.resize(mat->rows * mat->cols);
    f.read(reinterpret_cast<char *>(mat->data.data()), mat->rows * mat->cols * sizeof(f32));
  }

  f.read(reinterpret_cast<char *>(&ctx.loss), sizeof(f32));
  f.read(reinterpret_cast<char *>(&ctx.trainAcc), sizeof(f32));
  f.read(reinterpret_cast<char *>(&ctx.testAcc), sizeof(f32));
}

}  // namespace NN
