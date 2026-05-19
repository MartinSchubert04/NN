#include "Serializer.h"
#include "NeuralNetwork.h"

namespace NN {

void Serializer::save(const std::string &path, NeuralNetwork::ModelContext &ctx) {
  std::ofstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("Could not open file: " + path);

  u32 nBias = 0, nWeights = 0;
  for (auto &mat : ctx.bias) if (mat) nBias++;
  for (auto &mat : ctx.weights) if (mat) nWeights++;

  f.write(reinterpret_cast<const char *>(&nBias), sizeof(nBias));
  f.write(reinterpret_cast<const char *>(&nWeights), sizeof(nWeights));

  for (auto &mat : ctx.bias) {
    if (!mat) continue;
    u32 rows = mat->rows;
    u32 cols = mat->cols;
    f.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    f.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
    f.write(reinterpret_cast<const char *>(mat->data.data()), rows * cols * sizeof(f32));
  }

  for (auto &mat : ctx.weights) {
    if (!mat) continue;
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

  // index 0 is always null (input layer has no weights/bias)
  ctx.bias.resize(nBias + 1);
  ctx.weights.resize(nWeights + 1);

  for (u32 i = 1; i <= nBias; i++) {
    ctx.bias[i] = createRef<Matrix>();

    u32 fileRows, fileCols;
    f.read(reinterpret_cast<char *>(&fileRows), sizeof(fileRows));
    f.read(reinterpret_cast<char *>(&fileCols), sizeof(fileCols));

    ctx.bias[i]->rows = fileRows;
    ctx.bias[i]->cols = fileCols;
    ctx.bias[i]->data.resize(fileRows * fileCols);
    f.read(reinterpret_cast<char *>(ctx.bias[i]->data.data()), fileRows * fileCols * sizeof(f32));
  }

  for (u32 i = 1; i <= nWeights; i++) {
    ctx.weights[i] = createRef<Matrix>();

    u32 fileRows, fileCols;
    f.read(reinterpret_cast<char *>(&fileRows), sizeof(fileRows));
    f.read(reinterpret_cast<char *>(&fileCols), sizeof(fileCols));

    ctx.weights[i]->rows = fileRows;
    ctx.weights[i]->cols = fileCols;
    ctx.weights[i]->data.resize(fileRows * fileCols);
    f.read(reinterpret_cast<char *>(ctx.weights[i]->data.data()), fileRows * fileCols * sizeof(f32));
  }

  f.read(reinterpret_cast<char *>(&ctx.loss), sizeof(f32));
  f.read(reinterpret_cast<char *>(&ctx.trainAcc), sizeof(f32));
  f.read(reinterpret_cast<char *>(&ctx.testAcc), sizeof(f32));
}

}  // namespace NN
