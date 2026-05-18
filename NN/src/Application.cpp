#include "Application.h"
#include "NeuralNetwork.h"
#include "pch.h"
#include <chrono>
#include <vector>
#include <filesystem>

namespace NN {

Application::Application(const ApplicationSpec &spec) {
  _data.title = spec._title;
  _data.width = spec._windowWidth;
  _data.height = spec._windowHeight;
  _data.fps = spec._fps;

  InitWindow(_data.width, _data.height, _data.title.c_str());

  std::vector<u32> layers{784, 28, 32, 20, 16, 10};

  f32 step = 0.005;
  u32 epochs = 5000;

  nn = NN::NeuralNetwork(layers, step, epochs);

  ui.init(nn.getModelContext(), {_data.width, _data.height});

  const char *path = DATA_PATH "model.data";

  if (!std::filesystem::exists(path)) {
    auto start = std::chrono::high_resolution_clock::now();

    nn.train();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    ui.setTrainingDuration(duration.count() / 1000.f);

    auto context = nn.getModelContext();
    _serializer.save(path, context);
  } else {

    auto context = nn.getModelContext();
    _serializer.load(path, context);
    nn.setWeights(context.weights);
    nn.setBias(context.bias);
    nn.setMetrics(context.loss, context.trainAcc, context.testAcc);
  }

  loadImage(0);
  copyImageToGrid(0);
}

void Application::run() {

  while (!WindowShouldClose()) {

    SetWindowTitle(TextFormat("%s | FPS: %d", _data.title.c_str(), GetFPS()));

    BeginDrawing();
    ClearBackground({33, 60, 81});

    std::array<f32, 784> &gridData = ui.getGridData();
    std::vector<f32> gridVec(gridData.begin(), gridData.end());
    u16 gridPrediction = nn.predict(gridVec);

    ui.draw(gridPrediction, nn.getModelContext());

    onKeyPressed();

    EndDrawing();
  }
}

void Application::drawDigitsTerminal(std::vector<f32> data) {
  for (u32 y{0}; y < 28; y++) {
    for (u32 x{0}; x < 28; x++) {
      f32 num = data[x + y * 28];
      u32 col = 232 + (u32)(num * 23);
      printf("\x1b[48;5;%dm  ", col);
    }
    printf("\n");
  }
  printf("\x1b[0m");
}

u8 Application::getLabel(std::vector<f32> &data, u64 index) {
  for (size_t i{0}; i < 10; i++) {
    if (data[index * 10 + i] == 1.0f)
      return i;
  }

  return 0.0f;
}

void Application::loadImage(u32 index) {
  if (ui.isImgLoaded(index))
    return;

  std::vector<float> imageRawData(nn.getModelData().testImages.data.begin() + index * 784,
                                  nn.getModelData().testImages.data.begin() + index * 784 + 784);

  u8 label = getLabel(nn.getModelData().testLabels.data, index);
  ui.loadTexture(imageRawData, label, index);
}

void Application::onKeyPressed() {

  if (IsKeyPressed(KEY_LEFT)) {
    ui.moveImg(-1);
    u32 idx = ui.getCurrentImg();
    loadImage(idx);
    copyImageToGrid(idx);
  }
  if (IsKeyPressed(KEY_RIGHT)) {
    ui.moveImg(1);
    u32 idx = ui.getCurrentImg();
    loadImage(idx);
    copyImageToGrid(idx);
  }
}

void Application::copyImageToGrid(u32 index) {
  auto &gridData = ui.getGridData();
  auto &imgData = ui.images[index].data;
  for (int i = 0; i < 784; i++)
    gridData[i] = imgData[i];
}

}  // namespace NN
