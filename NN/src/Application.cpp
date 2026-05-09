#include "Application.h"

Application::Application(const ApplicationSpec &spec) {
  _data.title = spec._title;
  _data.width = spec._windowWidth;
  _data.height = spec._windowHeight;
  _data.fps = spec._fps;

  InitWindow(_data.width, _data.height, _data.title.c_str());

  NN::NeuralNetwork nn;

  trainImages = nn.loadFile(60000, 784, DATA_PATH "train_images.mat");
  testImages = nn.loadFile(10000, 784, DATA_PATH "test_images.mat");

  NN::Matrix trainLabelsFile = nn.loadFile(60000, 1, DATA_PATH "train_labels.mat");
  NN::Matrix testLabelsFile = nn.loadFile(10000, 1, DATA_PATH "test_labels.mat");

  trainLabels = NN::Matrix(60000, 10);
  testLabels = NN::Matrix(10000, 10);

  {
    // one hot encoding

    for (size_t i{0}; i < trainLabelsFile.data.size(); i++) {
      u16 num = trainLabelsFile.data[i];
      trainLabels.data[i * 10 + num] = 1.0f;
    }

    for (size_t i{0}; i < testLabelsFile.data.size(); i++) {
      u16 num = testLabelsFile.data[i];
      testLabels.data[i * 10 + num] = 1.0f;
    }
  }

  // drawDigitsTerminal(v);
  u32 index = 101;
  std::vector<float> imageRawData(trainImages.data.begin() + index * 784, trainImages.data.begin() + index * 784 + 784);

  u8 label = getLabel(trainLabels.data, index);
  ui.loadImage(imageRawData, label);
}

void Application::run() {

  while (!WindowShouldClose()) {

    SetWindowTitle(TextFormat("%s | FPS: %d", _data.title.c_str(), GetFPS()));

    BeginDrawing();
    ClearBackground(DARKPURPLE);

    ui.draw();

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
