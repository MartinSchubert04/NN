#include "Math.h"
#include "NeuralNetwork.h"
#include "Application.h"

#include "pch.h"
int main() {

  Application app({"NN", 1000, 1000, 60});

  NN::NeuralNetwork nn;

  NN::Matrix trainImages = nn.loadFile(60000, 784, DATA_PATH "train_images.mat");
  NN::Matrix testImages = nn.loadFile(10000, 784, DATA_PATH "test_images.mat");

  NN::Matrix trainLabelsFile = nn.loadFile(60000, 1, DATA_PATH "train_labels.mat");
  NN::Matrix testLabelsFile = nn.loadFile(10000, 1, DATA_PATH "test_labels.mat");

  NN::Matrix trainLabels(60000, 10);
  NN::Matrix testLabels(10000, 10);

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

  u32 index = 0;
  std::vector<float> v(trainImages.data.begin() + index * 784, trainImages.data.begin() + index * 784 + 784);
  app.drawDigits(v);

  for (u32 i{0}; i < 10; i++) {
    if (trainLabels.data[i] == 1.0f)
      std::cout << i;
  }

  app.run();

  return 0;
}
