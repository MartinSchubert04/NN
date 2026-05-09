#include "Math.h"
#include "NeuralNetwork.h"
#include "Application.h"
#include "pch.h"

int main() {
  NN::Application *app = new NN::Application({"NN", 1000, 1000, 60});

  app->run();

  delete app;

  return 0;
}
