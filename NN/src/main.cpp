#include "Math.h"
#include "NeuralNetwork.h"
#include "Application.h"
#include "pch.h"

int main() {

  Application app({"NN", 1000, 1000, 60});

  app.run();

  return 0;
}
