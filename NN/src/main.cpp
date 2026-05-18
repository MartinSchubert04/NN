#include "Application.h"

int main() {
  NN::Application *app = new NN::Application({"NN", 1500, 1000, 60});

  app->run();

  delete app;

  return 0;
}
