#include <iostream>
#include "Math.h"

int main() {

  NN::Matrix a(1, 3);

  a.fill(1);

  NN::Matrix b(3, 1);

  b.fill(1);

  auto x = a.T() * b.T();

  std::cout << x;

  return 0;
}
