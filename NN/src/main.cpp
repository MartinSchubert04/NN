#include <iostream>
#include "Math.h"

int main() {

  NN::Matrix a(1, 3);
  NN::Matrix b(3, 1);
  NN::Matrix c(1, 3);
  NN::Matrix d(1, 3);

  a.fill(1);
  b.fill(1);
  c.fill(1);
  d.fill(1);

  std::cout << a.T() * b.T();
  std::cout << c.T() * d;
  std::cout << c * d.T();

  return 0;
}
