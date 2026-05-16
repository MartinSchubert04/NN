#pragma once
#include <random>

inline std::mt19937 &rng() {
  static std::mt19937 gen(42);
  return gen;
}
