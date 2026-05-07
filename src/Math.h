#pragma once

#include <cstdint>
#include <stdint.h>
#include <iostream>
#include <stdfloat>

class Matrix {
public:
  uint32_t rows;
  uint32_t cols;
  float *data;

  Matrix() = default;
  Matrix(uint32_t rows, uint32_t cols) : rows(rows), cols(cols) {}

  void fill(float x);
  void scale(float x);
  void clear();

  void operator/(Matrix mat);
  void operator*(Matrix mat);
  void operator+(Matrix mat);
  void operator-(Matrix mat);
};
