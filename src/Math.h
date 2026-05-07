#pragma once

#include <cstdint>
#include <stdint.h>
#include <iostream>
#include <stdfloat>
#include <vector>

namespace NN {

class Matrix {

public:
  struct TransposeView {
    const Matrix *mat;
    TransposeView(const Matrix *m) : mat(m) {}
  };

  Matrix() = default;
  Matrix(uint32_t rows, uint32_t cols) : _rows(rows), _cols(cols) {}

  void fill(float x);
  void scale(float scale);
  void clear();
  Matrix copy();
  float sum();

  Matrix transpose();
  TransposeView T();

  Matrix operator+(const Matrix &mat);
  Matrix operator-(const Matrix &mat);

  Matrix operator*(const Matrix &mat);  // a * b
  Matrix operator*(const TransposeView &mat);  // a * b.T()
  friend Matrix operator*(const TransposeView &view, const Matrix &mat);  // a.T() * b
  friend Matrix operator*(const TransposeView &va, const TransposeView &vb);  // a.T() * b.T()

private:
  uint32_t _rows;
  uint32_t _cols;
  std::vector<float> _data;
};

void ReLU();
void softmax();
void crossentropy();

}  // namespace NN
