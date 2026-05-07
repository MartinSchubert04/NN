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
  Matrix(uint32_t rows, uint32_t cols) : _rows(rows), _cols(cols) { _data.resize(rows * cols, 0.0f); }

  void fill(float x);
  void scale(float scale);
  void clear();
  Matrix copy();
  float sum();

  std::vector<float> getData() const { return _data; };

  Matrix transpose();
  TransposeView T();

  Matrix operator+(const Matrix &mat);
  Matrix operator-(const Matrix &mat);

  Matrix operator*(float scalar);  // same as scale()
  Matrix operator*(const Matrix &mat);  // a * b
  Matrix operator*(const TransposeView &mat);  // a * b.T()
  friend Matrix operator*(const TransposeView &view, const Matrix &mat);  // a.T() * b
  friend Matrix operator*(const TransposeView &va, const TransposeView &vb);  // a.T() * b.T()

  friend std::ostream &operator<<(std::ostream &os, const Matrix &mat);

private:
  uint32_t _rows;
  uint32_t _cols;
  std::vector<float> _data;
};

void ReLU();
void softmax();
void crossentropy();

}  // namespace NN
