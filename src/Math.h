#pragma once

#include "pch.h"
#include <iostream>
#include <vector>

#define MAX(a, b) a > b ? a : b
#define MIN(a, b) a < b ? a : b
namespace NN {

class Matrix {

public:
  struct TransposeView {
    const Matrix *mat;
    TransposeView(const Matrix *m) : mat(m) {}
  };

  u32 rows;
  u32 cols;
  std::vector<f32> data;

  Matrix() = default;
  Matrix(u32 rows, u32 cols) : rows(rows), cols(cols) { data.resize(rows * cols, 0.0f); }

  void fill(f32 x);
  void scale(f32 scale);
  void clear();
  Matrix copy();
  f32 sum();

  std::vector<f32> getData() const { return data; };

  Matrix transpose();
  TransposeView T();

  Matrix operator+(const Matrix &mat);
  Matrix operator-(const Matrix &mat);

  Matrix operator*(f32 scalar);  // same as scale()
  Matrix operator*(const Matrix &mat);  // a * b
  Matrix operator*(const TransposeView &mat);  // a * b.T()
  friend Matrix operator*(const TransposeView &view, const Matrix &mat);  // a.T() * b
  friend Matrix operator*(const TransposeView &va, const TransposeView &vb);  // a.T() * b.T()

  friend std::ostream &operator<<(std::ostream &os, const Matrix &mat);

  Matrix relu();  // TODO move to nn class
  Matrix softmax();  // TODO move to nn class
};

}  // namespace NN
