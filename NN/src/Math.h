#pragma once

#include "pch.h"
#include <iostream>
#include <vector>

#define MAX(a, b) a > b ? a : b
#define MIN(a, b) a < b ? a : b

#define EPSILON 1e-8f

#define MAT_OP(expr) \
  [&]() { \
    try { \
      return (expr); \
    } catch (const std::exception &e) { \
      std::cerr << e.what() << "\n  en " << __FILE__ << ":" << __LINE__ << "\n"; \
      throw; \
    } \
  }()

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
  Matrix(u32 rows, u32 cols, std::vector<f32> data) : rows(rows), cols(cols), data(data) {}

  void fill(f32 x);
  void scale(f32 scale);
  void clear();
  u32 size();
  Matrix copy();
  f32 sum();
  Matrix elemMult(const Matrix &mat);

  std::vector<f32> getData() const { return data; };

  Matrix transpose();
  TransposeView T();

  Matrix operator+(const Matrix &mat);
  Matrix operator+(const f32 &scalar);
  Matrix operator-(const Matrix &mat);

  Matrix operator*(f32 scalar);  // same as scale()
  Matrix operator/(const Matrix &mat);  // a * b
  Matrix operator*(const Matrix &mat);  // a * b
  Matrix operator*(const TransposeView &mat);  // a * b.T()
  friend Matrix operator*(const TransposeView &view, const Matrix &mat);  // a.T() * b
  friend Matrix operator*(const TransposeView &va, const TransposeView &vb);  // a.T() * b.T()

  friend std::ostream &operator<<(std::ostream &os, const Matrix &mat);
};

// helper funs

Matrix sumXaxis(const Matrix &mat);
Matrix sqrt(const Matrix &mat);

}  // namespace NN
