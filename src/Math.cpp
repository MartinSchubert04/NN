#include "Math.h"
#include <cstring>

namespace NN {

void Matrix::fill(float x) {
  size_t size = _cols * _rows;

  for (size_t i{0}; i < size; i++) {
    _data[i] = x;
  }
}

void Matrix::scale(float scale) {
  size_t size = _cols * _rows;

  for (size_t i{0}; i < size; i++) {
    _data[i] *= scale;
  }
}

void Matrix::clear() {
  size_t size = _cols * _rows;

  for (size_t i{0}; i < size; i++) {
    _data[i] = 0;
  }
}

float Matrix::sum() {
  size_t size = _cols * _rows;

  float sum = 0.0f;
  for (size_t i{0}; i < size; i++) {
    sum += _data[i];
  }

  return sum;
}

Matrix Matrix::transpose() {
  Matrix result(this->_cols, this->_rows);

  for (size_t i{0}; i < this->_rows; i++)
    for (size_t j{0}; j < this->_rows; j++)
      result._data[i + j * this->_rows] = this->_data[j + i * this->_cols];

  return result;
}

Matrix::TransposeView Matrix::T() {
  return TransposeView(this);
}

Matrix Matrix::copy() {
  return Matrix(_rows, _cols);
}

Matrix Matrix::operator*(const Matrix &mat) {
  if (mat._cols != this->_rows) {
    throw std::invalid_argument("Dimesions do not aling for multiplication");
  }

  Matrix result(this->_rows, mat._cols);

  return result;
}

Matrix Matrix::operator*(const TransposeView &view) {
  if (view.mat->_cols != this->_rows) {
    throw std::invalid_argument("Dimesions do not aling for multiplication");
  }

  Matrix result(this->_rows, view.mat->_cols);

  return result;
}

Matrix Matrix::operator+(const Matrix &mat) {
  if (mat._rows != this->_rows || mat._cols != this->_cols) {
    throw std::invalid_argument("Dimesions do not aling for addition");
  }

  Matrix result(this->_rows, this->_cols);

  size_t size = _cols * _rows;

  for (size_t i{0}; i < size; i++) {
    result._data[i] = this->_data[i] + mat._data[i];
  }

  return result;
}

Matrix Matrix::operator-(const Matrix &mat) {
  if (mat._rows != this->_rows || mat._cols != this->_cols) {
    throw std::invalid_argument("Dimesions do not aling for substraction");
  }

  Matrix result;

  return result;
}

}  // namespace NN
