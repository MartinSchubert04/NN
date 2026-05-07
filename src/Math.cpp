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
    for (size_t j{0}; j < this->_cols; j++)
      result._data[i + j * this->_rows] = this->_data[j + i * this->_cols];

  return result;
}

Matrix::TransposeView Matrix::T() {
  return TransposeView(this);
}

Matrix Matrix::copy() {
  Matrix result(_rows, _cols);

  size_t size = _rows * _cols;

  for (size_t i{0}; i < size; i++)
    result._data[i] = this->_data[i];

  return result;
}

Matrix Matrix::operator*(const Matrix &mat) {
  if (this->_cols != mat._rows) {
    throw std::invalid_argument("Dimesions do not aling for multiplication");
  }

  Matrix result(this->_rows, mat._cols);

  size_t size = this->_rows * this->_cols;

  for (size_t i{0}; i < result._rows; i++)
    for (size_t k{0}; k < this->_cols; k++)
      for (size_t j{0}; j < result._cols; j++)
        result._data[j + i * result._cols] += this->_data[k + i * this->_cols] * mat._data[j + k * mat._cols];

  return result;
}

Matrix Matrix::operator*(const TransposeView &view) {
  // view _cols = view.mat _rows (logic cols = true rows)
  if (this->_cols != view.mat->_cols) {
    throw std::invalid_argument("Dimesions do not aling for multiplication");
  }

  Matrix result(this->_rows, view.mat->_rows);

  size_t size = this->_rows * this->_cols;

  for (size_t i{0}; i < result._rows; i++)
    for (size_t j{0}; j < result._cols; j++)
      for (size_t k{0}; k < this->_cols; k++)
        result._data[j + i * result._cols] +=
            this->_data[k + i * this->_cols] * view.mat->_data[k + j * view.mat->_cols];

  return result;
}

Matrix operator*(const Matrix::TransposeView &view, const Matrix &mat) {
  // view _rows = view.mat _cols (logic rows = true cols)
  if (view.mat->_rows != mat._rows) {
    throw std::invalid_argument("Dimesions do not aling for multiplication");
  }

  Matrix result(view.mat->_cols, mat._cols);

  for (size_t k{0}; k < view.mat->_rows; k++)
    for (size_t i{0}; i < result._rows; i++)
      for (size_t j{0}; j < result._cols; j++)
        result._data[j + i * result._cols] += view.mat->_data[i + k * view.mat->_cols] * mat._data[j + k * mat._cols];

  return result;
}
Matrix operator*(const Matrix::TransposeView &va, const Matrix::TransposeView &vb) {
  // va _rows = a _cols & vb _cols = b _rows (logic inversion)
  if (va.mat->_rows != vb.mat->_cols) {
    throw std::invalid_argument("Dimesions do not aling for multiplication");
  }

  Matrix result(va.mat->_cols, vb.mat->_rows);

  for (size_t i{0}; i < result._rows; i++)
    for (size_t j{0}; j < result._cols; j++)
      for (size_t k{0}; k < va.mat->_rows; k++)
        result._data[j + i * result._cols] +=
            va.mat->_data[i + k * va.mat->_cols] * vb.mat->_data[k + j * vb.mat->_cols];

  return result;
}

Matrix Matrix::operator*(float scalar) {
  Matrix result = this->copy();
  result.scale(scalar);
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

  Matrix result(this->_rows, this->_cols);

  size_t size = _cols * _rows;

  for (size_t i{0}; i < size; i++) {
    result._data[i] = this->_data[i] - mat._data[i];
  }

  return result;
}

std::ostream &operator<<(std::ostream &os, const Matrix &mat) {
  for (size_t i = 0; i < mat._rows; i++) {
    for (size_t j = 0; j < mat._cols; j++)
      os << mat._data[j + i * mat._cols] << " ";
    os << "\n";
  }
  return os;
}

}  // namespace NN
