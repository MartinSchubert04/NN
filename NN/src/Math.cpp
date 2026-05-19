#include "Math.h"
#include <cstring>

namespace NN {

// ──────────────────────────────────────────
// ──── MATRIX
// ──────────────────────────────────────────

void Matrix::fill(f32 x) {
  size_t size = cols * rows;

  for (size_t i{0}; i < size; i++) {
    data[i] = x;
  }
}

void Matrix::scale(f32 scale) {
  size_t size = cols * rows;

  for (size_t i{0}; i < size; i++) {
    data[i] *= scale;
  }
}

void Matrix::clear() {
  size_t size = cols * rows;

  for (size_t i{0}; i < size; i++) {
    data[i] = 0;
  }
}

u32 Matrix::size() {
  return this->data.size();
}

f32 Matrix::sum() {
  size_t size = cols * rows;

  f32 sum = 0.0f;
  for (size_t i{0}; i < size; i++) {
    sum += data[i];
  }

  return sum;
}

Matrix Matrix::elemMult(const Matrix &mat) {
  if (this->cols != mat.cols || this->rows != mat.rows) {
    throw std::invalid_argument("Dimesions do not align for element multiplication");
  }

  Matrix out(rows, cols);

  for (size_t i{0}; i < rows; i++)
    for (size_t j{0}; j < cols; j++)
      out.data[i * cols + j] = this->data[i * cols + j] * mat.data[i * cols + j];

  return out;
}

Matrix Matrix::transpose() {
  Matrix result(this->cols, this->rows);

  for (size_t i{0}; i < this->rows; i++)
    for (size_t j{0}; j < this->cols; j++)
      result.data[i + j * this->rows] = this->data[j + i * this->cols];

  return result;
}

Matrix::TransposeView Matrix::T() {
  return TransposeView(this);
}

Matrix Matrix::copy() {
  Matrix result(rows, cols);

  size_t size = rows * cols;

  for (size_t i{0}; i < size; i++)
    result.data[i] = this->data[i];

  return result;
}

Matrix Matrix::operator/(const Matrix &mat) {
  if (rows != mat.rows || cols != mat.cols)
    throw std::runtime_error("operator/ dimension mismatch: " + std::to_string(rows) + "x" + std::to_string(cols) +
                             " / " + std::to_string(mat.rows) + "x" + std::to_string(mat.cols));

  Matrix out(rows, cols);
  for (size_t i{0}; i < rows * cols; i++)
    out.data[i] = data[i] / mat.data[i];
  return out;
}

Matrix Matrix::operator*(const Matrix &mat) {
  if (this->cols != mat.rows) {
    throw std::invalid_argument(std::string("Dimensions do not align for multiplication: ") + "(" +
                                std::to_string(rows) + "x" + std::to_string(cols) + ") * " + "(" +
                                std::to_string(mat.rows) + "x" + std::to_string(mat.cols) + ")");
  }

  Matrix result(this->rows, mat.cols);

  size_t size = this->rows * this->cols;

  for (size_t i{0}; i < result.rows; i++)
    for (size_t k{0}; k < this->cols; k++)
      for (size_t j{0}; j < result.cols; j++)
        result.data[j + i * result.cols] += this->data[k + i * this->cols] * mat.data[j + k * mat.cols];

  return result;
}

Matrix Matrix::operator*(const TransposeView &view) {
  u32 bcols = view.mat->rows;  // a is transposed | a cols = a.T() rows
  u32 brows = view.mat->cols;  // a is transposed | a rows = a.T() cols

  if (this->cols != brows) {
    throw std::invalid_argument("Dimesions do not align for multiplication");
  }

  Matrix result(this->rows, bcols);

  size_t size = this->rows * this->cols;

  // for memory index use the real cols and rows
  for (size_t i{0}; i < result.rows; i++)
    for (size_t j{0}; j < result.cols; j++)
      for (size_t k{0}; k < this->cols; k++)
        result.data[j + i * result.cols] += this->data[k + i * this->cols] * view.mat->data[k + j * view.mat->cols];

  return result;
}

Matrix operator*(const Matrix::TransposeView &view, const Matrix &mat) {
  u32 acols = view.mat->rows;  // a is transposed | a cols = a.T() rows
  u32 arows = view.mat->cols;  // a is transposed | a rows = a.T() cols

  if (acols != mat.rows) {
    throw std::invalid_argument("Dimesions do not align for multiplication");
  }

  Matrix result(arows, mat.cols);

  // for memory index use the real cols and rows
  for (size_t k{0}; k < view.mat->rows; k++)
    for (size_t i{0}; i < result.rows; i++)
      for (size_t j{0}; j < result.cols; j++)
        result.data[j + i * result.cols] += view.mat->data[i + k * view.mat->cols] * mat.data[j + k * mat.cols];

  return result;
}

Matrix operator*(const Matrix::TransposeView &va, const Matrix::TransposeView &vb) {
  u32 acols = va.mat->rows;  // a is transposed | a cols = a.T() rows
  u32 arows = va.mat->cols;  // a is transposed | a rows = a.T() cols
  u32 brows = vb.mat->cols;  // b is transposed | b cols = b.T() rows
  u32 bcols = vb.mat->rows;  // b is transposed | b rows = b.T() cols

  if (acols != brows) {
    throw std::invalid_argument("Dimesions do not align for multiplication");
  }

  Matrix result(arows, bcols);

  // for memory index use the real cols and rows
  for (size_t i{0}; i < result.rows; i++)
    for (size_t j{0}; j < result.cols; j++)
      for (size_t k{0}; k < va.mat->rows; k++)
        result.data[j + i * result.cols] += va.mat->data[i + k * va.mat->cols] * vb.mat->data[k + j * vb.mat->cols];

  return result;
}

Matrix Matrix::operator*(f32 scalar) {
  Matrix result = this->copy();
  result.scale(scalar);
  return result;
}

Matrix Matrix::operator+(const Matrix &mat) {
  if (mat.rows != this->rows || mat.cols != this->cols) {
    throw std::invalid_argument(std::string("Dimensions do not align for additions: ") + "(" + std::to_string(rows) +
                                "x" + std::to_string(cols) + ") * " + "(" + std::to_string(mat.rows) + "x" +
                                std::to_string(mat.cols) + ")");
  }

  Matrix result(this->rows, this->cols);

  size_t size = cols * rows;

  for (size_t i{0}; i < size; i++) {
    result.data[i] = this->data[i] + mat.data[i];
  }

  return result;
}

Matrix Matrix::operator+(const f32 &scalar) {
  Matrix result(this->rows, this->cols);

  size_t size = cols * rows;

  for (size_t i{0}; i < size; i++) {
    result.data[i] = this->data[i] + scalar;
  }

  return result;
}

Matrix Matrix::operator-(const Matrix &mat) {
  if (mat.rows != this->rows || mat.cols != this->cols) {
    throw std::invalid_argument(std::string("Dimensions do not align for substraction: ") + "(" + std::to_string(rows) +
                                "x" + std::to_string(cols) + ") * " + "(" + std::to_string(mat.rows) + "x" +
                                std::to_string(mat.cols) + ")");
  }

  Matrix result(this->rows, this->cols);

  size_t size = cols * rows;

  for (size_t i{0}; i < size; i++) {
    result.data[i] = this->data[i] - mat.data[i];
  }

  return result;
}

std::ostream &operator<<(std::ostream &os, const Matrix &mat) {
  for (size_t i = 0; i < mat.rows; i++) {
    for (size_t j = 0; j < mat.cols; j++)
      os << mat.data[j + i * mat.cols] << " ";
    os << "\n";
  }
  return os;
}

// ──────────────────────────────────────────
// ──── MATRIX END
// ──────────────────────────────────────────

Matrix sumXaxis(const Matrix &mat) {
  Matrix out(mat.rows, 1);

  for (size_t i{0}; i < mat.rows; i++) {
    for (size_t j{0}; j < mat.cols; j++) {
      out.data[i] += mat.data[i * mat.cols + j];
    }
  }

  return out;
}

Matrix sqrt(const Matrix &mat) {
  Matrix out(mat.rows, mat.cols);
  for (size_t i{0}; i < mat.rows * mat.cols; i++)
    out.data[i] = std::sqrt(mat.data[i]);
  return out;
}

}  // namespace NN
