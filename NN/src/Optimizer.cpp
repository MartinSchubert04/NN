#include "Optimizer.h"

namespace NN {

Optimizer::Optimizer(std::vector<Ref<Matrix>> _weights, std::vector<Ref<Matrix>> _bias, u32 layers) :
    _weights(_weights), _bias(_bias), _layers(layers) {
  _mediaW.resize(_layers);
  _varianceW.resize(_layers);
  _mediaB.resize(_layers);
  _varianceB.resize(_layers);

  for (size_t i{1}; i < _layers; i++) {
    _mediaW[i] = Matrix(_weights[i]->rows, _weights[i]->cols);
    _varianceW[i] = Matrix(_weights[i]->rows, _weights[i]->cols);
    _mediaB[i] = Matrix(_bias[i]->rows, _bias[i]->cols);
    _varianceB[i] = Matrix(_bias[i]->rows, _bias[i]->cols);
  }
}

void Optimizer::step(std::unordered_map<std::string, Matrix> backwardOut) {
  _t++;

  for (size_t i{1}; i < _layers; i++) {
    Matrix &dW = backwardOut["dW" + toString(i)];
    Matrix &db = backwardOut["db" + toString(i)];

    // update momentum
    _mediaW[i] = (_mediaW[i] * _beta1) + (dW * (1.0f - _beta1));
    _varianceW[i] = (_varianceW[i] * _beta2) + (dW.elemMult(dW) * (1.0f - _beta2));
    _mediaB[i] = (_mediaB[i] * _beta1) + (db * (1.0f - _beta1));
    _varianceB[i] = (_varianceB[i] * _beta2) + (db.elemMult(db) * (1.0f - _beta2));

    // _bias correction
    f32 bc1 = 1.0f - pow(_beta1, (f32)_t);
    f32 bc2 = 1.0f - pow(_beta2, (f32)_t);

    Matrix _mediaW_hat = _mediaW[i] * (1.0f / bc1);
    Matrix _varianceW_hat = _varianceW[i] * (1.0f / bc2);
    Matrix _mediaB_hat = _mediaB[i] * (1.0f / bc1);
    Matrix _varianceB_hat = _varianceB[i] * (1.0f / bc2);

    *_weights[i] = *_weights[i] - (_mediaW_hat / (sqrt(_varianceW_hat) + EPSILON)) * _step;
    *_bias[i] = *_bias[i] - (_mediaB_hat / (sqrt(_varianceB_hat) + EPSILON)) * _step;
  }
}

}  // namespace NN
