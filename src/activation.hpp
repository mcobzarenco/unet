#pragma once

#include "typedefs.hpp"
#include "stan/agrad/rev/matrix.hpp"
#include "stan/agrad/rev/functions/tanh.hpp"
#include "stan/agrad/rev/functions/exp.hpp"

#include <algorithm>


namespace unet {

using std::exp;
using stan::agrad::exp;

using std::tanh;
using stan::agrad::tanh;

inline double sigmoid(double x) {
  // return 1.0 / (1.0 + std::exp(-2.0 * x));
  return tanh(x);
}

inline void sigmoid_in_place(Eigen::MatrixXd& x) {
  std::transform(x.data(), x.data() + x.size(), x.data(),
                 [] (const double x) {return sigmoid(x);});
}

inline double dsigmoid(double x) {
  // return 1.0 / (1.0 + std::exp(-2.0 * x));
  double t{tanh(x)};
  return 1.0 - t * t;
}

inline stan::agrad::var sigmoid(const stan::agrad::var& x) {
  return tanh(x);
  // return 1.0 / (1.0 + stan::agrad::exp(-2.0 * x));
  // return stan::agrad::inv_logit(x);
  // return 1.0 + stan::agrad::tanh(x / 2.0);
}

template<typename Scalar>
void softmax_in_place(DynamicMatrix<Scalar>& X) {
  std::transform(X.data(), X.data() + X.size(), X.data(),
                 [] (const Scalar& x) {return exp(x);});
  const DynamicVector<Scalar> exp_sum{X.colwise().sum()};
  for (int32_t i{0}; i < X.cols(); ++i) {
    X.col(i) /= exp_sum[i];
  }
}

}  // namespace unet
