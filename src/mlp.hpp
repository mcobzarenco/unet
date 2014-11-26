#pragma once

#include "init.hpp"
#include "stan/agrad/rev/matrix.hpp"
#include "stan/agrad/rev/functions/tanh.hpp"
#include "stan/agrad/rev/functions/exp.hpp"

#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/src/Core/NumTraits.h>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <iostream>


namespace unet {

using std::exp;
using stan::agrad::exp;

template<typename T>
using DynamicMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template<typename T>
using DynamicVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

inline double sigmoid(double x) {
  return 1.0 / (1.0 + std::exp(-2.0 * x));
  // return 1.0 + std::tanh(x / 2.0);
}

inline stan::agrad::var sigmoid(const stan::agrad::var& x) {
  return 1.0 / (1.0 + stan::agrad::exp(-2.0 * x));
  // return stan::agrad::inv_logit(x);
  // return 1.0 + stan::agrad::tanh(x / 2.0);
}

struct MLP {
public:
  MLP(uint32_t n_input, uint32_t n_hidden, uint32_t n_output,
      bool softmax, int32_t seed=0)
    : MLP(n_input, n_hidden, n_output, softmax,
          normal_weight_generator(0, .001, seed)) {}

  inline MLP(uint32_t n_input, uint32_t n_hidden, uint32_t n_output,
             bool softmax, std::function<double()> generate_weight);

  inline uint32_t num_params() const;

  Eigen::VectorXd& weights() { return weights_; }
  const Eigen::VectorXd& weights() const { return weights_; }

  template<typename T>
  inline DynamicMatrix<T> operator()(const DynamicMatrix<T>& X) const;

  template<typename T>
  inline DynamicMatrix<T> operator() (const DynamicVector<T>& weights,
                                      const DynamicMatrix<T>& X) const;

  const uint32_t n_input, n_hidden, n_output;
  const bool softmax;

private:
  template <typename T>
  static inline DynamicMatrix<T> function(
    uint32_t n_input, uint32_t n_hidden, uint32_t n_output, bool softmax,
    const DynamicVector<T>& weights, const DynamicMatrix<T>& X);

  Eigen::VectorXd weights_;
};

MLP::MLP(uint32_t n_input, uint32_t n_hidden, uint32_t n_output, bool softmax,
         std::function<double()> generate_weight)
  : n_input{n_input}, n_hidden{n_hidden}, n_output{n_output}, softmax{softmax} {
    CHECK(n_input > 0 && n_hidden > 0 && n_output > 0);
    CHECK(n_output > 1 || !softmax)
      << "There needs to be more than 1 output unit for softmax to make sense.";
    weights_.resize(num_params());
    for (int i = 0; i < weights_.size(); ++i) {
      weights_(i) = generate_weight();
    }
}

uint32_t MLP::num_params() const {
  return n_hidden * (n_input + 1) + n_output * (n_hidden + 1);
}

template<typename T>
DynamicMatrix<T> MLP::operator()(const DynamicMatrix<T>& X) const {
  return MLP::function(n_input, n_hidden, n_output, softmax, weights_, X);
}

template<typename T>
DynamicMatrix<T> MLP::operator()(
  const DynamicVector<T>& weights, const DynamicMatrix<T>& X) const {
  CHECK_EQ(num_params(), weights.size());
  return MLP::function(n_input, n_hidden, n_output, softmax, weights, X);
}

template <typename T>
DynamicMatrix<T> MLP::function(
  uint32_t n_input, uint32_t n_hidden, uint32_t n_output, bool softmax,
  const DynamicVector<T>& weights, const DynamicMatrix<T>& X) {
  using Matrix = DynamicMatrix<T>;
  using Vector = DynamicVector<T>;
  CHECK_EQ(n_input, X.rows());

  const T* H_start{weights.data()};
  const T* Hb_start{H_start + n_hidden * n_input};
  const T* V_start{Hb_start + n_hidden};
  const T* Vb_start{V_start + n_hidden * n_output};
  Eigen::Map<const Matrix> H{H_start, n_hidden, n_input};
  Eigen::Map<const Vector> Hb{Hb_start, n_hidden};
  Eigen::Map<const Matrix> V{V_start, n_output, n_hidden};
  Eigen::Map<const Vector> Vb{Vb_start, n_output};

  Matrix H_out{H * X};
  H_out.colwise() += Hb;
  std::transform(H_out.data(), H_out.data() + H_out.size(), H_out.data(),
                 [] (const T& x) {return sigmoid(x);});
  // std::cout << "s(H * X + Hb) = \n" << H_out << "\n";
  Matrix V_out{V * H_out};
  V_out.colwise() += Vb;
  if (softmax) {
    std::transform(V_out.data(), V_out.data() + V_out.size(), V_out.data(),
                   [] (const T& x) {return exp(x);});
    const Vector exp_sum{V_out.colwise().sum()};
    for (int32_t i{0}; i < V_out.cols(); ++i) {
      V_out.col(i) /= exp_sum[i];
    }
  }
  return V_out;
}

}  // unet namespace
