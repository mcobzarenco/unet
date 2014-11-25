#pragma once

#include "init.hpp"
#include "stan/agrad/rev/matrix.hpp"
#include "stan/agrad/rev/functions/inv_logit.hpp"
#include "stan/agrad/autodiff.hpp"

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
private:
  struct L2Error;

public:
  MLP(uint32_t n_input, uint32_t n_hidden, uint32_t n_output,
      bool softmax, int32_t seed=0)
    : MLP(n_input, n_hidden, n_output, softmax,
          normal_weight_generator(0, .001, seed)) {}

  inline MLP(uint32_t n_input, uint32_t n_hidden, uint32_t n_output,
             bool softmax, std::function<double()> generate_weight);

  inline uint32_t num_params() const;

  inline Eigen::MatrixXd operator()(const Eigen::MatrixXd& X) const;

  inline L2Error l2_error(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y);

  const uint32_t n_input, n_hidden, n_output;
  const bool softmax;

private:
  struct L2Error {
    L2Error(MLP& mlp, const Eigen::MatrixXd& X, const Eigen::MatrixXd& y)
      : mlp_(mlp), X_(X), y_(y) {}

    template <typename T>
    inline T operator()(Eigen::Matrix<T, Eigen::Dynamic, 1>& w) const;

    void minimize_gd(uint32_t max_iter=500);
  private:
    MLP& mlp_;
    const Eigen::MatrixXd& X_, y_;
  };

  template <typename T>
  static inline Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> function(
    uint32_t n_input, uint32_t n_hidden, uint32_t n_output, bool softmax,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& weights,
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& X);

  Eigen::Matrix<double, Eigen::Dynamic, 1> weights_;
};

MLP::MLP(uint32_t n_input, uint32_t n_hidden, uint32_t n_output, bool softmax,
         std::function<double()> generate_weight)
  : n_input{n_input}, n_hidden{n_hidden}, n_output{n_output}, softmax{softmax} {
    CHECK(n_input > 0 && n_hidden > 0 && n_output > 0);
    CHECK(n_output > 1 || !softmax)
      << "There needs to be more than one output unit for softmax to make sense.";
    weights_.resize(num_params());
    for (int i = 0; i < weights_.size(); ++i) {
      weights_(i) = generate_weight();
    }
}

uint32_t MLP::num_params() const {
  return n_hidden * (n_input + 1) + n_output * (n_hidden + 1);
}

Eigen::MatrixXd MLP::operator()(const Eigen::MatrixXd& X) const {
  return MLP::function(n_input, n_hidden, n_output, softmax, weights_, X);
}

MLP::L2Error MLP::l2_error(
  const Eigen::MatrixXd& X, const Eigen::MatrixXd& y) {
  return L2Error(*this, X, y);
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MLP::function(
  uint32_t n_input, uint32_t n_hidden, uint32_t n_output,
  bool softmax, const Eigen::Matrix<T, Eigen::Dynamic, 1>& weights,
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& X) {
  using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
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
  // std::cout << "s(H * X) = \n" << H_out << "\n";
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

template <typename T>
T MLP::L2Error::operator()(Eigen::Matrix<T, Eigen::Dynamic, 1>& weights) const {
  using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  CHECK_EQ(mlp_.num_params(), weights.size())
    << "Expected " << mlp_.num_params() << " parameters.";

  Matrix X{X_.cast<T>()};
  Matrix y{y_.cast<T>()};
  Matrix net_out{MLP::function(
      mlp_.n_input, mlp_.n_hidden, mlp_.n_output, mlp_.softmax, weights, X)};

  // std::cout << "net_out " << net_out.transpose() << std::endl;

  Matrix discrep{net_out - y};
  T err = (discrep * discrep.transpose()).trace();
  return err / net_out.cols();
}

void MLP::L2Error::minimize_gd(uint32_t max_iter) {
  Eigen::VectorXd grad_net;
  Eigen::VectorXd weight_update{Eigen::VectorXd::Zero(mlp_.weights_.size())};

  double error{0}, last_error{1e10};
  double alpha{0.01}, mu{0.8};
  std::cout << "alpha: " << alpha << "\n";
  for (int i = 0; i < max_iter; ++i) {
    stan::agrad::gradient(*this, mlp_.weights_, error, grad_net);
    LOG(INFO) << "alpha: " << alpha << " / error: " << error;
    // std::cout << "df(w): " << grad_net.transpose() << "\n";
    weight_update = -alpha * grad_net + mu * weight_update;
    mlp_.weights_ += weight_update;

    alpha *= (last_error > error ? 1.01 : 0.25);
    last_error = error;
  }
  std::cout << "alpha: " << alpha << "\n";
  std::cout << "Error: " << error << "\n";
  // std::cout << "w: " << mlp_.weights_.transpose() << "\n";
  // std::cout << "df(w): " << grad_net.transpose() << "\n";
  std::cout << "net:\n" << mlp_(X_) << "\n";
}

}  // unet namespace
