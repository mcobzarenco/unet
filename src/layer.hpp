#include "stan/agrad/rev/matrix.hpp"
#include "stan/agrad/autodiff.hpp"

#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/src/Core/NumTraits.h>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>


namespace unet {

using std::tanh;
using stan::agrad::tanh;

struct MLP {
private:
  struct L2Error;

public:
  MLP(uint32_t n_input, uint32_t n_hidden, uint32_t n_output);
  MLP(uint32_t n_input, uint32_t n_hidden, uint32_t n_output,
      std::function<double()> generate_weight);

  uint32_t num_params() const;

  Eigen::MatrixXd operator()(Eigen::MatrixXd& X) const;

  L2Error l2_error(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y);

  const uint32_t n_input, n_hidden, n_output;

private:
  struct L2Error {
    L2Error(MLP& mlp, const Eigen::MatrixXd& X, const Eigen::MatrixXd& y)
      : mlp_{mlp}, X_{X}, y_{y} {}

    template <typename T>
    T operator()(Eigen::Matrix<T, Eigen::Dynamic, 1>& w) const;

    void minimize_gd(uint32_t max_iter=500);
  private:
    MLP& mlp_;
    const Eigen::MatrixXd& X_, y_;
  };

  template <typename T>
  static Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> function(
    uint32_t n_input, uint32_t n_hidden, uint32_t n_output,
    Eigen::Matrix<T, Eigen::Dynamic, 1> weights,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& X);

  Eigen::Matrix<double, Eigen::Dynamic, 1> weights_;
};

MLP::MLP(uint32_t n_input, uint32_t n_hidden, uint32_t n_output)
  : n_input(n_input), n_hidden(n_hidden), n_output(n_output),
    weights_{num_params()} {
  std::random_device rd;
  std::mt19937 generator{rd()};
  std::normal_distribution<> normal(0, .1);
  for (int i = 0; i < weights_.size(); ++i) {
    weights_(i) = normal(generator);
  }
  LOG(INFO) << "initial weights: " << weights_;
}

MLP::MLP(uint32_t n_input, uint32_t n_hidden, uint32_t n_output,
    std::function<double()> generate_weight)
  : n_input(n_input), n_hidden(n_hidden), n_output(n_output),
    weights_{num_params()} {
  for (int i = 0; i < weights_.size(); ++i) {
    weights_(i) = generate_weight();
  }
}

uint32_t MLP::num_params() const {
  return n_hidden * n_input + n_hidden * n_output;
}

Eigen::MatrixXd MLP::operator()(Eigen::MatrixXd& X) const {
  return MLP::function(n_input, n_hidden, n_output, weights_, X);
}

MLP::L2Error MLP::l2_error(
  const Eigen::MatrixXd& X, const Eigen::MatrixXd& y) {
  return L2Error(*this, X, y);
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MLP::function(
  uint32_t n_input, uint32_t n_hidden, uint32_t n_output,
  Eigen::Matrix<T, Eigen::Dynamic, 1> weights,
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& X) {
  using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  CHECK_EQ(n_input, X.rows());

  T* H_start{weights.data()};
  T* V_start{H_start + n_hidden * n_input};
  Eigen::Map<Matrix> H{H_start, n_hidden, n_input};
  Eigen::Map<Matrix> V{V_start, n_output, n_hidden};

  Matrix H_out{H * X};
  T* H_out_data{H_out.data()};
  for(size_t i = 0; i < H_out.size(); ++i) {
    H_out_data[i] = 2 * tanh(H_out_data[i]) - 1;
  }
  // std::transform(h_out.data(), h_out.data() + h_out.size() * sizeof(T),
  // cout << "H * X = " << h_out << endl;
  Matrix V_out{V * H_out};
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
      mlp_.n_input, mlp_.n_hidden, mlp_.n_output, weights, X)};

  std::cout << "net_out " << net_out.transpose() << std::endl;

  Matrix discrep{net_out - y};
  T err = (discrep * discrep.transpose()).trace();
  return err;
}

void MLP::L2Error::minimize_gd(uint32_t max_iter) {
  Eigen::Matrix<double, Eigen::Dynamic, 1> grad_net;
  double error{0}, last_error{1e10};
  double alpha{0.01};
  for (int i = 0; i < max_iter; ++i) {
    stan::agrad::gradient(*this, mlp_.weights_, error, grad_net);

    LOG(INFO) << "alpha: " << alpha;
    mlp_.weights_ -= alpha * grad_net;

    alpha *= (last_error > error ? 1.01 : 0.5);
    last_error = error;
  }
  std::cout << "alpha: " << alpha << "\n";
  std::cout << "w: " << mlp_.weights_.transpose() << "\n";
  std::cout << "df(w): " << grad_net.transpose() << "\n";
  std::cout << "Error: " << error << "\n";
}


}  // unet namespace
