#pragma once

#include "typedefs.hpp"
#include "stan/agrad/rev/matrix.hpp"
#include "stan/agrad/autodiff.hpp"

#include <Eigen/Dense>
#include <glog/logging.h>

#include <chrono>
#include <iostream>


namespace unet {

/**** L2 Error Objective ****/

namespace internal {

template<typename Net>
struct L2ErrorNoGradient {
  L2ErrorNoGradient(
    const Net& net, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y)
    : net_(net), X_(X), Y_(Y) {}

  template <typename Scalar>
  inline Scalar operator()(const DynamicVector<Scalar>& weights) const {
    CHECK_EQ(net_.num_params(), weights.size())
      << "Expected " << net_.num_params() << " parameters.";

    DynamicMatrix<Scalar> net_out{net_(weights, X_.cast<Scalar>().eval())};
    DynamicMatrix<Scalar> discrepancy{net_out - Y_.cast<Scalar>()};
    Scalar err = discrepancy.squaredNorm() / net_out.size();
    return err;
  }

protected:
  const Net& net_;
  const Eigen::MatrixXd& X_;
  const Eigen::MatrixXd& Y_;
};

}  // namespace internal

namespace agrad {

template<typename Net>
struct L2Error : public internal::L2ErrorNoGradient<Net> {
  L2Error(const Net& net, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y)
    : internal::L2ErrorNoGradient<Net>{net, X, Y} {}

  void gradient(const Eigen::VectorXd& weights, double& error,
                Eigen::VectorXd& gradient) const {
    stan::agrad::gradient(*this, weights, error, gradient);
  }
};

}  // namespace agrad

template<typename Net>
struct L2Error : public internal::L2ErrorNoGradient<Net> {
  L2Error(const Net& net, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y)
    : internal::L2ErrorNoGradient<Net>{net, X, Y} {}

  void gradient(const Eigen::VectorXd& weights, double& error,
                Eigen::VectorXd& gradient) const {
    this->net_.gradient(weights, this->X_, this->Y_,
                        L2Error<Net>::output_gradient, error, gradient);
  }

  static void output_gradient(
    const Eigen::MatrixXd& network_out, const Eigen::MatrixXd& Y,
    double& error, Eigen::MatrixXd& out_grad) {
    out_grad = 2.0 * (network_out - Y) / network_out.size();
    error = network_out.size() * out_grad.squaredNorm() / 4.0;
  }
};

/**** Cross Entropy Objective ****/

template<typename Scalar>
DynamicVector<Scalar> labels_from_distribution(const DynamicMatrix<Scalar>& out) {
  DynamicVector<Scalar> classes{out.cols()};
  for (int i = 0; i < out.cols(); ++i) {
    out.col(i).maxCoeff(&classes[i]);
  }
  return classes;
}

namespace internal {

template<typename Net>
struct CrossEntropyNoGradient {
  CrossEntropyNoGradient(
    const Net& net, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y)
    : net_(net), X_(X), Y_(Y) {}

  template <typename Scalar>
  inline Scalar operator()(const DynamicVector<Scalar>& weights) const {
    CHECK_EQ(net_.num_params(), weights.size())
      << "Expected " << net_.num_params() << " parameters.";
    DynamicMatrix<Scalar> net_out{net_(weights, X_.cast<Scalar>().eval())};
    DynamicMatrix<Scalar> cross_entropy{
      -1.0 * net_out.array().log() * Y_.cast<Scalar>().array()};

    Scalar err = cross_entropy.sum() / cross_entropy.cols();
    return err;
  }

protected:
  const Net& net_;
  const Eigen::MatrixXd& X_, Y_;
};

}  // namespace internal

namespace agrad {

template<typename Net>
struct CrossEntropy : public internal::CrossEntropyNoGradient<Net> {
  CrossEntropy(const Net& net, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y)
    : internal::CrossEntropyNoGradient<Net>{net, X, Y} {}

  void gradient(const Eigen::VectorXd& weights, double& error,
                Eigen::VectorXd& gradient) const {
    stan::agrad::gradient(*this, weights, error, gradient);
  }
};

}  // namespace agrad

template<typename Net>
struct CrossEntropy : public internal::CrossEntropyNoGradient<Net> {
  CrossEntropy(const Net& net, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y)
    : internal::CrossEntropyNoGradient<Net>{net, X, Y} {}

  void gradient(const Eigen::VectorXd& weights, double& error,
                Eigen::VectorXd& gradient) const {
    this->net_.gradient(weights, this->X_, this->Y_,
                        CrossEntropy<Net>::output_gradient, error, gradient);
  }

  static void output_gradient(
    const Eigen::MatrixXd& net_out, const Eigen::MatrixXd& Y,
    double& error, Eigen::MatrixXd& out_grad) {
    out_grad = -(Y.array() - net_out.array()) / net_out.cols();
    error = (-1.0 * Y.array() * net_out.array().log()).sum() / net_out.cols();
  }
};


/**** Accuracy ****/

template<typename Net>
struct Accuracy {
  Accuracy(Net& net, const Eigen::MatrixXd& X, const Eigen::MatrixXd& y)
    : net_(net), X_(X), y_(y) {}

  template <typename Scalar>
  inline Scalar operator()(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& weights) const;

private:
  Net& net_;
  const Eigen::MatrixXd& X_, y_;
};

template<typename Net>
template <typename Scalar>
Scalar Accuracy<Net>::operator()(const DynamicVector<Scalar>& weights) const {
  CHECK_EQ(net_.num_params(), weights.size())
    << "Expected " << net_.num_params() << " parameters.";
  DynamicMatrix<Scalar> X{X_.cast<Scalar>()};
  DynamicMatrix<Scalar> y{y_.cast<Scalar>()};
  auto begin = std::chrono::high_resolution_clock::now();
  DynamicMatrix<Scalar> net_out{net_(weights, X)};
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
    std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
  LOG(INFO) << "Net eval took " << duration << " ms";

  DynamicVector<Scalar> classes = labels_from_distribution(y);
  DynamicVector<Scalar> pred_classes = labels_from_distribution(net_out);
  Scalar err = (pred_classes.array() == classes.array()).template cast<double>().sum()
    / classes.size();
  return err;
}

}  // namespace unet
