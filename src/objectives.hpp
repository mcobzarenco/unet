#pragma once

#include "typedefs.hpp"
#include "stan/agrad/rev/matrix.hpp"
#include "stan/agrad/autodiff.hpp"

#include <Eigen/Dense>
#include <glog/logging.h>

#include <chrono>
#include <iostream>


namespace unet {

template<typename Net>
struct L2Error {
  L2Error(Net& net, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y)
    : net_(net), X_(X), Y_(Y) {}

  template <typename Scalar>
  inline Scalar operator()(const DynamicVector<Scalar>& weights) const;

  inline void gradient(const Eigen::VectorXd& weights, double& error,
                       Eigen::VectorXd& gradient) const;

private:
  Net& net_;
  const Eigen::MatrixXd& X_, Y_;
};

template<typename Net>
template <typename Scalar>
Scalar L2Error<Net>::operator()(const DynamicVector<Scalar>& weights) const {
  CHECK_EQ(net_.num_params(), weights.size())
    << "Expected " << net_.num_params() << " parameters.";

  DynamicMatrix<Scalar> net_out{net_(weights, X_.cast<Scalar>().eval())};
  DynamicMatrix<Scalar> discrep{net_out - Y_.cast<Scalar>()};
  Scalar err = (discrep.transpose() * discrep).trace();
  return err / net_out.size();
}

template<typename Net>
inline void L2Error<Net>::gradient(
  const Eigen::VectorXd& weights, double& error,
  Eigen::VectorXd& gradient) const {
  stan::agrad::gradient(*this, weights, error, gradient);
}

template<typename Scalar>
DynamicVector<Scalar> labels_from_distribution(const DynamicMatrix<Scalar>& out) {
  DynamicVector<Scalar> classes{out.cols()};
  for (int i = 0; i < out.cols(); ++i) {
    out.col(i).maxCoeff(&classes[i]);
  }
  return classes;
}

template<typename Net>
struct CrossEntropy {
  CrossEntropy(Net& net, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y)
    : net_(net), X_(X), Y_(Y) {}

  template <typename Scalar>
  inline Scalar operator()(const DynamicVector<Scalar>& weights) const;

  inline void gradient(const Eigen::VectorXd& weights, double& error,
                       Eigen::VectorXd& gradient) const;

  Eigen::VectorXd& weights() { return net_.weights(); }
private:
  Net& net_;
  const Eigen::MatrixXd& X_, Y_;
};

template<typename Net>
template <typename Scalar>
Scalar CrossEntropy<Net>::operator()(const DynamicVector<Scalar>& weights) const {
  CHECK_EQ(net_.num_params(), weights.size())
    << "Expected " << net_.num_params() << " parameters.";
  DynamicMatrix<Scalar> net_out{net_(weights, X_.cast<Scalar>().eval())};
  DynamicMatrix<Scalar> cross_entropy{
    -1.0 * net_out.array().log() * Y_.cast<Scalar>().array()};

  Scalar err = cross_entropy.sum() / cross_entropy.cols();
  return err;
}

template<typename Net>
inline void CrossEntropy<Net>::gradient(
  const Eigen::VectorXd& weights, double& error,
  Eigen::VectorXd& gradient) const {
  stan::agrad::gradient(*this, weights, error, gradient);
}

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
