#pragma once

#include "typedefs.hpp"

#include <Eigen/Dense>
#include <glog/logging.h>

#include <iostream>


namespace unet {

template<typename Net>
struct L2Error {
  L2Error(Net& net, const Eigen::MatrixXd& X, const Eigen::MatrixXd& y)
    : net_(net), X_(X), y_(y) {}

  template <typename T>
  inline T operator()(const DynamicVector<T>& weights) const;

  Eigen::VectorXd& weights() { return net_.weights(); }
private:
  Net& net_;
  const Eigen::MatrixXd& X_, y_;
};

template<typename Net>
template <typename T>
T L2Error<Net>::operator()(const DynamicVector<T>& weights) const {
  CHECK_EQ(net_.num_params(), weights.size())
    << "Expected " << net_.num_params() << " parameters.";
  DynamicMatrix<T> X{X_.cast<T>()};
  DynamicMatrix<T> y{y_.cast<T>()};
  DynamicMatrix<T> net_out{net_(weights, X)};

  std::cout << "net_out\n" << net_out << std::endl;

  DynamicMatrix<T> discrep{net_out - y};
  T err = (discrep * discrep.transpose()).trace();
  return err / net_out.size();
}

template<typename T>
DynamicVector<T> labels_from_distribution(const DynamicMatrix<T>& out) {
  DynamicVector<T> classes{out.cols()};
  for (int i = 0; i < out.cols(); ++i) {
    out.col(i).maxCoeff(&classes[i]);
  }
  return classes;
}

template<typename Net>
struct CrossEntropy {
  CrossEntropy(Net& net, const Eigen::MatrixXd& X, const Eigen::MatrixXd& y)
    : net_(net), X_(X), y_(y) {}

  template <typename T>
  inline T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& weights) const;

  Eigen::VectorXd& weights() { return net_.weights(); }
private:
  Net& net_;
  const Eigen::MatrixXd& X_, y_;
};

template<typename Net>
template <typename T>
T CrossEntropy<Net>::operator()(const DynamicVector<T>& weights) const {
  CHECK_EQ(net_.num_params(), weights.size())
    << "Expected " << net_.num_params() << " parameters.";
  DynamicMatrix<T> X{X_.cast<T>()};
  DynamicMatrix<T> y{y_.cast<T>()};
  DynamicMatrix<T> net_out{net_(weights, X)};

  DynamicVector<T> classes = labels_from_distribution(y);
  DynamicVector<T> pred_classes = labels_from_distribution(net_out);

  // std::cout << "net_out\n" << pred_classes.transpose() << std::endl;
  // std::cout << "truth\n" << classes.transpose() << std::endl;

  LOG(INFO) << "accuracy = "
            << (pred_classes.array() == classes.array()).template cast<double>().sum() / classes.size();
  // std::cout << "net_out\n" << net_out << std::endl;
  // std::cout << "log(net_out)\n" << net_out.array().log() << std::endl;
  // std::cout << "-log(net_out) * y\n" << -1.0 * net_out.array().log() * y.array()
  //           << std::endl;

  DynamicMatrix<T> cross_entropy{-1.0 * net_out.array().log() * y.array()};

  T err = cross_entropy.sum() / cross_entropy.cols();
  return err;
}


}  // namespace unet
