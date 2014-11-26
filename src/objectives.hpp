#pragma once

#include <Eigen/Dense>
#include <iostream>


namespace unet {

template<typename Net>
struct L2Error {
  L2Error(Net& net, const Eigen::MatrixXd& X, const Eigen::MatrixXd& y)
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
T L2Error<Net>::operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& weights) const {
  using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  CHECK_EQ(net_.num_params(), weights.size())
    << "Expected " << net_.num_params() << " parameters.";

  Matrix X{X_.cast<T>()};
  Matrix y{y_.cast<T>()};
  Matrix net_out{net_(weights, X)};

  // std::cout << "net_out " << net_out.transpose() << std::endl;

  Matrix discrep{net_out - y};
  T err = (discrep * discrep.transpose()).trace();
  return err / net_out.cols();
}

}  // namespace unet
