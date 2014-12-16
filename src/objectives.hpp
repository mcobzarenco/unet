#pragma once

#include "typedefs.hpp"
#include "stan/agrad/rev/matrix.hpp"
#include "stan/agrad/autodiff.hpp"

#include <Eigen/Dense>
#include <glog/logging.h>

#include <chrono>
#include <iostream>


namespace unet {

/**** Regularizers ****/

struct NoRegularizer {
  template <typename Scalar>
  Scalar operator()(const DynamicVector<Scalar>& weights) const { return 0.0; }

  template <typename Scalar>
  void gradient(const DynamicVector<Scalar>& weights,
                double& objective, DynamicVector<Scalar>& gradient) const {}
};

/**** L2 Error Objective ****/

namespace internal {

template<typename Scalar, typename Net, typename Regularizer>
Scalar l2_objective(const Net& net, const Regularizer& regularizer,
                    const DynamicVector<Scalar>& weights,
                    const DynamicMatrix<Scalar>& X,
                    const DynamicMatrix<Scalar>& Y) {
  DynamicMatrix<Scalar> net_out{net(weights, X)};
  DynamicMatrix<Scalar> discrepancy{net_out - Y};
  Scalar err{discrepancy.squaredNorm() / net_out.size()};
  return err + regularizer(weights);
}

inline void l2_output_layer_grad(const Eigen::MatrixXd& network_out,
                                 const Eigen::MatrixXd& Y,
                                 double& out_error,
                                 Eigen::MatrixXd& out_grad) {
  out_grad = 2.0 * (network_out - Y) / network_out.size();
  out_error = network_out.size() * out_grad.squaredNorm() / 4.0;
};

template<typename Net, typename Regularizer>
void l2_objective_grad(const Net& net, const Regularizer& regularizer,
                       const Eigen::VectorXd& weights,
                       const Eigen::MatrixXd& X,
                       const Eigen::MatrixXd& Y,
                       double& error,
                       Eigen::VectorXd& gradient) {
  net.gradient(weights, X, Y, l2_output_layer_grad, error, gradient);
  regularizer.gradient(weights, error, gradient);
}

template<typename Net, typename Regularizer=NoRegularizer>
struct L2ErrorNoGradient {
  L2ErrorNoGradient(
    const Net& net, const Regularizer& regularizer,
    const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y)
    : net_(net), regularizer_{regularizer}, X_(X), Y_(Y) {}

  template <typename Scalar>
  Scalar operator()(const DynamicVector<Scalar>& weights) const {
    return l2_objective(net_, regularizer_, weights,
                        X_.cast<Scalar>().eval(), Y_.cast<Scalar>().eval());
  }

protected:
  const Net& net_;
  const Regularizer& regularizer_;
  const Eigen::MatrixXd& X_;
  const Eigen::MatrixXd& Y_;
};

}  // namespace internal

namespace agrad {

template<typename Net, typename Regularizer=NoRegularizer>
struct L2Error : public internal::L2ErrorNoGradient<Net, Regularizer> {
  L2Error(const Net& net, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y)
    : internal::L2ErrorNoGradient<Net, Regularizer>{net, Regularizer{}, X, Y} {}

  L2Error(const Net& net, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,
          const Regularizer& regularizer)
    : internal::L2ErrorNoGradient<Net, Regularizer>{net, regularizer, X, Y} {}

  void gradient(const Eigen::VectorXd& weights, double& error,
                Eigen::VectorXd& gradient) const {
    stan::agrad::gradient(*this, weights, error, gradient);
  }
};

}  // namespace agrad

template<typename Net, typename Regularizer=NoRegularizer>
struct L2Error : public internal::L2ErrorNoGradient<Net, Regularizer> {
  L2Error(const Net& net, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y)
    : internal::L2ErrorNoGradient<Net, Regularizer>{net, Regularizer{}, X, Y} {}

  L2Error(const Net& net, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,
          const Regularizer& regularizer)
    : internal::L2ErrorNoGradient<Net, Regularizer>{net, regularizer, X, Y} {}

  void gradient(const Eigen::VectorXd& weights, double& error,
                Eigen::VectorXd& gradient) const {
    internal::l2_objective_grad(this->net_, this->regularizer_, weights,
                                this->X_, this->Y_, error, gradient);
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

template<typename Scalar, typename Net, typename Regularizer>
Scalar log_objective(const Net& net, const Regularizer& regularizer,
                     const DynamicVector<Scalar>& weights,
                     const DynamicMatrix<Scalar>& X,
                     const DynamicMatrix<Scalar>& Y) {
  DynamicMatrix<Scalar> net_out{net(weights, X)};
  DynamicMatrix<Scalar> cross_entropy{-1.0 * net_out.array().log() * Y.array()};

  Scalar err = cross_entropy.sum() / cross_entropy.cols();
  return err + regularizer(weights);
}

inline void log_output_layer_grad(const Eigen::MatrixXd& net_out,
                                  const Eigen::MatrixXd& Y,
                                  double& out_error,
                                  Eigen::MatrixXd& out_grad) {
  out_grad = -(Y.array() - net_out.array()) / net_out.cols();
  out_error = (-1.0 * Y.array() * net_out.array().log()).sum() / net_out.cols();
};

template<typename Net, typename Regularizer>
void log_objective_grad(const Net& net, const Regularizer& regularizer,
                        const Eigen::VectorXd& weights,
                        const Eigen::MatrixXd& X,
                        const Eigen::MatrixXd& Y,
                        double& error,
                        Eigen::VectorXd& gradient) {
  net.gradient(weights, X, Y, log_output_layer_grad, error, gradient);
  regularizer.gradient(weights, error, gradient);
}

template<typename Net, typename Regularizer=NoRegularizer>
struct CrossEntropyNoGradient {
  CrossEntropyNoGradient(
    const Net& net, const Regularizer& regularizer,
    const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y)
    : net_(net), regularizer_{regularizer}, X_(X), Y_(Y) {}

  template <typename Scalar>
  Scalar operator()(const DynamicVector<Scalar>& weights) const {
    return log_objective(net_, regularizer_, weights,
                         X_.cast<Scalar>().eval(), Y_.cast<Scalar>().eval());
  }

protected:
  const Net& net_;
  const Regularizer& regularizer_;
  const Eigen::MatrixXd& X_;
  const Eigen::MatrixXd& Y_;
};

}  // namespace internal

namespace agrad {

template<typename Net, typename Regularizer=NoRegularizer>
struct CrossEntropy : public internal::CrossEntropyNoGradient<Net, Regularizer> {
  CrossEntropy(const Net& net, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y)
    : internal::CrossEntropyNoGradient<Net, Regularizer>{net, Regularizer{}, X, Y} {}

  CrossEntropy(const Net& net, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,
          const Regularizer& regularizer)
    : internal::CrossEntropyNoGradient<Net, Regularizer>{net, regularizer, X, Y} {}

  void gradient(const Eigen::VectorXd& weights, double& error,
                Eigen::VectorXd& gradient) const {
    stan::agrad::gradient(*this, weights, error, gradient);
  }
};

}  // namespace agrad

template<typename Net, typename Regularizer=NoRegularizer>
struct CrossEntropy : public internal::CrossEntropyNoGradient<Net, Regularizer> {
  CrossEntropy(const Net& net, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y)
    : internal::CrossEntropyNoGradient<Net, Regularizer>{net, Regularizer{}, X, Y} {}

  CrossEntropy(const Net& net, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,
          const Regularizer& regularizer)
    : internal::CrossEntropyNoGradient<Net, Regularizer>{net, regularizer, X, Y} {}

  void gradient(const Eigen::VectorXd& weights, double& error,
                Eigen::VectorXd& gradient) const {
    internal::log_objective_grad(this->net_, this->regularizer_, weights,
                                 this->X_, this->Y_, error, gradient);
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
