#include "activation.hpp"
#include "objectives.hpp"
#include "utilities.hpp"
#include "typedefs.hpp"
#include "stan/agrad/rev/matrix.hpp"
#include "stan/agrad/autodiff.hpp"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>


using namespace std;
using Eigen::Matrix;

namespace {
struct LinearNet {
  Eigen::VectorXd& weights() { return weights_; }

  template<typename T>
  unet::DynamicMatrix<T> operator() (const unet::DynamicVector<T>& weights,
                                     const unet::DynamicMatrix<T>& X) const {
    return X.array().colwise() * weights.array();
  }

  uint32_t num_params() const { return weights_.size(); }

  Eigen::VectorXd weights_;
};

struct FixedOutput {
  Eigen::VectorXd& weights() { return weights_; }

  template<typename T>
  unet::DynamicMatrix<T> operator() (const unet::DynamicVector<T>& weights,
                                     const unet::DynamicMatrix<T>& X) const {
    return output_.cast<T>();
  }

  uint32_t num_params() const { return weights_.size(); }

  Eigen::VectorXd weights_;
  Eigen::MatrixXd output_;
};
}  // anonymous namespace


TEST(Objectives, L2Error) {
  Eigen::MatrixXd X{3, 2};
  Eigen::MatrixXd y{3, 2};
  X <<
    1.0, 2,
    3, 4,
    5, 6;
  y <<
    0.7, 0.5,
    0.2, 0.2,
    0.1, 0.3;

  double error{-1};
  Eigen::VectorXd grad;

  FixedOutput fixed_out;
  fixed_out.output_.resize(3, 2);
  fixed_out.output_ = y;
  fixed_out.weights_ = Eigen::VectorXd::Random(4);
  unet::agrad::L2Error<FixedOutput> objective0{fixed_out, X, y};
  stan::agrad::gradient(objective0, fixed_out.weights_, error, grad);
  EXPECT_EQ(0, error);
  EXPECT_EQ(0, grad.transpose() * grad);

  fixed_out.output_.array() += 2.0;
  stan::agrad::gradient(objective0, fixed_out.weights_, error, grad);
  EXPECT_EQ(4.0, error);
  EXPECT_EQ(0, grad.transpose() * grad);
}

TEST(Objectives, CrossEntropy) {
  Eigen::MatrixXd X{3, 2};
  Eigen::MatrixXd y{3, 2};
  X <<
    1.0, 2,
    3, 4,
    5, 6;
  y <<
    0.7, 0.5,
    0.2, 0.2,
    0.1, 0.3;

  double error{-1};
  Eigen::VectorXd grad;
  double y_entropy{(-1.0 * y.array() * y.array().log()).sum() / y.cols()};
  EXPECT_GT(y_entropy, 0);

  FixedOutput fixed_out;
  fixed_out.output_.resize(3, 2);
  fixed_out.output_ = y;
  fixed_out.weights_ = Eigen::VectorXd::Random(4);
  unet::agrad::CrossEntropy<FixedOutput> objective0{fixed_out, X, y};
  stan::agrad::gradient(objective0, fixed_out.weights_, error, grad);
  EXPECT_EQ(y_entropy, error);
  EXPECT_EQ(0, grad.transpose() * grad);

  for (int iter = 0; iter < 100; ++iter) {
    fixed_out.output_ = Eigen::MatrixXd::Random(3, 2);
    unet::softmax_in_place(fixed_out.output_);
    stan::agrad::gradient(objective0, fixed_out.weights_, error, grad);
    ASSERT_TRUE(std::isfinite(error)) << "Error = " << error;
    ASSERT_LT(y_entropy, error);
    ASSERT_EQ(0, grad.transpose() * grad);
  }

  // unet::CrossEntropy<TestNet> objective{net, X, y};
  // stan::agrad::gradient(objective, net.weights_, error, grad);

  // TestNet net;
  // net.weights_.resize(3);
  // net.weights_ << 2, 1, 3;
}
