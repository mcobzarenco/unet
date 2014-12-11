#include "feedforward.hpp"
#include "objectives.hpp"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <iostream>


using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {
constexpr double EPS = 1e-14;

template<typename F>
VectorXd finite_diff(F f, const VectorXd& x) {
  double eps{0.000001};
  VectorXd grad{x.size()};
  VectorXd x_eps{x};
  for (uint32_t i = 0; i < x.size(); ++i) {
    x_eps[i] += eps;
    grad[i] = (f(x) - f(x_eps)) / eps;
    x_eps[i] -= eps;
  }
  return grad;
}

template<typename Net, typename UserObjective, typename AutoObjective>
void expect_agreement(const Net& net, const UserObjective& user_obj,
                    const AutoObjective& auto_obj) {
  double error_user{-1}, error_auto{-1};
  VectorXd grad_user, grad_auto;

  EXPECT_DOUBLE_EQ(auto_obj(net.weights()), user_obj(net.weights()));

  user_obj.gradient(net.weights(), error_user, grad_user);
  auto_obj.gradient(net.weights(), error_auto, grad_auto);
  ASSERT_EQ(grad_user.size(), grad_auto.size());
  EXPECT_NEAR(error_auto, error_user, EPS);
  EXPECT_NEAR(0, (grad_auto - grad_user).squaredNorm(), EPS);
}
}  // anonymous namespace

TEST(FeedForward, WeightsAsParams) {
  unet::FeedForward<unet::Tanh> net{{2, 2, 1}, false};
  VectorXd w{9};
  w << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  net.weights() = w;

  EXPECT_DEATH({
      net.weights() = w.segment(0, 8);
      net.weights_as_params();
    }, "");

  MatrixXd exp_W0{2, 2}, exp_W1{1, 2};
  VectorXd exp_b0{2}, exp_b1{1};
  // The weights are read in column-major order into the matrices:
  exp_W0 << 1, 3, 2, 4;
  exp_b0 << 5, 6;
  exp_W1 << 7, 8;
  exp_b1 << 9;

  auto params = net.weights_as_params();
  ASSERT_EQ(2, params.W.size());
  ASSERT_EQ(2, params.b.size());
  EXPECT_EQ(exp_W0, params.W[0])
    << "expected:\n" <<  exp_W0 << "\n"
    << "actual:\n" << params.W[0];
  EXPECT_EQ(exp_b0, params.b[0])
    << "expected:\n" <<  exp_W0 << "\n"
    << "actual:\n" << params.b[0];
  EXPECT_EQ(exp_W1, params.W[1])
    << "expected:\n" <<  exp_W1 << "\n"
    << "actual:\n" << params.W[1];
  EXPECT_EQ(exp_b1, params.b[1])
    << "expected:\n" <<  exp_W1 << "\n"
    << "actual:\n" << params.b[1];
}

TEST(FeedForward, CheckGradientForL2Error) {
  using unet::FeedForward;
  MatrixXd X, Y;

  for (uint32_t n_input = 50; n_input < 300; n_input += 23) {
    for (uint32_t n_output = 20; n_output < 300; n_output += 31) {
      FeedForward<unet::ReLU> net{{n_input, 40, 50, 20, n_output}, false, 1};
      X = MatrixXd::Random(n_input, 100);
      Y = MatrixXd::Random(n_output, 100);

      auto l2_error_user = net.l2_error(X, Y);
      unet::L2Error<FeedForward<unet::ReLU>> l2_error_auto{net, X, Y};
      expect_agreement(net, l2_error_user, l2_error_auto);
    }
  }
}

TEST(FeedForward, CheckGradientForCrossEntropy) {
  using unet::FeedForward;
  MatrixXd X, Y;
  VectorXd discrep;

  for (uint32_t n_input = 50; n_input < 300; n_input += 23) {
    for (uint32_t n_output = 20; n_output < 300; n_output += 31) {
      FeedForward<unet::Tanh> net{{n_input, 40, 50, 20, n_output}, true, 1};
      X = MatrixXd::Random(n_input, 100);
      Y = MatrixXd::Random(n_output, 100);
      unet::softmax_in_place(Y);

      auto cross_entropy_user = net.cross_entropy(X, Y);
      unet::CrossEntropy<FeedForward<unet::Tanh>> cross_entropy_auto{net, X, Y};
      expect_agreement(net, cross_entropy_user, cross_entropy_auto);
    }
  }
}

// TEST(FeedForward, CheckGradient) {
//   unet::FeedForward<4> net{{2, 2, 3, 4}, false, 1};
//   cout << net.structured_weights().W[0] << endl;

//   // MatrixXd X{2, 1};
//   // MatrixXd Y{4, 1};
//   // X <<
//   //   1,
//   //   2;
//   // Y <<
//   //   3,
//   //   1,
//   //   3,
//   //   1;
//   MatrixXd X{2, 3};
//   MatrixXd Y{4, 3};
//   X <<
//     1, 2, -1.2,
//     2, 4, -.8;
//   Y <<
//     3,  6,  1,
//     1,  2, -0.01,
//     3, -1,  0.56,
//     1,  1,  3;
//   cout << net(X) << endl;

//   double error_user{-1}, error_auto{-1};
//   Eigen::VectorXd grad_user = VectorXd::Zero(net.num_params());
//   Eigen::VectorXd grad_auto = VectorXd::Zero(net.num_params());
//   Eigen::VectorXd discrep;

//   net.l2_error(X, Y, error_user, grad_user);
//   unet::L2Error<unet::FeedForward<4>> l2_error{net, X, Y};
//   stan::agrad::gradient(l2_error, l2_error.weights(), error_auto, grad_auto);

//   ASSERT_EQ(grad_user.size(), grad_auto.size());
//   Eigen::MatrixXd grad{grad_user.size(), 2};
//   grad.col(0) = grad_auto;
//   grad.col(1) = grad_user / 2;

//   cout << "[l2] error_user=" << error_user << endl;
//   cout << "[l2] error_auto=" << error_auto << endl;
//   cout << "[l2] grad (auto, user)=\n" << grad << endl;

//   discrep = grad_auto - grad_user / 2;
//   cout << "[l2] discrep =\n" << discrep << endl;
//   EXPECT_LT(discrep.transpose() * discrep, 1e-10);
// }
