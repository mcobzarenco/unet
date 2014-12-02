#include "feedforward.hpp"
#include "objectives.hpp"

#include "stan/agrad/rev/matrix.hpp"
#include "stan/agrad/autodiff.hpp"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <iostream>


using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {
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
}

TEST(FeedForward, StructuredWeights) {
  unet::FeedForward<3> net{{2, 2, 1}, false};
  VectorXd w{9};
  w << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  net.weights() = w;

  EXPECT_DEATH({
      net.weights() = w.segment(0, 8);
      net.structured_weights();
    }, "");

  MatrixXd exp_W0{2, 2}, exp_W1{1, 2};
  VectorXd exp_b0{2}, exp_b1{1};
  // The weights are read in column-major order into the matrices:
  exp_W0 << 1, 3, 2, 4;
  exp_b0 << 5, 6;
  exp_W1 << 7, 8;
  exp_b1 << 9;

  auto params = net.structured_weights();
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

TEST(FeedForward, CheckGradient) {
  MatrixXd X, Y;
  double error_user{-1}, error_auto{-1};
  VectorXd grad_user, grad_auto;
  VectorXd discrep;

  for (uint32_t n_input = 50; n_input < 100; n_input += 23) {
    for (uint32_t n_output = 20; n_output < 100; n_output += 31) {
      for (uint32_t iter = 0; iter < 5; ++iter) {
        unet::FeedForward<5> net{{n_input, 40, 50, 20, n_output}, false, 1};
        X = MatrixXd::Random(n_input, 100);
        Y = MatrixXd::Random(n_output, 100);

        net.l2_error(X, Y, error_user, grad_user);
        unet::L2Error<unet::FeedForward<5>> l2_error{net, X, Y};
        stan::agrad::gradient(l2_error, l2_error.weights(), error_auto, grad_auto);
        ASSERT_EQ(grad_user.size(), grad_auto.size());
        ASSERT_LT((error_user - error_auto) *(error_user - error_auto), 1e-20);

        discrep = grad_auto - grad_user;
        // cout << "[l2] discrep =\n" << discrep.transpose() * discrep << endl;
        EXPECT_LT(discrep.transpose() * discrep, 1e-20);
      }
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
