#include "activation.hpp"
#include "stan/agrad/autodiff.hpp"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <random>


using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

template<typename NonLinearity>
void test_nonlinearity_consistency() {
  std::mt19937 generator{1337};
  std::normal_distribution<> normal(0, 10);
  double x;
  double double_value;
  stan::agrad::var agard_value;

  // Test it gives the same values for doubles and agrad::var
  for(uint32_t iter = 0; iter < 1000; ++iter) {
    x = normal(generator);
    double_value = NonLinearity::activation(x);
    agard_value = NonLinearity::activation(stan::agrad::var{x});
    ASSERT_EQ(double_value, agard_value);
  }

  // Test whether the matrix operations agree with their scalar counterparts
  MatrixXd X{MatrixXd::Random(200, 200)};

  MatrixXd X_activation{X};
  NonLinearity::activation_in_place(X_activation);

  MatrixXd X_derivative{X};
  NonLinearity::derivative_in_place(X_derivative);

  MatrixXd X_derivative_value{X_activation};
  NonLinearity::derivative_value_in_place(X_derivative_value);

  for(auto i = 0; i < X.rows(); ++i) {
    for(auto j = 0; j < X.cols(); ++j) {
      ASSERT_EQ(NonLinearity::activation(X(i, j)), X_activation(i, j));
      ASSERT_EQ(NonLinearity::derivative(X(i, j)), X_derivative(i, j));
      ASSERT_EQ(NonLinearity::derivative(X(i, j)), X_derivative_value(i, j));
    }
  }

  // Test that auto diff and the provided derivative agree
  VectorXd v_x{1}, auto_dx{1};
  double auto_fx, user_dx;
  auto nlin = [](const auto& x) { return NonLinearity::activation(x[0]); };
  for(uint32_t iter = 0; iter < 1000; ++iter) {
    v_x[0] = normal(generator);
    user_dx = NonLinearity::derivative(v_x[0]);
    stan::agrad::gradient(nlin, v_x, auto_fx, auto_dx);
    ASSERT_NEAR(auto_dx[0], user_dx, 1e-14);
    ASSERT_NEAR(auto_fx, NonLinearity::activation(v_x[0]), 1e-14);
  }
}

TEST(Activation, Tanh) {
  test_nonlinearity_consistency<unet::Tanh>();
  ASSERT_DOUBLE_EQ(-0.76159415595576485, unet::Tanh::activation(-1.0));
  ASSERT_DOUBLE_EQ(-0.46211715726000974, unet::Tanh::activation(-0.5));
  ASSERT_DOUBLE_EQ(0.0, unet::Tanh::activation(0.0));
  ASSERT_DOUBLE_EQ(0.46211715726000974, unet::Tanh::activation(0.5));
  ASSERT_DOUBLE_EQ(0.76159415595576485, unet::Tanh::activation(1.0));
}

TEST(Activation, ReLU) {
  test_nonlinearity_consistency<unet::ReLU>();
  ASSERT_DOUBLE_EQ(0, unet::ReLU::activation(-1.0));
  ASSERT_DOUBLE_EQ(0, unet::ReLU::activation(-0.5));
  ASSERT_DOUBLE_EQ(0, unet::ReLU::activation(0.0));
  ASSERT_DOUBLE_EQ(0.5, unet::ReLU::activation(0.5));
  ASSERT_DOUBLE_EQ(1.0, unet::ReLU::activation(1.0));
}

TEST(Activation, Softmax) {
  Eigen::MatrixXd mat1 = Eigen::MatrixXd::Random(5, 10);
  unet::softmax_in_place(mat1);

  EXPECT_TRUE((mat1.array() > 0).all());
  EXPECT_TRUE((mat1.colwise().sum().array() - 1.0 < 1e-10).all());

  Eigen::MatrixXd mat2 = Eigen::MatrixXd{3, 2};
  Eigen::MatrixXd exp2 = Eigen::MatrixXd{3, 2};
  mat2 <<
     1.0, 2.0,
    -2.0, 0.1,
    -0.5, 0.9;
  exp2 <<
    0.78559703,  0.67456369,
    0.03911257,  0.10089356,
    0.17529039,  0.22454275;
  unet::softmax_in_place(mat2);
  EXPECT_LT(((exp2 - mat2) * (exp2 - mat2).transpose()).trace(), 1e-10);
}
