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

namespace {
struct CalcSigmoid {
  template<typename T>
  T operator()(const unet::DynamicVector<T>& x) const { return unet::sigmoid(x[0]); }
};
}

TEST(Activation, SameSigmoid) {
  std::mt19937 generator{1337};
  std::normal_distribution<> normal(0, 10);
  double x;
  for(uint32_t iter = 0; iter < 1000; ++iter) {
    x = normal(generator);
    EXPECT_EQ(unet::sigmoid(x), unet::sigmoid(stan::agrad::var{x}));
  }
}

TEST(Activation, SigmoidDerivative) {
  std::mt19937 generator{1337};
  std::normal_distribution<> normal(0, 10);
  double discrep, fx;
  VectorXd x{1}, dx{1};

  for(uint32_t iter = 0; iter < 1000; ++iter) {
    x[0] = normal(generator);
    stan::agrad::gradient(CalcSigmoid{}, x, fx, dx);
    discrep = dx[0] - unet::dsigmoid(x[0]);
    EXPECT_LT(discrep * discrep, 1e-15);
  }
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
