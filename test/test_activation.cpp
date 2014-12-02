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
  Eigen::MatrixXd mat = Eigen::MatrixXd::Random(5, 10);
  unet::softmax_in_place(mat);

  EXPECT_TRUE((mat.array() > 0).all());
  EXPECT_TRUE((mat.colwise().sum().array() - 1.0 < 1e-10).all());
}
