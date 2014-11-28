#include "mlp.hpp"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <iostream>


using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

TEST(MLP, StructuredWeights) {
  unet::MLP mlp{2, 2, 1, false};
  VectorXd w{9};
  w << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  mlp.weights() = w;

  EXPECT_DEATH({
      mlp.weights() = w.segment(0, 8);
      mlp.structured_weights();
    }, "");

  MatrixXd exp_H{2, 2}, exp_V{1, 2};
  VectorXd exp_Hb{2}, exp_Vb{1};
  // The weights are read in column-major order into the matrices:
  exp_H  << 1, 3, 2, 4;
  exp_Hb << 5, 6;
  exp_V  << 7, 8;
  exp_Vb << 9;

  auto W = mlp.structured_weights();
  EXPECT_EQ(exp_H, W.H)
    << "expected:\n" <<  exp_H << "\n"
    << "actual:\n" << W.H;
  EXPECT_EQ(exp_Hb, W.Hb)
    << "expected:\n" <<  exp_Hb << "\n"
    << "actual:\n" << W.Hb;
  EXPECT_EQ(exp_V, W.V)
    << "expected:\n" <<  exp_V << "\n"
    << "actual:\n" << W.V;
  EXPECT_EQ(exp_Vb, W.Vb)
    << "expected:\n" <<  exp_Vb << "\n"
    << "actual:\n" << W.Vb;
}
