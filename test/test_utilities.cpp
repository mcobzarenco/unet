#include "utilities.hpp"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <vector>
#include <utility>


using namespace std;
using Eigen::Matrix;

TEST(Utilities, ParseRange) {
  const string range0;
  auto parsed0 = unet::parse_range(range0);
  EXPECT_FALSE(parsed0);

  const string range1{"10:91"};
  auto parsed1 = unet::parse_range(range1);
  ASSERT_TRUE(parsed1);
  EXPECT_EQ(make_pair(10, 91), parsed1);

  const string range2{"122:91"};
  auto parsed2 = unet::parse_range(range2);
  ASSERT_TRUE(parsed2);
  EXPECT_EQ(make_pair(122, 91), parsed2);

  const string range3{"afd"};
  auto parsed3 = unet::parse_range(range3);
  EXPECT_FALSE(parsed3);

  const string range4{"23:afsd"};
  auto parsed4 = unet::parse_range(range4);
  EXPECT_FALSE(parsed4);

  const string range5{"23:32:41"};
  auto parsed5 = unet::parse_range(range5);
  EXPECT_FALSE(parsed5);
}

TEST(Utilities, SplitStr) {
  string reals_str = " .1 ,4.3, -1.21,  -31.3, -.0011  ";
  string ints_str = "1 3 21 -31 -11";
  vector<double> exp_reals = {0.1, 4.3, -1.21, -31.3, -0.0011};
  vector<double> exp_ints = {1, 3, 21, -31, -11};

  Eigen::VectorXd reals_vec{unet::vector_from_str(reals_str, ',')};
  Eigen::VectorXd ints_vec{unet::vector_from_str(ints_str, ' ')};
  vector<double> reals{
    reals_vec.data(), reals_vec.data() + reals_vec.size()};
  vector<double> ints{
    ints_vec.data(), ints_vec.data() + ints_vec.size()};

  EXPECT_EQ(exp_reals, reals);
  EXPECT_EQ(exp_ints, ints);
}

TEST(Utilities, ReadBatch) {
}

TEST(Utilities, RangeSelector) {
}

TEST(Utilities, OneHotEncoder) {
  unet::OneHotEncoder encoder(2, 3);
  Eigen::VectorXd vec1{4}, vec2{4}, vec3{4};
  Eigen::VectorXd expect1{3}, expect2{3}, expect3{3};

  vec1 << 1.2, .4, 0, -18;
  expect1 << 1, 0, 0;
  auto encoded1 = encoder(vec1);
  EXPECT_EQ(expect1, encoded1)
    << "expected: " << expect1.transpose() << "\n"
    << "actual: " << encoded1.transpose();

  vec2 << 1.2, 3.4, 1, -1;
  expect2 << 0, 1, 0;
  auto encoded2 = encoder(vec2);
  EXPECT_EQ(expect2, encoded2)
    << "expected: " << expect2.transpose() << "\n"
    << "actual: " << encoded2.transpose();

  vec3 << 1.2, 3.4, 2, -1;
  expect3 << 0, 0, 1;
  auto encoded3 = encoder(vec3);
  EXPECT_EQ(expect3, encoded3)
    << "expected: " << expect3.transpose() << "\n"
    << "actual: " << encoded3.transpose();

  Eigen::VectorXd vec4{4};
  vec4 << 1.2, 3.4, 3, -1;
  EXPECT_DEATH({ encoder(vec4); }, "");

  Eigen::VectorXd vec5{4};
  vec5 << 0, 0, 3.3, -12;
  EXPECT_DEATH({ encoder(vec5); }, "");
}
