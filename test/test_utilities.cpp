#include "utilities.hpp"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>


using namespace std;
using Eigen::Matrix;

namespace unet {
inline bool operator==(const Batch& lhs, const Batch& rhs) {
  return lhs.n_input == rhs.n_input && lhs.n_output == rhs.n_output &&
    lhs.batch_size == rhs.batch_size && lhs.input.cols() == rhs.input.cols() &&
    lhs.input.rows() == rhs.input.rows() && lhs.input == rhs.input &&
    lhs.target.cols() == rhs.target.cols() &&
    lhs.target.rows() == rhs.target.rows() && lhs.target == rhs.target;
}

inline std::ostream& operator<<(std::ostream& os, const Batch& batch) {
  os << "Batch" << "<n_input=" << batch.n_input
     << ", n_output=" << batch.n_output << ", batch_size=" << batch.batch_size
     << ", input=\n" << batch.input << "\ntarget=\n" << batch.target << "\n>";
  return os;
}
}

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

  Eigen::VectorXd reals_vec{unet::eigen_vector_from_str(reals_str, ',')};
  Eigen::VectorXd ints_vec{unet::eigen_vector_from_str(ints_str, ' ')};
  vector<double> reals{
    reals_vec.data(), reals_vec.data() + reals_vec.size()};
  vector<double> ints{
    ints_vec.data(), ints_vec.data() + ints_vec.size()};

  EXPECT_EQ(exp_reals, reals);
  EXPECT_EQ(exp_ints, ints);
}

TEST(Utilities, SplitStrForArch) {
  auto parse_uint32 = [](const std::string& x) {
    return static_cast<uint32_t>(std::stoul(x));
  };
  auto arch_from_str = [&parse_uint32] (const std::string& arch) {
    return unet::vector_from_str(arch, '-', parse_uint32);
  };

  string arch1 = "748-700-300-100-10";
  vector<uint32_t> exp1 = {748, 700, 300, 100, 10};
  EXPECT_EQ(exp1, arch_from_str(arch1));
  EXPECT_EQ(unet::arch_from_str(arch1), arch_from_str(arch1));

  string arch2 = "2-100000-1";
  vector<uint32_t> exp2 = {2, 100000, 1};
  EXPECT_EQ(exp2, arch_from_str(arch2));
  EXPECT_EQ(unet::arch_from_str(arch2), arch_from_str(arch2));

  string arch3 = "748-10-";
  vector<uint32_t> exp3{};
  EXPECT_EQ(exp3, arch_from_str(arch3));
  EXPECT_EQ(unet::arch_from_str(arch3), arch_from_str(arch3));

  string arch4 = "-748-10x";
  vector<uint32_t> exp4{};
  EXPECT_EQ(exp4, arch_from_str(arch4));
  EXPECT_EQ(unet::arch_from_str(arch4), arch_from_str(arch4));
}

TEST(Utilities, ReadBatch) {
  stringstream train_in;
  train_in
    << "1, 2, 3, 4, 5\n"
    << "2, 3, 4, 5, 6\n"
    << "3, 4, 5, 6, 7\n"
    << "4, 5, 6, 7, 8\n";
  unet::RangeSelector input_transform{0, 5};
  unet::RangeSelector target_transform{1, 4};

  unet::Batch exp_batch1{5, 3, 3};
  exp_batch1.input << 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7;
  exp_batch1.target << 2, 3, 4, 3, 4, 5, 4, 5, 6;
  auto batch = unet::read_batch(train_in, 3, input_transform, target_transform);
  EXPECT_EQ(exp_batch1, batch);

  unet::Batch exp_batch2{5, 3, 3};
  exp_batch2.input << 4, 1, 2, 5, 2, 3, 6, 3, 4, 7, 4, 5, 8, 5, 6;
  exp_batch2.target << 5, 2, 3, 6, 3, 4, 7, 4, 5;
  batch = unet::read_batch(train_in, 3, input_transform, target_transform);
  EXPECT_EQ(exp_batch2, batch);

  unet::Batch exp_batch3{5, 3, 3};
  exp_batch3.input << 3, 4, 1, 4, 5, 2, 5, 6, 3, 6, 7, 4, 7, 8, 5;
  exp_batch3.target << 4, 5, 2, 5, 6, 3, 6, 7, 4;
  batch = unet::read_batch(train_in, 3, input_transform, target_transform);
  EXPECT_EQ(exp_batch3, batch);
}

TEST(Utilities, RangeSelector) {
  unet::RangeSelector ranger1(1, 3);
  Eigen::VectorXd vec1{4}, vec2{4}, expect1{2}, expect2{2};

  vec1 << 1.2, .4, 0, -18;
  expect1 << .4, 0;
  auto encoded1 = ranger1(vec1);
  EXPECT_EQ(expect1, encoded1)
    << "expected: " << expect1.transpose() << "\n"
    << "actual: " << encoded1.transpose();

  vec2 << 1.2, 3.4, 1.7, -1;
  expect2 << 3.4, 1.7;
  auto encoded2 = ranger1(vec2);
  EXPECT_EQ(expect2, encoded2)
    << "expected: " << expect2.transpose() << "\n"
    << "actual: " << encoded2.transpose();

  unet::RangeSelector ranger2(1, 4);
  Eigen::VectorXd vec3{4}, expect3{3};
  vec3 << 1.2, -131233, 2, -1;
  expect3 << -131233, 2, -1;
  auto encoded3 = ranger2(vec3);
  EXPECT_EQ(expect3, encoded3)
    << "expected: " << expect3.transpose() << "\n"
    << "actual: " << encoded3.transpose();

  Eigen::VectorXd vec5{3};
  vec5 << 9, 1, 3;
  EXPECT_DEATH({ ranger2(vec5); }, "");
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
