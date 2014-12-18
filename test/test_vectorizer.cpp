#include "vectorizer.hpp"

#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <iostream>
#include <sstream>
#include <vector>


using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

TEST(Vectorizer, MaskHashValue) {
  uint32_t value{65535};

  EXPECT_EQ(0, unet::internal::mask_hash_value(0, value));
  EXPECT_EQ(1, unet::internal::mask_hash_value(1, value));
  EXPECT_EQ(3, unet::internal::mask_hash_value(2, value));
  EXPECT_EQ(7, unet::internal::mask_hash_value(3, value));
  EXPECT_EQ(15, unet::internal::mask_hash_value(4, value));
  EXPECT_EQ(31, unet::internal::mask_hash_value(5, value));
  EXPECT_EQ(65535, unet::internal::mask_hash_value(16, value));
  EXPECT_EQ(65535, unet::internal::mask_hash_value(17, value));
  EXPECT_EQ(65535, unet::internal::mask_hash_value(22, value));
}
