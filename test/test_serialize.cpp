#include "serialize.hpp"

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

namespace {
struct MontyNetwork {
  MontyNetwork() = default;
  bool operator==(const MontyNetwork& rhs) const {
    return i == rhs.i && w.size() == rhs.w.size()
      && (w - rhs.w).transpose() * (w - rhs.w) < 1e-20;
  }

  MontyNetwork(int i, uint32_t w_size)
    : i{i} {
    w = VectorXd::Random(w_size);
  }

  template<typename Archive>
  void serialize(Archive& archive) {
    archive(i, w);
  }

  int i;
  VectorXd w{9};
};
}

TEST(Serialize, EigenDenseVectors) {
  VectorXd w{9}, w_bin, w_json;
  vector<int> vec;
  vec.push_back(10);
  vec.push_back(20);
  vec.push_back(31);

  w << 1, 2.00001, 3, 4, 5, 6, 7, 8, 9.123456789;

  std::stringstream bin_stream, json_stream;
  {
    cereal::BinaryOutputArchive bin_archive(bin_stream);
    cereal::JSONOutputArchive json_archive(json_stream);
    bin_archive(w);
    json_archive(w);
  }
  cereal::BinaryInputArchive in_binary(bin_stream);
  cereal::JSONInputArchive in_text(json_stream);
  in_binary(w_bin);
  in_text(w_json);
  EXPECT_EQ(w, w_bin);
  EXPECT_EQ(w, w_json);
}

TEST(Serialize, Helpers) {
  MontyNetwork net{-90, 10001};
  MontyNetwork net_bin, net_json;

  std::stringstream bin_stream;
  unet::save_to_binary(bin_stream, net);
  unet::load_from_binary(bin_stream, net_bin);
  EXPECT_EQ(net, net_bin);
  EXPECT_EQ(net.w, net_bin.w); // expect exact equality for binary
                               // format
  std::stringstream json_stream;
  unet::save_to_binary(json_stream, net);
  unet::load_from_binary(json_stream, net_json);
  EXPECT_EQ(net, net_json);
}
