#pragma once

#include "murmur3.hpp"

#include <Eigen/Dense>
#include <glog/logging.h>

#include <string>
#include <vector>
#include <unordered_map>


namespace unet {
namespace internal {

inline uint32_t mask_hash_value(uint32_t used_bits, uint32_t hash) {
  uint32_t mask{0};
  while(used_bits-- > 0) { mask = (mask << 1) | 1; }
  return mask & hash;
}

inline uint32_t hash_feature(
  const std::string& feature, uint32_t used_bits = 12, uint32_t seed = 1337) {
  uint32_t hash[4];
  MurmurHash3_x86_32(feature.data(), feature.size(), seed, hash);
  return mask_hash_value(used_bits, hash[0]);
}

}  // namespace internal

struct FeatureVectorizer {
  struct IdentityHash;
  using SparseWeights =
    std::unordered_map<uint32_t, Eigen::VectorXd, IdentityHash>;

  FeatureVectorizer(uint32_t subspace_size, uint32_t used_bits = 12,
                    uint32_t seed = 1337)
    : subspace_size_{subspace_size}, used_bits_{used_bits}, seed_{seed}  {}

  const SparseWeights& weights() const { return weights_; }

  uint32_t hash_feature(const std::string& feature) {
    return internal::hash_feature(feature, used_bits_, seed_);
  }

  Eigen::VectorXd operator()(const std::vector<std::string>& datapoint) {
    Eigen::VectorXd vectorized{
      static_cast<long>(datapoint.size() * subspace_size_)};
    for(auto i = 0; i < datapoint.size(); ++i) {
      auto& repr = weights_[hash_feature(datapoint[i])];
      if (repr.size() == 0) {
        repr = Eigen::VectorXd::Random(subspace_size_);
      }
      vectorized.segment(i * subspace_size_, subspace_size_) = repr;
    }
    return vectorized;
  }

  void set_weights(const std::vector<std::string>& datapoint,
                   const Eigen::VectorXd& weights) {
    CHECK_EQ(datapoint.size() * subspace_size_, weights.size());
    for(auto i = 0; i < datapoint.size(); ++i) {
      auto& repr = weights_[hash_feature(datapoint[i])];
      repr = weights.segment(i * subspace_size_, subspace_size_);
    }
  }

  struct IdentityHash {
    size_t operator()(const uint32_t& x) const {
      return static_cast<size_t>(x);
    }
  };

private:
  size_t subspace_size_;
  uint32_t used_bits_;
  uint32_t seed_;
  SparseWeights weights_;
};

}  // namespace unet
