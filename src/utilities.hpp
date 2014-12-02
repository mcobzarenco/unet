#pragma once

#include "typedefs.hpp"

#include <boost/optional.hpp>
#include <Eigen/Dense>
#include <glog/logging.h>

#include <algorithm>
#include <string>
#include <utility>
#include <iostream>


namespace unet {

template<typename T, typename U>
using MaybePair = boost::optional<std::pair<T, U>>;

inline MaybePair<int32_t, int32_t> parse_range(const std::string& range_str) {
  using Pair = MaybePair<int32_t, int32_t>;
  auto index = range_str.find_first_of(":");
  auto last = range_str.find_last_of(":");
  if (last != index)  return Pair{};
  try {
    if (index != std::string::npos) {
      return Pair{std::make_pair(
          std::stoi(range_str.substr(0, index)),
          std::stoi(range_str.substr(index + 1)))};
    }
  } catch (const std::exception&) {}
  return Pair{};
}

inline Eigen::VectorXd vector_from_str(
  const std::string& list, const char delim) {
  std::vector<double> elems;
  int32_t last_delim{-1};
  int32_t i{0};
  for (i = 0; i < list.length(); ++i) {
    if (list[i] == delim) {
      std::string elem_str{list.substr(last_delim + 1, i - last_delim - 1)};
      CHECK_GT(elem_str.length(), 0) << "Empty element";
      elems.push_back(std::stod(elem_str));
      last_delim = i;
    }
  }
  std::string elem_str{list.substr(last_delim + 1, last_delim - i - 1)};
  CHECK_GT(elem_str.length(), 0) << "Empty element";
  elems.push_back(std::stod(elem_str));

  uint32_t vec_len = elems.size();
  Eigen::Matrix<double, Eigen::Dynamic, 1> elems_vec{vec_len};
  uint32_t index{0};
  for(const auto& x:elems) {
    elems_vec[index++] = x;
  }
  return elems_vec;
}

struct Batch {
  Batch(uint32_t n_input, uint32_t n_output, uint32_t batch_size) :
    n_input{n_input}, n_output{n_output}, batch_size{batch_size},
    input{n_input, batch_size}, target{n_output, batch_size} {}

  uint32_t n_input;
  uint32_t n_output;
  uint32_t batch_size;

  Eigen::MatrixXd input;
  Eigen::MatrixXd target;
};

template<typename InputTransform, typename TargetTransform>
inline Batch read_batch(std::istream& in, uint32_t batch_size,
                        const InputTransform& input_transform,
                        const TargetTransform& target_transform) {
  CHECK_GT(batch_size, 0) << "The batch size should be a positive integer.";
  Batch batch{input_transform.out_size, target_transform.out_size, batch_size};
  Eigen::VectorXd input_vec;
  std::string line;
  uint32_t n_batched{0};
  while (n_batched != batch_size) {
    if (getline(in, line).eof()) {
      in.clear();
      in.seekg(0);
    } else {
      CHECK(in.good());
      input_vec = unet::vector_from_str(line, ',');
      batch.input.col(n_batched) = input_transform(input_vec);
      batch.target.col(n_batched++) = target_transform(input_vec);
    }
  }
  return batch;
}

struct RangeSelector {
  RangeSelector(std::pair<uint32_t, uint32_t> range)
    : RangeSelector(range.first, range.second) {}

  RangeSelector(uint32_t start, uint32_t end)
    : start{start}, end{end}, out_size{end - start} {}

  Eigen::VectorXd operator()(const Eigen::VectorXd& vec) const {
    CHECK_LE(start + out_size, vec.size());
    return vec.segment(start, out_size);
  }

  const uint32_t start;
  const uint32_t end;
  const uint32_t out_size;
};

struct OneHotEncoder {
  OneHotEncoder(uint32_t feature_index, uint32_t num_categories)
    : feature_index{feature_index}, out_size{num_categories} {
    CHECK_GT(num_categories, 0);
  }

  Eigen::VectorXd operator()(const Eigen::VectorXd& vec) const {
    CHECK_LT(feature_index, vec.size());
    uint32_t category{static_cast<uint32_t>(vec[feature_index])};
    double category_double{static_cast<double>(category)};
    CHECK_EQ(category, category_double)
      << "Column " << feature_index << " needs to be integer to encode a "
      << "category; " << vec[feature_index] << " is not an integer.";
    CHECK_LT(category, out_size)
      << "category " << category << " > num_categories = " << out_size;
    Eigen::VectorXd out{Eigen::VectorXd::Zero(out_size)};
    out[category] = 1.0;
    return out;
  }

  const uint32_t feature_index;
  const uint32_t out_size;
};



}  // namespace unet
