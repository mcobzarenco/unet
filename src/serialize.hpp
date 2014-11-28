#pragma once

#include "typedefs.hpp"

#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <Eigen/Dense>


namespace Eigen {

using cereal::traits::is_output_serializable;
using cereal::traits::is_input_serializable;

template<class Archive, class Scalar>
typename std::enable_if<
  is_output_serializable<cereal::BinaryData<Scalar>, Archive>::value, void>::type
save(Archive & archive, const unet::DynamicVector<Scalar>& vec) {
  cereal::size_type vec_size{static_cast<cereal::size_type>(vec.size())};
  archive(cereal::make_size_tag(vec_size));
  archive(cereal::binary_data(vec.data(), vec.size() * sizeof(Scalar)));
}

template<class Archive, class Scalar>
typename std::enable_if<
  is_input_serializable<cereal::BinaryData<Scalar>, Archive>::value, void>::type
load(Archive & archive, unet::DynamicVector<Scalar>& vec) {
  cereal::size_type vec_size;
  archive(cereal::make_size_tag(vec_size));
  vec.resize(vec_size);

  std::size_t num_bytes = static_cast<std::size_t>(vec_size) * sizeof(Scalar);
  archive(cereal::binary_data(vec.data(), num_bytes));
}

// Serialization for text formats:
template<class Archive, class Scalar>
typename std::enable_if<
  !is_output_serializable<cereal::BinaryData<Scalar>, Archive>::value, void>::type
save(Archive & archive, const unet::DynamicVector<Scalar>& vec) {
  cereal::size_type vec_size{static_cast<cereal::size_type>(vec.size())};
  archive(cereal::make_size_tag(vec_size));
  for (const Scalar* it = vec.data(), *end = vec.data() + vec.size();
       it != end; ++it) {
    archive(*it);
  }
}

template<class Archive, class Scalar>
typename std::enable_if<
  !is_input_serializable<cereal::BinaryData<Scalar>, Archive>::value, void>::type
load(Archive & archive, unet::DynamicVector<Scalar>& vec) {
  cereal::size_type vec_size;
  archive(cereal::make_size_tag(vec_size));
  vec.resize(vec_size);

  for (Scalar* it = vec.data(), *end = vec.data() + vec.size();
       it != end; ++it) {
    archive(*it);
  }
}

}  // namespace Eigen

namespace unet {

// Helpers for serializing object:

template<typename Archive, typename Net>
inline void save_net(std::ostream& out, const Net& net) {
  Archive out_archive(out);
  out_archive(net);
}

template<typename Archive, typename Net>
inline void load_net(std::istream& in, Net& net) {
  Archive in_archive(in);
  in_archive(net);
}

template<typename Net>
inline void save_to_binary(std::ostream& out, const Net& net) {
  save_net<cereal::BinaryOutputArchive, Net>(out, net);
}

template<typename Net>
inline void load_from_binary(std::istream& in, Net& net) {
  load_net<cereal::BinaryInputArchive, Net>(in, net);
}

template<typename Net>
inline void save_to_json(std::ostream& out, const Net& net) {
  save_net<cereal::JSONOutputArchive, Net>(out, net);
}

template<typename Net>
inline void load_from_json(std::istream& in, Net& net) {
  load_net<cereal::JSONInputArchive, Net>(in, net);
}

}  // namespace unet
