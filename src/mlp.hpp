#pragma once

#include "activation.hpp"
#include "init.hpp"
#include "typedefs.hpp"
#include "utilities.hpp"
#include "stan/agrad/rev/matrix.hpp"
#include "stan/agrad/rev/functions/tanh.hpp"
#include "stan/agrad/rev/functions/exp.hpp"

#include <cereal/cereal.hpp>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/src/Core/NumTraits.h>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <iostream>


namespace unet {

struct MLP {
private:
  template<typename T> struct WeightsView;

public:
  MLP() = default;

  MLP(uint32_t n_input, uint32_t n_hidden, uint32_t n_output,
      bool softmax, int32_t seed=0)
    : MLP(n_input, n_hidden, n_output, softmax,
          normal_weight_generator(0, .01, seed)) {}

  inline MLP(uint32_t n_input, uint32_t n_hidden, uint32_t n_output,
             bool softmax, std::function<double()> generate_weight);

  inline uint32_t num_params() const;

  uint32_t n_input() const { return n_input_; }
  uint32_t n_hidden() const { return n_hidden_; }
  uint32_t n_output() const { return n_output_; }
  uint32_t softmax() const { return softmax_; }

  Eigen::VectorXd& weights() { return weights_; }
  const Eigen::VectorXd& weights() const { return weights_; }

  // inline WeightsView formatted_weights();
  inline WeightsView<double> structured_weights() const;

  template<typename T>
  inline DynamicMatrix<T> operator()(const DynamicMatrix<T>& X) const;
  template<typename T>
  inline DynamicMatrix<T> operator()(const DynamicVector<T>& weights,
                                     const DynamicMatrix<T>& X) const;

  template<class Archive>
  inline void serialize(Archive& archive);

  static inline uint32_t num_params(
    uint32_t n_input, uint32_t n_hidden, uint32_t n_output) {
    return n_hidden * (n_input + 1) + n_output * (n_hidden + 1);
  }

  template <typename T>
  static inline DynamicMatrix<T> function(
    uint32_t n_input, uint32_t n_hidden, uint32_t n_output, bool softmax,
    const DynamicVector<T>& weights, const DynamicMatrix<T>& X);

private:
  template<typename T>
  static inline WeightsView<T> extract_weights(
    const DynamicVector<T>& weights, const uint32_t n_input,
    const uint32_t n_hidden, const uint32_t n_output);

  template<typename T>
  struct WeightsView {
    WeightsView(const T* H_start,  uint32_t H_rows, uint32_t H_cols,
                const T* Hb_start, uint32_t Hb_size,
                const T* V_start,  uint32_t V_rows, uint32_t V_cols,
                const T* Vb_start, uint32_t Vb_size) :
      H{H_start, H_rows, H_cols}, Hb{Hb_start, Hb_size},
      V{V_start, V_rows, V_cols}, Vb{Vb_start, Vb_size} {}

    Eigen::Map<const DynamicMatrix<T>> H;   // hidden weight matrix
    Eigen::Map<const DynamicVector<T>> Hb;  // hidden bias
    Eigen::Map<const DynamicMatrix<T>> V;   // output weight matrix
    Eigen::Map<const DynamicVector<T>> Vb;  // output bias
  };

  uint32_t n_input_, n_hidden_, n_output_;
  bool softmax_;
  Eigen::VectorXd weights_;
};

MLP::MLP(uint32_t n_input, uint32_t n_hidden, uint32_t n_output, bool softmax,
         std::function<double()> generate_weight)
  : n_input_{n_input}, n_hidden_{n_hidden}, n_output_{n_output}, softmax_{softmax} {
    CHECK(n_input > 0 && n_hidden > 0 && n_output > 0);
    CHECK(n_output > 1 || !softmax)
      << "There needs to be more than 1 output unit for softmax to make sense.";
    weights_.resize(num_params());
    for (int i = 0; i < weights_.size(); ++i) {
      weights_(i) = generate_weight();
    }
}

uint32_t MLP::num_params() const {
  return MLP::num_params(n_input_, n_hidden_, n_output_);
}

MLP::WeightsView<double> MLP::structured_weights() const {
  return MLP::extract_weights(weights_, n_input_, n_hidden_, n_output_);
}

template<typename T>
DynamicMatrix<T> MLP::operator()(const DynamicMatrix<T>& X) const {
  return MLP::function(n_input_, n_hidden_, n_output_, softmax_, weights_, X);
}

template<typename T>
DynamicMatrix<T> MLP::operator()(
  const DynamicVector<T>& weights, const DynamicMatrix<T>& X) const {
  CHECK_EQ(num_params(), weights.size());
  return MLP::function(n_input_, n_hidden_, n_output_, softmax_, weights, X);
}

template<class Archive>
void MLP::serialize(Archive& archive) {
  archive(cereal::make_nvp("n_input", n_input_),
          cereal::make_nvp("n_hidden", n_hidden_),
          cereal::make_nvp("n_output", n_output_),
          cereal::make_nvp("softmax", softmax_),
          cereal::make_nvp("weights", weights_));
}

template<typename T>
MLP::WeightsView<T> MLP::extract_weights(
  const DynamicVector<T>& weights, const uint32_t n_input,
  const uint32_t n_hidden, const uint32_t n_output) {
  using Matrix = DynamicMatrix<T>;
  using Vector = DynamicVector<T>;
  CHECK_EQ(weights.size(), MLP::num_params(n_input, n_hidden, n_output));

  const T* H_start{weights.data()};
  const T* Hb_start{H_start + n_hidden * n_input};
  const T* V_start{Hb_start + n_hidden};
  const T* Vb_start{V_start + n_hidden * n_output};

  return MLP::WeightsView<T>{
    H_start, n_hidden, n_input, Hb_start, n_hidden,
    V_start, n_output, n_hidden, Vb_start, n_output};
}

template <typename T>
DynamicMatrix<T> MLP::function(
  uint32_t n_input, uint32_t n_hidden, uint32_t n_output, bool softmax,
  const DynamicVector<T>& weights, const DynamicMatrix<T>& X) {
  using Matrix = DynamicMatrix<T>;
  using Vector = DynamicVector<T>;
  CHECK_EQ(n_input, X.rows());

  auto W = MLP::extract_weights(weights, n_input, n_hidden, n_output);
  const Matrix& H{W.H}, V{W.V};
  const Vector& Hb{W.Hb}, Vb{W.Vb};

  Matrix H_out{H * X};
  H_out.colwise() += Hb;
  std::transform(H_out.data(), H_out.data() + H_out.size(), H_out.data(),
                 [] (const T& x) {return sigmoid(x);});
  // std::cout << "s(H * X + Hb) = \n" << H_out << "\n";
  Matrix V_out{V * H_out};
  V_out.colwise() += Vb;
  if (softmax) {
    softmax_in_place(V_out);
  }
  return V_out;
}

}  // unet namespace
