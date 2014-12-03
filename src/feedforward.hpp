#pragma once

#include "activation.hpp"
#include "init.hpp"
#include "typedefs.hpp"
#include "utilities.hpp"

#include <cereal/cereal.hpp>
#include <Eigen/Dense>
#include <Eigen/src/Core/NumTraits.h>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <vector>


namespace unet {

struct FeedForward {
private:
  using Layers = std::vector<uint32_t>;
  template<typename Scalar> struct ImmutableParams;
  template<typename Scalar> struct MutableParams;

public:
  FeedForward() = default;

  FeedForward(const std::initializer_list<uint32_t>& layers, const bool softmax,
              const int32_t seed=0)
    : FeedForward{layers, softmax, normal_weight_generator(0, .1, seed)} {}

  inline FeedForward(const std::initializer_list<uint32_t>& layers, const bool softmax,
                     std::function<double()> generate_weight);

  uint32_t num_params() const { return FeedForward::num_params(layers_); };

  Eigen::VectorXd& weights() { return weights_; }
  const Eigen::VectorXd& weights() const { return weights_; }

  inline MutableParams<double> weights_as_params();
  inline ImmutableParams<double> weights_as_params() const;

  template<typename T>
  inline DynamicMatrix<T> operator()(const DynamicMatrix<T>& X) const;
  template<typename T>
  inline DynamicMatrix<T> operator()(const DynamicVector<T>& weights,
                                     const DynamicMatrix<T>& X) const;

  inline void l2_error(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,
                       double& error, Eigen::VectorXd& grad);

  template<class Archive>
  inline void serialize(Archive& archive);

  static inline uint32_t num_params(const Layers& layers);

  template <typename Scalar>
  static inline DynamicMatrix<Scalar> function(
    const Layers& layers, const bool softmax,
    const DynamicVector<Scalar>& weights, const DynamicMatrix<Scalar>& X);

  static inline void l2_error(
    const Layers& layers, const bool softmax,
    const Eigen::VectorXd& weights, const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y, double& error, Eigen::VectorXd& gradient);

private:
  template<typename Scalar>
  struct ImmutableParams {
    std::vector<Eigen::Map<const DynamicMatrix<Scalar>>> W;
    std::vector<Eigen::Map<const DynamicVector<Scalar>>> b;
  };

  template<typename Scalar>
  struct MutableParams {
    std::vector<Eigen::Map<DynamicMatrix<Scalar>>> W;
    std::vector<Eigen::Map<DynamicVector<Scalar>>> b;
  };

  template<typename ScalarPtr, typename Params>
  static Params map_weights_as_params(
    const Layers& layers, Params& params, ScalarPtr head) {
    for (uint32_t layer = 0; layer < layers.size() - 1; ++layer) {
      params.W.emplace_back(head, layers[layer + 1], layers[layer]);
      head += layers[layer + 1] * layers[layer];

      params.b.emplace_back(head, layers[layer + 1]);
      head += layers[layer + 1];
    };
    return params;
  }

  template<typename Scalar>
  static ImmutableParams<Scalar> weights_as_params(
    const Layers& layers, const DynamicVector<Scalar>& weights) {
    CHECK_EQ(weights.size(), FeedForward::num_params(layers));
    ImmutableParams<Scalar> params;
    map_weights_as_params(layers, params, weights.data());
    return params;
  }

  template<typename Scalar>
  static MutableParams<Scalar> weights_as_params(
    const Layers& layers, DynamicVector<Scalar>& weights) {
    CHECK_EQ(weights.size(), FeedForward::num_params(layers));
    MutableParams<Scalar> params;
    map_weights_as_params(layers, params, weights.data());
    return params;
  }

  // Feedforward state:

  bool softmax_;
  std::vector<uint32_t> layers_;
  Eigen::VectorXd weights_;
};

FeedForward::FeedForward(
  const std::initializer_list<uint32_t>& layers, const bool softmax,
  std::function<double()> generate_weight) : softmax_{softmax} {
  CHECK_GT(layers.size(), 2)
    << layers.size() << " layers specified, at least 3 are required.";
  layers_.resize(layers.size());
  uint32_t index{0};
  for (const auto& n:layers) { layers_[index++] = n; }

  weights_.resize(num_params());
  auto params = FeedForward::weights_as_params(layers, weights_);
  for (uint32_t layer = 0; layer < params.W.size() ; ++layer) {
    auto& W = params.W[layer];
    std::transform(W.data(), W.data() + W.size(), W.data(),
                   [&] (const double&) {
                     // if (layer == 0 || layer >= params.W.size() - 2 ||
                     //     generate_weight() < -0.15)
                     return generate_weight();
                     // else
                     //   return 0.0;
                   } );
    params.b[layer] = Eigen::VectorXd::Zero(params.b[layer].size());
  }
}

FeedForward::MutableParams<double> FeedForward::weights_as_params() {
  return FeedForward::weights_as_params(layers_, weights_);
}

FeedForward::ImmutableParams<double> FeedForward::weights_as_params() const {
  return FeedForward::weights_as_params(layers_, weights_);
}

template<typename Scalar>
DynamicMatrix<Scalar> FeedForward::operator()(
  const DynamicMatrix<Scalar>& X) const {
  return FeedForward::function(layers_, softmax_, weights_, X);
}

template<typename Scalar>
DynamicMatrix<Scalar> FeedForward::operator()(
  const DynamicVector<Scalar>& weights, const DynamicMatrix<Scalar>& X) const {
  CHECK_EQ(num_params(), weights.size());
  return FeedForward::function(layers_, softmax_, weights, X);
}

void FeedForward::l2_error(
  const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,
  double& error, Eigen::VectorXd& grad) {
  return FeedForward::l2_error(layers_, softmax_, weights_, X, Y, error, grad);
}

template<class Archive>
void FeedForward::serialize(Archive& archive) {
  archive(cereal::make_nvp("layers", layers_),
          cereal::make_nvp("softmax", softmax_),
          cereal::make_nvp("weights", weights_));
}

/**  Static methods  **/

uint32_t FeedForward::num_params(const Layers& layers) {
  uint32_t param_count{0};
  for (uint32_t index = 0; index < layers.size() - 1; ++index) {
    param_count += layers[index + 1] * (layers[index] + 1);
  }
  return param_count;
}

template <typename Scalar>
DynamicMatrix<Scalar> FeedForward::function(
  const Layers& layers, const bool softmax,
  const DynamicVector<Scalar>& weights, const DynamicMatrix<Scalar>& X) {
  using Matrix = DynamicMatrix<Scalar>;
  using Vector = DynamicVector<Scalar>;
  CHECK_EQ(layers[0], X.rows());

  const auto view = FeedForward::weights_as_params(layers, weights);
  Matrix out{X};
  uint32_t layer{0};
  for (layer = 0; layer < layers.size() - 2; ++layer) {
    out = (view.W[layer] * out).colwise() + view.b[layer];
    sigmoid_in_place(out);
  }
  out = (view.W[layer] * out).colwise() + view.b[layer];  // Linear layer
  if (softmax) { softmax_in_place(out); }

  return out;
}

void FeedForward::l2_error(
  const Layers& layers, const bool softmax,
  const Eigen::VectorXd& weights, const Eigen::MatrixXd& X,
  const Eigen::MatrixXd& Y, double& error, Eigen::VectorXd& gradient) {
  CHECK_EQ(layers[0], X.rows());
  const size_t n_layers{layers.size()};

  if (gradient.size() != weights.size()) {
    gradient.resize(weights.size());
  }
  const auto params = FeedForward::weights_as_params(layers, weights);
  auto grad = FeedForward::weights_as_params(layers, gradient);

  std::vector<Eigen::MatrixXd> outs;
  outs.resize(n_layers);
  outs[0] = X;
  int32_t layer{0};
  for (layer = 1; layer < n_layers - 1; ++layer) {
    outs[layer] = (params.W[layer - 1] * outs[layer - 1]).colwise() +
      params.b[layer - 1];
    sigmoid_in_place(outs[layer]);
  }
  outs[n_layers - 1] = (params.W[n_layers - 2] * outs[n_layers - 2]).colwise() +
    params.b[n_layers - 2];
  Eigen::MatrixXd& net_out{outs[n_layers - 1]};
  if (softmax) { softmax_in_place(net_out); }

  Eigen::MatrixXd S{2.0 * (net_out - Y) / net_out.size()};
  error = net_out.size() * (S.transpose() * S).trace() / 4.0;

  grad.W[n_layers - 2] = S * outs[n_layers - 2].transpose();
  grad.b[n_layers - 2] = S.rowwise().sum();
  for (layer = n_layers - 3; layer >= 0; --layer) {
    Eigen::MatrixXd D{
      (1 - outs[layer + 1].array() * outs[layer + 1].array())};
    S = (params.W[layer + 1].transpose() * S).array() * D.array();

    grad.W[layer] = S * outs[layer].transpose();
    grad.b[layer] = S.rowwise().sum();
  }

  // if (softmax) { softmax_in_place(out); }
}

}  // unet namespace
