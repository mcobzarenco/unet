#pragma once

#include "activation.hpp"
#include "init.hpp"
#include "typedefs.hpp"
#include "objectives.hpp"
#include "utilities.hpp"

#include <cereal/cereal.hpp>
#include <Eigen/Dense>
#include <Eigen/src/Core/NumTraits.h>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>


namespace unet {

template<typename NonLinearity>
struct FeedForward {
private:
  using Layers = std::vector<uint32_t>;
  template<typename Scalar> struct Params;
  using L2ErrorObjective = L2Error<FeedForward<NonLinearity>>;
  using CrossEntropyObjective = CrossEntropy<FeedForward<NonLinearity>>;

public:
  FeedForward() = default;

  FeedForward(const std::vector<uint32_t>& layers, const bool softmax,
              const int32_t seed=0)
    : FeedForward{layers, softmax, normal_weight_generator(0, .1, seed)} {}

  inline FeedForward(const std::vector<uint32_t>& layers, const bool softmax,
                     std::function<double()> generate_weight);

  const Layers& layers() const { return layers_; };
  size_t n_layers() const { return layers_.size(); };
  uint32_t n_input() const { return layers_[0]; };
  uint32_t n_output() const { return layers_[layers_.size() - 1]; };

  uint32_t num_params() const { return FeedForward::num_params(layers_); };

  Eigen::VectorXd& weights() { return weights_; }
  const Eigen::VectorXd& weights() const { return weights_; }

  inline Params<Eigen::VectorXd> weights_as_params();
  inline Params<const Eigen::VectorXd> weights_as_params() const;

  template<typename T>
  inline DynamicMatrix<T> operator()(const DynamicMatrix<T>& X) const;
  template<typename T>
  inline DynamicMatrix<T> operator()(const DynamicVector<T>& weights,
                                     const DynamicMatrix<T>& X) const;

  L2ErrorObjective l2_error(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) {
    return L2ErrorObjective{*this, X, Y};
  }
  CrossEntropyObjective cross_entropy(const Eigen::MatrixXd& X,
                                      const Eigen::MatrixXd& Y) {
    return CrossEntropyObjective{*this, X, Y};
  }

  template<class Archive>
  inline void serialize(Archive& archive);

  template<typename Objective>
  void gradient(const Eigen::VectorXd& weights, const Eigen::MatrixXd& X,
                const Eigen::MatrixXd& Y, const Objective& objective,
                double& error, Eigen::VectorXd& gradient) const {
    FeedForward::gradient(layers_, softmax_, weights,
                          X, Y, objective, error, gradient);
  }

  static inline uint32_t num_params(const Layers& layers);

  template <typename Scalar>
  static inline DynamicMatrix<Scalar> function(
    const Layers& layers, const bool softmax,
    const DynamicVector<Scalar>& weights, const DynamicMatrix<Scalar>& X);

private:

  // Weights as parameters utilities:
  template<typename Scalar>
  struct Params;

  template<typename Scalar>
  struct Params<DynamicVector<Scalar>> {
    std::vector<Eigen::Map<DynamicMatrix<Scalar>>> W;
    std::vector<Eigen::Map<DynamicVector<Scalar>>> b;
  };

  template<typename Scalar>
  struct Params<const DynamicVector<Scalar>> {
    std::vector<Eigen::Map<const DynamicMatrix<Scalar>>> W;
    std::vector<Eigen::Map<const DynamicVector<Scalar>>> b;
  };

  template<typename T>
  static inline Params<T> weights_as_params(
    const Layers& layers, T& weights) {
    CHECK_EQ(weights.size(), FeedForward::num_params(layers));
    Params<T> params;
    auto head = weights.data();
    for (uint32_t layer = 0; layer < layers.size() - 1; ++layer) {
      params.W.emplace_back(head, layers[layer + 1], layers[layer]);
      head += layers[layer + 1] * layers[layer];

      params.b.emplace_back(head, layers[layer + 1]);
      head += layers[layer + 1];
    };
    return params;
  }

  // Backpropagation and objective functions:

  template<typename OutputGradient>
  static inline void gradient(
    const Layers& layers, const bool softmax,
    const Eigen::VectorXd& weights, const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y, const OutputGradient& output_gradient,
    double& error, Eigen::VectorXd& gradient);


  // Feedforward network state:

  bool softmax_;
  Layers layers_;
  Eigen::VectorXd weights_;
};

template<typename NonLinearity>
FeedForward<NonLinearity>::FeedForward(
  const std::vector<uint32_t>& layers, const bool softmax,
  std::function<double()> generate_weight) : softmax_{softmax}, layers_{layers} {
  CHECK_GT(layers.size(), 2)
    << layers.size() << " layers specified, at least 3 are required.";
  weights_.resize(num_params());
  auto params = FeedForward::weights_as_params(layers, weights_);
  for (uint32_t layer = 0; layer < params.W.size() ; ++layer) {
    auto& W = params.W[layer];
    std::transform(W.data(), W.data() + W.size(), W.data(),
                   [&] (const double&) {
                     // if (generate_weight() < -0.1)
                       return generate_weight();
                     // else
                     //   return 0.0;
                   } );
    params.b[layer] = Eigen::VectorXd::Zero(params.b[layer].size());
  }
}

template<typename NonLinearity>
FeedForward<NonLinearity>::Params<Eigen::VectorXd>
FeedForward<NonLinearity>::weights_as_params() {
  return FeedForward<NonLinearity>::weights_as_params(layers_, weights_);
}

template<typename NonLinearity>
FeedForward<NonLinearity>::Params<const Eigen::VectorXd>
FeedForward<NonLinearity>::weights_as_params() const {
  return FeedForward<NonLinearity>::weights_as_params(layers_, weights_);
}

template<typename NonLinearity>
template<typename Scalar>
DynamicMatrix<Scalar> FeedForward<NonLinearity>::operator()(
  const DynamicMatrix<Scalar>& X) const {
  return FeedForward<NonLinearity>::function(layers_, softmax_, weights_, X);
}

template<typename NonLinearity>
template<typename Scalar>
DynamicMatrix<Scalar> FeedForward<NonLinearity>::operator()(
  const DynamicVector<Scalar>& weights, const DynamicMatrix<Scalar>& X) const {
  CHECK_EQ(num_params(), weights.size());
  return FeedForward::function(layers_, softmax_, weights, X);
}

template<typename NonLinearity>
template<class Archive>
void FeedForward<NonLinearity>::serialize(Archive& archive) {
  archive(cereal::make_nvp("layers", layers_),
          cereal::make_nvp("softmax", softmax_),
          cereal::make_nvp("weights", weights_));
}

/**  Static methods  **/

template<typename NonLinearity>
uint32_t FeedForward<NonLinearity>::num_params(const Layers& layers) {
  uint32_t param_count{0};
  for (uint32_t index = 0; index < layers.size() - 1; ++index) {
    param_count += layers[index + 1] * (layers[index] + 1);
  }
  return param_count;
}

template<typename NonLinearity>
template <typename Scalar>
DynamicMatrix<Scalar> FeedForward<NonLinearity>::function(
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
    NonLinearity::activation_in_place(out);
  }
  out = (view.W[layer] * out).colwise() + view.b[layer];  // Linear layer
  if (softmax) { softmax_in_place(out); }

  return out;
}

template<typename NonLinearity>
template<typename OutputGradient>
void FeedForward<NonLinearity>::gradient(
  const Layers& layers, const bool softmax,
  const Eigen::VectorXd& weights, const Eigen::MatrixXd& X,
  const Eigen::MatrixXd& Y, const OutputGradient& output_gradient,
  double& error, Eigen::VectorXd& gradient) {
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
    NonLinearity::activation_in_place(outs[layer]);
  }
  outs[n_layers - 1] = (params.W[n_layers - 2] * outs[n_layers - 2]).colwise() +
    params.b[n_layers - 2];
  if (softmax) { softmax_in_place(outs[n_layers - 1]); }

  Eigen::MatrixXd D;  // holds derivatives
  Eigen::MatrixXd S;  // back propagates gradient
  output_gradient(outs[n_layers - 1], Y, error, S);

  grad.W[n_layers - 2] = S * outs[n_layers - 2].transpose();
  grad.b[n_layers - 2] = S.rowwise().sum();
  for (layer = n_layers - 3; layer >= 0; --layer) {
    D = outs[layer + 1];
    NonLinearity::derivative_value_in_place(D);
    S = (params.W[layer + 1].transpose() * S).array() * D.array();

    grad.W[layer] = S * outs[layer].transpose();
    grad.b[layer] = S.rowwise().sum();
  }
}

}  // unet namespace
