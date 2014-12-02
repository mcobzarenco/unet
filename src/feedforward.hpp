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

template<uint32_t N_LAYERS>
struct FeedForward {
private:
  template<typename Scalar, bool Mutable> struct WeightsView {};

public:
  FeedForward() = default;

  FeedForward(std::initializer_list<uint32_t> layers, bool softmax,
              int32_t seed=0)
    : FeedForward{layers, softmax, normal_weight_generator(0, .1, seed)} {}

  FeedForward(std::initializer_list<uint32_t> layers, bool softmax,
              std::function<double()> generate_weight) : softmax_{softmax} {
    static_assert(N_LAYERS >= 3, "At least 3 layers are required.");
    CHECK_EQ(layers.size(), N_LAYERS)
      << layers.size() << " layers specified != " << N_LAYERS << " required.";
    uint32_t index{0};
    for (auto n:layers) { layers_[index++] = n; }
    weights_.resize(num_params());
    for (int i = 0; i < weights_.size(); ++i) {
      weights_(i) = generate_weight();
    }
  }

  uint32_t num_params() const { return FeedForward::num_params(layers_); };

  Eigen::VectorXd& weights() { return weights_; }
  const Eigen::VectorXd& weights() const { return weights_; }

  // inline WeightsView formatted_weights();
  inline WeightsView<double, false> structured_weights() const {
    return weights_view<double>(layers_, weights_);
  }

  template<typename T>
  inline DynamicMatrix<T> operator()(const DynamicMatrix<T>& X) const;
  template<typename T>
  inline DynamicMatrix<T> operator()(const DynamicVector<T>& weights,
                                     const DynamicMatrix<T>& X) const;

  inline void l2_error(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,
                       double& error, Eigen::VectorXd& grad);

  template<class Archive>
  inline void serialize(Archive& archive);

  static inline uint32_t num_params(const std::array<uint32_t, N_LAYERS>& layers);

  template <typename Scalar>
  static inline DynamicMatrix<Scalar> function(
    std::array<uint32_t, N_LAYERS> layers, bool softmax,
    const DynamicVector<Scalar>& weights, const DynamicMatrix<Scalar>& X);

  static inline void l2_error(
    const std::array<uint32_t, N_LAYERS>& layers, bool softmax,
    const Eigen::VectorXd& weights, const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y, double& error, Eigen::VectorXd& gradient);

private:
  template<typename Scalar>
  struct WeightsView<Scalar, false> {
    std::vector<Eigen::Map<const DynamicMatrix<Scalar>>> W;
    std::vector<Eigen::Map<const DynamicVector<Scalar>>> b;
  };

  template<typename Scalar>
  struct WeightsView<Scalar, true> {
    std::vector<Eigen::Map<DynamicMatrix<Scalar>>> W;
    std::vector<Eigen::Map<DynamicVector<Scalar>>> b;
  };

  template<typename Scalar>
  static WeightsView<Scalar, false> weights_view(
    const std::array<uint32_t, N_LAYERS>& layers,
    const DynamicVector<Scalar>& weights) {
    CHECK_EQ(weights.size(), FeedForward::num_params(layers));

    WeightsView<Scalar, false> view;
    const Scalar* head{weights.data()};
    for (uint32_t layer = 0; layer < N_LAYERS - 1; ++layer) {
      view.W.emplace_back(head, layers[layer + 1], layers[layer]);
      head += layers[layer + 1] * layers[layer];

      view.b.emplace_back(head, layers[layer + 1]);
      head += layers[layer + 1];
    };
    return view;
  }

  template<typename Scalar>
  static WeightsView<Scalar, true> weights_view(
    const std::array<uint32_t, N_LAYERS>& layers, DynamicVector<Scalar>& weights) {
    CHECK_EQ(weights.size(), FeedForward::num_params(layers));

    WeightsView<Scalar, true> view;
    Scalar* head{weights.data()};
    for (uint32_t layer = 0; layer < N_LAYERS - 1; ++layer) {
      view.W.emplace_back(head, layers[layer + 1], layers[layer]);
      head += layers[layer + 1] * layers[layer];

      view.b.emplace_back(head, layers[layer + 1]);
      head += layers[layer + 1];
    };
    return view;
  }

  bool softmax_;
  std::array<uint32_t, N_LAYERS> layers_;
  Eigen::VectorXd weights_;
};

template<uint32_t N_LAYERS>
template<typename Scalar>
DynamicMatrix<Scalar> FeedForward<N_LAYERS>::operator()(
  const DynamicMatrix<Scalar>& X) const {
  return FeedForward::function(layers_, softmax_, weights_, X);
}

template<uint32_t N_LAYERS>
template<typename Scalar>
DynamicMatrix<Scalar> FeedForward<N_LAYERS>::operator()(
  const DynamicVector<Scalar>& weights, const DynamicMatrix<Scalar>& X) const {
  CHECK_EQ(num_params(), weights.size());
  return FeedForward::function(layers_, softmax_, weights, X);
}

template<uint32_t N_LAYERS>
void FeedForward<N_LAYERS>::l2_error(
  const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,
  double& error, Eigen::VectorXd& grad) {
  return FeedForward<N_LAYERS>::l2_error(
    layers_, softmax_, weights_, X, Y, error, grad);
}

template<uint32_t N_LAYERS>
template<class Archive>
void FeedForward<N_LAYERS>::serialize(Archive& archive) {
  archive(cereal::make_nvp("layers", layers_),
          cereal::make_nvp("softmax", softmax_),
          cereal::make_nvp("weights", weights_));
}

/**  Static methods  **/

template<uint32_t N_LAYERS>
uint32_t FeedForward<N_LAYERS>::num_params(
  const std::array<uint32_t, N_LAYERS>& layers) {
  uint32_t param_count{0};
  for (uint32_t index = 0; index < N_LAYERS - 1; ++index) {
    param_count += layers[index + 1] * (layers[index] + 1);
  }
  return param_count;
}

template<uint32_t N_LAYERS>
template <typename Scalar>
DynamicMatrix<Scalar> FeedForward<N_LAYERS>::function(
  std::array<uint32_t, N_LAYERS> layers, bool softmax,
  const DynamicVector<Scalar>& weights, const DynamicMatrix<Scalar>& X) {
  using Matrix = DynamicMatrix<Scalar>;
  using Vector = DynamicVector<Scalar>;
  CHECK_EQ(layers[0], X.rows());

  const auto view = FeedForward<N_LAYERS>::weights_view(layers, weights);
  Matrix out{X};
  uint32_t layer{0};
  for (layer = 0; layer < N_LAYERS - 2; ++layer) {
    // LOG(INFO) << "W: " << view.W[layer].cols() << " x " << view.W[layer].rows()
    //           << " / out: " << out.cols() << " x " << out.rows();
    out = (view.W[layer] * out).colwise() + view.b[layer];
    std::transform(out.data(), out.data() + out.size(), out.data(),
                   [] (const Scalar& x) {return sigmoid(x);});
  }
  out = (view.W[layer] * out).colwise() + view.b[layer];  // Linear layer
  if (softmax) { softmax_in_place(out); }

  return out;
}

template<uint32_t N_LAYERS>
void FeedForward<N_LAYERS>::l2_error(
  const std::array<uint32_t, N_LAYERS>& layers, bool softmax,
  const Eigen::VectorXd& weights, const Eigen::MatrixXd& X,
  const Eigen::MatrixXd& Y, double& error, Eigen::VectorXd& gradient) {
  CHECK_EQ(layers[0], X.rows());

  if (gradient.size() != weights.size()) { gradient.resize(weights.size()); }

  auto params = FeedForward<N_LAYERS>::weights_view(layers, weights);
  auto grad = FeedForward<N_LAYERS>::weights_view(layers, gradient);
  CHECK_EQ(params.W.size(), N_LAYERS - 1);
  CHECK_EQ(params.b.size(), N_LAYERS - 1);
  CHECK_EQ(grad.W.size(), N_LAYERS - 1);
  CHECK_EQ(grad.b.size(), N_LAYERS - 1);

  std::vector<Eigen::MatrixXd> outs;
  outs.resize(N_LAYERS);

  Eigen::MatrixXd out{X};
  int32_t layer{0};
  outs[0] = X;
  // std::cerr << "mean_out[" << 0 << "]=\n" << outs[layer] << "\n";
  for (layer = 1; layer < N_LAYERS - 1; ++layer) {
    // LOG(INFO) << "W" << params.W[layer - 1].cols() << " x " << params.W[layer - 1].rows()
    //           << " / out: " << out.cols() << " x " << out.rows();
    // LOG(INFO) << "grad - W: " << grad.W[layer - 1].cols() << " x " << grad.W[layer - 1].rows()
    //           << " / out: " << out.cols() << " x " << out.rows();

    out = (params.W[layer - 1] * out).colwise() + params.b[layer - 1];
    std::transform(out.data(), out.data() + out.size(), out.data(),
                   [] (const double x) {return sigmoid(x);});
    outs[layer] = out;
  }
  out = (params.W[layer - 1] * out).colwise() + params.b[layer - 1];
  outs[layer] = out;

  Eigen::MatrixXd S{2.0 * (out - Y) / out.size()};
  error = out.size() * (S.transpose() * S).trace() / 4.0;

  grad.W[N_LAYERS - 2] = S * outs[N_LAYERS - 2].transpose();
  grad.b[N_LAYERS - 2] = S.rowwise().sum();
  for (layer = N_LAYERS - 3; layer >= 0; --layer) {
    Eigen::MatrixXd D{
      (1 - outs[layer + 1].array() * outs[layer + 1].array())};
    // std::cerr << "D=\n" << D << "\n";
    // std::cerr << "W=\n" << params.W[layer + 1] << "\n";
    // std::cerr << "S=\n" << S << "\n";
    // std::cerr << "mean_out=\n" << outs[layer + 1] << "\n";

    S = (params.W[layer + 1].transpose() * S).array() * D.array();
    // std::cerr << "S_new=\n" << S << "\n";
    // std::cerr << "dW=\n" << S * outs[layer].rowwise().sum().transpose() << "\n";
    // std::cerr << "dW.size()=" << grad.W[layer].rows() << "\n";
    // CHECK_EQ(S.cols(), mean_outs[layer].size());
    grad.W[layer] = S * outs[layer].transpose();
    grad.b[layer] = S.rowwise().sum();
  }

  // if (softmax) { softmax_in_place(out); }
}

// template<uint32_t N_LAYERS>
// void FeedForward<N_LAYERS>::l2_error(
//   const std::array<uint32_t, N_LAYERS>& layers, bool softmax,
//   const Eigen::VectorXd& weights, const Eigen::MatrixXd& X,
//   const Eigen::MatrixXd& Y, double& error, Eigen::VectorXd& gradient) {
//   CHECK_EQ(layers[0], X.rows());

//   if (gradient.size() != weights.size()) {
//     gradient.resize(weights.size());
//   }

//   auto params = FeedForward<N_LAYERS>::weights_view(layers, weights);
//   auto grad = FeedForward<N_LAYERS>::weights_view(layers, gradient);
//   CHECK_EQ(params.W.size(), N_LAYERS - 1);
//   CHECK_EQ(params.b.size(), N_LAYERS - 1);
//   CHECK_EQ(grad.W.size(), N_LAYERS - 1);
//   CHECK_EQ(grad.b.size(), N_LAYERS - 1);

//   std::vector<Eigen::VectorXd> mean_outs;
//   mean_outs.resize(N_LAYERS);

//   Eigen::MatrixXd out{X};
//   int32_t layer{0};
//   mean_outs[0] = X.rowwise().mean();
//   std::cerr << "mean_out[" << 0 << "]=\n" << mean_outs[layer] << "\n";
//   for (layer = 1; layer < N_LAYERS - 1; ++layer) {
//     LOG(INFO) << "W" << params.W[layer - 1].cols() << " x " << params.W[layer - 1].rows()
//               << " / out: " << out.cols() << " x " << out.rows();
//     LOG(INFO) << "grad - W: " << grad.W[layer - 1].cols() << " x " << grad.W[layer - 1].rows()
//               << " / out: " << out.cols() << " x " << out.rows();

//     out = (params.W[layer - 1] * out).colwise() + params.b[layer - 1];
//     std::transform(out.data(), out.data() + out.size(), out.data(),
//                    [] (const double x) {return sigmoid(x);});
//     mean_outs[layer] = out.rowwise().mean();

//     std::cerr << "mean_out[" << layer << "]=\n" << mean_outs[layer] << "\n";
//   }
//   out = (params.W[layer - 1] * out).colwise() + params.b[layer - 1];
//   mean_outs[layer] = out.rowwise().mean();
//   std::cerr << "mean_out[" << layer << "]=\n" << mean_outs[layer] << "\n";

//   Eigen::MatrixXd deviation{out - Y};
//   Eigen::VectorXd mean_deviation{deviation.rowwise().mean()};
//   error = (deviation.transpose() * deviation).trace();
//   std::cerr << "error=" << error << "\n";

//   Eigen::VectorXd S{mean_deviation};
//   std::cerr << "S=\n" << Eigen::MatrixXd{S.asDiagonal()} << "\n";

//   grad.W[N_LAYERS - 2] = S * mean_outs[N_LAYERS - 2].transpose();
//   for (layer = N_LAYERS - 3; layer >= 0; --layer) {
//     Eigen::VectorXd D{(1 - mean_outs[layer + 1].array() * mean_outs[layer + 1].array())};
//     std::cerr << "D=\n" << Eigen::MatrixXd{D.asDiagonal()} << "\n";
//     std::cerr << "W=\n" << params.W[layer + 1] << "\n";
//     std::cerr << "S=\n" << S << "\n";
//     std::cerr << "mean_out=\n" << mean_outs[layer] << "\n";

//     S = D.asDiagonal() * params.W[layer + 1].transpose() * S;
//     std::cerr << "dW=\n" << S * mean_outs[layer].transpose() << "\n";
//     std::cerr << "dW.size()=" << grad.W[layer].rows() << "\n";
//     // CHECK_EQ(S.cols(), mean_outs[layer].size());
//     grad.W[layer] = S * mean_outs[layer].transpose();
//   }
//   // if (softmax) { softmax_in_place(out); }
// }


}  // unet namespace
