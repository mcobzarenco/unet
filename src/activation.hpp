#pragma once

#include "typedefs.hpp"
#include "stan/agrad/rev/matrix.hpp"
#include "stan/agrad/rev/functions/tanh.hpp"
#include "stan/agrad/rev/functions/exp.hpp"

#include <algorithm>


namespace unet {
namespace internal {

using std::tanh;
using stan::agrad::tanh;

template<typename NonLinearity>
struct ApplyInPlace : public NonLinearity {
  static double derivative(const double x) {
    return NonLinearity::derivative_value(NonLinearity::activation(x));
  }

  template<typename Scalar>
  static void activation_in_place(DynamicMatrix<Scalar>& x) {
    auto nlin = [] (const Scalar& x) {return NonLinearity::activation(x);};
    std::transform(x.data(), x.data() + x.size(), x.data(), nlin);
  }

  template<typename Scalar>
  static void derivative_in_place(DynamicMatrix<Scalar>& x) {
    auto nlin = [] (const Scalar& x) {return NonLinearity::derivative(x);};
    std::transform(x.data(), x.data() + x.size(), x.data(), nlin);
  }

  template<typename Scalar>
  static void derivative_value_in_place(DynamicMatrix<Scalar>& x) {
    auto nlin = [] (const Scalar& x) {return NonLinearity::derivative_value(x);};
    std::transform(x.data(), x.data() + x.size(), x.data(), nlin);
  }
};

struct Tanh {

  static const char* name() {
    static const char* NAME{"tanh"};
    return NAME;
  }

  template<typename Scalar>
  static Scalar activation(const Scalar& x) { return tanh(x); }

  static double derivative_value(double fx) {
    return 1.0 - fx * fx;
  }
};

struct ReLU {
  static const char* name() {
    static const char* NAME{"relu"};
    return NAME;
  }

  template<typename Scalar>
  static Scalar activation(const Scalar& x) {
    return std::max(Scalar{0.0}, x);
  }

  static double derivative_value(double fx) {
    return static_cast<double>(fx > 0.0);
  }
};

}  // namespace internal

using Tanh = internal::ApplyInPlace<internal::Tanh>;
using ReLU = internal::ApplyInPlace<internal::ReLU>;

template<typename Scalar>
void softmax_in_place(DynamicMatrix<Scalar>& X) {
  std::transform(X.data(), X.data() + X.size(), X.data(),
                 [] (const Scalar& x) {return exp(x);});
  const DynamicVector<Scalar> exp_sum{X.colwise().sum()};
  for (int32_t i{0}; i < X.cols(); ++i) {
    X.col(i) /= exp_sum[i];
  }
}

}  // namespace unet
