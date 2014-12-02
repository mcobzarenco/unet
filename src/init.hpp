#pragma once

#include <functional>
#include <random>


namespace unet {

inline std::function<double()> normal_weight_generator(
  double mean, double stddev, int32_t seed) {
  if (seed == 0) {
    std::random_device rd;
    seed = rd();
  }
  std::mt19937 generator{seed};
  std::normal_distribution<> normal(mean, stddev);

  return [generator, normal] () mutable { return normal(generator); };
}

}  // namespace unet
