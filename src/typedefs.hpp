#pragma once

#include <Eigen/Dense>


namespace unet {

template<typename T>
using DynamicMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template<typename T>
using DynamicVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

}  // namespace unet
