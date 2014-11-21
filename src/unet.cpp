#include <iostream>

#include "layer.hpp"

using namespace std;

using Eigen::Matrix;
using Eigen::Dynamic;


int main() {
  std::random_device rd;
  std::mt19937 generator{rd()};
  std::normal_distribution<> normal(0, .1);

  unet::MLP net(2, 3, 1, [&] () {return normal(generator);});

  Matrix<double, Dynamic, Dynamic> x{2, 5}, y{1, 5};
  x <<
    0, 0, 1, 1, .9,
    0, 1, 0, 1, .1;
  y << 0, 1, 1, 0, .9;

  auto l2_error = net.l2_error(x, y);
  l2_error.minimize_gd(1000);
}
