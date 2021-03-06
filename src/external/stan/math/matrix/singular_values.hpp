#ifndef STAN__MATH__MATRIX__SINGULAR_VALUES_HPP
#define STAN__MATH__MATRIX__SINGULAR_VALUES_HPP

#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace math {

    /**
     * Return the vector of the singular values of the specified matrix
     * in decreasing order of magnitude.
     * <p>See the documentation for <code>svd()</code> for
     * information on the signular values.
     * @param m Specified matrix.
     * @return Singular values of the matrix.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1>
    singular_values(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& m) {
      return Eigen::JacobiSVD<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> >(m)
        .singularValues();
    }

  }
}
#endif
