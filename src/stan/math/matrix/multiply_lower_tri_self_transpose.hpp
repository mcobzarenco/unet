#ifndef STAN__MATH__MATRIX__MULTIPLY_LOWER_TRI_SELF_HPP
#define STAN__MATH__MATRIX__MULTIPLY_LOWER_TRI_SELF_HPP

#include <stan/math/matrix/typedefs.hpp>

namespace stan {
  namespace math {

    /**
     * Returns the result of multiplying the lower triangular
     * portion of the input matrix by its own transpose.
     * @param L Matrix to multiply.
     * @return The lower triangular values in L times their own
     * transpose.
     * @throw std::domain_error If the input matrix is not square.
     */
    inline matrix_d
    multiply_lower_tri_self_transpose(const matrix_d& L) {
      int K = L.rows();
      int J = L.cols();
      int k;
      matrix_d LLt(K,K);
      matrix_d Lt = L.transpose();

      if (K == 0)
        return matrix_d(0,0);
      if (K == 1) {
        matrix_d result(1,1);
        result(0,0) = L(0,0) * L(0,0);
        return result;
      }

      for (int m = 0; m < K; ++m) {
        k = (J < m + 1) ? J : m + 1;
        LLt(m,m) = Lt.col(m).head(k).squaredNorm();
        for (int n = (m + 1); n < K; ++n) {
          LLt(n,m) = LLt(m,n) = Lt.col(m).head(k).dot(Lt.col(n).head(k));
        }
      }
      return LLt;
    }

  }
}
#endif
