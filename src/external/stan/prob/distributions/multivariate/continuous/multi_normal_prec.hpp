#ifndef STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__MULTI_NORMAL_PREC_HPP
#define STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__MULTI_NORMAL_PREC_HPP

#include <stan/error_handling/matrix/check_ldlt_factor.hpp>
#include <stan/error_handling/matrix/check_size_match.hpp>
#include <stan/error_handling/matrix/check_symmetric.hpp>
#include <stan/error_handling/scalar/check_finite.hpp>
#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/error_handling/scalar/check_positive.hpp>
#include <stan/math/matrix/columns_dot_product.hpp>
#include <stan/math/matrix/columns_dot_self.hpp>
#include <stan/math/matrix/dot_product.hpp>
#include <stan/math/matrix/dot_self.hpp>
#include <stan/math/matrix/log_determinant_ldlt.hpp>
#include <stan/math/matrix/log.hpp>
#include <stan/math/matrix/log_determinant.hpp>
#include <stan/math/matrix/mdivide_left_spd.hpp>
#include <stan/math/matrix/mdivide_left_tri_low.hpp>
#include <stan/math/matrix/multiply.hpp>
#include <stan/math/matrix/subtract.hpp>
#include <stan/math/matrix/sum.hpp>
#include <stan/math/matrix/trace_quad_form.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    template <bool propto,
              typename T_y, typename T_loc, typename T_covar>
    typename boost::math::tools::promote_args<typename scalar_type<T_y>::type, typename scalar_type<T_loc>::type, T_covar>::type
    multi_normal_prec_log(const T_y& y,
                          const T_loc& mu,
                          const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& Sigma) {
      static const std::string function("stan::prob::multi_normal_prec_log");
      typedef typename boost::math::tools::promote_args<typename scalar_type<T_y>::type, typename scalar_type<T_loc>::type, T_covar>::type lp_type;
      lp_type lp(0.0);
      
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_symmetric;
      using stan::error_handling::check_size_match;
      using stan::error_handling::check_positive;
      using stan::error_handling::check_finite;
      using stan::math::sum;
      using stan::math::trace_quad_form;
      using stan::math::log_determinant_ldlt;
      using stan::math::LDLT_factor;
      using stan::error_handling::check_ldlt_factor;
      
      check_size_match(function, 
                       "Rows of precision parameter", Sigma.rows(), 
                       "columns of precision parameter", Sigma.cols());
      check_positive(function, "Precision matrix rows", Sigma.rows());
      check_symmetric(function, "Precision matrix", Sigma);
      
      LDLT_factor<T_covar,Eigen::Dynamic,Eigen::Dynamic> ldlt_Sigma(Sigma);
      check_ldlt_factor(function, "LDLT_Factor of precision parameter", ldlt_Sigma);

      using Eigen::Matrix;
      using Eigen::Dynamic;
      using std::vector;
      VectorViewMvt<const T_y> y_vec(y);
      VectorViewMvt<const T_loc> mu_vec(mu);
      //size of std::vector of Eigen vectors
      size_t size_vec = max_size_mvt(y, mu);
      
      
      //Check if every vector of the array has the same size
      int size_y = y_vec[0].size();
      int size_mu = mu_vec[0].size();
      if (size_vec > 1) {
        int size_y_old = size_y;
        int size_y_new;
        for (size_t i = 1, size_ = length_mvt(y); i < size_; i++) {
          int size_y_new = y_vec[i].size();
          check_size_match(function, 
                           "Size of one of the vectors of the random variable", size_y_new, 
                           "Size of another vector of the random variable", size_y_old);
          size_y_old = size_y_new;
        }
        int size_mu_old = size_mu;
        int size_mu_new;
        for (size_t i = 1, size_ = length_mvt(mu); i < size_; i++) {
          int size_mu_new = mu_vec[i].size();
          check_size_match(function, 
                           "Size of one of the vectors of the location variable", size_mu_new,
                           "Size of another vector of the location variable", size_mu_old);
          size_mu_old = size_mu_new;
        }
        (void) size_y_old;
        (void) size_y_new;
        (void) size_mu_old;
        (void) size_mu_new;
      }

      check_size_match(function, 
                       "Size of random variable", size_y,
                       "size of location parameter", size_mu);
      check_size_match(function, 
                       "Size of random variable", size_y,
                       "rows of covariance parameter", Sigma.rows());
      check_size_match(function, 
                       "Size of random variable", size_y,
                       "columns of covariance parameter", Sigma.cols());
  
      for (size_t i = 0; i < size_vec; i++) {      
        check_finite(function, "Location parameter", mu_vec[i]);
        check_not_nan(function, "Random variable", y_vec[i]);
      } 
      
      if (size_y == 0) //y_vec[0].size() == 0
        return lp;
      
      if (include_summand<propto,T_covar>::value)
        lp += 0.5 * log_determinant_ldlt(ldlt_Sigma) * size_vec;

      if (include_summand<propto>::value) 
        lp += NEG_LOG_SQRT_TWO_PI * size_y * size_vec;

      if (include_summand<propto,T_y,T_loc,T_covar>::value) {
        lp_type sum_lp_vec(0.0);
        for (size_t i = 0; i < size_vec; i++) {
          Matrix<typename 
              boost::math::tools::promote_args<typename scalar_type<T_y>::type, typename scalar_type<T_loc>::type>::type,
              Dynamic, 1> y_minus_mu(size_y);
          for (int j = 0; j < size_y; j++)
            y_minus_mu(j) = y_vec[i](j)-mu_vec[i](j);
          sum_lp_vec += trace_quad_form(Sigma,y_minus_mu);
        }
        lp -= 0.5*sum_lp_vec;
      }
      return lp;
    }
    
    template <typename T_y, typename T_loc, typename T_covar>
    inline
    typename boost::math::tools::promote_args<typename scalar_type<T_y>::type, typename scalar_type<T_loc>::type, T_covar>::type
    multi_normal_prec_log(const T_y& y,
                          const T_loc& mu,
                          const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& Sigma) {
      return multi_normal_prec_log<false>(y,mu,Sigma);
    }

  }
}
#endif
  
