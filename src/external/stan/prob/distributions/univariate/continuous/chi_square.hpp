#ifndef STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__CHI_SQUARE_HPP
#define STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__CHI_SQUARE_HPP

#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/agrad/partials_vari.hpp>
#include <stan/error_handling/scalar/check_consistent_sizes.hpp>
#include <stan/error_handling/scalar/check_nonnegative.hpp>
#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/error_handling/scalar/check_positive_finite.hpp>
#include <stan/math/constants.hpp>
#include <stan/math/functions/multiply_log.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/internal_math.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    /**
     * The log of a chi-squared density for y with the specified
     * degrees of freedom parameter.
     * The degrees of freedom prarameter must be greater than 0.
     * y must be greater than or equal to 0.
     * 
     \f{eqnarray*}{
     y &\sim& \chi^2_\nu \\
     \log (p (y \,|\, \nu)) &=& \log \left( \frac{2^{-\nu / 2}}{\Gamma (\nu / 2)} y^{\nu / 2 - 1} \exp^{- y / 2} \right) \\
     &=& - \frac{\nu}{2} \log(2) - \log (\Gamma (\nu / 2)) + (\frac{\nu}{2} - 1) \log(y) - \frac{y}{2} \\
     & & \mathrm{ where } \; y \ge 0
     \f}
     * @param y A scalar variable.
     * @param nu Degrees of freedom.
     * @throw std::domain_error if nu is not greater than or equal to 0
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_dof Type of degrees of freedom.
     */
    template <bool propto,
              typename T_y, typename T_dof>
    typename return_type<T_y,T_dof>::type
    chi_square_log(const T_y& y, const T_dof& nu) {
      static const std::string function("stan::prob::chi_square_log");

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(nu)))
        return 0.0;
      
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_nonnegative;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::math::value_of;
      
      double logp(0.0);
      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y);
      check_positive_finite(function, "Degrees of freedom parameter", nu);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Degrees of freedom parameter", nu);
    
      
      // set up template expressions wrapping scalars into vector views
      VectorView<const T_y> y_vec(y);
      VectorView<const T_dof> nu_vec(nu);
      size_t N = max_size(y, nu);
      
      for (size_t n = 0; n < length(y); n++) 
        if (value_of(y_vec[n]) < 0)
          return LOG_ZERO;

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_dof>::value)
        return 0.0;

      using boost::math::digamma;
      using boost::math::lgamma;
      using stan::math::multiply_log;

      DoubleVectorView<include_summand<propto,T_y,T_dof>::value,
        is_vector<T_y>::value> log_y(length(y));
      for (size_t i = 0; i < length(y); i++)
        if (include_summand<propto,T_y,T_dof>::value)
          log_y[i] = log(value_of(y_vec[i]));

      DoubleVectorView<include_summand<propto,T_y>::value,
        is_vector<T_y>::value> inv_y(length(y));
      for (size_t i = 0; i < length(y); i++)
        if (include_summand<propto,T_y>::value)
          inv_y[i] = 1.0 / value_of(y_vec[i]);

      DoubleVectorView<include_summand<propto,T_dof>::value,
        is_vector<T_dof>::value> lgamma_half_nu(length(nu));
      DoubleVectorView<!is_constant_struct<T_dof>::value,
        is_vector<T_dof>::value> digamma_half_nu_over_two(length(nu));

      for (size_t i = 0; i < length(nu); i++) {
        double half_nu = 0.5 * value_of(nu_vec[i]);
        if (include_summand<propto,T_dof>::value)
          lgamma_half_nu[i] = lgamma(half_nu);
        if (!is_constant_struct<T_dof>::value)
          digamma_half_nu_over_two[i] = digamma(half_nu) * 0.5;
      }


      agrad::OperandsAndPartials<T_y,T_dof> operands_and_partials(y, nu);

      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double half_y = 0.5 * y_dbl;
        const double nu_dbl = value_of(nu_vec[n]);
        const double half_nu = 0.5 * nu_dbl;
        if (include_summand<propto,T_dof>::value)
          logp += nu_dbl * NEG_LOG_TWO_OVER_TWO - lgamma_half_nu[n];
        if (include_summand<propto,T_y,T_dof>::value)
          logp += (half_nu-1.0) * log_y[n];
        if (include_summand<propto,T_y>::value)
          logp -= half_y;
  
        if (!is_constant_struct<T_y>::value) {
          operands_and_partials.d_x1[n] += (half_nu-1.0)*inv_y[n] - 0.5;
        }
        if (!is_constant_struct<T_dof>::value) {
          operands_and_partials.d_x2[n] 
            += NEG_LOG_TWO_OVER_TWO - digamma_half_nu_over_two[n] + log_y[n]*0.5; 
        }
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_y, typename T_dof>
    inline
    typename return_type<T_y,T_dof>::type
    chi_square_log(const T_y& y, const T_dof& nu) {
      return chi_square_log<false>(y,nu);
    }

    /** 
     * Calculates the chi square cumulative distribution function for the given
     * variate and degrees of freedom.
     * 
     * y A scalar variate.
     * nu Degrees of freedom.
     * 
     * @return The cdf of the chi square distribution
     */
    template <typename T_y, typename T_dof>
    typename return_type<T_y,T_dof>::type
    chi_square_cdf(const T_y& y, const T_dof& nu) {
      static const std::string function("stan::prob::chi_square_cdf");

      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_nonnegative;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::math::value_of;

      double cdf(1.0);

     // Size checks
      if (!(stan::length(y) && stan::length(nu))) 
        return cdf;

      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y);
      check_positive_finite(function, "Degrees of freedom parameter", nu);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Degrees of freedom parameter", nu);

      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_dof> nu_vec(nu);
      size_t N = max_size(y, nu);
          
      agrad::OperandsAndPartials<T_y, T_dof> 
        operands_and_partials(y, nu);
          
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
      for (size_t i = 0; i < stan::length(y); i++) {
        if (value_of(y_vec[i]) == 0) 
          return operands_and_partials.to_var(0.0);
      }
          
      // Compute CDF and its gradients
      using boost::math::gamma_p_derivative;
      using boost::math::gamma_p;
      using boost::math::digamma;
      using boost::math::tgamma;
          
      // Cache a few expensive function calls if nu is a parameter
      DoubleVectorView<!is_constant_struct<T_dof>::value,
                       is_vector<T_dof>::value> 
        gamma_vec(stan::length(nu));
      DoubleVectorView<!is_constant_struct<T_dof>::value,
                       is_vector<T_dof>::value> 
        digamma_vec(stan::length(nu));
          
      if (!is_constant_struct<T_dof>::value) {
        for (size_t i = 0; i < stan::length(nu); i++) {
          const double alpha_dbl = value_of(nu_vec[i]) * 0.5;
          gamma_vec[i] = tgamma(alpha_dbl);
          digamma_vec[i] = digamma(alpha_dbl);
        }
      }
          
      // Compute vectorized CDF and gradient
      for (size_t n = 0; n < N; n++) {
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(y_vec[n]) == std::numeric_limits<double>::infinity())
          continue;
              
        // Pull out values
        const double y_dbl = value_of(y_vec[n]);
        const double alpha_dbl = value_of(nu_vec[n]) * 0.5;
        const double beta_dbl = 0.5;
              
        // Compute
        const double Pn = gamma_p(alpha_dbl, beta_dbl * y_dbl);
              
        cdf *= Pn;
              
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] 
            += beta_dbl * gamma_p_derivative(alpha_dbl, beta_dbl * y_dbl) 
            / Pn;
        if (!is_constant_struct<T_dof>::value)
          operands_and_partials.d_x2[n] 
            -= 0.5 * stan::math::gradRegIncGamma(alpha_dbl, beta_dbl
                                                 * y_dbl, gamma_vec[n],
                                                 digamma_vec[n]) / Pn;
      }
          
      if (!is_constant_struct<T_y>::value)
        for (size_t n = 0; n < stan::length(y); ++n) 
          operands_and_partials.d_x1[n] *= cdf;
      if (!is_constant_struct<T_dof>::value)
        for (size_t n = 0; n < stan::length(nu); ++n) 
          operands_and_partials.d_x2[n] *= cdf;
          
      return operands_and_partials.to_var(cdf);
    }

    template <typename T_y, typename T_dof>
    typename return_type<T_y,T_dof>::type
    chi_square_cdf_log(const T_y& y, const T_dof& nu) {
      static const std::string function("stan::prob::chi_square_cdf_log");

      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_nonnegative;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::math::value_of;

      double cdf_log(0.0);

     // Size checks
      if (!(stan::length(y) && stan::length(nu))) 
        return cdf_log;

      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y);
      check_positive_finite(function, "Degrees of freedom parameter", nu);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Degrees of freedom parameter", nu);

      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_dof> nu_vec(nu);
      size_t N = max_size(y, nu);
          
      agrad::OperandsAndPartials<T_y, T_dof> 
        operands_and_partials(y, nu);
          
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
      for (size_t i = 0; i < stan::length(y); i++) {
        if (value_of(y_vec[i]) == 0) 
          return operands_and_partials.to_var(stan::math::negative_infinity());
      }
          
      // Compute cdf_log and its gradients
      using boost::math::gamma_p_derivative;
      using boost::math::gamma_p;
      using boost::math::digamma;
      using boost::math::tgamma;
          
      // Cache a few expensive function calls if nu is a parameter
      DoubleVectorView<!is_constant_struct<T_dof>::value,
                       is_vector<T_dof>::value> 
        gamma_vec(stan::length(nu));
      DoubleVectorView<!is_constant_struct<T_dof>::value,
                       is_vector<T_dof>::value> 
        digamma_vec(stan::length(nu));
          
      if (!is_constant_struct<T_dof>::value) {
        for (size_t i = 0; i < stan::length(nu); i++) {
          const double alpha_dbl = value_of(nu_vec[i]) * 0.5;
          gamma_vec[i] = tgamma(alpha_dbl);
          digamma_vec[i] = digamma(alpha_dbl);
        }
      }
          
      // Compute vectorized cdf_log and gradient
      for (size_t n = 0; n < N; n++) {
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(y_vec[n]) == std::numeric_limits<double>::infinity())
          return operands_and_partials.to_var(0.0);
              
        // Pull out values
        const double y_dbl = value_of(y_vec[n]);
        const double alpha_dbl = value_of(nu_vec[n]) * 0.5;
        const double beta_dbl = 0.5;
              
        // Compute
        const double Pn = gamma_p(alpha_dbl, beta_dbl * y_dbl);
              
        cdf_log += log(Pn);
              
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] 
            += beta_dbl * gamma_p_derivative(alpha_dbl, beta_dbl * y_dbl) 
            / Pn;
        if (!is_constant_struct<T_dof>::value)
          operands_and_partials.d_x2[n] 
            -= 0.5 * stan::math::gradRegIncGamma(alpha_dbl, beta_dbl
                                                 * y_dbl, gamma_vec[n],
                                                 digamma_vec[n]) / Pn;
      }
          
      return operands_and_partials.to_var(cdf_log);
    }

    template <typename T_y, typename T_dof>
    typename return_type<T_y,T_dof>::type
    chi_square_ccdf_log(const T_y& y, const T_dof& nu) {
      static const std::string function("stan::prob::chi_square_ccdf_log");

      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_nonnegative;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::math::value_of;

      double ccdf_log(0.0);

     // Size checks
      if (!(stan::length(y) && stan::length(nu))) 
        return ccdf_log;

      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y);
      check_positive_finite(function, "Degrees of freedom parameter", nu);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Degrees of freedom parameter", nu);

      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_dof> nu_vec(nu);
      size_t N = max_size(y, nu);
          
      agrad::OperandsAndPartials<T_y, T_dof> 
        operands_and_partials(y, nu);
          
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
      for (size_t i = 0; i < stan::length(y); i++) {
        if (value_of(y_vec[i]) == 0) 
          return operands_and_partials.to_var(0.0);
      }
          
      // Compute ccdf_log and its gradients
      using boost::math::gamma_p_derivative;
      using boost::math::gamma_p;
      using boost::math::digamma;
      using boost::math::tgamma;
          
      // Cache a few expensive function calls if nu is a parameter
      DoubleVectorView<!is_constant_struct<T_dof>::value,
                       is_vector<T_dof>::value> 
        gamma_vec(stan::length(nu));
      DoubleVectorView<!is_constant_struct<T_dof>::value,
                       is_vector<T_dof>::value> 
        digamma_vec(stan::length(nu));
          
      if (!is_constant_struct<T_dof>::value) {
        for (size_t i = 0; i < stan::length(nu); i++) {
          const double alpha_dbl = value_of(nu_vec[i]) * 0.5;
          gamma_vec[i] = tgamma(alpha_dbl);
          digamma_vec[i] = digamma(alpha_dbl);
        }
      }
          
      // Compute vectorized ccdf_log and gradient
      for (size_t n = 0; n < N; n++) {
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(y_vec[n]) == std::numeric_limits<double>::infinity())
          return operands_and_partials.to_var(stan::math::negative_infinity());
              
        // Pull out values
        const double y_dbl = value_of(y_vec[n]);
        const double alpha_dbl = value_of(nu_vec[n]) * 0.5;
        const double beta_dbl = 0.5;
              
        // Compute
        const double Pn = 1.0 - gamma_p(alpha_dbl, beta_dbl * y_dbl);
              
        ccdf_log += log(Pn);
              
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] 
            -= beta_dbl * gamma_p_derivative(alpha_dbl, beta_dbl * y_dbl) 
            / Pn;
        if (!is_constant_struct<T_dof>::value)
          operands_and_partials.d_x2[n] 
            += 0.5 * stan::math::gradRegIncGamma(alpha_dbl, beta_dbl
                                                 * y_dbl, gamma_vec[n],
                                                 digamma_vec[n]) / Pn;
      }
          
      return operands_and_partials.to_var(ccdf_log);
    }

    template <class RNG>
    inline double
    chi_square_rng(const double nu,
                   RNG& rng) {
      using boost::variate_generator;
      using boost::random::chi_squared_distribution;

      static const std::string function("stan::prob::chi_square_rng");

      using stan::error_handling::check_positive_finite;
      
      check_positive_finite(function, "Degrees of freedom parameter", nu);

      variate_generator<RNG&, chi_squared_distribution<> >
        chi_square_rng(rng, chi_squared_distribution<>(nu));
      return chi_square_rng();
    }
  }
}

#endif

