#pragma once

#include "stan/agrad/rev/matrix.hpp"
#include "stan/agrad/autodiff.hpp"

#include <Eigen/Dense>
#include <glog/logging.h>

#include <cmath>
#include <iostream>


namespace unet {

struct UpdateSummary {

};

struct MomentumGD {
  MomentumGD(double alpha = 1.01, double alpha_increase = 1.01, double alpha_decay = 0.8,
             double mu = 0.8, double mu_decay = 0.9999, uint32_t iter = 5)
    : alpha{alpha}, alpha_increase{alpha_increase}, alpha_decay{alpha_decay},
      mu{mu}, mu_decay{mu_decay}, iter{iter} {}

  double alpha, alpha_increase, alpha_decay;
  double mu, mu_decay;
  uint32_t iter;

  template<class Func>
  void fit_batch(Func f) { fit_batch(f, f.weights()); }

  template<class Func>
  void fit_batch(Func f, Eigen::VectorXd& w) {
    if (w_update_.size() == 0) { w_update_.resize(w.size()); }
    CHECK_EQ(w_update_.size(), w.size())
      << "fit_partial was used with vectors of different sizes";

    Eigen::VectorXd grad;
    double error{0}, last_error{1e10};
    for (uint32_t i = 0; i < iter; ++i) {
      stan::agrad::gradient(f, w, error, grad);
      if (std::isnan(error)) {
        LOG(WARNING) << "Error is NaN. Trying to backtrack. ";
        w -= w_update_;
        alpha *= alpha_decay;
        continue;
      }
      w_update_ = -alpha * grad + mu * w_update_;
      w += w_update_;

      LOG(INFO) << "alpha: " << alpha
                << " / mu: " << mu
                << " / error: " << error;

      if (error > last_error) {
        w -= w_update_;
        alpha *= alpha_decay;
      } else {
        alpha *= alpha_increase;
      }
      last_error = error;
      mu *= mu_decay;
    }

    // std::cout << "df(w): " << grad.transpose() << "\n";
  }

private:
  Eigen::VectorXd w_update_;
};

struct NesterovGD {
  NesterovGD(double alpha = 1.01, double alpha_increase = 1.01, double alpha_decay = 0.8,
             double mu = 0.8, double mu_decay = 0.9999, uint32_t iter = 5)
    : alpha{alpha}, alpha_increase{alpha_increase}, alpha_decay{alpha_decay},
      mu{mu}, mu_decay{mu_decay}, iter{iter} {}

  double alpha, alpha_increase, alpha_decay;
  double mu, mu_decay;
  uint32_t iter;

  template<class Func>
  void fit_batch(Func f) { fit_batch(f, f.weights()); }

  template<class Func>
  void fit_batch(Func f, Eigen::VectorXd& w) {
    if (w_update_.size() == 0) {
      w_update_ = Eigen::VectorXd::Zero(w.size());
    }
    CHECK_EQ(w_update_.size(), w.size())
      << "fit_partial was used with vectors of different sizes";

    Eigen::VectorXd grad;
    double error{0}, last_error{1e10};
    for (uint32_t i = 0; i < iter; ++i) {
      Eigen::VectorXd probe{w + w_update_};
      stan::agrad::gradient(f, probe, error, grad);
      if (std::isnan(error)) {
        LOG(WARNING) << "Error is NaN. Trying to backtrack. ";
        w -= w_update_;
        alpha *= alpha_decay;
        continue;
      }
      w_update_ = mu * w_update_ - alpha * grad;
      w += w_update_;

      LOG(INFO) << "alpha: " << alpha
                << " / mu: " << mu
                << " / error: " << error;

      if (error > last_error) {
        w -= w_update_;
        alpha *= alpha_decay;
      } else {
        alpha *= alpha_increase;
      }
      last_error = error;
      mu *= mu_decay;

      // if (alpha + mu >= 0.98) {
      //   double sum{(alpha + mu) / 0.98};
      //   alpha /= sum;
      //   mu /= sum;
      //   LOG(INFO) << "new alpha = " << alpha << " new mu = " << mu;
      // }
    }

    // std::cout << "df(w): " << grad.transpose() << "\n";
  }

private:
  Eigen::VectorXd w_update_;
};

}
