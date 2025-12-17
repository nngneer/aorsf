/*-----------------------------------------------------------------------------
 This file is part of aorsf.
 Author: Byron C Jaeger
 aorsf may be modified and distributed under the terms of the MIT license.
#----------------------------------------------------------------------------*/

#ifndef AORSF_RCPP_RMATH_H_
#define AORSF_RCPP_RMATH_H_

#include <Rcpp.h>
#include "../core/RMath.h"

namespace aorsf {

/**
 * @brief R-specific statistical distribution functions using R's internal functions.
 */
class RcppStatDistributions : public StatDistributions {
public:
    double pt(double x, double df) override {
        // R::pt(x, df, lower_tail, log_p)
        return R::pt(x, df, 1, 0);
    }

    double pchisq(double x, double df) override {
        // R::pchisq(x, df, lower_tail, log_p)
        return R::pchisq(x, df, 1, 0);
    }
};

/**
 * @brief Initialize R statistical distributions.
 * Call this at the beginning of R entry points.
 */
inline void init_r_stat() {
    StatManager::set_distributions(std::make_shared<RcppStatDistributions>());
}

} // namespace aorsf

#endif // AORSF_RCPP_RMATH_H_
