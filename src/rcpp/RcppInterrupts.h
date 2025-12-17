/*-----------------------------------------------------------------------------
 This file is part of aorsf.
 Author: Byron C Jaeger
 aorsf may be modified and distributed under the terms of the MIT license.
#----------------------------------------------------------------------------*/

#ifndef AORSF_RCPP_INTERRUPTS_H_
#define AORSF_RCPP_INTERRUPTS_H_

#include <RcppArmadillo.h>
#include "../core/Interrupts.h"

namespace aorsf {

/**
 * @brief R-specific interrupt handler using Rcpp::checkUserInterrupt.
 */
class RcppInterrupt : public InterruptHandler {
public:
    bool check() override {
        try {
            Rcpp::checkUserInterrupt();
            return false;
        } catch (...) {
            return true;
        }
    }
};

/**
 * @brief Initialize the R interrupt handler.
 * Call this at the beginning of R entry points.
 */
inline void init_r_interrupt() {
    InterruptManager::set_handler(std::make_shared<RcppInterrupt>());
}

} // namespace aorsf

#endif // AORSF_RCPP_INTERRUPTS_H_
