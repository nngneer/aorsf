/*-----------------------------------------------------------------------------
 This file is part of aorsf.
 Author: Byron C Jaeger
 aorsf may be modified and distributed under the terms of the MIT license.
#----------------------------------------------------------------------------*/

#ifndef AORSF_RCPP_OUTPUT_H_
#define AORSF_RCPP_OUTPUT_H_

#include <RcppArmadillo.h>
#include "../core/Output.h"

namespace aorsf {

/**
 * @brief Output handler that uses Rcpp::Rcout for R console output.
 *
 * This implementation bridges the abstract OutputHandler interface
 * to R's console output system via Rcpp.
 */
class RcppOutput : public OutputHandler {
public:
    void print(const std::string& msg) override {
        Rcpp::Rcout << msg;
    }

    void println(const std::string& msg) override {
        Rcpp::Rcout << msg << std::endl;
    }
};

/**
 * @brief Initialize the output system for R.
 *
 * Call this function at the start of any Rcpp exported function
 * that may produce console output.
 */
inline void init_r_output() {
    OutputManager::set_handler(std::make_shared<RcppOutput>());
}

} // namespace aorsf

#endif // AORSF_RCPP_OUTPUT_H_
