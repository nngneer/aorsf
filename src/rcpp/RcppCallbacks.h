/*-----------------------------------------------------------------------------
 This file is part of aorsf.
 Author: Byron C Jaeger
 aorsf may be modified and distributed under the terms of the MIT license.
#----------------------------------------------------------------------------*/

#ifndef AORSF_RCPP_CALLBACKS_H_
#define AORSF_RCPP_CALLBACKS_H_

#include <RcppArmadillo.h>
#include "../core/Callbacks.h"

namespace aorsf {

/**
 * @brief Create a LinCombCallback from an R function for user-defined linear combinations.
 *
 * @param r_func R function object (can be NULL/nil)
 * @return LinCombCallback that wraps the R function, or nullptr if r_func is NULL
 */
inline LinCombCallback make_lincomb_callback(Rcpp::RObject r_func) {
    if (r_func.isNULL()) {
        return nullptr;
    }

    Rcpp::Function f = Rcpp::as<Rcpp::Function>(r_func);
    return [f](const arma::mat& x, const arma::mat& y, const arma::vec& w) -> arma::mat {
        Rcpp::NumericMatrix xx = Rcpp::wrap(x);
        Rcpp::NumericMatrix yy = Rcpp::wrap(y);
        Rcpp::NumericVector ww = Rcpp::wrap(w);
        Rcpp::NumericMatrix result = f(xx, yy, ww);
        return Rcpp::as<arma::mat>(result);
    };
}

/**
 * @brief Create a LinCombCallback from an R function for glmnet-style linear combinations.
 *
 * This variant passes additional parameters (alpha, df_target) to the R function.
 *
 * @param r_func R function object (can be NULL/nil)
 * @param alpha The alpha parameter for glmnet
 * @param df_target The target degrees of freedom for glmnet
 * @return LinCombCallback that wraps the R function, or nullptr if r_func is NULL
 */
inline LinCombCallback make_glmnet_callback(Rcpp::RObject r_func,
                                            double alpha,
                                            arma::uword df_target) {
    if (r_func.isNULL()) {
        return nullptr;
    }

    Rcpp::Function f = Rcpp::as<Rcpp::Function>(r_func);
    return [f, alpha, df_target](const arma::mat& x, const arma::mat& y, const arma::vec& w) -> arma::mat {
        Rcpp::NumericMatrix xx = Rcpp::wrap(x);
        Rcpp::NumericMatrix yy = Rcpp::wrap(y);
        Rcpp::NumericVector ww = Rcpp::wrap(w);
        Rcpp::NumericMatrix result = f(xx, yy, ww, alpha, df_target);
        return Rcpp::as<arma::mat>(result);
    };
}

/**
 * @brief Create an OobagEvalCallback from an R function.
 *
 * @param r_func R function object (can be NULL/nil)
 * @return OobagEvalCallback that wraps the R function, or nullptr if r_func is NULL
 */
inline OobagEvalCallback make_oobag_callback(Rcpp::RObject r_func) {
    if (r_func.isNULL()) {
        return nullptr;
    }

    Rcpp::Function f = Rcpp::as<Rcpp::Function>(r_func);
    return [f](const arma::mat& y, const arma::vec& w, const arma::vec& p) -> double {
        Rcpp::NumericMatrix yy = Rcpp::wrap(y);
        Rcpp::NumericVector ww = Rcpp::wrap(w);
        Rcpp::NumericVector pp = Rcpp::wrap(p);
        Rcpp::NumericVector result = f(yy, ww, pp);
        return result[0];
    };
}

} // namespace aorsf

#endif // AORSF_RCPP_CALLBACKS_H_
