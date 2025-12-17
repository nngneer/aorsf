/*-----------------------------------------------------------------------------
 This file is part of aorsf.
 Author: Byron C Jaeger
 aorsf may be modified and distributed under the terms of the MIT license.
#----------------------------------------------------------------------------*/

#ifndef AORSF_CALLBACKS_H_
#define AORSF_CALLBACKS_H_

#include <armadillo>
#include <functional>

namespace aorsf {

/**
 * @brief Callback type for custom linear combination functions.
 *
 * This callback computes coefficients for linear combinations of predictors
 * at a node during tree growing. It replaces the R function callback mechanism.
 *
 * @param x Predictor matrix at the node (n_obs x n_predictors)
 * @param y Outcome matrix at the node (n_obs x n_outcomes)
 * @param w Weight vector at the node (n_obs)
 * @return Matrix of coefficients (n_predictors x 1 typically)
 */
using LinCombCallback = std::function<arma::mat(
    const arma::mat& x,
    const arma::mat& y,
    const arma::vec& w
)>;

/**
 * @brief Callback type for custom out-of-bag evaluation functions.
 *
 * This callback computes a custom accuracy metric for out-of-bag predictions.
 * It replaces the R function callback mechanism.
 *
 * @param y True outcome matrix (n_obs x n_outcomes)
 * @param w Weight vector (n_obs)
 * @param p Prediction vector (n_obs)
 * @return Scalar accuracy metric
 */
using OobagEvalCallback = std::function<double(
    const arma::mat& y,
    const arma::vec& w,
    const arma::vec& p
)>;

} // namespace aorsf

#endif // AORSF_CALLBACKS_H_
