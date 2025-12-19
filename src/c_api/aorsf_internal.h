#ifndef AORSF_INTERNAL_H
#define AORSF_INTERNAL_H

/**
 * @file aorsf_internal.h
 * @brief Internal structures shared between aorsf_c.cpp and aorsf_serialize.cpp
 */

#include "aorsf_c.h"
#include "../core/arma_config.h"
#include "../core/Forest.h"
#include <memory>
#include <vector>
#include <string>

/* Thread-local error handling */
extern void set_error(const std::string& msg);

/**
 * @brief Internal forest wrapper structure
 *
 * This structure holds the C++ Forest object along with configuration
 * and cached results for easy retrieval.
 */
struct aorsf_forest_t {
    std::unique_ptr<aorsf::Forest> forest;
    aorsf_config_t config;
    bool is_fitted;
    int32_t n_features;
    int32_t n_class;
    std::vector<double> importance;
    std::vector<double> unique_times;  /* survival only */
    double oob_error;

    // Store tree structure for prediction
    std::vector<std::vector<double>> cutpoints;
    std::vector<std::vector<arma::uword>> child_left;
    std::vector<std::vector<arma::vec>> coef_values;
    std::vector<std::vector<arma::uvec>> coef_indices;
    std::vector<std::vector<double>> leaf_summary;

    // Type-specific leaf data (classification and regression)
    std::vector<std::vector<arma::vec>> leaf_pred_prob;

    // Type-specific leaf data (survival only)
    std::vector<std::vector<arma::vec>> leaf_pred_indx;
    std::vector<std::vector<arma::vec>> leaf_pred_chaz;

    // OOB data (for partial dependence, etc.)
    std::vector<arma::uvec> rows_oobag;  // Per-tree OOB indices
    std::vector<double> oobag_denom;     // OOB denominator per observation

    // Metadata (feature names, normalization info)
    std::vector<std::string> feature_names;
    std::vector<double> feature_means;    // For potential denormalization
    std::vector<double> feature_stds;     // For potential denormalization

    aorsf_forest_t() : is_fitted(false), n_features(0), n_class(0), oob_error(0.0) {}
};

/**
 * @brief Internal data wrapper structure
 */
struct aorsf_data_t {
    arma::mat x;
    arma::mat y;
    arma::vec w;
    int32_t n_class;

    aorsf_data_t() : n_class(0) {}
};

#endif /* AORSF_INTERNAL_H */
