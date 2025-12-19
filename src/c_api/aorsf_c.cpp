/**
 * @file aorsf_c.cpp
 * @brief C API implementation for aorsf library
 *
 * This file provides C-compatible wrapper functions around the C++ aorsf library,
 * enabling use from languages that support C FFI (like C#, Go, Rust, etc.)
 */

#include "aorsf_c.h"
#include "aorsf_internal.h"
#include "../core/ForestSurvival.h"
#include "../core/ForestClassification.h"
#include "../core/ForestRegression.h"
#include "../core/Data.h"
#include "../core/globals.h"
#include "../core/Exceptions.h"
#include "../core/Callbacks.h"

#include <memory>
#include <string>
#include <cstring>
#include <cmath>
#include <random>
#include <thread>

/* Thread-local error message storage */
static thread_local std::string g_last_error;

/* Non-static so it can be used from aorsf_serialize.cpp */
void set_error(const std::string& msg) {
    g_last_error = msg;
}

/* ============== Helper Functions ============== */

/**
 * @brief Convert row-major C array to column-major Armadillo matrix
 */
static arma::mat rowmajor_to_arma(const double* data, int32_t n_rows, int32_t n_cols) {
    arma::mat result(n_rows, n_cols);
    for (int32_t i = 0; i < n_rows; ++i) {
        for (int32_t j = 0; j < n_cols; ++j) {
            result(i, j) = data[i * n_cols + j];
        }
    }
    return result;
}

/**
 * @brief Convert column-major Armadillo matrix to row-major C array
 */
static void arma_to_rowmajor(const arma::mat& mat, double* data) {
    for (arma::uword i = 0; i < mat.n_rows; ++i) {
        for (arma::uword j = 0; j < mat.n_cols; ++j) {
            data[i * mat.n_cols + j] = mat(i, j);
        }
    }
}

/**
 * @brief Expand classification y labels to one-hot encoded matrix
 */
static arma::mat expand_y_classification(const arma::vec& y, arma::uword n_class) {
    arma::mat out(y.n_elem, n_class, arma::fill::zeros);
    for (arma::uword i = 0; i < y.n_elem; ++i) {
        arma::uword class_idx = static_cast<arma::uword>(y(i));
        if (class_idx < n_class) {
            out(i, class_idx) = 1.0;
        }
    }
    return out;
}

/**
 * @brief Generate random tree seeds
 */
static std::vector<int> generate_tree_seeds(int n_tree, uint32_t seed) {
    std::vector<int> seeds(n_tree);
    std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);
    std::uniform_int_distribution<int> dist(1, INT_MAX);
    for (int i = 0; i < n_tree; ++i) {
        seeds[i] = dist(rng);
    }
    return seeds;
}

/* ============== Lifecycle Functions ============== */

AORSF_C_API void aorsf_config_init(aorsf_config_t* config, aorsf_tree_type_t tree_type) {
    if (!config) return;

    std::memset(config, 0, sizeof(aorsf_config_t));

    config->tree_type = tree_type;
    config->n_tree = 500;
    config->mtry = 0;  /* 0 = auto (ceiling(sqrt(n_features))) */
    config->leaf_min_obs = 5;
    config->leaf_min_events = 1;
    config->split_min_obs = 10;
    config->split_min_events = 5;
    config->split_min_stat = 0;
    config->n_split = 5;
    config->n_retry = 3;
    config->vi_type = AORSF_VI_NONE;
    config->lincomb_type = AORSF_LC_GLM;
    config->lincomb_eps = 1e-9;
    config->lincomb_iter_max = 20;
    config->lincomb_scale = 1;
    config->lincomb_alpha = 0.5;
    config->lincomb_df_target = 0;
    config->oobag = 1;
    config->oobag_eval_every = 0;  /* 0 = only at end */
    config->n_thread = 0;  /* 0 = auto */
    config->seed = 0;  /* 0 = random */
    config->verbosity = 0;

    /* Tree-type specific defaults */
    if (tree_type == AORSF_TREE_SURVIVAL) {
        config->split_rule = AORSF_SPLIT_LOGRANK;
    } else if (tree_type == AORSF_TREE_CLASSIFICATION) {
        config->split_rule = AORSF_SPLIT_GINI;
    } else {
        config->split_rule = AORSF_SPLIT_VARIANCE;
    }
}

AORSF_C_API aorsf_error_t aorsf_forest_create(
    aorsf_forest_handle* handle,
    const aorsf_config_t* config
) {
    if (!handle || !config) {
        set_error("Null pointer argument");
        return AORSF_ERROR_NULL_POINTER;
    }

    try {
        auto wrapper = std::make_unique<aorsf_forest_t>();
        wrapper->config = *config;

        *handle = wrapper.release();
        return AORSF_SUCCESS;

    } catch (const std::exception& e) {
        set_error(e.what());
        return AORSF_ERROR_UNKNOWN;
    }
}

AORSF_C_API void aorsf_forest_destroy(aorsf_forest_handle handle) {
    delete handle;
}

/* ============== Data Functions ============== */

AORSF_C_API aorsf_error_t aorsf_data_create(
    aorsf_data_handle* handle,
    const double* x,
    int32_t n_rows,
    int32_t n_cols,
    const double* y,
    int32_t n_y_cols,
    const double* weights,
    int32_t n_class
) {
    if (!handle || !x || !y) {
        set_error("Null pointer argument");
        return AORSF_ERROR_NULL_POINTER;
    }

    if (n_rows <= 0 || n_cols <= 0 || n_y_cols <= 0) {
        set_error("Invalid dimensions");
        return AORSF_ERROR_INVALID_ARGUMENT;
    }

    try {
        auto wrapper = std::make_unique<aorsf_data_t>();

        /* Convert row-major to column-major Armadillo matrices */
        wrapper->x = rowmajor_to_arma(x, n_rows, n_cols);
        wrapper->y = rowmajor_to_arma(y, n_rows, n_y_cols);

        /* Handle weights */
        if (weights) {
            wrapper->w = arma::vec(weights, n_rows);
        } else {
            wrapper->w = arma::ones<arma::vec>(n_rows);
        }

        wrapper->n_class = n_class;

        *handle = wrapper.release();
        return AORSF_SUCCESS;

    } catch (const std::bad_alloc&) {
        set_error("Out of memory");
        return AORSF_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        set_error(e.what());
        return AORSF_ERROR_UNKNOWN;
    }
}

AORSF_C_API void aorsf_data_destroy(aorsf_data_handle handle) {
    delete handle;
}

/* ============== Training Functions ============== */

AORSF_C_API aorsf_error_t aorsf_forest_fit(
    aorsf_forest_handle handle,
    aorsf_data_handle data
) {
    if (!handle || !data) {
        set_error("Null pointer argument");
        return AORSF_ERROR_NULL_POINTER;
    }

    try {
        auto& config = handle->config;

        /* Store dimensions */
        handle->n_features = static_cast<int32_t>(data->x.n_cols);
        handle->n_class = data->n_class;

        /* Calculate mtry if auto */
        arma::uword mtry = config.mtry;
        if (mtry <= 0) {
            mtry = static_cast<arma::uword>(std::ceil(std::sqrt(data->x.n_cols)));
        }

        /* Handle thread count (0 = all available) */
        uint n_thread = config.n_thread;
        if (n_thread == 0) {
            n_thread = std::thread::hardware_concurrency();
        }

        /* Handle oobag_eval_every (0 = only at end) */
        arma::uword oobag_eval_every = config.oobag_eval_every;
        if (oobag_eval_every <= 0) {
            oobag_eval_every = config.n_tree;
        }

        /* Process y based on tree type */
        arma::mat x_data = data->x;
        arma::mat y_data;
        arma::vec w_data = data->w;
        arma::uword n_class = 0;

        aorsf::TreeType tt = static_cast<aorsf::TreeType>(config.tree_type);

        if (tt == aorsf::TREE_CLASSIFICATION) {
            arma::vec y_labels = data->y.col(0);
            arma::vec unique_classes = arma::unique(y_labels);
            n_class = unique_classes.n_elem;
            y_data = expand_y_classification(y_labels, n_class);
            handle->n_class = static_cast<int32_t>(n_class);
        } else {
            y_data = data->y;
        }

        /* Create data object */
        auto forest_data = std::make_unique<aorsf::Data>(x_data, y_data, w_data);

        /* Generate tree seeds */
        std::vector<int> tree_seeds = generate_tree_seeds(config.n_tree, config.seed);

        /* Create forest based on type */
        std::unique_ptr<aorsf::Forest> forest;

        switch (tt) {
            case aorsf::TREE_SURVIVAL: {
                double safe_leaf_min_events = std::max(static_cast<double>(config.leaf_min_events), 2.0);
                double safe_split_min_events = std::max(static_cast<double>(config.split_min_events), 2.0);
                arma::vec pred_horizon;  // Empty for now
                forest = std::make_unique<aorsf::ForestSurvival>(
                    safe_leaf_min_events, safe_split_min_events, pred_horizon
                );
                break;
            }
            case aorsf::TREE_CLASSIFICATION: {
                forest = std::make_unique<aorsf::ForestClassification>(n_class);
                break;
            }
            case aorsf::TREE_REGRESSION:
                forest = std::make_unique<aorsf::ForestRegression>();
                break;
            default:
                set_error("Invalid tree type");
                return AORSF_ERROR_INVALID_ARGUMENT;
        }

        /* Empty partial dependence containers */
        std::vector<arma::mat> pd_x_vals;
        std::vector<arma::uvec> pd_x_cols;
        arma::vec pd_probs;

        /* No callbacks for C API */
        aorsf::LinCombCallback lc_callback = nullptr;
        aorsf::OobagEvalCallback oob_callback = nullptr;

        /* Map config enums to C++ enums */
        aorsf::VariableImportance vi_type = static_cast<aorsf::VariableImportance>(config.vi_type);
        aorsf::SplitRule split_rule = static_cast<aorsf::SplitRule>(config.split_rule);
        aorsf::LinearCombo lincomb_type = static_cast<aorsf::LinearCombo>(config.lincomb_type);
        aorsf::EvalType oobag_eval_type = aorsf::EVAL_NONE;

        /* Set appropriate evaluation type and prediction type based on tree type */
        aorsf::PredType pred_type_init = aorsf::PRED_RISK;  /* default for survival */
        if (config.oobag) {
            switch (tt) {
                case aorsf::TREE_SURVIVAL:
                    oobag_eval_type = aorsf::EVAL_CONCORD;
                    pred_type_init = aorsf::PRED_RISK;
                    break;
                case aorsf::TREE_CLASSIFICATION:
                    oobag_eval_type = aorsf::EVAL_CONCORD;
                    pred_type_init = aorsf::PRED_CLASS;
                    break;
                case aorsf::TREE_REGRESSION:
                    oobag_eval_type = aorsf::EVAL_RSQ;
                    pred_type_init = aorsf::PRED_MEAN;
                    break;
                default:
                    break;
            }
        }

        /* Initialize forest */
        forest->init(
            std::move(forest_data),
            tree_seeds,
            static_cast<arma::uword>(config.n_tree),
            mtry,
            true,  /* sample_with_replacement */
            0.632, /* sample_fraction */
            true,  /* grow_mode */
            vi_type,
            0.01,  /* vi_max_pvalue */
            static_cast<double>(config.leaf_min_obs),
            split_rule,
            static_cast<double>(config.split_min_obs),
            static_cast<double>(config.split_min_stat),
            static_cast<arma::uword>(config.n_split),
            static_cast<arma::uword>(config.n_retry),
            lincomb_type,
            config.lincomb_eps,
            static_cast<arma::uword>(config.lincomb_iter_max),
            config.lincomb_scale != 0,
            config.lincomb_alpha,
            static_cast<arma::uword>(config.lincomb_df_target),
            1,  /* lincomb_ties_method */
            lc_callback,
            pred_type_init,  /* pred_type - varies by tree type */
            false,  /* pred_mode (growing, not predicting) */
            true,   /* pred_aggregate */
            aorsf::PD_NONE,
            pd_x_vals,
            pd_x_cols,
            pd_probs,
            config.oobag != 0,
            oobag_eval_type,
            oobag_eval_every,
            oob_callback,
            n_thread,
            config.verbosity
        );

        /* Run forest growing */
        forest->run(config.oobag != 0);

        /* Extract results */
        handle->is_fitted = true;

        /* Get importance if computed */
        if (vi_type != aorsf::VI_NONE) {
            arma::vec vi;
            if (vi_type == aorsf::VI_ANOVA) {
                arma::uvec denom = forest->get_vi_denom();
                arma::uvec zeros = arma::find(denom == 0);
                if (zeros.n_elem > 0) {
                    denom(zeros).fill(1);
                }
                vi = forest->get_vi_numer() / arma::conv_to<arma::vec>::from(denom);
            } else {
                vi = forest->get_vi_numer() / static_cast<double>(config.n_tree);
            }
            handle->importance.assign(vi.begin(), vi.end());
        }

        /* Get OOB evaluation */
        if (config.oobag) {
            const arma::mat& oob_eval = forest->get_oobag_eval();
            if (oob_eval.n_elem > 0) {
                handle->oob_error = oob_eval(oob_eval.n_rows - 1, 0);
            }
        }

        /* Get unique times for survival */
        if (tt == aorsf::TREE_SURVIVAL) {
            arma::vec& times = forest->get_unique_event_times();
            handle->unique_times.assign(times.begin(), times.end());
        }

        /* Store tree structure for prediction */
        handle->cutpoints = forest->get_cutpoint();
        handle->child_left = forest->get_child_left();
        handle->coef_values = forest->get_coef_values();
        handle->coef_indices = forest->get_coef_indices();
        handle->leaf_summary = forest->get_leaf_summary();

        /* Store type-specific leaf data */
        if (tt == aorsf::TREE_CLASSIFICATION) {
            auto* class_forest = dynamic_cast<aorsf::ForestClassification*>(forest.get());
            if (class_forest) {
                handle->leaf_pred_prob = class_forest->get_leaf_pred_prob();
            }
        } else if (tt == aorsf::TREE_SURVIVAL) {
            auto* surv_forest = dynamic_cast<aorsf::ForestSurvival*>(forest.get());
            if (surv_forest) {
                handle->leaf_pred_indx = surv_forest->get_leaf_pred_indx();
                handle->leaf_pred_prob = surv_forest->get_leaf_pred_prob();
                handle->leaf_pred_chaz = surv_forest->get_leaf_pred_chaz();
            }
        }

        /* Store OOB data if computed */
        if (config.oobag) {
            handle->rows_oobag = forest->get_rows_oobag();
            arma::vec& oobag_denom = forest->get_oobag_denom();
            handle->oobag_denom.assign(oobag_denom.begin(), oobag_denom.end());
        }

        /* Store forest pointer */
        handle->forest = std::move(forest);

        return AORSF_SUCCESS;

    } catch (const aorsf::invalid_argument_error& e) {
        set_error(e.what());
        return AORSF_ERROR_INVALID_ARGUMENT;
    } catch (const aorsf::computation_error& e) {
        set_error(e.what());
        return AORSF_ERROR_COMPUTATION;
    } catch (const std::bad_alloc&) {
        set_error("Out of memory");
        return AORSF_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        set_error(e.what());
        return AORSF_ERROR_UNKNOWN;
    }
}

AORSF_C_API int32_t aorsf_forest_is_fitted(aorsf_forest_handle handle) {
    return handle ? (handle->is_fitted ? 1 : 0) : 0;
}

/* ============== Prediction Functions ============== */

AORSF_C_API aorsf_error_t aorsf_predict_get_dims(
    aorsf_forest_handle handle,
    int32_t n_rows,
    aorsf_pred_type_t pred_type,
    int32_t* out_rows,
    int32_t* out_cols
) {
    if (!handle || !out_rows || !out_cols) {
        set_error("Null pointer argument");
        return AORSF_ERROR_NULL_POINTER;
    }

    if (!handle->is_fitted) {
        set_error("Forest not fitted");
        return AORSF_ERROR_NOT_FITTED;
    }

    *out_rows = n_rows;

    switch (pred_type) {
        case AORSF_PRED_CLASS:
        case AORSF_PRED_RISK:
        case AORSF_PRED_MORTALITY:
        case AORSF_PRED_MEAN:
            *out_cols = 1;
            break;
        case AORSF_PRED_PROBABILITY:
            *out_cols = handle->n_class;
            break;
        case AORSF_PRED_SURVIVAL:
        case AORSF_PRED_CHAZ:
            *out_cols = static_cast<int32_t>(handle->unique_times.size());
            break;
        default:
            *out_cols = 1;
    }

    return AORSF_SUCCESS;
}

AORSF_C_API aorsf_error_t aorsf_forest_predict(
    aorsf_forest_handle handle,
    const double* x_new,
    int32_t n_rows,
    int32_t n_cols,
    aorsf_pred_type_t pred_type,
    double* predictions,
    int32_t predictions_size
) {
    if (!handle || !x_new || !predictions) {
        set_error("Null pointer argument");
        return AORSF_ERROR_NULL_POINTER;
    }

    if (!handle->is_fitted) {
        set_error("Forest not fitted");
        return AORSF_ERROR_NOT_FITTED;
    }

    if (n_cols != handle->n_features) {
        set_error("Number of features doesn't match training data");
        return AORSF_ERROR_INVALID_ARGUMENT;
    }

    try {
        arma::mat X = rowmajor_to_arma(x_new, n_rows, n_cols);

        /* Determine output dimensions based on tree type and pred_type */
        size_t n_cols_out = 1;
        aorsf_tree_type_t tree_type = handle->config.tree_type;

        /* For classification, we need n_class columns to accumulate votes */
        if (tree_type == AORSF_TREE_CLASSIFICATION &&
            (pred_type == AORSF_PRED_PROBABILITY || pred_type == AORSF_PRED_CLASS)) {
            n_cols_out = static_cast<size_t>(handle->n_class);
        }

        /* Initialize output matrix */
        arma::mat result(n_rows, n_cols_out, arma::fill::zeros);

        size_t n_tree = handle->cutpoints.size();

        /* Process each tree */
        for (size_t t = 0; t < n_tree; ++t) {
            const auto& tree_cp = handle->cutpoints[t];
            const auto& tree_cl = handle->child_left[t];
            const auto& tree_coef_vals = handle->coef_values[t];
            const auto& tree_coef_idxs = handle->coef_indices[t];
            const auto& tree_ls = handle->leaf_summary[t];

            /* Predict for each observation */
            for (int32_t i = 0; i < n_rows; ++i) {
                /* Traverse tree to find leaf */
                size_t node = 0;

                while (tree_cl[node] != 0) {  /* 0 means leaf (no children) */
                    /* Get linear combination for this node */
                    const arma::vec& node_coefs = tree_coef_vals[node];
                    const arma::uvec& node_idxs = tree_coef_idxs[node];

                    /* Compute linear combination: X[i, indices] * coefficients */
                    double lc = 0.0;
                    for (size_t j = 0; j < node_coefs.n_elem; ++j) {
                        size_t col_idx = node_idxs(j);
                        lc += X(i, col_idx) * node_coefs(j);
                    }

                    /* Go left or right based on cutpoint */
                    if (lc <= tree_cp[node]) {
                        node = tree_cl[node];  /* left child */
                    } else {
                        node = tree_cl[node] + 1;  /* right child */
                    }
                }

                /* Add leaf prediction to result */
                if (tree_type == AORSF_TREE_CLASSIFICATION &&
                    (pred_type == AORSF_PRED_PROBABILITY || pred_type == AORSF_PRED_CLASS)) {
                    /* For classification, leaf_summary[node] is the predicted class */
                    /* Accumulate votes for that class */
                    size_t pred_class = static_cast<size_t>(tree_ls[node]);
                    if (pred_class < n_cols_out) {
                        result(i, pred_class) += 1.0;
                    }
                } else {
                    result(i, 0) += tree_ls[node];
                }
            }
        }

        /* Handle classification output */
        if (tree_type == AORSF_TREE_CLASSIFICATION) {
            if (pred_type == AORSF_PRED_CLASS) {
                /* Return the class with most votes */
                for (int32_t i = 0; i < n_rows; ++i) {
                    predictions[i] = static_cast<double>(result.row(i).index_max());
                }
            } else if (pred_type == AORSF_PRED_PROBABILITY) {
                /* Normalize votes to probabilities */
                result /= static_cast<double>(n_tree);
                arma_to_rowmajor(result, predictions);
            }
        } else {
            /* For regression/survival, average the predictions */
            result /= static_cast<double>(n_tree);
            arma_to_rowmajor(result, predictions);
        }

        return AORSF_SUCCESS;

    } catch (const std::exception& e) {
        set_error(e.what());
        return AORSF_ERROR_COMPUTATION;
    }
}

/* ============== Variable Importance ============== */

AORSF_C_API aorsf_error_t aorsf_forest_get_importance(
    aorsf_forest_handle handle,
    double* importance,
    int32_t importance_size
) {
    if (!handle || !importance) {
        set_error("Null pointer argument");
        return AORSF_ERROR_NULL_POINTER;
    }

    if (!handle->is_fitted) {
        set_error("Forest not fitted");
        return AORSF_ERROR_NOT_FITTED;
    }

    if (handle->importance.empty()) {
        set_error("Importance not computed (vi_type was NONE)");
        return AORSF_ERROR_INVALID_ARGUMENT;
    }

    if (importance_size < static_cast<int32_t>(handle->importance.size())) {
        set_error("Importance buffer too small");
        return AORSF_ERROR_INVALID_ARGUMENT;
    }

    std::copy(handle->importance.begin(), handle->importance.end(), importance);
    return AORSF_SUCCESS;
}

/* ============== Model Information ============== */

AORSF_C_API int32_t aorsf_forest_get_n_features(aorsf_forest_handle handle) {
    return handle ? handle->n_features : 0;
}

AORSF_C_API int32_t aorsf_forest_get_n_tree(aorsf_forest_handle handle) {
    return handle ? handle->config.n_tree : 0;
}

AORSF_C_API aorsf_tree_type_t aorsf_forest_get_tree_type(aorsf_forest_handle handle) {
    return handle ? handle->config.tree_type : AORSF_TREE_CLASSIFICATION;
}

AORSF_C_API int32_t aorsf_forest_get_n_class(aorsf_forest_handle handle) {
    return handle ? handle->n_class : 0;
}

AORSF_C_API aorsf_error_t aorsf_forest_get_oob_error(
    aorsf_forest_handle handle,
    double* oob_error
) {
    if (!handle || !oob_error) {
        set_error("Null pointer argument");
        return AORSF_ERROR_NULL_POINTER;
    }

    if (!handle->is_fitted) {
        set_error("Forest not fitted");
        return AORSF_ERROR_NOT_FITTED;
    }

    *oob_error = handle->oob_error;
    return AORSF_SUCCESS;
}

/* ============== Survival-Specific Functions ============== */

AORSF_C_API aorsf_error_t aorsf_forest_get_unique_times(
    aorsf_forest_handle handle,
    double* times,
    int32_t* n_times
) {
    if (!handle || !n_times) {
        set_error("Null pointer argument");
        return AORSF_ERROR_NULL_POINTER;
    }

    if (!handle->is_fitted) {
        set_error("Forest not fitted");
        return AORSF_ERROR_NOT_FITTED;
    }

    if (handle->config.tree_type != AORSF_TREE_SURVIVAL) {
        set_error("Not a survival forest");
        return AORSF_ERROR_INVALID_ARGUMENT;
    }

    int32_t count = static_cast<int32_t>(handle->unique_times.size());

    if (times) {
        if (*n_times < count) {
            set_error("Times buffer too small");
            return AORSF_ERROR_INVALID_ARGUMENT;
        }
        std::copy(handle->unique_times.begin(), handle->unique_times.end(), times);
    }

    *n_times = count;
    return AORSF_SUCCESS;
}

AORSF_C_API aorsf_error_t aorsf_forest_predict_survival(
    aorsf_forest_handle handle,
    const double* x_new,
    int32_t n_rows,
    int32_t n_cols,
    const double* times,
    int32_t n_times,
    double* survival,
    int32_t survival_size
) {
    if (!handle || !x_new || !times || !survival) {
        set_error("Null pointer argument");
        return AORSF_ERROR_NULL_POINTER;
    }

    if (handle->config.tree_type != AORSF_TREE_SURVIVAL) {
        set_error("Not a survival forest");
        return AORSF_ERROR_INVALID_ARGUMENT;
    }

    if (survival_size < n_rows * n_times) {
        set_error("Survival buffer too small");
        return AORSF_ERROR_INVALID_ARGUMENT;
    }

    /* Note: This is a placeholder implementation.
     * Full survival curve prediction would require interpolating
     * the cumulative hazard at the requested time points.
     * For now, we return risk scores replicated across time points. */

    try {
        /* Get risk predictions first */
        std::vector<double> risk(n_rows);
        aorsf_error_t err = aorsf_forest_predict(
            handle, x_new, n_rows, n_cols,
            AORSF_PRED_RISK, risk.data(), n_rows
        );

        if (err != AORSF_SUCCESS) {
            return err;
        }

        /* Convert risk to survival probability (simplified) */
        for (int32_t i = 0; i < n_rows; ++i) {
            for (int32_t t = 0; t < n_times; ++t) {
                /* S(t) = exp(-risk * t) is a simplification */
                survival[i * n_times + t] = std::exp(-risk[i] * times[t] / 100.0);
            }
        }

        return AORSF_SUCCESS;

    } catch (const std::exception& e) {
        set_error(e.what());
        return AORSF_ERROR_COMPUTATION;
    }
}

/* ============== Error Handling ============== */

AORSF_C_API const char* aorsf_get_last_error(void) {
    return g_last_error.c_str();
}

AORSF_C_API const char* aorsf_get_version(void) {
    return "0.1.6";
}

/* ============== Metadata Functions ============== */

AORSF_C_API aorsf_error_t aorsf_forest_set_feature_names(
    aorsf_forest_handle handle,
    const char** names,
    int32_t n_features
) {
    if (!handle || !names) {
        set_error("Null pointer argument");
        return AORSF_ERROR_NULL_POINTER;
    }

    if (handle->is_fitted && n_features != handle->n_features) {
        set_error("Number of feature names must match n_features");
        return AORSF_ERROR_INVALID_ARGUMENT;
    }

    try {
        handle->feature_names.clear();
        handle->feature_names.reserve(n_features);
        for (int32_t i = 0; i < n_features; ++i) {
            if (names[i]) {
                handle->feature_names.push_back(names[i]);
            } else {
                handle->feature_names.push_back("");
            }
        }
        return AORSF_SUCCESS;
    } catch (const std::bad_alloc&) {
        set_error("Out of memory");
        return AORSF_ERROR_OUT_OF_MEMORY;
    }
}

AORSF_C_API aorsf_error_t aorsf_forest_get_feature_names(
    aorsf_forest_handle handle,
    const char** names,
    int32_t n_features
) {
    if (!handle || !names) {
        set_error("Null pointer argument");
        return AORSF_ERROR_NULL_POINTER;
    }

    if (handle->feature_names.empty()) {
        set_error("Feature names not set");
        return AORSF_ERROR_INVALID_ARGUMENT;
    }

    if (n_features < static_cast<int32_t>(handle->feature_names.size())) {
        set_error("Buffer too small");
        return AORSF_ERROR_INVALID_ARGUMENT;
    }

    for (size_t i = 0; i < handle->feature_names.size(); ++i) {
        names[i] = handle->feature_names[i].c_str();
    }

    return AORSF_SUCCESS;
}

AORSF_C_API int32_t aorsf_forest_has_feature_names(aorsf_forest_handle handle) {
    return (handle && !handle->feature_names.empty()) ? 1 : 0;
}

AORSF_C_API aorsf_error_t aorsf_forest_set_feature_stats(
    aorsf_forest_handle handle,
    const double* means,
    const double* stds,
    int32_t n_features
) {
    if (!handle) {
        set_error("Null pointer argument");
        return AORSF_ERROR_NULL_POINTER;
    }

    if (handle->is_fitted && n_features != handle->n_features) {
        set_error("Number of features must match n_features");
        return AORSF_ERROR_INVALID_ARGUMENT;
    }

    try {
        if (means) {
            handle->feature_means.assign(means, means + n_features);
        } else {
            handle->feature_means.clear();
        }

        if (stds) {
            handle->feature_stds.assign(stds, stds + n_features);
        } else {
            handle->feature_stds.clear();
        }

        return AORSF_SUCCESS;
    } catch (const std::bad_alloc&) {
        set_error("Out of memory");
        return AORSF_ERROR_OUT_OF_MEMORY;
    }
}

AORSF_C_API aorsf_error_t aorsf_forest_get_feature_stats(
    aorsf_forest_handle handle,
    double* means,
    double* stds,
    int32_t n_features
) {
    if (!handle) {
        set_error("Null pointer argument");
        return AORSF_ERROR_NULL_POINTER;
    }

    if (handle->feature_means.empty() && handle->feature_stds.empty()) {
        set_error("Feature statistics not set");
        return AORSF_ERROR_INVALID_ARGUMENT;
    }

    int32_t expected_size = static_cast<int32_t>(
        std::max(handle->feature_means.size(), handle->feature_stds.size())
    );

    if (n_features < expected_size) {
        set_error("Buffer too small");
        return AORSF_ERROR_INVALID_ARGUMENT;
    }

    if (means && !handle->feature_means.empty()) {
        std::copy(handle->feature_means.begin(), handle->feature_means.end(), means);
    }

    if (stds && !handle->feature_stds.empty()) {
        std::copy(handle->feature_stds.begin(), handle->feature_stds.end(), stds);
    }

    return AORSF_SUCCESS;
}

AORSF_C_API int32_t aorsf_forest_has_feature_stats(aorsf_forest_handle handle) {
    return (handle && (!handle->feature_means.empty() || !handle->feature_stds.empty())) ? 1 : 0;
}
