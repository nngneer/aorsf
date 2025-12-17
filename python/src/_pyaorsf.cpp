/**
 * @brief nanobind bindings for pyaorsf
 *
 * This file creates Python bindings for the aorsf C++ core library
 * using nanobind for NumPy array conversion.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>

// Include aorsf core headers
#include "arma_config.h"
#include "globals.h"
#include "Data.h"
#include "Forest.h"
#include "ForestSurvival.h"
#include "ForestClassification.h"
#include "ForestRegression.h"
#include "Exceptions.h"
#include "utility.h"

// Python-specific adapters
#include "python/PythonOutput.h"
#include "python/PythonInterrupts.h"
#include "python/PythonRMath.h"

namespace nb = nanobind;
using namespace aorsf;

// =============================================================================
// NumPy <-> Armadillo Conversion Helpers
// =============================================================================

/**
 * @brief Convert a 2D NumPy array to Armadillo matrix.
 * Data is copied to ensure memory safety.
 */
arma::mat numpy_to_arma_mat(nb::ndarray<double, nb::ndim<2>, nb::c_contig> arr) {
    size_t nrows = arr.shape(0);
    size_t ncols = arr.shape(1);
    const double* data = arr.data();

    // Create Armadillo matrix (column-major) from NumPy array (row-major)
    // We need to transpose during copy
    arma::mat result(nrows, ncols);
    for (size_t i = 0; i < nrows; ++i) {
        for (size_t j = 0; j < ncols; ++j) {
            result(i, j) = data[i * ncols + j];
        }
    }
    return result;
}

/**
 * @brief Convert a 1D NumPy array to Armadillo vector.
 */
arma::vec numpy_to_arma_vec(nb::ndarray<double, nb::ndim<1>, nb::c_contig> arr) {
    size_t n = arr.shape(0);
    const double* data = arr.data();
    return arma::vec(data, n);
}

/**
 * @brief Convert Armadillo matrix to NumPy array.
 */
nb::ndarray<nb::numpy, double, nb::ndim<2>> arma_mat_to_numpy(const arma::mat& m) {
    size_t nrows = m.n_rows;
    size_t ncols = m.n_cols;

    // Allocate memory
    double* data = new double[nrows * ncols];

    // Copy data (transposing from column-major to row-major)
    for (size_t i = 0; i < nrows; ++i) {
        for (size_t j = 0; j < ncols; ++j) {
            data[i * ncols + j] = m(i, j);
        }
    }

    // Create capsule to handle memory deallocation
    nb::capsule owner(data, [](void* p) noexcept {
        delete[] static_cast<double*>(p);
    });

    return nb::ndarray<nb::numpy, double, nb::ndim<2>>(
        data, {nrows, ncols}, owner
    );
}

/**
 * @brief Convert Armadillo vector to NumPy array.
 */
nb::ndarray<nb::numpy, double, nb::ndim<1>> arma_vec_to_numpy(const arma::vec& v) {
    size_t n = v.n_elem;

    double* data = new double[n];
    std::memcpy(data, v.memptr(), n * sizeof(double));

    nb::capsule owner(data, [](void* p) noexcept {
        delete[] static_cast<double*>(p);
    });

    return nb::ndarray<nb::numpy, double, nb::ndim<1>>(data, {n}, owner);
}

/**
 * @brief Convert Armadillo uvec to NumPy array of doubles.
 */
nb::ndarray<nb::numpy, double, nb::ndim<1>> arma_uvec_to_numpy(const arma::uvec& v) {
    size_t n = v.n_elem;

    double* data = new double[n];
    for (size_t i = 0; i < n; ++i) {
        data[i] = static_cast<double>(v(i));
    }

    nb::capsule owner(data, [](void* p) noexcept {
        delete[] static_cast<double*>(p);
    });

    return nb::ndarray<nb::numpy, double, nb::ndim<1>>(data, {n}, owner);
}

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * @brief Expand classification y labels to one-hot encoded matrix.
 * Input: vector of class labels (0, 1, ..., n_class-1)
 * Output: matrix of shape (n_samples, n_class) with 1s in the appropriate columns
 */
arma::mat expand_y_classification(const arma::vec& y, arma::uword n_class) {
    arma::mat out(y.n_elem, n_class, arma::fill::zeros);
    for (arma::uword i = 0; i < y.n_elem; ++i) {
        arma::uword class_idx = static_cast<arma::uword>(y(i));
        if (class_idx < n_class) {
            out(i, class_idx) = 1.0;
        }
    }
    return out;
}

// =============================================================================
// Utility Function Bindings
// =============================================================================

/**
 * @brief Compute concordance statistic for survival data.
 */
double compute_cstat_surv_py(
    nb::ndarray<double, nb::ndim<2>, nb::c_contig> y_np,
    nb::ndarray<double, nb::ndim<1>, nb::c_contig> w_np,
    nb::ndarray<double, nb::ndim<1>, nb::c_contig> p_np,
    bool pred_is_risklike
) {
    arma::mat y = numpy_to_arma_mat(y_np);
    arma::vec w = numpy_to_arma_vec(w_np);
    arma::vec p = numpy_to_arma_vec(p_np);

    return compute_cstat_surv(y, w, p, pred_is_risklike);
}

/**
 * @brief Compute concordance statistic for classification data.
 */
double compute_cstat_clsf_py(
    nb::ndarray<double, nb::ndim<1>, nb::c_contig> y_np,
    nb::ndarray<double, nb::ndim<1>, nb::c_contig> w_np,
    nb::ndarray<double, nb::ndim<1>, nb::c_contig> p_np
) {
    arma::vec y = numpy_to_arma_vec(y_np);
    arma::vec w = numpy_to_arma_vec(w_np);
    arma::vec p = numpy_to_arma_vec(p_np);

    return compute_cstat_clsf(y, w, p);
}

/**
 * @brief Compute log-rank statistic.
 */
double compute_logrank_py(
    nb::ndarray<double, nb::ndim<2>, nb::c_contig> y_np,
    nb::ndarray<double, nb::ndim<1>, nb::c_contig> w_np,
    nb::ndarray<uint32_t, nb::ndim<1>, nb::c_contig> g_np
) {
    arma::mat y = numpy_to_arma_mat(y_np);
    arma::vec w = numpy_to_arma_vec(w_np);

    // Convert g to uvec
    size_t n = g_np.shape(0);
    arma::uvec g(n);
    const uint32_t* gdata = g_np.data();
    for (size_t i = 0; i < n; ++i) {
        g(i) = gdata[i];
    }

    return compute_logrank(y, w, g);
}

/**
 * @brief Compute Gini impurity.
 */
double compute_gini_py(
    nb::ndarray<double, nb::ndim<1>, nb::c_contig> y_np,
    nb::ndarray<double, nb::ndim<1>, nb::c_contig> w_np,
    nb::ndarray<uint32_t, nb::ndim<1>, nb::c_contig> g_np
) {
    arma::vec y = numpy_to_arma_vec(y_np);
    arma::vec w = numpy_to_arma_vec(w_np);

    size_t n = g_np.shape(0);
    arma::uvec g(n);
    const uint32_t* gdata = g_np.data();
    for (size_t i = 0; i < n; ++i) {
        g(i) = gdata[i];
    }

    return compute_gini(y, w, g);
}

// =============================================================================
// Low-Level Forest Training Function
// =============================================================================

/**
 * @brief Fit an oblique random forest.
 *
 * This is the low-level training function called by the Python API classes.
 */
nb::dict fit_forest(
    nb::ndarray<double, nb::ndim<2>, nb::c_contig> x_np,
    nb::ndarray<double, nb::ndim<2>, nb::c_contig> y_np,
    nb::ndarray<double, nb::ndim<1>, nb::c_contig> w_np,
    int tree_type,
    std::vector<int> tree_seeds,
    int n_tree,
    int mtry,
    bool sample_with_replacement,
    double sample_fraction,
    int vi_type,
    double vi_max_pvalue,
    double leaf_min_events,
    double leaf_min_obs,
    int split_rule,
    double split_min_events,
    double split_min_obs,
    double split_min_stat,
    int split_max_cuts,
    int split_max_retry,
    int lincomb_type,
    double lincomb_eps,
    int lincomb_iter_max,
    bool lincomb_scale,
    double lincomb_alpha,
    int lincomb_df_target,
    int lincomb_ties_method,
    std::vector<double> pred_horizon,
    int pred_type,
    bool oobag,
    int oobag_eval_type,
    int oobag_eval_every,
    int n_thread,
    int verbose
) {
    // Initialize Python handlers
    init_python_output(verbose > 0);
    init_python_interrupt();
    init_python_stat();

    try {
        // Debug: print tree_seeds info
        if (verbose > 0) {
            AORSF_OUT.println("DEBUG: tree_seeds.size() = " + std::to_string(tree_seeds.size()));
            AORSF_OUT.println("DEBUG: n_tree = " + std::to_string(n_tree));
            AORSF_OUT.println("DEBUG: tree_type = " + std::to_string(tree_type));
            AORSF_OUT.println("DEBUG: pred_horizon.size() = " + std::to_string(pred_horizon.size()));
        }

        // Convert numpy arrays to Armadillo
        if (verbose > 0) AORSF_OUT.println("DEBUG: Converting numpy arrays...");
        arma::mat x = numpy_to_arma_mat(x_np);
        if (verbose > 0) AORSF_OUT.println("DEBUG: x converted, shape: " + std::to_string(x.n_rows) + "x" + std::to_string(x.n_cols));
        arma::mat y_input = numpy_to_arma_mat(y_np);
        if (verbose > 0) AORSF_OUT.println("DEBUG: y_input converted, shape: " + std::to_string(y_input.n_rows) + "x" + std::to_string(y_input.n_cols));
        arma::vec w = numpy_to_arma_vec(w_np);
        if (verbose > 0) AORSF_OUT.println("DEBUG: w converted, size: " + std::to_string(w.n_elem));

        TreeType tt = static_cast<TreeType>(tree_type);
        arma::uword n_class = 0;

        // Process y based on tree type
        arma::mat y;
        if (tt == TREE_CLASSIFICATION) {
            // For classification, y should be a single column of class labels (0, 1, ..., n-1)
            // We expand it to one-hot encoded matrix
            arma::vec y_labels = y_input.col(0);
            arma::vec unique_classes = arma::unique(y_labels);
            n_class = unique_classes.n_elem;
            y = expand_y_classification(y_labels, n_class);
        } else {
            // For survival and regression, use y as-is
            y = y_input;
        }

        // Create data object
        auto data = std::make_unique<Data>(x, y, w);
        arma::uword n_obs = data->n_rows;
        arma::uword n_cols_x = data->n_cols_x;

        // Handle thread count (0 = all available)
        if (n_thread == 0) {
            n_thread = std::thread::hardware_concurrency();
        }

        // Handle oobag_eval_every (0 = only at end, avoid division by zero)
        if (oobag_eval_every <= 0) {
            oobag_eval_every = n_tree;
        }

        // Create forest based on type
        if (verbose > 0) AORSF_OUT.println("DEBUG: Creating forest...");
        std::unique_ptr<Forest> forest;

        switch (tt) {
            case TREE_SURVIVAL: {
                if (verbose > 0) AORSF_OUT.println("DEBUG: Creating ForestSurvival...");

                // Ensure leaf_min_events and split_min_events >= 2 for survival
                // because compute_max_leaves() divides by (split_min_events - 1)
                // and (leaf_min_events - 1), which would cause division by zero
                double safe_leaf_min_events = std::max(leaf_min_events, 2.0);
                double safe_split_min_events = std::max(split_min_events, 2.0);

                arma::vec horizon(pred_horizon);
                if (verbose > 0) AORSF_OUT.println("DEBUG: horizon vec created, size: " + std::to_string(horizon.n_elem));
                forest = std::make_unique<ForestSurvival>(
                    safe_leaf_min_events, safe_split_min_events, horizon
                );
                if (verbose > 0) AORSF_OUT.println("DEBUG: ForestSurvival created");
                break;
            }
            case TREE_CLASSIFICATION: {
                forest = std::make_unique<ForestClassification>(n_class);
                break;
            }
            case TREE_REGRESSION:
                forest = std::make_unique<ForestRegression>();
                break;
            default:
                throw invalid_argument_error("Unknown tree type");
        }

        // Empty partial dependence containers
        std::vector<arma::mat> pd_x_vals;
        std::vector<arma::uvec> pd_x_cols;
        arma::vec pd_probs;

        // No custom callbacks for now (will be added in Phase 6)
        LinCombCallback lc_callback = nullptr;
        OobagEvalCallback oob_callback = nullptr;

        // Initialize forest
        if (verbose > 0) AORSF_OUT.println("DEBUG: Calling forest->init()...");
        forest->init(
            std::move(data),
            tree_seeds,
            static_cast<arma::uword>(n_tree),
            static_cast<arma::uword>(mtry),
            sample_with_replacement,
            sample_fraction,
            true,  // grow_mode
            static_cast<VariableImportance>(vi_type),
            vi_max_pvalue,
            leaf_min_obs,
            static_cast<SplitRule>(split_rule),
            split_min_obs,
            split_min_stat,
            static_cast<arma::uword>(split_max_cuts),
            static_cast<arma::uword>(split_max_retry),
            static_cast<LinearCombo>(lincomb_type),
            lincomb_eps,
            static_cast<arma::uword>(lincomb_iter_max),
            lincomb_scale,
            lincomb_alpha,
            static_cast<arma::uword>(lincomb_df_target),
            static_cast<arma::uword>(lincomb_ties_method),
            lc_callback,
            static_cast<PredType>(pred_type),
            false,  // pred_mode (growing, not predicting)
            true,   // pred_aggregate
            PD_NONE,
            pd_x_vals,
            pd_x_cols,
            pd_probs,
            oobag,
            static_cast<EvalType>(oobag_eval_type),
            static_cast<arma::uword>(oobag_eval_every),
            oob_callback,
            static_cast<uint>(n_thread),
            verbose
        );

        // Run forest growing
        if (verbose > 0) AORSF_OUT.println("DEBUG: Calling forest->run()...");
        forest->run(oobag);
        if (verbose > 0) AORSF_OUT.println("DEBUG: forest->run() completed");

        // Build result dictionary
        nb::dict result;

        // Basic info
        result["n_obs"] = n_obs;
        result["n_features"] = n_cols_x;
        result["n_tree"] = n_tree;
        result["tree_type"] = tree_type;

        // OOB predictions - only if predictions were generated
        const arma::mat& preds = forest->get_predictions();
        if (oobag && preds.n_elem > 0) {
            result["oob_predictions"] = arma_mat_to_numpy(preds);
        }

        // OOB evaluation
        const arma::mat& oob_eval = forest->get_oobag_eval();
        if (oob_eval.n_elem > 0) {
            result["oob_eval"] = arma_mat_to_numpy(oob_eval);
        }

        // Variable importance
        if (vi_type != VI_NONE) {
            arma::vec vi;
            if (vi_type == VI_ANOVA) {
                arma::uvec denom = forest->get_vi_denom();
                arma::uvec zeros = arma::find(denom == 0);
                if (zeros.n_elem > 0) {
                    denom(zeros).fill(1);
                }
                vi = forest->get_vi_numer() / arma::conv_to<arma::vec>::from(denom);
            } else {
                vi = forest->get_vi_numer() / static_cast<double>(n_tree);
            }
            result["importance"] = arma_vec_to_numpy(vi);
        }

        // Store forest structure for serialization
        result["oobag_denom"] = arma_vec_to_numpy(forest->get_oobag_denom());

        // Store full tree structure for prediction on new data
        // cutpoint: list of vectors, one per tree
        auto cutpoints = forest->get_cutpoint();
        nb::list cutpoint_list;
        for (const auto& cp : cutpoints) {
            nb::list tree_cp;
            for (double v : cp) tree_cp.append(v);
            cutpoint_list.append(tree_cp);
        }
        result["cutpoint"] = cutpoint_list;

        // child_left: list of vectors, one per tree
        auto child_lefts = forest->get_child_left();
        nb::list child_left_list;
        for (const auto& cl : child_lefts) {
            nb::list tree_cl;
            for (auto v : cl) tree_cl.append(static_cast<int>(v));
            child_left_list.append(tree_cl);
        }
        result["child_left"] = child_left_list;

        // coef_values: list of list of vectors (one per node per tree)
        auto coef_vals = forest->get_coef_values();
        nb::list coef_values_list;
        for (const auto& tree_coefs : coef_vals) {
            nb::list node_list;
            for (const auto& node_coef : tree_coefs) {
                node_list.append(arma_vec_to_numpy(node_coef));
            }
            coef_values_list.append(node_list);
        }
        result["coef_values"] = coef_values_list;

        // coef_indices: list of list of vectors (one per node per tree)
        auto coef_idxs = forest->get_coef_indices();
        nb::list coef_indices_list;
        for (const auto& tree_idxs : coef_idxs) {
            nb::list node_list;
            for (const auto& node_idx : tree_idxs) {
                node_list.append(arma_uvec_to_numpy(node_idx));
            }
            coef_indices_list.append(node_list);
        }
        result["coef_indices"] = coef_indices_list;

        // leaf_summary: list of vectors (one per tree)
        auto leaf_sums = forest->get_leaf_summary();
        nb::list leaf_summary_list;
        for (const auto& ls : leaf_sums) {
            nb::list tree_ls;
            for (double v : ls) tree_ls.append(v);
            leaf_summary_list.append(tree_ls);
        }
        result["leaf_summary"] = leaf_summary_list;

        // For survival forests, also store unique event times
        if (tt == TREE_SURVIVAL) {
            result["unique_event_times"] = arma_vec_to_numpy(forest->get_unique_event_times());
        }

        // Type-specific data
        if (tt == TREE_CLASSIFICATION) {
            result["n_class"] = n_class;
        }

        return result;

    } catch (const aorsf_error& e) {
        throw std::runtime_error(e.what());
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Internal error: ") + e.what());
    }
}

// =============================================================================
// Prediction Function
// =============================================================================

/**
 * @brief Predict using a fitted forest structure.
 *
 * Takes the tree structure from fit_forest and predicts on new data.
 */
nb::ndarray<nb::numpy, double, nb::ndim<2>> predict_forest(
    nb::ndarray<double, nb::ndim<2>, nb::c_contig> x,  // New data to predict
    nb::list cutpoint,       // List of lists: cutpoints per tree
    nb::list child_left,     // List of lists: child indices per tree
    nb::list coef_values,    // List of list of arrays: coefficients per node per tree
    nb::list coef_indices,   // List of list of arrays: feature indices per node per tree
    nb::list leaf_summary,   // List of lists: leaf values per tree
    int tree_type,           // Type of tree
    int n_class,             // Number of classes (for classification)
    bool aggregate           // Whether to aggregate predictions
) {
    // Convert input data
    arma::mat X = numpy_to_arma_mat(x);
    size_t n_obs = X.n_rows;
    size_t n_tree = nb::len(cutpoint);

    // Determine output dimensions based on tree type
    size_t n_cols_out = 1;
    if (tree_type == TREE_CLASSIFICATION) {
        n_cols_out = static_cast<size_t>(n_class);
    }

    // Initialize output matrix
    arma::mat result;
    if (aggregate) {
        result.zeros(n_obs, n_cols_out);
    } else {
        result.zeros(n_obs, n_tree);
    }

    // Process each tree
    for (size_t t = 0; t < n_tree; ++t) {
        // Get tree structure
        nb::list tree_cp = nb::cast<nb::list>(cutpoint[t]);
        nb::list tree_cl = nb::cast<nb::list>(child_left[t]);
        nb::list tree_coef_vals = nb::cast<nb::list>(coef_values[t]);
        nb::list tree_coef_idxs = nb::cast<nb::list>(coef_indices[t]);
        nb::list tree_ls = nb::cast<nb::list>(leaf_summary[t]);

        size_t n_nodes = nb::len(tree_cp);

        // Convert to vectors for faster access
        std::vector<double> cp(n_nodes);
        std::vector<int> cl(n_nodes);
        for (size_t i = 0; i < n_nodes; ++i) {
            cp[i] = nb::cast<double>(tree_cp[i]);
            cl[i] = nb::cast<int>(tree_cl[i]);
        }

        // Predict for each observation
        for (size_t i = 0; i < n_obs; ++i) {
            // Traverse tree to find leaf
            size_t node = 0;

            while (cl[node] != 0) {  // 0 means leaf (no children)
                // Get linear combination for this node
                nb::ndarray<double> node_coefs = nb::cast<nb::ndarray<double>>(tree_coef_vals[node]);
                nb::ndarray<double> node_idxs = nb::cast<nb::ndarray<double>>(tree_coef_idxs[node]);

                // Compute linear combination: X[i, indices] * coefficients
                double lc = 0.0;
                size_t n_coefs = node_coefs.shape(0);
                const double* coef_ptr = node_coefs.data();
                const double* idx_ptr = node_idxs.data();

                for (size_t j = 0; j < n_coefs; ++j) {
                    size_t col_idx = static_cast<size_t>(idx_ptr[j]);
                    lc += X(i, col_idx) * coef_ptr[j];
                }

                // Go left or right based on cutpoint
                if (lc <= cp[node]) {
                    node = static_cast<size_t>(cl[node]);  // left child
                } else {
                    node = static_cast<size_t>(cl[node]) + 1;  // right child
                }
            }

            // Get leaf prediction
            double leaf_val = nb::cast<double>(tree_ls[node]);

            if (aggregate) {
                if (tree_type == TREE_CLASSIFICATION) {
                    // For classification, leaf_val is the predicted class
                    size_t pred_class = static_cast<size_t>(leaf_val);
                    if (pred_class < n_cols_out) {
                        result(i, pred_class) += 1.0;
                    }
                } else {
                    result(i, 0) += leaf_val;
                }
            } else {
                result(i, t) = leaf_val;
            }
        }
    }

    // Average predictions if aggregating
    if (aggregate) {
        result /= static_cast<double>(n_tree);
    }

    return arma_mat_to_numpy(result);
}

// =============================================================================
// Module Definition
// =============================================================================

NB_MODULE(_pyaorsf, m) {
    m.doc() = "Python bindings for aorsf C++ core library";
    m.attr("__version__") = "0.1.0";

    // Expose constants
    m.attr("DEFAULT_N_TREE") = DEFAULT_N_TREE;
    m.attr("DEFAULT_LEAF_MIN_OBS") = DEFAULT_LEAF_MIN_OBS;
    m.attr("DEFAULT_SPLIT_MIN_OBS") = DEFAULT_SPLIT_MIN_OBS;
    m.attr("DEFAULT_SPLIT_MIN_STAT") = DEFAULT_SPLIT_MIN_STAT;
    m.attr("DEFAULT_SPLIT_MAX_CUTS") = static_cast<int>(DEFAULT_SPLIT_MAX_CUTS);
    m.attr("DEFAULT_LINCOMB_EPS") = DEFAULT_LINCOMB_EPS;
    m.attr("DEFAULT_LINCOMB_ITER_MAX") = static_cast<int>(DEFAULT_LINCOMB_ITER_MAX);

    // Expose enums
    nb::enum_<TreeType>(m, "TreeType")
        .value("CLASSIFICATION", TREE_CLASSIFICATION)
        .value("REGRESSION", TREE_REGRESSION)
        .value("SURVIVAL", TREE_SURVIVAL);

    nb::enum_<VariableImportance>(m, "VariableImportance")
        .value("NONE", VI_NONE)
        .value("NEGATE", VI_NEGATE)
        .value("PERMUTE", VI_PERMUTE)
        .value("ANOVA", VI_ANOVA);

    nb::enum_<SplitRule>(m, "SplitRule")
        .value("LOGRANK", SPLIT_LOGRANK)
        .value("CONCORD", SPLIT_CONCORD)
        .value("GINI", SPLIT_GINI)
        .value("VARIANCE", SPLIT_VARIANCE);

    nb::enum_<LinearCombo>(m, "LinearCombo")
        .value("GLM", LC_GLM)
        .value("RANDOM", LC_RANDOM_COEFS)
        .value("GLMNET", LC_GLMNET)
        .value("CUSTOM", LC_R_FUNCTION);

    nb::enum_<PredType>(m, "PredType")
        .value("NONE", PRED_NONE)
        .value("RISK", PRED_RISK)
        .value("SURVIVAL", PRED_SURVIVAL)
        .value("CHAZ", PRED_CHAZ)
        .value("MORTALITY", PRED_MORTALITY)
        .value("MEAN", PRED_MEAN)
        .value("PROBABILITY", PRED_PROBABILITY)
        .value("CLASS", PRED_CLASS)
        .value("TERMINAL_NODES", PRED_TERMINAL_NODES)
        .value("TIME", PRED_TIME);

    nb::enum_<EvalType>(m, "EvalType")
        .value("NONE", EVAL_NONE)
        .value("CONCORD", EVAL_CONCORD)
        .value("CUSTOM", EVAL_R_FUNCTION)
        .value("MSE", EVAL_MSE)
        .value("RSQ", EVAL_RSQ);

    // Utility functions
    m.def("compute_cstat_surv", &compute_cstat_surv_py,
          "Compute concordance statistic for survival data",
          nb::arg("y"), nb::arg("w"), nb::arg("p"), nb::arg("pred_is_risklike"));

    m.def("compute_cstat_clsf", &compute_cstat_clsf_py,
          "Compute concordance statistic for classification data",
          nb::arg("y"), nb::arg("w"), nb::arg("p"));

    m.def("compute_logrank", &compute_logrank_py,
          "Compute log-rank statistic",
          nb::arg("y"), nb::arg("w"), nb::arg("g"));

    m.def("compute_gini", &compute_gini_py,
          "Compute Gini impurity",
          nb::arg("y"), nb::arg("w"), nb::arg("g"));

    // Main training function
    m.def("fit_forest", &fit_forest,
          "Fit an oblique random forest",
          nb::arg("x"), nb::arg("y"), nb::arg("w"),
          nb::arg("tree_type"),
          nb::arg("tree_seeds"),
          nb::arg("n_tree"),
          nb::arg("mtry"),
          nb::arg("sample_with_replacement"),
          nb::arg("sample_fraction"),
          nb::arg("vi_type"),
          nb::arg("vi_max_pvalue"),
          nb::arg("leaf_min_events"),
          nb::arg("leaf_min_obs"),
          nb::arg("split_rule"),
          nb::arg("split_min_events"),
          nb::arg("split_min_obs"),
          nb::arg("split_min_stat"),
          nb::arg("split_max_cuts"),
          nb::arg("split_max_retry"),
          nb::arg("lincomb_type"),
          nb::arg("lincomb_eps"),
          nb::arg("lincomb_iter_max"),
          nb::arg("lincomb_scale"),
          nb::arg("lincomb_alpha"),
          nb::arg("lincomb_df_target"),
          nb::arg("lincomb_ties_method"),
          nb::arg("pred_horizon"),
          nb::arg("pred_type"),
          nb::arg("oobag"),
          nb::arg("oobag_eval_type"),
          nb::arg("oobag_eval_every"),
          nb::arg("n_thread"),
          nb::arg("verbose"));

    // Prediction function
    m.def("predict_forest", &predict_forest,
          "Predict using a fitted forest structure",
          nb::arg("x"),
          nb::arg("cutpoint"),
          nb::arg("child_left"),
          nb::arg("coef_values"),
          nb::arg("coef_indices"),
          nb::arg("leaf_summary"),
          nb::arg("tree_type"),
          nb::arg("n_class"),
          nb::arg("aggregate") = true);
}
