#ifndef AORSF_C_H
#define AORSF_C_H

#include "aorsf_export.h"
#include <stddef.h>
#include <stdint.h>

/* Opaque handle types */
typedef struct aorsf_forest_t* aorsf_forest_handle;
typedef struct aorsf_data_t* aorsf_data_handle;

/* Error codes */
typedef enum {
    AORSF_SUCCESS = 0,
    AORSF_ERROR_NULL_POINTER = -1,
    AORSF_ERROR_INVALID_ARGUMENT = -2,
    AORSF_ERROR_NOT_FITTED = -3,
    AORSF_ERROR_COMPUTATION = -4,
    AORSF_ERROR_OUT_OF_MEMORY = -5,
    AORSF_ERROR_IO = -6,
    AORSF_ERROR_FORMAT = -7,
    AORSF_ERROR_UNKNOWN = -99
} aorsf_error_t;

/* Tree type enum (mirrors globals.h) */
typedef enum {
    AORSF_TREE_CLASSIFICATION = 1,
    AORSF_TREE_REGRESSION = 2,
    AORSF_TREE_SURVIVAL = 3
} aorsf_tree_type_t;

/* Variable importance enum */
typedef enum {
    AORSF_VI_NONE = 0,
    AORSF_VI_NEGATE = 1,
    AORSF_VI_PERMUTE = 2,
    AORSF_VI_ANOVA = 3
} aorsf_vi_type_t;

/* Split rule enum */
typedef enum {
    AORSF_SPLIT_LOGRANK = 1,
    AORSF_SPLIT_CONCORD = 2,
    AORSF_SPLIT_GINI = 3,
    AORSF_SPLIT_VARIANCE = 4
} aorsf_split_rule_t;

/* Linear combination method enum */
typedef enum {
    AORSF_LC_GLM = 1,
    AORSF_LC_RANDOM = 2,
    AORSF_LC_GLMNET = 3
} aorsf_lincomb_type_t;

/* Prediction type enum */
typedef enum {
    AORSF_PRED_RISK = 1,
    AORSF_PRED_SURVIVAL = 2,
    AORSF_PRED_CHAZ = 3,
    AORSF_PRED_MORTALITY = 4,
    AORSF_PRED_MEAN = 5,
    AORSF_PRED_PROBABILITY = 6,
    AORSF_PRED_CLASS = 7
} aorsf_pred_type_t;

/* Forest configuration structure */
typedef struct {
    aorsf_tree_type_t tree_type;
    int32_t n_tree;
    int32_t mtry;
    int32_t leaf_min_obs;
    int32_t leaf_min_events;      /* survival only */
    int32_t split_min_obs;
    int32_t split_min_events;     /* survival only */
    int32_t split_min_stat;
    int32_t n_split;
    int32_t n_retry;
    aorsf_vi_type_t vi_type;
    aorsf_split_rule_t split_rule;
    aorsf_lincomb_type_t lincomb_type;
    double lincomb_eps;
    int32_t lincomb_iter_max;
    int32_t lincomb_scale;
    double lincomb_alpha;
    int32_t lincomb_df_target;
    int32_t oobag;
    int32_t oobag_eval_every;
    int32_t n_thread;
    uint32_t seed;
    int32_t verbosity;
} aorsf_config_t;

/* ============== Lifecycle Functions ============== */

/* Get default configuration */
AORSF_C_API void aorsf_config_init(aorsf_config_t* config, aorsf_tree_type_t tree_type);

/* Create forest handle */
AORSF_C_API aorsf_error_t aorsf_forest_create(
    aorsf_forest_handle* handle,
    const aorsf_config_t* config
);

/* Destroy forest handle */
AORSF_C_API void aorsf_forest_destroy(aorsf_forest_handle handle);

/* ============== Data Functions ============== */

/* Create data handle from arrays (data is copied) */
AORSF_C_API aorsf_error_t aorsf_data_create(
    aorsf_data_handle* handle,
    const double* x,           /* Feature matrix, row-major [n_rows x n_cols] */
    int32_t n_rows,
    int32_t n_cols,
    const double* y,           /* Outcome matrix, row-major [n_rows x n_y_cols] */
    int32_t n_y_cols,          /* 1 for regression/classification, 2 for survival (time, status) */
    const double* weights,     /* Sample weights, NULL for uniform [n_rows] */
    int32_t n_class            /* Number of classes (classification only, 0 otherwise) */
);

/* Destroy data handle */
AORSF_C_API void aorsf_data_destroy(aorsf_data_handle handle);

/* ============== Training Functions ============== */

/* Fit forest to data */
AORSF_C_API aorsf_error_t aorsf_forest_fit(
    aorsf_forest_handle handle,
    aorsf_data_handle data
);

/* Check if forest is fitted */
AORSF_C_API int32_t aorsf_forest_is_fitted(aorsf_forest_handle handle);

/* ============== Prediction Functions ============== */

/* Get prediction output dimensions */
AORSF_C_API aorsf_error_t aorsf_predict_get_dims(
    aorsf_forest_handle handle,
    int32_t n_rows,
    aorsf_pred_type_t pred_type,
    int32_t* out_rows,
    int32_t* out_cols
);

/* Predict on new data */
AORSF_C_API aorsf_error_t aorsf_forest_predict(
    aorsf_forest_handle handle,
    const double* x_new,       /* Feature matrix, row-major [n_rows x n_cols] */
    int32_t n_rows,
    int32_t n_cols,
    aorsf_pred_type_t pred_type,
    double* predictions,       /* Output array, pre-allocated by caller */
    int32_t predictions_size   /* Size of predictions array */
);

/* ============== Variable Importance ============== */

/* Get variable importance (after fitting with vi_type != NONE) */
AORSF_C_API aorsf_error_t aorsf_forest_get_importance(
    aorsf_forest_handle handle,
    double* importance,        /* Output array [n_features] */
    int32_t importance_size
);

/* ============== Model Information ============== */

/* Get number of features */
AORSF_C_API int32_t aorsf_forest_get_n_features(aorsf_forest_handle handle);

/* Get number of trees */
AORSF_C_API int32_t aorsf_forest_get_n_tree(aorsf_forest_handle handle);

/* Get tree type */
AORSF_C_API aorsf_tree_type_t aorsf_forest_get_tree_type(aorsf_forest_handle handle);

/* Get number of classes (classification only) */
AORSF_C_API int32_t aorsf_forest_get_n_class(aorsf_forest_handle handle);

/* Get OOB evaluation metric */
AORSF_C_API aorsf_error_t aorsf_forest_get_oob_error(
    aorsf_forest_handle handle,
    double* oob_error
);

/* ============== Survival-Specific Functions ============== */

/* Get unique event times (survival only) */
AORSF_C_API aorsf_error_t aorsf_forest_get_unique_times(
    aorsf_forest_handle handle,
    double* times,             /* Output array */
    int32_t* n_times           /* In: array size, Out: actual count */
);

/* Predict survival function at specific times */
AORSF_C_API aorsf_error_t aorsf_forest_predict_survival(
    aorsf_forest_handle handle,
    const double* x_new,
    int32_t n_rows,
    int32_t n_cols,
    const double* times,       /* Times at which to evaluate */
    int32_t n_times,
    double* survival,          /* Output: [n_rows x n_times] row-major */
    int32_t survival_size
);

/* ============== Error Handling ============== */

/* Get last error message (thread-local) */
AORSF_C_API const char* aorsf_get_last_error(void);

/* Get version string */
AORSF_C_API const char* aorsf_get_version(void);

/* ============== Serialization Functions ============== */

/* Serialization format enum */
typedef enum {
    AORSF_FORMAT_BINARY = 0,
    AORSF_FORMAT_JSON = 1
} aorsf_format_t;

/* Serialization flags */
#define AORSF_FLAG_HAS_IMPORTANCE  0x01
#define AORSF_FLAG_HAS_OOB         0x02
#define AORSF_FLAG_HAS_METADATA    0x04

/**
 * Get the size needed to serialize a fitted forest.
 *
 * @param handle    Fitted forest handle
 * @param format    Serialization format (BINARY or JSON)
 * @param flags     Flags controlling what to include (HAS_IMPORTANCE, HAS_OOB, HAS_METADATA)
 * @param size      Output: size in bytes needed
 * @return          AORSF_SUCCESS or error code
 */
AORSF_C_API aorsf_error_t aorsf_forest_get_save_size(
    aorsf_forest_handle handle,
    aorsf_format_t format,
    uint32_t flags,
    size_t* size
);

/**
 * Save a fitted forest to a memory buffer.
 *
 * @param handle      Fitted forest handle
 * @param format      Serialization format
 * @param flags       Flags controlling what to include
 * @param buffer      Output buffer (caller allocated)
 * @param buffer_size Size of buffer
 * @param written     Output: bytes actually written
 * @return            AORSF_SUCCESS or error code
 */
AORSF_C_API aorsf_error_t aorsf_forest_save(
    aorsf_forest_handle handle,
    aorsf_format_t format,
    uint32_t flags,
    unsigned char* buffer,
    size_t buffer_size,
    size_t* written
);

/**
 * Load a forest from a memory buffer.
 *
 * @param handle      Output: new forest handle
 * @param buffer      Input buffer containing serialized forest
 * @param buffer_size Size of buffer
 * @return            AORSF_SUCCESS or error code
 */
AORSF_C_API aorsf_error_t aorsf_forest_load(
    aorsf_forest_handle* handle,
    const unsigned char* buffer,
    size_t buffer_size
);

/**
 * Save a fitted forest to a file.
 *
 * @param handle   Fitted forest handle
 * @param filepath Output file path
 * @param format   Serialization format
 * @param flags    Flags controlling what to include
 * @return         AORSF_SUCCESS or error code
 */
AORSF_C_API aorsf_error_t aorsf_forest_save_file(
    aorsf_forest_handle handle,
    const char* filepath,
    aorsf_format_t format,
    uint32_t flags
);

/**
 * Load a forest from a file.
 *
 * @param handle   Output: new forest handle
 * @param filepath Input file path
 * @return         AORSF_SUCCESS or error code
 */
AORSF_C_API aorsf_error_t aorsf_forest_load_file(
    aorsf_forest_handle* handle,
    const char* filepath
);

/* ============== Metadata Functions ============== */

/**
 * Set feature names for the forest.
 *
 * @param handle        Forest handle
 * @param names         Array of null-terminated strings [n_features]
 * @param n_features    Number of feature names (must match forest's n_features)
 * @return              AORSF_SUCCESS or error code
 */
AORSF_C_API aorsf_error_t aorsf_forest_set_feature_names(
    aorsf_forest_handle handle,
    const char** names,
    int32_t n_features
);

/**
 * Get feature names from the forest.
 *
 * @param handle        Forest handle
 * @param names         Output array of string pointers [n_features]
 * @param n_features    Size of names array
 * @return              AORSF_SUCCESS or error code
 *
 * Note: Returned pointers are valid until the handle is destroyed or names are modified.
 */
AORSF_C_API aorsf_error_t aorsf_forest_get_feature_names(
    aorsf_forest_handle handle,
    const char** names,
    int32_t n_features
);

/**
 * Check if feature names are available.
 *
 * @param handle  Forest handle
 * @return        1 if names are available, 0 otherwise
 */
AORSF_C_API int32_t aorsf_forest_has_feature_names(aorsf_forest_handle handle);

/**
 * Set feature normalization statistics (means and standard deviations).
 *
 * @param handle        Forest handle
 * @param means         Array of feature means [n_features]
 * @param stds          Array of feature standard deviations [n_features]
 * @param n_features    Number of features
 * @return              AORSF_SUCCESS or error code
 */
AORSF_C_API aorsf_error_t aorsf_forest_set_feature_stats(
    aorsf_forest_handle handle,
    const double* means,
    const double* stds,
    int32_t n_features
);

/**
 * Get feature normalization statistics.
 *
 * @param handle        Forest handle
 * @param means         Output array for means [n_features] (can be NULL)
 * @param stds          Output array for stds [n_features] (can be NULL)
 * @param n_features    Size of output arrays
 * @return              AORSF_SUCCESS or error code
 */
AORSF_C_API aorsf_error_t aorsf_forest_get_feature_stats(
    aorsf_forest_handle handle,
    double* means,
    double* stds,
    int32_t n_features
);

/**
 * Check if feature statistics are available.
 *
 * @param handle  Forest handle
 * @return        1 if stats are available, 0 otherwise
 */
AORSF_C_API int32_t aorsf_forest_has_feature_stats(aorsf_forest_handle handle);

#endif /* AORSF_C_H */
