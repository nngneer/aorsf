#include "../aorsf_c.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define ASSERT_SUCCESS(expr) do { \
    aorsf_error_t err = (expr); \
    if (err != AORSF_SUCCESS) { \
        fprintf(stderr, "FAILED at %s:%d: %s (error=%d: %s)\n", \
                __FILE__, __LINE__, #expr, err, aorsf_get_last_error()); \
        exit(1); \
    } \
} while(0)

#define ASSERT_TRUE(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAILED at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        exit(1); \
    } \
} while(0)

/* Generate simple test data */
void generate_classification_data(
    double* x, double* y, int n_rows, int n_cols
) {
    srand(42);
    for (int i = 0; i < n_rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < n_cols; j++) {
            x[i * n_cols + j] = (double)rand() / RAND_MAX * 2.0 - 1.0;
            if (j < 2) sum += x[i * n_cols + j];
        }
        y[i] = sum > 0 ? 1.0 : 0.0;
    }
}

void generate_survival_data(
    double* x, double* y, int n_rows, int n_cols
) {
    srand(42);
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            x[i * n_cols + j] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        }
        /* y has 2 columns: time, status */
        y[i * 2 + 0] = (double)rand() / RAND_MAX * 100.0 + 1.0;  /* time */
        y[i * 2 + 1] = rand() % 2;  /* status: 0 or 1 */
    }
}

void test_config_init(void) {
    printf("Testing config init... ");

    aorsf_config_t config;
    aorsf_config_init(&config, AORSF_TREE_CLASSIFICATION);

    ASSERT_TRUE(config.tree_type == AORSF_TREE_CLASSIFICATION);
    ASSERT_TRUE(config.n_tree == 500);
    ASSERT_TRUE(config.split_rule == AORSF_SPLIT_GINI);

    aorsf_config_init(&config, AORSF_TREE_SURVIVAL);
    ASSERT_TRUE(config.split_rule == AORSF_SPLIT_LOGRANK);

    printf("PASSED\n");
}

void test_forest_create_destroy(void) {
    printf("Testing forest create/destroy... ");

    aorsf_config_t config;
    aorsf_config_init(&config, AORSF_TREE_CLASSIFICATION);

    aorsf_forest_handle handle = NULL;
    ASSERT_SUCCESS(aorsf_forest_create(&handle, &config));
    ASSERT_TRUE(handle != NULL);
    ASSERT_TRUE(aorsf_forest_is_fitted(handle) == 0);

    aorsf_forest_destroy(handle);

    printf("PASSED\n");
}

void test_data_create_destroy(void) {
    printf("Testing data create/destroy... ");

    int n_rows = 100, n_cols = 5;
    double* x = malloc(n_rows * n_cols * sizeof(double));
    double* y = malloc(n_rows * sizeof(double));

    generate_classification_data(x, y, n_rows, n_cols);

    aorsf_data_handle data = NULL;
    ASSERT_SUCCESS(aorsf_data_create(&data, x, n_rows, n_cols, y, 1, NULL, 2));
    ASSERT_TRUE(data != NULL);

    aorsf_data_destroy(data);
    free(x);
    free(y);

    printf("PASSED\n");
}

void test_classification_fit_predict(void) {
    printf("Testing classification fit/predict... ");

    int n_rows = 200, n_cols = 5;
    double* x = malloc(n_rows * n_cols * sizeof(double));
    double* y = malloc(n_rows * sizeof(double));

    generate_classification_data(x, y, n_rows, n_cols);

    /* Create config */
    aorsf_config_t config;
    aorsf_config_init(&config, AORSF_TREE_CLASSIFICATION);
    config.n_tree = 10;  /* Small for testing */
    config.vi_type = AORSF_VI_NEGATE;

    /* Create forest */
    aorsf_forest_handle forest = NULL;
    ASSERT_SUCCESS(aorsf_forest_create(&forest, &config));

    /* Create data */
    aorsf_data_handle data = NULL;
    ASSERT_SUCCESS(aorsf_data_create(&data, x, n_rows, n_cols, y, 1, NULL, 2));

    /* Fit */
    ASSERT_SUCCESS(aorsf_forest_fit(forest, data));
    ASSERT_TRUE(aorsf_forest_is_fitted(forest) == 1);
    ASSERT_TRUE(aorsf_forest_get_n_features(forest) == n_cols);
    ASSERT_TRUE(aorsf_forest_get_n_class(forest) == 2);

    /* Predict */
    int32_t out_rows, out_cols;
    ASSERT_SUCCESS(aorsf_predict_get_dims(forest, n_rows, AORSF_PRED_CLASS, &out_rows, &out_cols));
    ASSERT_TRUE(out_rows == n_rows);
    ASSERT_TRUE(out_cols == 1);

    double* pred = malloc(out_rows * out_cols * sizeof(double));
    ASSERT_SUCCESS(aorsf_forest_predict(forest, x, n_rows, n_cols, AORSF_PRED_CLASS, pred, out_rows * out_cols));

    /* Check predictions are valid classes */
    for (int i = 0; i < n_rows; i++) {
        ASSERT_TRUE(pred[i] == 0.0 || pred[i] == 1.0);
    }

    /* Get importance */
    double* importance = malloc(n_cols * sizeof(double));
    ASSERT_SUCCESS(aorsf_forest_get_importance(forest, importance, n_cols));

    /* Importance should be computed */
    int has_nonzero = 0;
    for (int i = 0; i < n_cols; i++) {
        if (fabs(importance[i]) > 1e-10) has_nonzero = 1;
    }
    ASSERT_TRUE(has_nonzero);

    /* Cleanup */
    free(importance);
    free(pred);
    aorsf_data_destroy(data);
    aorsf_forest_destroy(forest);
    free(x);
    free(y);

    printf("PASSED\n");
}

void test_survival_fit_predict(void) {
    printf("Testing survival fit/predict... ");

    int n_rows = 200, n_cols = 5;
    double* x = malloc(n_rows * n_cols * sizeof(double));
    double* y = malloc(n_rows * 2 * sizeof(double));  /* time, status */

    generate_survival_data(x, y, n_rows, n_cols);

    /* Create config */
    aorsf_config_t config;
    aorsf_config_init(&config, AORSF_TREE_SURVIVAL);
    config.n_tree = 10;

    /* Create forest */
    aorsf_forest_handle forest = NULL;
    ASSERT_SUCCESS(aorsf_forest_create(&forest, &config));

    /* Create data */
    aorsf_data_handle data = NULL;
    ASSERT_SUCCESS(aorsf_data_create(&data, x, n_rows, n_cols, y, 2, NULL, 0));

    /* Fit */
    ASSERT_SUCCESS(aorsf_forest_fit(forest, data));
    ASSERT_TRUE(aorsf_forest_is_fitted(forest) == 1);
    ASSERT_TRUE(aorsf_forest_get_tree_type(forest) == AORSF_TREE_SURVIVAL);

    /* Get unique times */
    int32_t n_times = 0;
    ASSERT_SUCCESS(aorsf_forest_get_unique_times(forest, NULL, &n_times));
    ASSERT_TRUE(n_times > 0);

    double* times = malloc(n_times * sizeof(double));
    ASSERT_SUCCESS(aorsf_forest_get_unique_times(forest, times, &n_times));

    /* Predict risk */
    double* risk = malloc(n_rows * sizeof(double));
    ASSERT_SUCCESS(aorsf_forest_predict(forest, x, n_rows, n_cols, AORSF_PRED_RISK, risk, n_rows));

    /* Cleanup */
    free(risk);
    free(times);
    aorsf_data_destroy(data);
    aorsf_forest_destroy(forest);
    free(x);
    free(y);

    printf("PASSED\n");
}

void test_error_handling(void) {
    printf("Testing error handling... ");

    /* Null pointer should fail */
    aorsf_error_t err = aorsf_forest_create(NULL, NULL);
    ASSERT_TRUE(err == AORSF_ERROR_NULL_POINTER);
    ASSERT_TRUE(strlen(aorsf_get_last_error()) > 0);

    /* Invalid dimensions should fail */
    aorsf_data_handle data = NULL;
    double x[1] = {1.0};
    double y[1] = {0.0};
    err = aorsf_data_create(&data, x, 0, 1, y, 1, NULL, 0);  /* 0 rows */
    ASSERT_TRUE(err == AORSF_ERROR_INVALID_ARGUMENT);

    /* Predict on unfitted forest should fail */
    aorsf_config_t config;
    aorsf_config_init(&config, AORSF_TREE_CLASSIFICATION);
    aorsf_forest_handle forest = NULL;
    ASSERT_SUCCESS(aorsf_forest_create(&forest, &config));

    double pred[1];
    err = aorsf_forest_predict(forest, x, 1, 1, AORSF_PRED_CLASS, pred, 1);
    ASSERT_TRUE(err == AORSF_ERROR_NOT_FITTED);

    aorsf_forest_destroy(forest);

    printf("PASSED\n");
}

void test_version(void) {
    printf("Testing version... ");

    const char* version = aorsf_get_version();
    ASSERT_TRUE(version != NULL);
    ASSERT_TRUE(strlen(version) > 0);
    printf("(version=%s) ", version);

    printf("PASSED\n");
}

int main(void) {
    printf("=== aorsf C API Tests ===\n\n");

    test_version();
    test_config_init();
    test_forest_create_destroy();
    test_data_create_destroy();
    test_classification_fit_predict();
    test_survival_fit_predict();
    test_error_handling();

    printf("\n=== All tests passed! ===\n");
    return 0;
}
