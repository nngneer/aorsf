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
    free(pred);

    /* Test probability prediction */
    int32_t prob_rows, prob_cols;
    ASSERT_SUCCESS(aorsf_predict_get_dims(forest, n_rows, AORSF_PRED_PROBABILITY, &prob_rows, &prob_cols));
    ASSERT_TRUE(prob_rows == n_rows);
    ASSERT_TRUE(prob_cols == 2);  /* Binary classification */

    double* prob = malloc(prob_rows * prob_cols * sizeof(double));
    ASSERT_SUCCESS(aorsf_forest_predict(forest, x, n_rows, n_cols, AORSF_PRED_PROBABILITY, prob, prob_rows * prob_cols));

    /* Check probabilities sum to 1 */
    for (int i = 0; i < n_rows; i++) {
        double sum = prob[i * 2 + 0] + prob[i * 2 + 1];
        ASSERT_TRUE(fabs(sum - 1.0) < 0.01);  /* Should sum to ~1 */
    }
    free(prob);

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

void test_serialization_binary(void) {
    printf("Testing binary serialization... ");

    int n_rows = 200, n_cols = 5;
    double* x = malloc(n_rows * n_cols * sizeof(double));
    double* y = malloc(n_rows * sizeof(double));

    generate_classification_data(x, y, n_rows, n_cols);

    /* Create and fit forest */
    aorsf_config_t config;
    aorsf_config_init(&config, AORSF_TREE_CLASSIFICATION);
    config.n_tree = 10;
    config.vi_type = AORSF_VI_NEGATE;

    aorsf_forest_handle forest = NULL;
    ASSERT_SUCCESS(aorsf_forest_create(&forest, &config));

    aorsf_data_handle data = NULL;
    ASSERT_SUCCESS(aorsf_data_create(&data, x, n_rows, n_cols, y, 1, NULL, 2));
    ASSERT_SUCCESS(aorsf_forest_fit(forest, data));

    /* Get predictions before save */
    double* pred_before = malloc(n_rows * sizeof(double));
    ASSERT_SUCCESS(aorsf_forest_predict(forest, x, n_rows, n_cols, AORSF_PRED_CLASS, pred_before, n_rows));

    /* Get save size */
    size_t save_size;
    ASSERT_SUCCESS(aorsf_forest_get_save_size(forest, AORSF_FORMAT_BINARY, AORSF_FLAG_HAS_IMPORTANCE, &save_size));
    ASSERT_TRUE(save_size > 0);

    /* Save to buffer */
    unsigned char* buffer = malloc(save_size);
    size_t written;
    ASSERT_SUCCESS(aorsf_forest_save(forest, AORSF_FORMAT_BINARY, AORSF_FLAG_HAS_IMPORTANCE, buffer, save_size, &written));
    ASSERT_TRUE(written > 0);

    /* Load from buffer */
    aorsf_forest_handle loaded = NULL;
    ASSERT_SUCCESS(aorsf_forest_load(&loaded, buffer, written));
    ASSERT_TRUE(loaded != NULL);
    ASSERT_TRUE(aorsf_forest_is_fitted(loaded) == 1);
    ASSERT_TRUE(aorsf_forest_get_n_features(loaded) == n_cols);
    ASSERT_TRUE(aorsf_forest_get_n_class(loaded) == 2);

    /* Get predictions after load */
    double* pred_after = malloc(n_rows * sizeof(double));
    ASSERT_SUCCESS(aorsf_forest_predict(loaded, x, n_rows, n_cols, AORSF_PRED_CLASS, pred_after, n_rows));

    /* Predictions should match */
    for (int i = 0; i < n_rows; i++) {
        ASSERT_TRUE(fabs(pred_before[i] - pred_after[i]) < 0.001);
    }

    /* Test file I/O */
    const char* test_file = "/tmp/claude/test_forest.bin";
    ASSERT_SUCCESS(aorsf_forest_save_file(forest, test_file, AORSF_FORMAT_BINARY, AORSF_FLAG_HAS_IMPORTANCE));

    aorsf_forest_handle file_loaded = NULL;
    ASSERT_SUCCESS(aorsf_forest_load_file(&file_loaded, test_file));
    ASSERT_TRUE(aorsf_forest_is_fitted(file_loaded) == 1);

    /* Cleanup */
    free(pred_before);
    free(pred_after);
    free(buffer);
    aorsf_forest_destroy(file_loaded);
    aorsf_forest_destroy(loaded);
    aorsf_data_destroy(data);
    aorsf_forest_destroy(forest);
    free(x);
    free(y);

    printf("PASSED\n");
}

void test_serialization_json(void) {
    printf("Testing JSON serialization... ");

    int n_rows = 200, n_cols = 5;
    double* x = malloc(n_rows * n_cols * sizeof(double));
    double* y = malloc(n_rows * sizeof(double));

    generate_classification_data(x, y, n_rows, n_cols);

    /* Create and fit forest */
    aorsf_config_t config;
    aorsf_config_init(&config, AORSF_TREE_CLASSIFICATION);
    config.n_tree = 10;
    config.vi_type = AORSF_VI_NEGATE;

    aorsf_forest_handle forest = NULL;
    ASSERT_SUCCESS(aorsf_forest_create(&forest, &config));

    aorsf_data_handle data = NULL;
    ASSERT_SUCCESS(aorsf_data_create(&data, x, n_rows, n_cols, y, 1, NULL, 2));
    ASSERT_SUCCESS(aorsf_forest_fit(forest, data));

    /* Get predictions before save */
    double* pred_before = malloc(n_rows * sizeof(double));
    ASSERT_SUCCESS(aorsf_forest_predict(forest, x, n_rows, n_cols, AORSF_PRED_CLASS, pred_before, n_rows));

    /* Get save size */
    size_t save_size;
    ASSERT_SUCCESS(aorsf_forest_get_save_size(forest, AORSF_FORMAT_JSON, AORSF_FLAG_HAS_IMPORTANCE, &save_size));
    ASSERT_TRUE(save_size > 0);

    /* Save to buffer (add extra space for pretty printing) */
    size_t buffer_size = save_size * 2;
    unsigned char* buffer = malloc(buffer_size);
    size_t written;
    ASSERT_SUCCESS(aorsf_forest_save(forest, AORSF_FORMAT_JSON, AORSF_FLAG_HAS_IMPORTANCE, buffer, buffer_size, &written));
    ASSERT_TRUE(written > 0);

    /* Load from buffer */
    aorsf_forest_handle loaded = NULL;
    ASSERT_SUCCESS(aorsf_forest_load(&loaded, buffer, written));
    ASSERT_TRUE(loaded != NULL);
    ASSERT_TRUE(aorsf_forest_is_fitted(loaded) == 1);
    ASSERT_TRUE(aorsf_forest_get_n_features(loaded) == n_cols);
    ASSERT_TRUE(aorsf_forest_get_n_class(loaded) == 2);

    /* Get predictions after load */
    double* pred_after = malloc(n_rows * sizeof(double));
    ASSERT_SUCCESS(aorsf_forest_predict(loaded, x, n_rows, n_cols, AORSF_PRED_CLASS, pred_after, n_rows));

    /* Predictions should match */
    for (int i = 0; i < n_rows; i++) {
        ASSERT_TRUE(fabs(pred_before[i] - pred_after[i]) < 0.001);
    }

    /* Cleanup */
    free(pred_before);
    free(pred_after);
    free(buffer);
    aorsf_forest_destroy(loaded);
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

void test_metadata(void) {
    printf("Testing metadata... ");

    int n_rows = 200, n_cols = 5;
    double* x = malloc(n_rows * n_cols * sizeof(double));
    double* y = malloc(n_rows * sizeof(double));

    generate_classification_data(x, y, n_rows, n_cols);

    /* Create and fit forest */
    aorsf_config_t config;
    aorsf_config_init(&config, AORSF_TREE_CLASSIFICATION);
    config.n_tree = 10;
    config.vi_type = AORSF_VI_NEGATE;  /* Enable importance for testing */

    aorsf_forest_handle forest = NULL;
    ASSERT_SUCCESS(aorsf_forest_create(&forest, &config));

    aorsf_data_handle data = NULL;
    ASSERT_SUCCESS(aorsf_data_create(&data, x, n_rows, n_cols, y, 1, NULL, 2));
    ASSERT_SUCCESS(aorsf_forest_fit(forest, data));

    /* Test feature names */
    ASSERT_TRUE(aorsf_forest_has_feature_names(forest) == 0);

    const char* names[5] = {"feature_a", "feature_b", "feature_c", "feature_d", "feature_e"};
    ASSERT_SUCCESS(aorsf_forest_set_feature_names(forest, names, n_cols));
    ASSERT_TRUE(aorsf_forest_has_feature_names(forest) == 1);

    const char* retrieved_names[5];
    ASSERT_SUCCESS(aorsf_forest_get_feature_names(forest, retrieved_names, n_cols));
    for (int i = 0; i < n_cols; i++) {
        ASSERT_TRUE(strcmp(retrieved_names[i], names[i]) == 0);
    }

    /* Test feature stats */
    ASSERT_TRUE(aorsf_forest_has_feature_stats(forest) == 0);

    double means[5] = {0.0, 1.0, 2.0, 3.0, 4.0};
    double stds[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    ASSERT_SUCCESS(aorsf_forest_set_feature_stats(forest, means, stds, n_cols));
    ASSERT_TRUE(aorsf_forest_has_feature_stats(forest) == 1);

    double retrieved_means[5];
    double retrieved_stds[5];
    ASSERT_SUCCESS(aorsf_forest_get_feature_stats(forest, retrieved_means, retrieved_stds, n_cols));
    for (int i = 0; i < n_cols; i++) {
        ASSERT_TRUE(fabs(retrieved_means[i] - means[i]) < 1e-10);
        ASSERT_TRUE(fabs(retrieved_stds[i] - stds[i]) < 1e-10);
    }

    /* Test metadata serialization (binary) */
    size_t save_size;
    ASSERT_SUCCESS(aorsf_forest_get_save_size(forest, AORSF_FORMAT_BINARY,
        AORSF_FLAG_HAS_IMPORTANCE | AORSF_FLAG_HAS_METADATA, &save_size));
    ASSERT_TRUE(save_size > 0);

    unsigned char* buffer = malloc(save_size);
    size_t written;
    ASSERT_SUCCESS(aorsf_forest_save(forest, AORSF_FORMAT_BINARY,
        AORSF_FLAG_HAS_IMPORTANCE | AORSF_FLAG_HAS_METADATA, buffer, save_size, &written));

    /* Load and verify metadata preserved */
    aorsf_forest_handle loaded = NULL;
    ASSERT_SUCCESS(aorsf_forest_load(&loaded, buffer, written));
    ASSERT_TRUE(aorsf_forest_has_feature_names(loaded) == 1);
    ASSERT_TRUE(aorsf_forest_has_feature_stats(loaded) == 1);

    const char* loaded_names[5];
    ASSERT_SUCCESS(aorsf_forest_get_feature_names(loaded, loaded_names, n_cols));
    for (int i = 0; i < n_cols; i++) {
        ASSERT_TRUE(strcmp(loaded_names[i], names[i]) == 0);
    }

    double loaded_means[5];
    double loaded_stds[5];
    ASSERT_SUCCESS(aorsf_forest_get_feature_stats(loaded, loaded_means, loaded_stds, n_cols));
    for (int i = 0; i < n_cols; i++) {
        ASSERT_TRUE(fabs(loaded_means[i] - means[i]) < 1e-10);
        ASSERT_TRUE(fabs(loaded_stds[i] - stds[i]) < 1e-10);
    }

    free(buffer);
    aorsf_forest_destroy(loaded);

    /* Test metadata serialization (JSON) */
    ASSERT_SUCCESS(aorsf_forest_get_save_size(forest, AORSF_FORMAT_JSON,
        AORSF_FLAG_HAS_IMPORTANCE | AORSF_FLAG_HAS_METADATA, &save_size));
    buffer = malloc(save_size * 2);  /* Extra space for safety */
    ASSERT_SUCCESS(aorsf_forest_save(forest, AORSF_FORMAT_JSON,
        AORSF_FLAG_HAS_IMPORTANCE | AORSF_FLAG_HAS_METADATA, buffer, save_size * 2, &written));

    ASSERT_SUCCESS(aorsf_forest_load(&loaded, buffer, written));
    ASSERT_TRUE(aorsf_forest_has_feature_names(loaded) == 1);
    ASSERT_TRUE(aorsf_forest_has_feature_stats(loaded) == 1);

    /* Cleanup */
    free(buffer);
    aorsf_forest_destroy(loaded);
    aorsf_data_destroy(data);
    aorsf_forest_destroy(forest);
    free(x);
    free(y);

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
    test_serialization_binary();
    test_serialization_json();
    test_metadata();
    test_error_handling();

    printf("\n=== All tests passed! ===\n");
    return 0;
}
