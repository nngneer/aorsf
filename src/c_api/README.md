# aorsf C API

C API for Accelerated Oblique Random Forests, enabling integration with C#, Go, Rust, and other languages via FFI.

## Building

```bash
cd src/c_api
mkdir build && cd build
cmake ..
make
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_SHARED` | ON | Build shared library (vs static) |
| `BUILD_C_API_TESTS` | OFF | Build test executable |

### Dependencies

- Armadillo (matrix library)
- nlohmann/json (fetched automatically via CMake)
- OpenMP (optional, for parallelization)

## Quick Start

```c
#include "aorsf_c.h"

// Create configuration
aorsf_config_t config;
aorsf_config_init(&config, AORSF_TREE_CLASSIFICATION);
config.n_tree = 100;
config.vi_type = AORSF_VI_NEGATE;

// Create forest
aorsf_forest_handle forest;
aorsf_forest_create(&forest, &config);

// Create data (row-major arrays)
aorsf_data_handle data;
aorsf_data_create(&data, x, n_rows, n_cols, y, 1, NULL, n_class);

// Fit
aorsf_forest_fit(forest, data);

// Predict
double* predictions = malloc(n_rows * sizeof(double));
aorsf_forest_predict(forest, x_new, n_rows, n_cols,
                     AORSF_PRED_CLASS, predictions, n_rows);

// Cleanup
free(predictions);
aorsf_data_destroy(data);
aorsf_forest_destroy(forest);
```

## API Reference

### Forest Types

```c
typedef enum {
    AORSF_TREE_CLASSIFICATION = 1,
    AORSF_TREE_REGRESSION = 2,
    AORSF_TREE_SURVIVAL = 3
} aorsf_tree_type_t;
```

### Variable Importance Types

```c
typedef enum {
    AORSF_VI_NONE = 0,
    AORSF_VI_NEGATE = 1,    // Recommended
    AORSF_VI_PERMUTE = 2,
    AORSF_VI_ANOVA = 3
} aorsf_vi_type_t;
```

### Error Codes

```c
AORSF_SUCCESS           =  0   // Operation successful
AORSF_ERROR_NULL_POINTER = -1  // Null pointer argument
AORSF_ERROR_INVALID_ARGUMENT = -2
AORSF_ERROR_NOT_FITTED  = -3   // Forest not fitted
AORSF_ERROR_COMPUTATION = -4
AORSF_ERROR_OUT_OF_MEMORY = -5
AORSF_ERROR_IO          = -6   // File I/O error
AORSF_ERROR_FORMAT      = -7   // Invalid file format
```

### Lifecycle Functions

```c
// Initialize configuration with defaults
void aorsf_config_init(aorsf_config_t* config, aorsf_tree_type_t type);

// Create/destroy forest handle
aorsf_error_t aorsf_forest_create(aorsf_forest_handle* handle,
                                   const aorsf_config_t* config);
void aorsf_forest_destroy(aorsf_forest_handle handle);

// Create/destroy data handle
aorsf_error_t aorsf_data_create(aorsf_data_handle* handle,
                                 const double* x, int n_rows, int n_cols,
                                 const double* y, int n_y_cols,
                                 const double* weights, int n_class);
void aorsf_data_destroy(aorsf_data_handle handle);
```

### Training and Prediction

```c
// Fit forest to data
aorsf_error_t aorsf_forest_fit(aorsf_forest_handle handle,
                                aorsf_data_handle data);

// Check if fitted
int aorsf_forest_is_fitted(aorsf_forest_handle handle);

// Get prediction dimensions
aorsf_error_t aorsf_predict_get_dims(aorsf_forest_handle handle,
                                      int n_rows, aorsf_pred_type_t type,
                                      int* out_rows, int* out_cols);

// Predict on new data
aorsf_error_t aorsf_forest_predict(aorsf_forest_handle handle,
                                    const double* x_new, int n_rows, int n_cols,
                                    aorsf_pred_type_t type,
                                    double* predictions, int pred_size);
```

### Model Information

```c
int aorsf_forest_get_n_features(aorsf_forest_handle handle);
int aorsf_forest_get_n_tree(aorsf_forest_handle handle);
int aorsf_forest_get_n_class(aorsf_forest_handle handle);
aorsf_tree_type_t aorsf_forest_get_tree_type(aorsf_forest_handle handle);

// Get OOB evaluation metric
aorsf_error_t aorsf_forest_get_oob_error(aorsf_forest_handle handle,
                                          double* oob_error);

// Get variable importance (requires vi_type != NONE during fit)
aorsf_error_t aorsf_forest_get_importance(aorsf_forest_handle handle,
                                           double* importance, int size);
```

## Serialization

Models can be saved and loaded in binary or JSON format.

### Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `AORSF_FORMAT_BINARY` | Compact binary format | Production, fast |
| `AORSF_FORMAT_JSON` | Human-readable JSON | Debugging, inspection |

### Flags

Combine flags with `|` to control what's included:

```c
AORSF_FLAG_HAS_IMPORTANCE  // Include variable importance
AORSF_FLAG_HAS_OOB         // Include OOB data (for partial dependence)
AORSF_FLAG_HAS_METADATA    // Include feature names, means, stds
```

### Save/Load to Memory

```c
// Get required buffer size
size_t size;
aorsf_forest_get_save_size(forest, AORSF_FORMAT_BINARY,
                            AORSF_FLAG_HAS_IMPORTANCE, &size);

// Save to buffer
unsigned char* buffer = malloc(size);
size_t written;
aorsf_forest_save(forest, AORSF_FORMAT_BINARY,
                   AORSF_FLAG_HAS_IMPORTANCE, buffer, size, &written);

// Load from buffer (auto-detects format)
aorsf_forest_handle loaded;
aorsf_forest_load(&loaded, buffer, written);

free(buffer);
```

### Save/Load to File

```c
// Save to file
aorsf_forest_save_file(forest, "model.orsf",
                        AORSF_FORMAT_BINARY, AORSF_FLAG_HAS_IMPORTANCE);

// Load from file (auto-detects format)
aorsf_forest_handle loaded;
aorsf_forest_load_file(&loaded, "model.orsf");
```

## Metadata

Store feature names and normalization statistics with the model.

### Feature Names

```c
// Set feature names
const char* names[] = {"age", "weight", "height", "bmi"};
aorsf_forest_set_feature_names(forest, names, 4);

// Check if names are available
if (aorsf_forest_has_feature_names(forest)) {
    const char* retrieved[4];
    aorsf_forest_get_feature_names(forest, retrieved, 4);
}
```

### Feature Statistics

Store means and standard deviations for denormalization:

```c
double means[] = {45.0, 70.0, 175.0, 24.0};
double stds[] = {15.0, 15.0, 10.0, 4.0};
aorsf_forest_set_feature_stats(forest, means, stds, 4);

// Retrieve later
if (aorsf_forest_has_feature_stats(forest)) {
    double loaded_means[4], loaded_stds[4];
    aorsf_forest_get_feature_stats(forest, loaded_means, loaded_stds, 4);
}
```

### Serialization with Metadata

```c
// Save with all metadata
uint32_t flags = AORSF_FLAG_HAS_IMPORTANCE |
                 AORSF_FLAG_HAS_OOB |
                 AORSF_FLAG_HAS_METADATA;

aorsf_forest_save_file(forest, "model.orsf", AORSF_FORMAT_BINARY, flags);

// Load - metadata is automatically restored
aorsf_forest_handle loaded;
aorsf_forest_load_file(&loaded, "model.orsf");

// Feature names and stats are available
const char* names[4];
aorsf_forest_get_feature_names(loaded, names, 4);
```

## Survival-Specific Functions

```c
// Get unique event times
int n_times = 0;
aorsf_forest_get_unique_times(forest, NULL, &n_times);  // Get count
double* times = malloc(n_times * sizeof(double));
aorsf_forest_get_unique_times(forest, times, &n_times);  // Get values

// Predict survival probability at specific times
double* survival = malloc(n_rows * n_times * sizeof(double));
aorsf_forest_predict_survival(forest, x_new, n_rows, n_cols,
                               times, n_times, survival, n_rows * n_times);
```

## Error Handling

```c
aorsf_error_t err = aorsf_forest_fit(forest, data);
if (err != AORSF_SUCCESS) {
    const char* msg = aorsf_get_last_error();
    fprintf(stderr, "Error: %s\n", msg);
}
```

## File Format

### Binary Format

The binary format uses:
- Magic bytes: `ORSF`
- Version: 1.0
- 32-byte header with metadata
- Tree structures with nodes, coefficients, cutpoints
- Optional: importance, OOB data, metadata

### JSON Format

Human-readable JSON with:
- Format identifier: `"format": "aorsf"`
- Version info
- Tree type and configuration
- Full tree structure
- Optional: importance, OOB data, metadata

The loader auto-detects format (JSON starts with `{`, binary with `ORSF`).

## Running Tests

```bash
cd build
cmake -DBUILD_C_API_TESTS=ON ..
make
./tests/test_c_api
```

## Language Bindings

### C#

See `csharp/Aorsf/` for the .NET wrapper with:
- `ObliqueForestClassifier`
- `ObliqueForestRegressor`
- `ObliqueForestSurvival`

### Python

See `python/` for the Python wrapper using nanobind.

## License

MIT License - see LICENSE file for details.
