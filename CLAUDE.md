# CLAUDE.md

## Package Overview

**Package:** aorsf (Accelerated Oblique Random Forests)  
**Version:** 0.1.6  
**Purpose:** Fit, interpret, and predict with oblique random forests. Specializes in oblique random survival forests using accelerated Cox regression.

**Key Innovation:** Uses oblique (linear combination) splits instead of axis-based splits, resulting in more efficient trees and smoother decision boundaries.

## Architecture

```
aorsf/
├── R/           # User interface, coordination
├── src/         # C++ computational engine (tree growing, splits)
├── man/         # Documentation (auto-generated)
└── tests/       # testthat unit tests
```

**Design Pattern:** C++ handles computation-intensive operations (tree growing, matrix operations), R provides user interface and high-level coordination.

## Core Concepts

### Forest Types
- **Classification**: Predict categorical outcomes (uses accelerated logistic regression)
- **Regression**: Predict continuous outcomes (uses accelerated linear regression)
- **Survival**: Predict time-to-event with censoring (uses accelerated Cox regression)

### Key Features
- **Oblique splits**: Linear combinations of variables (not single variables)
- **Variable importance**: Three methods - negation (new/fast), permutation (standard), ANOVA (efficient)
- **Partial dependence**: Marginal effects with `orsf_pd_oob()` and `orsf_pd_inb()`
- **Interaction scores**: `orsf_vint()` detects variable interactions
- **OOB evaluation**: Unbiased performance estimates without separate validation set

### Main Functions
- `orsf()` - Fit oblique random forest
- `orsf_update()` - Modify fitted forest efficiently
- `predict()` - Generate predictions
- `orsf_vi_*()` - Variable importance (negate, permute, anova)
- `orsf_pd_*()` - Partial dependence (oob, inb)
- `orsf_vint()` - Interaction scores
- `orsf_summarize_uni()` - Summary statistics

## Development Workflow

```r
# Standard workflow
devtools::load_all()        # Compile C++ and load
devtools::test()            # Run tests
devtools::check()           # R CMD check

# After C++ changes
Rcpp::compileAttributes()   # CRITICAL: Update exports
devtools::load_all()        # Reload

# Documentation
devtools::document()        # Update .Rd files from roxygen2
```

## R Package Best Practices

### Documentation
- Complete roxygen2 for all exported functions (`@param`, `@return`, `@examples`)
- Use `@seealso` and `@family` for related functions
- Keep README concise, use vignettes for depth
- Document all dataset variables

### Function Design
- Consistent naming: `orsf_*` prefix for main functions
- Sensible defaults for all optional arguments
- Validate inputs early with clear error messages: `stop(msg, call. = FALSE)`
- Return consistent types; use S3 classes for complex objects

### Testing
```r
test_that("function handles edge cases", {
  expect_error(orsf(bad_input), "specific error message")
  expect_s3_class(result, "orsf")
  expect_equal(result$n_tree, 500)
})
```
- Test edge cases: empty data, single row, all NA, etc.
- One test per logical unit
- Use `skip_on_cran()` for slow tests

### Dependencies
- Minimize dependencies (only add when necessary)
- Use `Imports:` for direct dependencies, `Suggests:` for tests/vignettes
- `LinkingTo:` for C++ headers (Rcpp, RcppArmadillo)

### NAMESPACE
- Only export user-facing functions (use `@export` in roxygen2)
- Import only specific functions needed: `@importFrom pkg func`
- Let roxygen2 manage NAMESPACE (don't edit manually)

## C++ Best Practices

### Memory Management
```cpp
// Use smart pointers (automatic cleanup)
std::unique_ptr<TreeNode> node = std::make_unique<TreeNode>();

// Pass large objects by const reference
void process_data(const arma::mat& X) { }

// Armadillo objects have automatic memory management
arma::mat X(100, 50);  // No manual delete needed
```

### Armadillo Efficiency
```cpp
// Use subviews (no copy)
arma::vec col = X.col(j);
X.col(j) *= 2.0;

// Pre-allocate when size known
arma::mat results(n_rows, n_cols);

// Column-major iteration (cache-friendly)
for (size_t j = 0; j < X.n_cols; ++j) {
  for (size_t i = 0; i < X.n_rows; ++i) {
    process(X(i, j));
  }
}
```

### Rcpp Interface
```cpp
// [[Rcpp::export]]
arma::vec rcpp_function(arma::mat X, arma::vec y) {
  // Types auto-converted
  try {
    return arma::solve(X, y);
  } catch (std::exception& e) {
    Rcpp::stop("Error: " + std::string(e.what()));
  }
}

// Return complex objects as Rcpp::List
return Rcpp::List::create(
  Rcpp::Named("coef") = coefficients,
  Rcpp::Named("fitted") = fitted_values
);
```

### Code Quality
```cpp
// Const correctness
size_t get_depth() const { return depth; }  // Doesn't modify object

// Meaningful names
void grow_trees(const arma::mat& training_data, int n_trees);

// Initialize variables
int n_nodes = 0;  // Not: int n_nodes;

// Modern C++ features
auto result = compute_value();
for (const auto& node : nodes) { process(node); }
```

### Common Pitfalls
- **Integer division**: `double avg = n_correct / static_cast<double>(n_total);`
- **Dangling references**: Return by value, not reference to local variable
- **Off-by-one**: Armadillo uses 0-based indexing
- **Uninitialized variables**: Always initialize

## Package-Specific Notes

### Survival Analysis
- Requires `Surv(time, status)` in formula
- OOB metric: Harrell's C-index (default)
- Predictions are risk scores (higher = higher risk)
- Handles right-censored data

### Key Parameters
- `n_tree`: Number of trees (500 recommended)
- `mtry`: Predictors per split (1 = axis-based, >1 = oblique)
- `leaf_min_events`: Minimum events in leaf (survival only)
- `tree_seeds`: For reproducibility

### Variable Importance
- **Negation**: Fast, emphasizes coefficients in linear combinations (newer)
- **Permutation**: Standard, flexible, has known limitations
- **ANOVA**: Most efficient, based on p-values of coefficients

### Performance Tips
- Use `n_thread` for parallelization
- For large datasets: reduce `n_tree` initially, increase for final model
- C++ handles computation, R handles interface/validation

## CRAN Submission Checklist
- [ ] `devtools::check()` with 0 errors, 0 warnings, 0 notes
- [ ] Update version in DESCRIPTION and NEWS.md
- [ ] Examples run in < 5 seconds (use `\donttest{}` if needed)
- [ ] Test on win-builder and rhub
- [ ] No C++ compilation warnings
- [ ] All URLs valid

## Common Issues

**C++ Compilation**
- Always run `Rcpp::compileAttributes()` after modifying exports
- Check for memory leaks with large datasets
- Enable warnings: Add `PKG_CXXFLAGS = -Wall` to src/Makevars

**Testing**
- OOB predictions require careful handling
- Test with edge cases: heavy censoring, small n, all missing
- Survival tests need `Surv()` objects

**Performance**
- Profile with `profvis::profvis()` to find bottlenecks
- Move tight loops to C++
- Use Armadillo for matrix operations (not manual loops)

## AI Assistant Guidelines

When helping with aorsf:

1. **After C++ changes**: Always remind to run `Rcpp::compileAttributes()` then `devtools::load_all()`
2. **Survival formulas**: Must use `Surv(time, status)` from survival package
3. **OOB vs in-bag**: Clarify which data is being used (OOB = unbiased)
4. **Variable importance**: Explain trade-offs between methods when asked
5. **Memory**: Watch for Armadillo subviews becoming invalid
6. **Reproducibility**: Suggest `tree_seeds` for consistent results
7. **Examples**: Keep `n_tree` small (5-10) in examples for speed

### Common User Questions
- **"Why oblique?"** → Fewer splits needed, smoother boundaries, better separation
- **"Which VI method?"** → Negation (new/fast), permutation (standard/flexible), ANOVA (efficient)
- **"What's accelerated?"** → Fast approximations to regression, not full GLM fits
- **"OOB error?"** → Unbiased estimate without separate validation set

## References

**Primary Paper:** Jaeger et al. (2023). "Accelerated and interpretable oblique random survival forests." *Journal of Computational and Graphical Statistics*. DOI: 10.1080/10618600.2023.2231048

**Documentation:** https://docs.ropensci.org/aorsf/