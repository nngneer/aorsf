#!/usr/bin/env python3
"""Test fit_forest function."""
import numpy as np
from pyaorsf import _pyaorsf

np.random.seed(42)
n_samples = 100
n_features = 5

# Create synthetic classification data
X = np.random.randn(n_samples, n_features).astype(np.float64)
y = (X[:, 0] + X[:, 1] > 0).astype(np.float64).reshape(-1, 1)
w = np.ones(n_samples, dtype=np.float64)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"y class distribution: {np.bincount(y.astype(int).flatten())}")

tree_seeds = list(range(1, 11))  # 10 trees

# Test classification with OOB
print("\n=== Test: Classification with OOB ===")
try:
    result = _pyaorsf.fit_forest(
        x=X, y=y, w=w,
        tree_type=_pyaorsf.TreeType.CLASSIFICATION.value,
        tree_seeds=tree_seeds,
        n_tree=10,
        mtry=3,
        sample_with_replacement=True,
        sample_fraction=0.632,
        vi_type=_pyaorsf.VariableImportance.NONE.value,
        vi_max_pvalue=0.01,
        leaf_min_events=1.0,
        leaf_min_obs=5.0,
        split_rule=_pyaorsf.SplitRule.GINI.value,
        split_min_events=1.0,
        split_min_obs=5.0,
        split_min_stat=0.0,
        split_max_cuts=5,
        split_max_retry=3,
        lincomb_type=_pyaorsf.LinearCombo.RANDOM.value,
        lincomb_eps=1e-9,
        lincomb_iter_max=20,
        lincomb_scale=True,
        lincomb_alpha=0.5,
        lincomb_df_target=0,
        lincomb_ties_method=0,
        pred_horizon=[],
        pred_type=_pyaorsf.PredType.PROBABILITY.value,
        oobag=True,
        oobag_eval_type=_pyaorsf.EvalType.NONE.value,
        oobag_eval_every=0,  # Should be auto-fixed to n_tree
        n_thread=1,
        verbose=0
    )
    print("SUCCESS!")
    print(f"  n_obs: {result['n_obs']}, n_features: {result['n_features']}")
    print(f"  n_tree: {result['n_tree']}, n_class: {result.get('n_class', 'N/A')}")
    if 'oob_predictions' in result:
        print(f"  OOB predictions shape: {result['oob_predictions'].shape}")
        # Calculate OOB accuracy
        oob_pred = result['oob_predictions']
        oob_class = np.argmax(oob_pred, axis=1)
        accuracy = np.mean(oob_class == y.flatten())
        print(f"  OOB accuracy: {accuracy:.3f}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")

# Test classification without OOB
print("\n=== Test: Classification without OOB ===")
try:
    result = _pyaorsf.fit_forest(
        x=X, y=y, w=w,
        tree_type=_pyaorsf.TreeType.CLASSIFICATION.value,
        tree_seeds=tree_seeds,
        n_tree=10,
        mtry=3,
        sample_with_replacement=True,
        sample_fraction=0.632,
        vi_type=_pyaorsf.VariableImportance.NONE.value,
        vi_max_pvalue=0.01,
        leaf_min_events=1.0,
        leaf_min_obs=5.0,
        split_rule=_pyaorsf.SplitRule.GINI.value,
        split_min_events=1.0,
        split_min_obs=5.0,
        split_min_stat=0.0,
        split_max_cuts=5,
        split_max_retry=3,
        lincomb_type=_pyaorsf.LinearCombo.RANDOM.value,
        lincomb_eps=1e-9,
        lincomb_iter_max=20,
        lincomb_scale=True,
        lincomb_alpha=0.5,
        lincomb_df_target=0,
        lincomb_ties_method=0,
        pred_horizon=[],
        pred_type=_pyaorsf.PredType.PROBABILITY.value,
        oobag=False,
        oobag_eval_type=_pyaorsf.EvalType.NONE.value,
        oobag_eval_every=0,
        n_thread=1,
        verbose=0
    )
    print("SUCCESS!")
    print(f"  Result keys: {list(result.keys())}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")

# Test regression
print("\n=== Test: Regression ===")
y_reg = X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1
y_reg = y_reg.reshape(-1, 1).astype(np.float64)
try:
    result = _pyaorsf.fit_forest(
        x=X, y=y_reg, w=w,
        tree_type=_pyaorsf.TreeType.REGRESSION.value,
        tree_seeds=tree_seeds,
        n_tree=10,
        mtry=3,
        sample_with_replacement=True,
        sample_fraction=0.632,
        vi_type=_pyaorsf.VariableImportance.NONE.value,
        vi_max_pvalue=0.01,
        leaf_min_events=1.0,
        leaf_min_obs=5.0,
        split_rule=_pyaorsf.SplitRule.VARIANCE.value,
        split_min_events=1.0,
        split_min_obs=5.0,
        split_min_stat=0.0,
        split_max_cuts=5,
        split_max_retry=3,
        lincomb_type=_pyaorsf.LinearCombo.RANDOM.value,
        lincomb_eps=1e-9,
        lincomb_iter_max=20,
        lincomb_scale=True,
        lincomb_alpha=0.5,
        lincomb_df_target=0,
        lincomb_ties_method=0,
        pred_horizon=[],
        pred_type=_pyaorsf.PredType.MEAN.value,
        oobag=True,
        oobag_eval_type=_pyaorsf.EvalType.NONE.value,
        oobag_eval_every=0,
        n_thread=1,
        verbose=0
    )
    print("SUCCESS!")
    print(f"  Result keys: {list(result.keys())}")
    if 'oob_predictions' in result:
        print(f"  OOB predictions shape: {result['oob_predictions'].shape}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")

# Test survival
print("\n=== Test: Survival ===")
time = np.random.exponential(10, n_samples)
status = np.random.binomial(1, 0.7, n_samples)

# Sort by time (required by aorsf)
sort_idx = np.argsort(time)
time = time[sort_idx]
status = status[sort_idx]
X_surv = X[sort_idx]
w_surv = w[sort_idx]

y_surv = np.column_stack([time, status]).astype(np.float64)
try:
    result = _pyaorsf.fit_forest(
        x=X_surv, y=y_surv, w=w_surv,
        tree_type=_pyaorsf.TreeType.SURVIVAL.value,
        tree_seeds=tree_seeds,
        n_tree=10,
        mtry=3,
        sample_with_replacement=True,
        sample_fraction=0.632,
        vi_type=_pyaorsf.VariableImportance.NONE.value,
        vi_max_pvalue=0.01,
        leaf_min_events=1.0,
        leaf_min_obs=5.0,
        split_rule=_pyaorsf.SplitRule.LOGRANK.value,
        split_min_events=1.0,
        split_min_obs=5.0,
        split_min_stat=0.0,
        split_max_cuts=5,
        split_max_retry=3,
        lincomb_type=_pyaorsf.LinearCombo.RANDOM.value,
        lincomb_eps=1e-9,
        lincomb_iter_max=20,
        lincomb_scale=True,
        lincomb_alpha=0.5,
        lincomb_df_target=0,
        lincomb_ties_method=0,
        pred_horizon=[5.0, 10.0],  # Prediction horizons for survival
        pred_type=_pyaorsf.PredType.SURVIVAL.value,
        oobag=True,
        oobag_eval_type=_pyaorsf.EvalType.NONE.value,
        oobag_eval_every=0,
        n_thread=1,
        verbose=0
    )
    print("SUCCESS!")
    print(f"  Result keys: {list(result.keys())}")
    if 'oob_predictions' in result:
        print(f"  OOB predictions shape: {result['oob_predictions'].shape}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")

print("\n=== All tests complete ===")
